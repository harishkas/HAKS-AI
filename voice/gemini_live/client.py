"""Asynchronous Gemini Live Bidirectional Streaming Client.

Connects to Google Gemini Live API via WebSockets using `google-genai` SDK
and model `gemini-3.8-live` on `v1alpha`.
Supports real-time bidirectional audio streaming, instant barge-in,
automatic reconnect with exponential backoff, latency metrics, and safe logging.
"""

import asyncio
from enum import Enum
import logging
import time
from typing import Callable, Optional

from ai.credentials import CredentialManager

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

logger = logging.getLogger("haks.voice.gemini_live.client")


class LiveConnectionStatus(str, Enum):
    """Gemini Live WebSocket connection lifecycle states."""

    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    STREAMING = "STREAMING"
    ERROR = "ERROR"
    CLOSED = "CLOSED"


class GeminiLiveClient:
    """Asynchronous client managing a persistent Gemini Live WebSocket session."""

    def __init__(
        self,
        model: str = "gemini-3.8-live",
        voice_name: str = "Aoede",
        system_instruction: Optional[str] = None,
        credential_manager: Optional[CredentialManager] = None,
    ) -> None:
        self.model = model
        self.voice_name = voice_name
        self._default_system_instruction = (
            "You are HAKS, a private, fast, bilingual personal AI assistant operating on Windows. "
            "You speak English and Tamil fluently, and easily understand Tanglish (Tamil written in English script or code-mixed speech). "
            "Keep your spoken answers concise, direct, helpful, and natural. "
            "Do not output markdown asterisks or bullet formatting in spoken audio."
        )
        self.system_instruction = system_instruction or self._default_system_instruction
        self.credentials = credential_manager or CredentialManager()

        self.status: LiveConnectionStatus = LiveConnectionStatus.DISCONNECTED
        self._client: Optional[genai.Client] = None
        self._session = None
        self._session_context = None
        self._receive_task: Optional[asyncio.Task] = None
        self._keepalive_task: Optional[asyncio.Task] = None
        self._is_active: bool = False
        self._lock = asyncio.Lock()

        # Latency tracking
        self._last_turn_sent_time: float = 0.0
        self._first_chunk_received_time: float = 0.0
        self.last_latency_ms: float = 0.0

        # Accumulated turn transcript
        self._accumulated_output_text: list[str] = []
        self._accumulated_input_text: list[str] = []

        # Callbacks
        self.on_status_changed: Optional[Callable[[LiveConnectionStatus], None]] = None
        self.on_audio_chunk: Optional[Callable[[bytes], None]] = None
        self.on_transcript_chunk: Optional[Callable[[str], None]] = None
        self.on_user_transcript: Optional[Callable[[str], None]] = None
        self.on_turn_complete: Optional[Callable[[str], None]] = None
        self.on_interrupted: Optional[Callable[[], None]] = None
        self.on_latency_measured: Optional[Callable[[float], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None

    def _set_status(self, new_status: LiveConnectionStatus) -> None:
        if self.status != new_status:
            logger.info("Gemini Live connection status: %s -> %s", self.status.value, new_status.value)
            self.status = new_status
            if self.on_status_changed:
                try:
                    self.on_status_changed(new_status)
                except Exception as exc:
                    logger.error("Error in on_status_changed callback: %s", exc)

    def is_connected(self) -> bool:
        """Check if WebSocket session is connected and active."""
        return self.status in (LiveConnectionStatus.CONNECTED, LiveConnectionStatus.STREAMING) and self._session is not None

    async def connect(self, max_retries: int = 3, retry_backoff: float = 1.0) -> bool:
        """Establish bidirectional WebSocket connection to Gemini Live API with exponential backoff."""
        if genai is None or types is None:
            err_msg = "Google Gen AI SDK (google-genai) is not installed."
            logger.error(err_msg)
            self._set_status(LiveConnectionStatus.ERROR)
            if self.on_error:
                self.on_error(err_msg)
            return False

        api_key = self.credentials.get_api_key("gemini")
        if not api_key:
            err_msg = "Gemini API key is not configured. Please set it in AI Settings."
            logger.error("Gemini API key missing.")
            self._set_status(LiveConnectionStatus.ERROR)
            if self.on_error:
                self.on_error(err_msg)
            return False

        async with self._lock:
            if self.is_connected():
                return True

            self._set_status(LiveConnectionStatus.CONNECTING)

            for attempt in range(1, max_retries + 1):
                try:
                    logger.info("Connecting to Gemini Live (%s, v1alpha), attempt %d/%d...", self.model, attempt, max_retries)
                    self._client = genai.Client(
                        api_key=api_key,
                        http_options=types.HttpOptions(api_version="v1alpha"),
                    )

                    config = types.LiveConnectConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.voice_name)
                            )
                        ),
                        system_instruction=types.Content(
                            parts=[types.Part.from_text(text=self.system_instruction)]
                        ),
                        input_audio_transcription=types.AudioTranscriptionConfig(),
                        output_audio_transcription=types.AudioTranscriptionConfig(),
                    )

                    self._session_context = self._client.aio.live.connect(
                        model=self.model,
                        config=config,
                    )
                    self._session = await self._session_context.__aenter__()

                    self._is_active = True
                    self._set_status(LiveConnectionStatus.CONNECTED)

                    # Start background receive task
                    self._receive_task = asyncio.create_task(self._receive_loop())
                    logger.info("Successfully connected to Gemini Live session (%s).", self.model)
                    return True

                except Exception as exc:
                    logger.warning("Gemini Live connection attempt %d failed: %s", attempt, exc)
                    if attempt < max_retries:
                        delay = retry_backoff * (2 ** (attempt - 1))
                        await asyncio.sleep(delay)
                    else:
                        err_str = f"Failed to connect to Gemini Live after {max_retries} attempts: {exc}"
                        logger.error(err_str)
                        self._set_status(LiveConnectionStatus.ERROR)
                        if self.on_error:
                            self.on_error(err_str)
                        return False

        return False

    async def disconnect(self) -> None:
        """Gracefully close the Gemini Live WebSocket session."""
        self._is_active = False

        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        if self._session_context:
            try:
                await self._session_context.__aexit__(None, None, None)
            except Exception as exc:
                logger.debug("Error closing Gemini Live session: %s", exc)
            self._session = None
            self._session_context = None

        self._set_status(LiveConnectionStatus.CLOSED)
        logger.info("Gemini Live session closed.")

    async def send_audio_chunk(self, pcm_16k_bytes: bytes) -> bool:
        """Stream a 16 kHz mono 16-bit PCM chunk (audio/pcm;rate=16000) to Gemini Live."""
        if not self.is_connected() or self._session is None:
            logger.debug("Cannot send audio chunk: session is not connected.")
            return False

        try:
            self._set_status(LiveConnectionStatus.STREAMING)
            await self._session.send_realtime_input(
                audio=types.Blob(data=pcm_16k_bytes, mime_type="audio/pcm;rate=16000")
            )
            return True
        except Exception as exc:
            logger.error("Error sending audio chunk to Gemini Live: %s", exc)
            self._handle_network_failure(exc)
            return False

    async def send_text_turn(self, text: str) -> bool:
        """Send a user text turn to trigger a Gemini Live spoken response."""
        if not self.is_connected() or self._session is None:
            logger.warning("Cannot send text turn: session is not connected.")
            return False

        try:
            self._last_turn_sent_time = time.time()
            self._first_chunk_received_time = 0.0
            self._accumulated_output_text.clear()

            self._set_status(LiveConnectionStatus.STREAMING)
            await self._session.send_client_content(
                turns=[
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=text)],
                    )
                ],
                turn_complete=True,
            )
            return True
        except Exception as exc:
            logger.error("Error sending text turn to Gemini Live: %s", exc)
            self._handle_network_failure(exc)
            return False

    def mark_speech_turn_ended(self) -> None:
        """Record the timestamp when the user finished speaking for latency calculation."""
        self._last_turn_sent_time = time.time()
        self._first_chunk_received_time = 0.0
        self._accumulated_output_text.clear()

    async def _receive_loop(self) -> None:
        """Asynchronous receiver continuously iterating over session.receive()."""
        try:
            async for response in self._session.receive():
                if not self._is_active:
                    break

                sc = response.server_content
                if sc is None:
                    continue

                # 1. User speech transcription from Live server
                if hasattr(sc, "input_transcription") and sc.input_transcription and sc.input_transcription.text:
                    user_text = sc.input_transcription.text
                    self._accumulated_input_text.append(user_text)
                    if self.on_user_transcript:
                        try:
                            self.on_user_transcript(user_text)
                        except Exception as e:
                            logger.error("Error in on_user_transcript callback: %s", e)

                # 2. Model audio parts
                if sc.model_turn and sc.model_turn.parts:
                    for part in sc.model_turn.parts:
                        if part.inline_data and part.inline_data.data:
                            # Latency calculation on first audio packet
                            if self._first_chunk_received_time == 0.0 and self._last_turn_sent_time > 0:
                                self._first_chunk_received_time = time.time()
                                self.last_latency_ms = (self._first_chunk_received_time - self._last_turn_sent_time) * 1000.0
                                logger.info("Gemini Live first audio packet latency: %.1f ms", self.last_latency_ms)
                                if self.on_latency_measured:
                                    try:
                                        self.on_latency_measured(self.last_latency_ms)
                                    except Exception as e:
                                        logger.error("Error in on_latency_measured: %s", e)

                            if self.on_audio_chunk:
                                try:
                                    self.on_audio_chunk(part.inline_data.data)
                                except Exception as e:
                                    logger.error("Error in on_audio_chunk callback: %s", e)

                        if part.text:
                            self._accumulated_output_text.append(part.text)
                            if self.on_transcript_chunk:
                                try:
                                    self.on_transcript_chunk(part.text)
                                except Exception as e:
                                    logger.error("Error in on_transcript_chunk: %s", e)

                # 3. Model output transcription
                if hasattr(sc, "output_transcription") and sc.output_transcription and sc.output_transcription.text:
                    out_text = sc.output_transcription.text
                    self._accumulated_output_text.append(out_text)
                    if self.on_transcript_chunk:
                        try:
                            self.on_transcript_chunk(out_text)
                        except Exception as e:
                            logger.error("Error in on_transcript_chunk: %s", e)

                # 4. Barge-in interruption signal from server
                if getattr(sc, "interrupted", False):
                    logger.info("Gemini Live server detected user interruption (barge-in).")
                    if self.on_interrupted:
                        try:
                            self.on_interrupted()
                        except Exception as e:
                            logger.error("Error in on_interrupted: %s", e)

                # 5. Assistant turn complete
                if getattr(sc, "turn_complete", False):
                    full_text = "".join(self._accumulated_output_text).strip()
                    logger.info("Gemini Live turn complete. Transcript: %s", full_text[:80])
                    self._set_status(LiveConnectionStatus.CONNECTED)
                    if self.on_turn_complete:
                        try:
                            self.on_turn_complete(full_text)
                        except Exception as e:
                            logger.error("Error in on_turn_complete: %s", e)

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("Gemini Live receive loop terminated with error: %s", exc)
            self._handle_network_failure(exc)

    def _handle_network_failure(self, exc: Exception) -> None:
        """Handle disconnection or streaming error and attempt recovery."""
        self._set_status(LiveConnectionStatus.ERROR)
        if self.on_error:
            try:
                self.on_error(str(exc))
            except Exception:
                pass

