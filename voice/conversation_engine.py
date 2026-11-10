"""Continuous Bilingual Voice Assistant Engine for HAKS.

Implements the complete Phase 6 state machine:
STANDBY -> WAKE_DETECTED -> LISTENING -> PROCESSING -> THINKING -> SPEAKING -> (INTERRUPTED / SLEEP / STOPPED)

Integrates:
- Audio capture via sounddevice
- VAD speech segmentation
- 'Hey HAKS' wake word detection (Local acoustic & Porcupine)
- Faster-Whisper bilingual STT (Tamil, English, Mixed)
- Phase 5 AI Routing & Privacy Policy Enforcement
- Phase 4 Memory Retrieval (approved memories only)
- Phase 3 AI Response Generation
- Edge / SAPI Bilingual TTS Synthesis
- Real-time barge-in speech interruption & acoustic loopback suppression
- Voice command parsing ('Stop HAKS', 'Sleep HAKS', 'Wake up', 'Repeat')
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import re
import threading
import time
from typing import TYPE_CHECKING, Optional
import numpy as np

from config.settings import AppConfig, get_settings
from voice.audio_capture import AudioCaptureService
from voice.models import (
    TranscriptionResult,
    VoiceCommandType,
    VoiceState,
)
from voice.player import AudioPlayerService
from voice.echo_cancellation import AcousticEchoSuppressor
from voice.gemini_live import (
    AudioStreamerBridge,
    GeminiLiveClient,
    LiveConnectionStatus,
    PCMStreamPlayer,
)
from voice.stt.faster_whisper_stt import FasterWhisperSTT
from voice.tts.router import LanguageAwareTTSRouter
from voice.vad import VoiceActivityDetector
from voice.wake_word.local_detector import LocalAcousticWakeDetector
from voice.wake_word.porcupine_detector import PorcupineWakeDetector

if TYPE_CHECKING:
    from core.app_controller import AppController

logger = logging.getLogger("haks.voice.engine")

# Voice command recognition patterns in English and Tamil
COMMAND_PATTERNS = [
    (
        VoiceCommandType.STOP,
        [
            re.compile(r"\b(stop haks|stop assistant)\b", re.IGNORECASE),
            re.compile(r"(?:^|[\s.,!?;:])(நிறுத்து|நில்)(?:$|[\s.,!?;:])"),
        ],
    ),
    (
        VoiceCommandType.SLEEP,
        [
            re.compile(r"\b(sleep haks|go to sleep)\b", re.IGNORECASE),
            re.compile(r"(?:^|[\s.,!?;:])(தூங்கு|ஓய்வெடு)(?:$|[\s.,!?;:])"),
        ],
    ),
    (
        VoiceCommandType.WAKE,
        [
            re.compile(r"\b(wake up haks|wake up)\b", re.IGNORECASE),
            re.compile(r"(?:^|[\s.,!?;:])(எழுந்திரு|விழி)(?:$|[\s.,!?;:])"),
        ],
    ),
    (
        VoiceCommandType.CANCEL,
        [
            re.compile(r"\b(cancel|cancel request)\b", re.IGNORECASE),
            re.compile(r"(?:^|[\s.,!?;:])(ரத்து செய்|வேண்டாம்)(?:$|[\s.,!?;:])"),
        ],
    ),
    (
        VoiceCommandType.REPEAT,
        [
            re.compile(r"\b(repeat|say that again)\b", re.IGNORECASE),
            re.compile(r"(?:^|[\s.,!?;:])(மீண்டும் சொல்|மறுபடியும் சொல்)(?:$|[\s.,!?;:])"),
        ],
    ),
    (
        VoiceCommandType.STOP_SPEAKING,
        [
            re.compile(r"\b(stop speaking|be quiet|shut up)\b", re.IGNORECASE),
            re.compile(r"(?:^|[\s.,!?;:])(பேசாதே|பேசுவதை நிறுத்து)(?:$|[\s.,!?;:])"),
        ],
    ),
]


class ContinuousVoiceEngine:
    """Central Voice Assistant coordinator and state machine."""

    def __init__(
        self,
        controller: Optional["AppController"] = None,
        config: Optional[AppConfig] = None,
    ) -> None:
        self.config = config or get_settings()
        self.controller = controller

        # Audio Capture & Unified DSP Pipeline (DC norm, Rumble HPF, Hum notch, AEC, NS, Enhancer, VAD)
        self.audio_capture = AudioCaptureService(self.config)
        self.echo_suppressor = self.audio_capture.pipeline.echo_canceller

        self.vad = VoiceActivityDetector(
            sample_rate=self.config.voice.input_sample_rate,
            silence_timeout_ms=self.config.voice.silence_timeout_ms,
            min_speech_duration_ms=self.config.voice.min_speech_duration_ms,
            pre_speech_padding_ms=self.config.voice.pre_roll_ms,
            vad_sensitivity=self.config.voice.vad_sensitivity,
            echo_suppressor=self.echo_suppressor,
        )
        self.local_wake_detector = LocalAcousticWakeDetector(
            sensitivity=self.config.voice.wake_sensitivity
        )
        self.porcupine_wake_detector = PorcupineWakeDetector(
            access_key=self.config.voice.porcupine_access_key,
            keyword_path=self.config.voice.porcupine_keyword_path,
            sensitivity=self.config.voice.wake_sensitivity,
        )
        self.stt = FasterWhisperSTT(self.config)
        self.tts = LanguageAwareTTSRouter(self.config)
        self.player = AudioPlayerService(
            output_device_index=self.config.voice.output_device_index,
            echo_suppressor=self.echo_suppressor,
        )

        # Gemini Live native real-time speech-to-speech subsystems
        self.gemini_live_client = GeminiLiveClient(
            model=self.config.voice.gemini_live_model,
            voice_name=self.config.voice.gemini_live_voice,
        )
        self.pcm_player = PCMStreamPlayer(
            output_device_index=self.config.voice.output_device_index,
            echo_suppressor=self.echo_suppressor,
        )
        self.audio_streamer = AudioStreamerBridge(
            client=self.gemini_live_client
        )

        # State machine
        self._state: VoiceState = VoiceState.STOPPED
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._last_user_transcript: str = ""
        self._last_spoken_response: str = ""
        self._conversation_turn_active: bool = False
        self._last_turn_time: float = 0.0

        # Latency metrics
        self.last_speech_detection_latency_ms: float = 0.0
        self.last_interruption_latency_ms: float = 0.0
        self.last_gemini_response_latency_ms: float = 0.0
        self.last_total_turnaround_latency_ms: float = 0.0
        self._speech_onset_time: float = 0.0

        # Background async worker
        self._async_loop: Optional[asyncio.AbstractEventLoop] = None
        self._async_thread: Optional[threading.Thread] = None
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="voice_worker")

        # Wire internal callbacks
        self._wire_pipeline()

    @property
    def state(self) -> VoiceState:
        return self._state

    def set_controller(self, controller: "AppController") -> None:
        """Attach AppController reference."""
        self.controller = controller

    def _set_state(self, new_state: VoiceState) -> None:
        """Thread-safe transition of voice state and signal emission."""
        with self._state_lock:
            old_state = self._state
            if old_state == new_state:
                return
            self._state = new_state

        try:
            logger.info("Voice State: %s -> %s", old_state.value, new_state.value)
        except Exception:
            pass
        if self.controller and hasattr(self.controller, "events"):
            try:
                self.controller.events.voice_state_changed.emit(new_state.value)
            except Exception:
                pass

    def _wire_pipeline(self) -> None:
        """Connect audio capture chunks to VAD, wake word detector, and player."""
        self.audio_capture.add_chunk_callback(self._on_audio_chunk)
        self.audio_capture.add_level_callback(self._on_audio_level)
        self.audio_capture.add_confidence_callback(self._on_speech_confidence)

        self.vad.on_speech_started = self._on_vad_speech_started
        self.vad.on_speech_ended = self._on_vad_speech_ended

        self.player.on_playback_started = self._on_player_started
        self.player.on_playback_finished = self._on_player_finished
        self.player.on_playback_interrupted = self._on_player_interrupted

        # Gemini Live & PCM Stream callbacks
        self.gemini_live_client.on_status_changed = self._on_live_status_changed
        self.gemini_live_client.on_latency_measured = self._on_live_latency_measured
        self.gemini_live_client.on_audio_chunk = self._on_live_audio_chunk
        self.gemini_live_client.on_transcript_chunk = self._on_live_transcript_chunk
        self.gemini_live_client.on_user_transcript = self._on_live_user_transcript
        self.gemini_live_client.on_turn_complete = self._on_live_turn_complete
        self.gemini_live_client.on_interrupted = self._on_live_interrupted
        self.gemini_live_client.on_error = self._on_live_error

        self.pcm_player.on_playback_started = self._on_player_started
        self.pcm_player.on_playback_finished = self._on_pcm_playback_finished
        self.pcm_player.on_playback_interrupted = self._on_player_interrupted

    # --------------------------------------------------------------------------
    # Audio Stream Processing & Interruption Monitoring
    # --------------------------------------------------------------------------

    def _on_audio_chunk(self, chunk: np.ndarray) -> None:
        """Executed per 32ms audio frame (512 samples float32)."""
        current_state = self._state

        if current_state == VoiceState.STOPPED:
            return

        # 1. State: SLEEP — Only listen for 'Wake up' commands
        if current_state == VoiceState.SLEEP:
            return

        # 2. State: STANDBY — Feed to wake word detectors
        if current_state == VoiceState.STANDBY:
            # Check configured wake detector
            is_wake = False
            if self.config.voice.wake_word_engine == "porcupine" and self.porcupine_wake_detector.is_available():
                is_wake = self.porcupine_wake_detector.process_frame(chunk)
            else:
                is_wake = self.local_wake_detector.process_frame(chunk)

            if is_wake:
                self._handle_wake_word_detected()
            return

        # 3. State: SPEAKING — Check for Barge-in Interruption using Echo-Cancelled VAD
        if current_state == VoiceState.SPEAKING:
            if self.config.voice.voice_mode == "gemini_live":
                # Feed frame into streamer (held safely in pre-roll during playback)
                self.audio_streamer.feed_frame(chunk)

            if self.config.voice.enable_barge_in:
                t_eval_start = time.perf_counter()
                is_human, reason = self.vad.is_confirmed_human_speech(
                    chunk,
                    check_echo=self.config.voice.enable_echo_cancellation,
                )
                if is_human:
                    t_eval_end = time.perf_counter()
                    self.last_speech_detection_latency_ms = (t_eval_end - t_eval_start) * 1000.0
                    logger.info("Confirmed human speech barge-in (%s)! Halting HAKS playback.", reason)
                    self._handle_barge_in_interruption()
            return

        # 4. State: LISTENING — Feed to VAD to detect user speech turn
        if current_state == VoiceState.LISTENING:
            if self.config.voice.voice_mode == "gemini_live":
                self.audio_streamer.feed_frame(chunk)

            self.vad.process_frame(chunk)

            # Check continuous conversation timeout (e.g. 45s of idling in LISTENING)
            if self.config.voice.continuous_mode and self._conversation_turn_active:
                if time.time() - self._last_turn_time > self.config.voice.conversation_timeout_seconds:
                    logger.info("Continuous conversation timeout. Returning to STANDBY.")
                    self._conversation_turn_active = False
                    if self.config.voice.voice_mode == "gemini_live":
                        self.audio_streamer.stop_streaming()
                    self._set_state(VoiceState.STANDBY)

    def _on_audio_level(self, level: float) -> None:
        """Broadcast live VU meter level to UI."""
        if self.controller and hasattr(self.controller, "events"):
            try:
                self.controller.events.voice_audio_level_updated.emit(level)
            except Exception:
                pass

    def _on_speech_confidence(self, confidence: float) -> None:
        """Broadcast live speech confidence (0.0 to 1.0) to UI."""
        if self.controller and hasattr(self.controller, "events") and hasattr(self.controller.events, "voice_speech_confidence_updated"):
            try:
                self.controller.events.voice_speech_confidence_updated.emit(confidence)
            except Exception:
                pass

    # --------------------------------------------------------------------------
    # VAD Event Handlers
    # --------------------------------------------------------------------------

    def _on_vad_speech_started(self) -> None:
        """VAD confirmed user began speaking."""
        self._speech_onset_time = time.time()
        if self._state == VoiceState.LISTENING:
            logger.info("VAD: User speech initiated.")
            if self.controller and hasattr(self.controller, "events"):
                self.controller.events.voice_speech_started.emit()

    def _on_vad_speech_ended(self, speech_audio: np.ndarray) -> None:
        """VAD confirmed user finished speaking turn."""
        if self._state != VoiceState.LISTENING:
            return

        speech_duration_sec = len(speech_audio) / 16000.0
        logger.info("VAD: User speech completed (%.2fs audio).", speech_duration_sec)

        # In Gemini Live mode, mark speech end for latency metrics
        if self.config.voice.voice_mode == "gemini_live" and self.gemini_live_client.is_connected():
            self.gemini_live_client.mark_speech_turn_ended()
            self._set_state(VoiceState.THINKING)
            return

        self._set_state(VoiceState.PROCESSING)
        if self.controller and hasattr(self.controller, "events"):
            self.controller.events.voice_speech_ended.emit()

        # Schedule async processing for pipeline/offline modes or when Gemini Live is offline
        if self._async_loop and self._async_loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._process_speech_turn(speech_audio),
                self._async_loop,
            )

    # --------------------------------------------------------------------------
    # Wake & Interruption Handlers
    # --------------------------------------------------------------------------

    def _handle_wake_word_detected(self) -> None:
        """Transition from STANDBY to WAKE_DETECTED then LISTENING."""
        logger.info("Wake word detected! Activating conversation channel.")
        self._set_state(VoiceState.WAKE_DETECTED)
        self._conversation_turn_active = True
        self._last_turn_time = time.time()
        self.vad.reset()

        if self.config.voice.voice_mode == "gemini_live":
            if self._async_loop and self._async_loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._ensure_gemini_live_ready(),
                    self._async_loop,
                )

        # Automatically transition to LISTENING after 300ms
        def _activate_listening():
            if not self._stop_event.wait(0.3) and not self._stop_event.is_set():
                if self._state == VoiceState.WAKE_DETECTED:
                    self._set_state(VoiceState.LISTENING)
                    if self.config.voice.voice_mode == "gemini_live":
                        self.audio_streamer.start_streaming(self._async_loop)

        self._executor.submit(_activate_listening)

    def _handle_barge_in_interruption(self) -> None:
        """User spoke while HAKS was speaking. Immediately stop playback."""
        t_stop_start = time.perf_counter()
        self._set_state(VoiceState.INTERRUPTED)
        self.player.stop()
        self.pcm_player.stop()
        self.audio_streamer.set_playback_active(False)
        self.audio_streamer.flush_held_frames()
        t_stop_end = time.perf_counter()
        self.last_interruption_latency_ms = (t_stop_end - t_stop_start) * 1000.0
        logger.info("Barge-in playback stop latency: %.2f ms", self.last_interruption_latency_ms)

        if self.controller and hasattr(self.controller, "events"):
            self.controller.events.voice_interrupted.emit()

        self.vad.reset()
        self._last_turn_time = time.time()
        # Return to listening after short debounce
        def _back_to_listen():
            if not self._stop_event.wait(0.08) and not self._stop_event.is_set():
                if self._state == VoiceState.INTERRUPTED:
                    self._set_state(VoiceState.LISTENING)
                    if self.config.voice.voice_mode == "gemini_live":
                        self.audio_streamer.start_streaming(self._async_loop)

        self._executor.submit(_back_to_listen)

    def _on_player_started(self) -> None:
        if self._state == VoiceState.SPEAKING:
            self.audio_streamer.set_playback_active(True)
            if self.controller and hasattr(self.controller, "events"):
                self.controller.events.voice_speaking_started.emit(self._last_spoken_response)

    def _on_player_finished(self) -> None:
        """Audio playback completed naturally."""
        self.audio_streamer.set_playback_active(False)
        if self._state == VoiceState.SPEAKING:
            if self.controller and hasattr(self.controller, "events"):
                self.controller.events.voice_speaking_finished.emit()

            if self.config.voice.continuous_mode:
                logger.info("Continuous mode active: returning to LISTENING for next turn.")
                self.vad.reset()
                self._last_turn_time = time.time()
                self._set_state(VoiceState.LISTENING)
                if self.config.voice.voice_mode == "gemini_live":
                    self.audio_streamer.start_streaming(self._async_loop)
            else:
                self._conversation_turn_active = False
                self._set_state(VoiceState.STANDBY)
                if self.config.voice.voice_mode == "gemini_live":
                    self.audio_streamer.stop_streaming()

    def _on_player_interrupted(self) -> None:
        logger.info("Audio player interrupted by barge-in.")

    # --------------------------------------------------------------------------
    # Core Speech Turn Processing
    # --------------------------------------------------------------------------

    async def _process_speech_turn(self, audio_data: np.ndarray) -> None:
        """Pipeline executing STT, command checks, AI routing, memory, generation, & TTS."""
        try:
            # 1. Transcribe audio with Faster-Whisper
            transcription = await asyncio.to_thread(
                self.stt.transcribe,
                audio_data,
                language=self.config.voice.preferred_language,
            )

            text = transcription.text.strip()
            if not text or text.startswith("["):
                logger.warning("Empty or error transcript. Returning to listening.")
                self.vad.reset()
                self._set_state(VoiceState.LISTENING if self.config.voice.continuous_mode else VoiceState.STANDBY)
                return

            self._last_user_transcript = text
            logger.info("User Voice Input: '%s' (Lang: %s)", text, transcription.language)

            if self.controller and hasattr(self.controller, "events"):
                self.controller.events.voice_transcript_received.emit(text, transcription.language)

            # 2. Check for Voice Control Commands
            command_type = self._parse_voice_command(text)
            if command_type != VoiceCommandType.NONE:
                await self._execute_voice_command(command_type)
                return

            # 3. Check for Phase 7 Computer Control Operations
            if self.controller and hasattr(self.controller, "windows_control"):
                try:
                    plan = self.controller.windows_control.interpreter.interpret(text, source="voice")
                    if plan.operation_type in ("launch_app", "restart", "shutdown", "lock", "pip_install", "storage_analysis", "network_diagnostics"):
                        self._set_state(VoiceState.THINKING)
                        logger.info("Executing voice-triggered computer control action: %s", plan.objective)
                        result = await self.controller.windows_control.execute_plan(plan)

                        if result.status.value == "confirmation_required":
                            response_text = f"Action '{plan.objective}' requires confirmation: {plan.confirmation_message}"
                        elif result.exit_code == 0:
                            response_text = f"Completed: {plan.objective}."
                        else:
                            response_text = f"Action '{plan.objective}' failed: {result.stderr or result.error or 'Execution error'}"

                        self._last_spoken_response = response_text
                        self._set_state(VoiceState.SPEAKING)
                        spoken = await self.tts.synthesize(
                            response_text,
                            speed=self.config.voice.speech_speed,
                            volume=self.config.voice.speech_volume,
                        )
                        if spoken.audio_path:
                            self.player.play_file(spoken.audio_path)
                        else:
                            self._on_player_finished()
                        return
                except Exception as c_exc:
                    logger.warning("Error evaluating voice computer control: %s", c_exc)

            # 4. Transition to THINKING
            self._set_state(VoiceState.THINKING)
            self._last_turn_time = time.time()

            # 4. Integrate with Shared Controller, AI Routing, & Memory Architecture
            response_text = await self._generate_ai_response(text)
            if not response_text:
                logger.warning("Empty response received for voice prompt.")
                response_text = "I'm listening, but I didn't get a response from the AI engine. How can I help you?"

            self._last_spoken_response = response_text

            # 5. Transition to SPEAKING and Synthesize TTS
            self._set_state(VoiceState.SPEAKING)
            spoken = await self.tts.synthesize(
                response_text,
                speed=self.config.voice.speech_speed,
                volume=self.config.voice.speech_volume,
            )

            # 6. Play Audio Output (Non-blocking with barge-in support)
            if spoken.audio_path:
                self.player.play_file(spoken.audio_path)
            else:
                logger.warning("No audio path returned from TTS router.")
                self._on_player_finished()

        except Exception as exc:
            logger.error("Error during speech turn processing: %s", exc)
            if self.controller and hasattr(self.controller, "events"):
                self.controller.events.voice_error_occurred.emit(str(exc))
            self._set_state(VoiceState.STANDBY)

    def _parse_voice_command(self, text: str) -> VoiceCommandType:
        """Identify if transcription is a voice control command."""
        clean = text.strip()
        for cmd_type, patterns in COMMAND_PATTERNS:
            for pattern in patterns:
                if pattern.search(clean):
                    return cmd_type
        return VoiceCommandType.NONE

    async def _execute_voice_command(self, cmd: VoiceCommandType) -> None:
        """Handle execution of built-in voice commands."""
        logger.info("Executing voice control command: %s", cmd.value)

        if cmd == VoiceCommandType.STOP:
            self._conversation_turn_active = False
            self.player.stop()
            self._set_state(VoiceState.STANDBY)
            await self._speak_quick_confirmation("Stopping voice interaction.")

        elif cmd == VoiceCommandType.SLEEP:
            self._conversation_turn_active = False
            self.player.stop()
            self._set_state(VoiceState.SLEEP)
            await self._speak_quick_confirmation("Going to sleep. Say 'Wake up HAKS' when you need me.")

        elif cmd == VoiceCommandType.WAKE:
            self._conversation_turn_active = True
            self._set_state(VoiceState.STANDBY)
            await self._speak_quick_confirmation("I am awake and standing by.")

        elif cmd == VoiceCommandType.CANCEL:
            self.player.stop()
            self._set_state(VoiceState.LISTENING if self.config.voice.continuous_mode else VoiceState.STANDBY)
            await self._speak_quick_confirmation("Cancelled.")

        elif cmd == VoiceCommandType.REPEAT:
            if self._last_spoken_response:
                self._set_state(VoiceState.SPEAKING)
                spoken = await self.tts.synthesize(self._last_spoken_response)
                if spoken.audio_path:
                    self.player.play_file(spoken.audio_path)
            else:
                await self._speak_quick_confirmation("I have nothing to repeat yet.")

        elif cmd == VoiceCommandType.STOP_SPEAKING:
            self.player.stop()
            self._set_state(VoiceState.LISTENING if self.config.voice.continuous_mode else VoiceState.STANDBY)

    async def _speak_quick_confirmation(self, text: str) -> None:
        """Synthesize and play a brief confirmation utterance."""
        try:
            spoken = await self.tts.synthesize(text)
            if spoken.audio_path:
                self.player.play_file(spoken.audio_path)
        except Exception as exc:
            logger.warning("Could not speak quick confirmation: %s", exc)

    async def _generate_ai_response(self, prompt_text: str) -> str:
        """Route request through Phase 5 Router, inject Phase 4 memories, and generate response."""
        if not self.controller:
            return "App controller is not connected."

        # 1. Record user turn in active conversation database
        self.controller.send_message(prompt_text, role="user")

        # 2. Build conversation context messages
        messages = self.controller.get_conversation_ai_messages(limit=10)

        # 3. Retrieve relevant approved memory context
        extra_gen_kwargs = {}
        try:
            mem_context = self.controller.context_builder.build_context(
                query=prompt_text,
                include_personal=True,
                include_work=True,
                top_k_per_domain=2,
                min_relevance=0.25,
            )
            if mem_context:
                extra_gen_kwargs["system_prompt"] = (
                    "You are HAKS, a helpful, private, local-first personal AI assistant.\n\n"
                    + mem_context
                )
        except Exception as mem_exc:
            logger.warning("Could not inject memory context: %s", mem_exc)

        # 4. Phase 5 AI Routing & Privacy Policy Enforcement
        ai_mgr = self.controller.get_ai_manager()
        engine_type = "cloud"
        try:
            decision = self.controller.route_request(user_prompt=prompt_text)
            if decision.is_fallback_needed and decision.fallback_engine is None:
                err_msg = f"Gemini is currently unavailable ({decision.availability_message}). Please check connectivity or retry."
                logger.warning("Voice AI Gemini unavailable without auto-fallback: %s", err_msg)
                return err_msg
            engine_type = decision.selected_engine
            logger.info(
                "Voice AI Router selected engine: %s (Privacy: %s, Complexity: %s)",
                engine_type,
                decision.privacy_level.value,
                decision.complexity_level.value,
            )
        except Exception as r_exc:
            logger.warning("Voice routing evaluation failed: %s. Using default.", r_exc)
            engine_type = self.controller.get_selected_ai_engine()

        # 5. Generate response
        try:
            response = await ai_mgr.generate(
                messages=messages,
                engine_type=engine_type,
                **extra_gen_kwargs,
            )
            resp_content = response.content.strip() if response and response.content else ""

            if resp_content:
                # Record assistant turn in database
                self.controller.send_message(
                    resp_content,
                    role="assistant",
                    metadata={"engine": engine_type, "model": response.model},
                )
                return resp_content
            else:
                fallback_msg = "I'm listening, but I didn't get a response from the AI engine. How can I help you?"
                logger.warning("AI engine returned empty content for voice prompt (engine=%s)", engine_type)
                return fallback_msg
        except Exception as gen_exc:
            err_msg = f"I encountered an error generating a response: {gen_exc}"
            logger.error("AI Generation failed for voice turn: %s", gen_exc)
            self.controller.send_message(err_msg, role="assistant", metadata={"error": True})
            return err_msg

    # --------------------------------------------------------------------------
    # Lifecycle Controls
    # --------------------------------------------------------------------------

    def start(self) -> bool:
        """Start the continuous voice assistant pipeline."""
        if self._state not in (VoiceState.STOPPED, VoiceState.SLEEP):
            return True

        self._stop_event.clear()
        logger.info("Starting Continuous Voice Assistant...")
        # Start dedicated asyncio background thread if needed
        if self._async_loop is None or not self._async_loop.is_running():
            self._start_async_worker()

        started = self.audio_capture.start()
        if not started:
            logger.error("Could not start audio capture.")
            self._set_state(VoiceState.STOPPED)
            return False

        self._set_state(VoiceState.STANDBY)
        return True

    def stop(self) -> None:
        """Stop microphone capture, interrupt playback, and shut down voice pipeline."""
        self._stop_event.set()
        logger.info("Stopping Continuous Voice Assistant...")
        self.player.stop()
        self.pcm_player.stop()
        self.audio_streamer.stop_streaming()
        self.audio_capture.stop()
        self.vad.reset()
        self._conversation_turn_active = False
        self._set_state(VoiceState.STOPPED)

    def shutdown(self) -> None:
        """Complete teardown of voice engine resources."""
        try:
            self.stop()
        except Exception:
            pass
        if hasattr(self, "gemini_live_client") and self.gemini_live_client:
            if self._async_loop and self._async_loop.is_running():
                try:
                    asyncio.run_coroutine_threadsafe(
                        self.gemini_live_client.disconnect(),
                        self._async_loop,
                    )
                except Exception:
                    pass
        if hasattr(self, "_executor") and self._executor is not None:
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
        if hasattr(self, "tts") and self.tts is not None:
            try:
                self.tts.shutdown()
            except Exception:
                pass

    def __del__(self) -> None:
        try:
            self.controller = None
            self._stop_event.set()
            if hasattr(self, "_executor") and self._executor is not None:
                try:
                    self._executor.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
        except Exception:
            pass

    def trigger_push_to_talk(self) -> None:
        """Manual trigger directly entering LISTENING state without wake word."""
        if self._state == VoiceState.STOPPED:
            self.start()
        logger.info("Push-to-talk triggered. Activating LISTENING state.")
        self.player.stop()
        self.pcm_player.stop()
        self.vad.reset()
        self._conversation_turn_active = True
        self._last_turn_time = time.time()
        self._set_state(VoiceState.LISTENING)

        if self.config.voice.voice_mode == "gemini_live":
            if self._async_loop and self._async_loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._ensure_gemini_live_ready(),
                    self._async_loop,
                )
            self.audio_streamer.start_streaming(self._async_loop)

    def toggle_mute(self) -> bool:
        """Toggle microphone mute."""
        new_mute = not self.audio_capture.is_muted
        self.audio_capture.set_muted(new_mute)
        if self.controller and hasattr(self.controller, "events"):
            self.controller.events.voice_mute_toggled.emit(new_mute)
        return new_mute

    def set_voice_mode(self, mode: str) -> None:
        """Update active voice processing mode ('gemini_live', 'pipeline', 'offline')."""
        if mode not in ("gemini_live", "pipeline", "offline"):
            logger.warning("Invalid voice mode '%s'; defaulting to 'gemini_live'.", mode)
            mode = "gemini_live"

        self.config.voice.voice_mode = mode
        logger.info("Voice Assistant mode changed to: %s", mode)

        if self.controller:
            self.controller.set_setting("voice_mode", mode, section="voice")

        if mode == "gemini_live" and self._state == VoiceState.LISTENING:
            if self._async_loop and self._async_loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._ensure_gemini_live_ready(),
                    self._async_loop,
                )
                self.audio_streamer.start_streaming(self._async_loop)
        elif mode != "gemini_live":
            self.audio_streamer.stop_streaming()

    # --------------------------------------------------------------------------
    # Gemini Live Callbacks & Helpers
    # --------------------------------------------------------------------------

    def _build_live_system_instruction(self) -> str:
        """Construct bilingual persona with approved memory context and privacy protections."""
        base = (
            "You are HAKS, a private, fast, bilingual personal AI assistant operating on Windows. "
            "You speak English and Tamil fluently, and easily understand Tanglish (Tamil written in English script or code-mixed speech). "
            "Keep your spoken answers concise, direct, helpful, and natural. "
            "Do not output markdown asterisks or bullet formatting in spoken audio."
        )
        if self.controller and hasattr(self.controller, "context_builder"):
            try:
                mem_context = self.controller.context_builder.build_context(
                    query="user profile preferences work background",
                    include_personal=True,
                    include_work=True,
                    top_k_per_domain=2,
                    min_relevance=0.25,
                )
                if mem_context:
                    base += f"\n\nContext and approved memories:\n{mem_context}"
            except Exception as mem_exc:
                logger.debug("Could not inject memory context for Gemini Live: %s", mem_exc)
        return base

    async def _ensure_gemini_live_ready(self) -> bool:
        """Ensure GeminiLiveClient is connected with current approved system instructions & memories."""
        if self.gemini_live_client.is_connected():
            return True

        system_instruction = self._build_live_system_instruction()
        self.gemini_live_client.system_instruction = system_instruction
        self.gemini_live_client.model = self.config.voice.gemini_live_model
        self.gemini_live_client.voice_name = self.config.voice.gemini_live_voice

        success = await self.gemini_live_client.connect()
        if not success:
            logger.warning("Gemini Live connection failed; falling back to pipeline mode.")
            return False
        return True

    def _on_live_status_changed(self, status: LiveConnectionStatus) -> None:
        """Forward Live API status to UI."""
        if self.controller and hasattr(self.controller, "events"):
            self.controller.events.voice_live_status_changed.emit(status.value)

    def _on_live_latency_measured(self, latency_ms: float) -> None:
        """Forward response latency to UI and calculate turnaround latency."""
        self.last_gemini_response_latency_ms = latency_ms
        if self._speech_onset_time > 0:
            self.last_total_turnaround_latency_ms = latency_ms + self.config.voice.silence_timeout_ms
        if self.controller and hasattr(self.controller, "events"):
            self.controller.events.voice_live_latency_updated.emit(latency_ms)

    def _on_live_audio_chunk(self, data: bytes) -> None:
        """Handle incoming 24 kHz linear PCM chunk from Gemini Live."""
        if self._state != VoiceState.SPEAKING:
            self._set_state(VoiceState.SPEAKING)
            self.audio_streamer.set_playback_active(True)
        self.pcm_player.write_chunk(data)

    def _on_live_transcript_chunk(self, text: str) -> None:
        """Accumulate assistant response text chunks."""
        pass

    def _on_live_user_transcript(self, text: str) -> None:
        """Handle real-time user speech transcript from Live API."""
        cleaned = text.strip()
        if not cleaned:
            return
        self._last_user_transcript = cleaned
        logger.info("Gemini Live User Speech: '%s'", cleaned)
        if self.controller and hasattr(self.controller, "events"):
            self.controller.events.voice_transcript_received.emit(cleaned, self.config.voice.preferred_language)

        cmd_type = self._parse_voice_command(cleaned)
        if cmd_type != VoiceCommandType.NONE:
            if self._async_loop and self._async_loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._execute_voice_command(cmd_type),
                    self._async_loop,
                )

    def _on_live_turn_complete(self, full_text: str) -> None:
        """Handle Gemini Live turn completion."""
        self._last_spoken_response = full_text
        self.pcm_player.mark_turn_complete()
        if self.controller and full_text:
            try:
                if self._last_user_transcript:
                    self.controller.send_message(self._last_user_transcript, role="user")
                self.controller.send_message(
                    full_text,
                    role="assistant",
                    metadata={"engine": "gemini_live", "model": self.gemini_live_client.model},
                )
            except Exception as e:
                logger.debug("Could not record live turn in conversation db: %s", e)

    def _on_live_interrupted(self) -> None:
        """Server-side barge-in signal received from Gemini Live."""
        is_playing = self.pcm_player.is_playing or self.player.is_playing
        if is_playing and self.echo_suppressor.is_playback_active():
            if self.vad._consecutive_barge_in_speech_frames < 2:
                logger.info("Ignoring spurious Gemini Live server-side interruption caused by speaker acoustic feedback.")
                return

        logger.info("Gemini Live server-side barge-in confirmed.")
        self._handle_barge_in_interruption()

    def _on_live_error(self, err_msg: str) -> None:
        """Log Live error and notify controller."""
        logger.warning("Gemini Live encountered error: %s", err_msg)
        if self.controller and hasattr(self.controller, "events"):
            self.controller.events.voice_error_occurred.emit(err_msg)

    def _on_pcm_playback_finished(self) -> None:
        """24 kHz audio playback completed naturally."""
        self.audio_streamer.set_playback_active(False)
        if self._state == VoiceState.SPEAKING:
            if self.controller and hasattr(self.controller, "events"):
                self.controller.events.voice_speaking_finished.emit()

            if self.config.voice.continuous_mode:
                logger.info("Continuous mode: returning to LISTENING for next turn.")
                self.vad.reset()
                self._last_turn_time = time.time()
                self._set_state(VoiceState.LISTENING)
                if self.config.voice.voice_mode == "gemini_live":
                    self.audio_streamer.start_streaming(self._async_loop)
            else:
                self._conversation_turn_active = False
                self._set_state(VoiceState.STANDBY)
                if self.config.voice.voice_mode == "gemini_live":
                    self.audio_streamer.stop_streaming()

    def _start_async_worker(self) -> None:
        """Start a dedicated background thread running an asyncio event loop."""
        ready_event = threading.Event()

        def _worker():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._async_loop = loop
            ready_event.set()
            loop.run_forever()

        self._async_thread = threading.Thread(
            target=_worker,
            name="voice_async_worker",
            daemon=True,
        )
        self._async_thread.start()
        ready_event.wait(timeout=3.0)
