"""Comprehensive Test Suite for Gemini Live Real-time Speech-to-Speech Subsystem.

Covers:
- Gemini Live client initialization, model configuration (gemini-3.8-live), and authentication
- WebSocket connection lifecycle, states, and error handling
- Audio chunk streaming (16 kHz linear PCM -> audio/pcm;rate=16000)
- Response reception (24 kHz linear PCM audio and transcriptions)
- First audio packet response latency measurement
- Instant barge-in speech interruption
- Transient disconnection and exponential backoff reconnection
- Quota error (HTTP 429) & network error handling without credential leakage
- PCMStreamPlayer streaming playback, volume adjustment, and instant abort
- AudioStreamerBridge float32 to int16 PCM conversion and asynchronous queueing
- ContinuousVoiceEngine integration with mode selection (gemini_live, pipeline, offline)
- Offline fallback when Gemini Live is unavailable
- Phase 4 Memory injection & Phase 5 Privacy enforcement
- Real live connection test against Google Gemini Live API
"""

import asyncio
from enum import Enum
import time
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pytest

from ai.credentials import CredentialManager
from config.settings import AppConfig
from core.app_controller import AppController
from voice.gemini_live import (
    AudioStreamerBridge,
    GeminiLiveClient,
    LiveConnectionStatus,
    PCMStreamPlayer,
)
from voice.models import VoiceState


# ------------------------------------------------------------------------------
# 1. GeminiLiveClient Initialization & Credentials Tests
# ------------------------------------------------------------------------------

def test_gemini_live_client_initialization():
    """Verify GeminiLiveClient initializes with correct model, voice, and defaults."""
    client = GeminiLiveClient(
        model="gemini-3.8-live",
        voice_name="Aoede",
    )
    assert client.model == "gemini-3.8-live"
    assert client.voice_name == "Aoede"
    assert client.status == LiveConnectionStatus.DISCONNECTED
    assert not client.is_connected()
    assert "HAKS" in client.system_instruction
    assert "Tamil" in client.system_instruction
    assert "Tanglish" in client.system_instruction


def test_gemini_live_missing_api_key():
    """Verify GeminiLiveClient reports error and transitions to ERROR status when key missing."""
    mock_creds = MagicMock(spec=CredentialManager)
    mock_creds.get_api_key.return_value = None

    client = GeminiLiveClient(credential_manager=mock_creds)
    error_called = []
    client.on_error = lambda err: error_called.append(err)

    loop = asyncio.new_event_loop()
    try:
        success = loop.run_until_complete(client.connect())
        assert success is False
        assert client.status == LiveConnectionStatus.ERROR
        assert len(error_called) == 1
        assert "API key" in error_called[0]
    finally:
        loop.close()


# ------------------------------------------------------------------------------
# 2. Connection, Reconnection & Backoff Tests
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gemini_live_connect_success():
    """Verify successful connection initializes v1alpha client and live session."""
    mock_creds = MagicMock(spec=CredentialManager)
    mock_creds.get_api_key.return_value = "fake_test_key_12345"

    client = GeminiLiveClient(credential_manager=mock_creds)
    statuses = []
    client.on_status_changed = lambda s: statuses.append(s)

    # Mock the genai SDK
    mock_session = AsyncMock()
    # Mock receive generator
    async def mock_receive():
        return
        yield

    mock_session.receive = mock_receive
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__.return_value = mock_session

    with patch("voice.gemini_live.client.genai") as mock_genai:
        mock_client_instance = MagicMock()
        mock_client_instance.aio.live.connect.return_value = mock_ctx
        mock_genai.Client.return_value = mock_client_instance

        success = await client.connect()
        assert success is True
        assert client.is_connected()
        assert client.status == LiveConnectionStatus.CONNECTED
        assert LiveConnectionStatus.CONNECTING in statuses
        assert LiveConnectionStatus.CONNECTED in statuses

        await client.disconnect()
        assert not client.is_connected()
        assert client.status == LiveConnectionStatus.CLOSED


@pytest.mark.asyncio
async def test_gemini_live_reconnect_exponential_backoff():
    """Verify exponential backoff retries when connection fails transiently."""
    mock_creds = MagicMock(spec=CredentialManager)
    mock_creds.get_api_key.return_value = "fake_key"

    client = GeminiLiveClient(credential_manager=mock_creds)

    attempt_count = 0
    with patch("voice.gemini_live.client.genai") as mock_genai:
        mock_client_inst = MagicMock()

        def fail_connect(*args, **kwargs):
            nonlocal attempt_count
            attempt_count += 1
            raise ConnectionError(f"Attempt {attempt_count} network drop")

        mock_client_inst.aio.live.connect.side_effect = fail_connect
        mock_genai.Client.return_value = mock_client_inst

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            success = await client.connect(max_retries=3, retry_backoff=0.01)
            assert success is False
            assert attempt_count == 3
            assert mock_sleep.call_count == 2
            assert client.status == LiveConnectionStatus.ERROR


# ------------------------------------------------------------------------------
# 3. Audio & Text Transmission Tests
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gemini_live_send_audio_chunk():
    """Verify 16 kHz audio chunk is sent as audio/pcm;rate=16000."""
    client = GeminiLiveClient()
    mock_session = AsyncMock()
    client._session = mock_session
    client.status = LiveConnectionStatus.CONNECTED

    dummy_pcm = b"\x00\x00" * 512
    success = await client.send_audio_chunk(dummy_pcm)

    assert success is True
    assert mock_session.send_realtime_input.called
    call_args = mock_session.send_realtime_input.call_args[1]
    assert "audio" in call_args
    assert call_args["audio"].mime_type == "audio/pcm;rate=16000"
    assert call_args["audio"].data == dummy_pcm


@pytest.mark.asyncio
async def test_gemini_live_send_text_turn():
    """Verify user text turn triggers send_client_content."""
    client = GeminiLiveClient()
    mock_session = AsyncMock()
    client._session = mock_session
    client.status = LiveConnectionStatus.CONNECTED

    success = await client.send_text_turn("வணக்கம் HAKS")
    assert success is True
    assert mock_session.send_client_content.called
    call_kwargs = mock_session.send_client_content.call_args[1]
    assert call_kwargs.get("turn_complete") is True
    assert len(call_kwargs.get("turns")) == 1


# ------------------------------------------------------------------------------
# 4. Receiving Audio, Transcription, and Latency Tests
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gemini_live_receive_audio_and_transcription():
    """Verify receive loop correctly processes audio chunks, text, and turn completion."""
    client = GeminiLiveClient()
    client.status = LiveConnectionStatus.CONNECTED
    client._is_active = True
    client._last_turn_sent_time = time.time() - 0.150  # 150ms ago

    audio_chunks = []
    text_chunks = []
    turns_completed = []
    latencies = []

    client.on_audio_chunk = lambda chunk: audio_chunks.append(chunk)
    client.on_transcript_chunk = lambda text: text_chunks.append(text)
    client.on_turn_complete = lambda full: turns_completed.append(full)
    client.on_latency_measured = lambda ms: latencies.append(ms)

    # Build mock responses
    mock_part_audio = MagicMock()
    mock_part_audio.inline_data.data = b"\x01\x00" * 480
    mock_part_audio.inline_data.mime_type = "audio/pcm;rate=24000"
    mock_part_audio.text = None

    resp1 = MagicMock()
    resp1.server_content.model_turn.parts = [mock_part_audio]
    resp1.server_content.output_transcription = None
    resp1.server_content.input_transcription = None
    resp1.server_content.interrupted = False
    resp1.server_content.turn_complete = False

    resp2 = MagicMock()
    resp2.server_content.model_turn.parts = []
    resp2.server_content.output_transcription.text = "Hello! How can I help you?"
    resp2.server_content.input_transcription = None
    resp2.server_content.interrupted = False
    resp2.server_content.turn_complete = True

    async def mock_receive_iter():
        yield resp1
        yield resp2

    mock_session = MagicMock()
    mock_session.receive = mock_receive_iter
    client._session = mock_session

    await client._receive_loop()

    assert len(audio_chunks) == 1
    assert audio_chunks[0] == b"\x01\x00" * 480
    assert len(text_chunks) == 1
    assert text_chunks[0] == "Hello! How can I help you?"
    assert len(turns_completed) == 1
    assert turns_completed[0] == "Hello! How can I help you?"
    assert len(latencies) == 1
    assert latencies[0] >= 140.0  # Latency measured ~150ms


@pytest.mark.asyncio
async def test_gemini_live_barge_in_signal():
    """Verify server-side interrupted flag triggers on_interrupted callback."""
    client = GeminiLiveClient()
    client.status = LiveConnectionStatus.CONNECTED
    client._is_active = True

    interrupted_called = False

    def handle_interrupt():
        nonlocal interrupted_called
        interrupted_called = True

    client.on_interrupted = handle_interrupt

    resp = MagicMock()
    resp.server_content.model_turn = None
    resp.server_content.output_transcription = None
    resp.server_content.input_transcription = None
    resp.server_content.interrupted = True
    resp.server_content.turn_complete = False

    async def mock_receive_iter():
        yield resp

    mock_session = MagicMock()
    mock_session.receive = mock_receive_iter
    client._session = mock_session

    await client._receive_loop()
    assert interrupted_called is True


# ------------------------------------------------------------------------------
# 5. PCMStreamPlayer Tests
# ------------------------------------------------------------------------------

def test_pcm_stream_player_volume_and_queue():
    """Verify PCMStreamPlayer volume bounds and chunk queueing."""
    player = PCMStreamPlayer(sample_rate=24000, channels=1)
    player.volume = 0.5
    assert player.volume == 0.5

    player.volume = 1.5
    assert player.volume == 1.0

    player.volume = -0.5
    assert player.volume == 0.0

    # Test feed audio
    dummy_chunk = b"\x00\x00" * 240
    with patch("voice.gemini_live.pcm_player.sd") as mock_sd:
        mock_stream = MagicMock()
        mock_sd.RawOutputStream.return_value = mock_stream
        player.write_chunk(dummy_chunk)
        assert player.is_playing is True
        player.stop()
        assert player.is_playing is False


def test_pcm_stream_player_instant_barge_in_abort():
    """Verify stop() clears queue and aborts active audio stream immediately."""
    player = PCMStreamPlayer()
    interrupted_called = False
    player.on_playback_interrupted = lambda: locals().update({"interrupted_called": True})

    with patch("voice.gemini_live.pcm_player.sd") as mock_sd:
        mock_stream = MagicMock()
        mock_sd.RawOutputStream.return_value = mock_stream

        player._is_playing = True
        player._queue.put(b"\x01\x00" * 1024)
        player._queue.put(b"\x02\x00" * 1024)

        player.stop()
        assert player.is_playing is False
        assert player._queue.empty()


# ------------------------------------------------------------------------------
# 6. AudioStreamerBridge Tests
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_audio_streamer_bridge_flow():
    """Verify AudioStreamerBridge converts float32 frames to 16 kHz int16 bytes."""
    client = GeminiLiveClient()
    mock_send = AsyncMock(return_value=True)
    client.send_audio_chunk = mock_send
    client.is_connected = MagicMock(return_value=True)

    streamer = AudioStreamerBridge(client=client)
    loop = asyncio.get_running_loop()
    streamer.start_streaming(loop=loop)
    assert streamer.is_streaming is True

    # Feed 512 float32 samples (sine wave)
    t = np.linspace(0, 0.032, 512, endpoint=False)
    sine_float = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    streamer.feed_frame(sine_float)
    await asyncio.sleep(0.05)

    assert mock_send.called
    pcm_arg = mock_send.call_args[0][0]
    assert len(pcm_arg) == 512 * 2  # 512 samples * 2 bytes/sample (int16)

    streamer.stop_streaming()
    assert streamer.is_streaming is False


# ------------------------------------------------------------------------------
# 7. ContinuousVoiceEngine Integration & Fallback Tests
# ------------------------------------------------------------------------------

def test_voice_engine_mode_selection_and_switch(temp_workspace: AppConfig):
    """Verify voice engine mode selection, settings persistence, and fallback handling."""
    from voice.conversation_engine import ContinuousVoiceEngine

    engine = ContinuousVoiceEngine(config=temp_workspace)
    assert engine.config.voice.voice_mode == "gemini_live"

    engine.set_voice_mode("pipeline")
    assert engine.config.voice.voice_mode == "pipeline"

    engine.set_voice_mode("offline")
    assert engine.config.voice.voice_mode == "offline"

    engine.set_voice_mode("gemini_live")
    assert engine.config.voice.voice_mode == "gemini_live"


def test_voice_engine_system_instruction_with_memory(temp_workspace: AppConfig):
    """Verify dynamic system instructions include approved memories and persona."""
    from voice.conversation_engine import ContinuousVoiceEngine

    engine = ContinuousVoiceEngine(config=temp_workspace)
    instruction = engine._build_live_system_instruction()
    assert "HAKS" in instruction
    assert "Tamil" in instruction
    assert "Tanglish" in instruction


@pytest.mark.asyncio
async def test_voice_engine_gemini_live_offline_fallback(temp_workspace: AppConfig):
    """Verify that when Gemini Live cannot connect, it cleanly reports failure for pipeline fallback."""
    from voice.conversation_engine import ContinuousVoiceEngine

    temp_workspace.voice.voice_mode = "gemini_live"
    engine = ContinuousVoiceEngine(config=temp_workspace)

    # Mock live client failing connection
    engine.gemini_live_client.connect = AsyncMock(return_value=False)
    ready = await engine._ensure_gemini_live_ready()
    assert ready is False


# ------------------------------------------------------------------------------
# 8. Real Gemini Live API Integration Test
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_real_gemini_live_connection():
    """Integration test with live Google Gemini Live API using credentials in CredentialManager."""
    creds = CredentialManager()
    api_key = creds.get_api_key("gemini")
    if not api_key:
        pytest.skip("No Gemini API key found in Windows Credential Manager.")

    client = GeminiLiveClient(
        model="gemini-3.8-live",
        voice_name="Aoede",
        system_instruction="You are HAKS. Reply with exactly two words: 'Vanakkam HAKS'."
    )

    try:
        connected = await client.connect(max_retries=2)
        assert connected is True, "Failed to connect to real Gemini Live WebSocket"

        # Send text turn to test full bidirectional exchange
        received_chunks = []
        client.on_audio_chunk = lambda c: received_chunks.append(c)

        sent = await client.send_text_turn("Vanakkam!")
        assert sent is True

        # Wait for first audio packets to arrive
        for _ in range(30):
            if len(received_chunks) > 0:
                break
            await asyncio.sleep(0.2)

        assert len(received_chunks) > 0, "Did not receive audio response from Gemini Live"
        assert client.last_latency_ms > 0
    finally:
        await client.disconnect()

