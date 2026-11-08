"""Comprehensive tests for HAKS Phase 6 — Advanced Voice Assistant.

Covers:
- Audio capture, device enumeration, and RMS level monitoring
- Voice Activity Detection (VAD) speech start, end, padding, and noise rejection
- Local acoustic wake word detection ('Hey HAKS') & Porcupine adapter
- Faster-Whisper lazy loading, language detection, and transcription
- Multi-engine TTS (Edge Neural Tamil/English, Windows SAPI, Piper)
- Language-aware voice routing (Tamil vs English script)
- Instant barge-in speech interruption and thread-safe audio playback
- Continuous conversation engine state machine & voice control commands
- Integration with AppController, AI engines, and memory architecture
"""

import asyncio
from pathlib import Path
import time
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pytest

from ai.models import AIResponse
from config.settings import AppConfig
from core.app_controller import AppController
from storage.database import DatabaseManager
from voice.audio_capture import AudioCaptureService
from voice.conversation_engine import ContinuousVoiceEngine
from voice.models import (
    AudioDevice,
    SpokenUtterance,
    TranscriptionResult,
    VoiceCommandType,
    VoiceState,
)
from voice.player import AudioPlayerService
from voice.stt.faster_whisper_stt import FasterWhisperSTT
from voice.tts.edge_tts_engine import EdgeTTSEngine
from voice.tts.router import LanguageAwareTTSRouter
from voice.tts.sapi_tts_engine import SapiTTSEngine
from voice.vad import VoiceActivityDetector
from voice.wake_word.local_detector import LocalAcousticWakeDetector
from voice.wake_word.porcupine_detector import PorcupineWakeDetector


@pytest.fixture
def voice_config(temp_workspace: AppConfig) -> AppConfig:
    """Fixture providing isolated AppConfig with voice settings."""
    temp_workspace.voice.continuous_mode = True
    temp_workspace.voice.wake_sensitivity = 0.65
    temp_workspace.voice.silence_timeout_ms = 400  # Faster timeout for tests
    return temp_workspace


# ------------------------------------------------------------------------------
# 1. Audio Capture Tests
# ------------------------------------------------------------------------------

def test_audio_capture_device_listing(voice_config: AppConfig):
    """Verify audio capture service enumerates available microphones."""
    devices = AudioCaptureService.list_input_devices()
    assert isinstance(devices, list)
    if devices:
        dev = devices[0]
        assert isinstance(dev, AudioDevice)
        assert hasattr(dev, "index")
        assert hasattr(dev, "name")
        assert dev.max_input_channels > 0


def test_audio_capture_buffering_and_level(voice_config: AppConfig):
    """Verify audio capture accumulates frames in ring buffer and calculates audio level."""
    service = AudioCaptureService(voice_config)
    service._is_capturing = True
    levels = []
    service.add_level_callback(lambda lvl: levels.append(lvl))

    # Simulate feeding 5 audio frames
    for _ in range(5):
        frame = np.random.uniform(-0.1, 0.1, 512).astype(np.float32)
        service._audio_callback(frame[:, np.newaxis], 512, None, None)

    assert len(service._ring_buffer) == 5
    assert len(levels) == 5
    assert all(0.0 <= lvl <= 1.0 for lvl in levels)

    buffered = service.get_buffered_audio(duration_seconds=0.5)
    assert len(buffered) > 0

    service.clear_buffer()
    assert len(service._ring_buffer) == 0


def test_audio_capture_mute(voice_config: AppConfig):
    """Verify muting replaces audio with silence and drops level to 0."""
    service = AudioCaptureService(voice_config)
    service._is_capturing = True
    service.set_muted(True)
    assert service.is_muted is True

    frame = np.ones((512, 1), dtype=np.float32) * 0.5
    service._audio_callback(frame, 512, None, None)

    # When muted, current audio level must be 0.0
    assert service._current_audio_level == 0.0
    # Last stored frame in ring buffer must be all zeros
    last_frame = list(service._ring_buffer)[-1]
    assert np.all(last_frame == 0.0)


# ------------------------------------------------------------------------------
# 2. Voice Activity Detection (VAD) Tests
# ------------------------------------------------------------------------------

def test_vad_speech_start_and_end_detection():
    """Verify VAD triggers speech start, buffers pre-speech padding, and detects speech end."""
    vad = VoiceActivityDetector(sample_rate=16000, frame_size=512, silence_timeout_ms=100)
    started_calls = []
    ended_calls = []

    vad.on_speech_started = lambda: started_calls.append(True)
    vad.on_speech_ended = lambda audio: ended_calls.append(audio)

    # 1. Feed low-level silence (10 frames)
    for _ in range(10):
        vad.process_frame(np.zeros(512, dtype=np.float32))
    assert not vad.is_speech_active
    assert len(started_calls) == 0

    # 2. Feed sustained speech burst (10 frames of 0.2 amplitude)
    for _ in range(10):
        vad.process_frame(np.ones(512, dtype=np.float32) * 0.2)

    assert vad.is_speech_active is True
    assert len(started_calls) == 1

    # 3. Feed trailing silence (5 frames to exceed silence_timeout_ms=100ms)
    for _ in range(5):
        vad.process_frame(np.zeros(512, dtype=np.float32))

    assert vad.is_speech_active is False
    assert len(ended_calls) == 1
    assert len(ended_calls[0]) > 0


def test_vad_ignores_background_noise():
    """Verify VAD adapts noise floor and does not falsely trigger on continuous low noise."""
    vad = VoiceActivityDetector(sample_rate=16000, frame_size=512)
    started = []
    vad.on_speech_started = lambda: started.append(True)

    # Feed steady ambient hiss (RMS ~0.003)
    for _ in range(25):
        noise = np.random.uniform(-0.005, 0.005, 512).astype(np.float32)
        vad.process_frame(noise)

    assert vad.is_speech_active is False
    assert len(started) == 0


# ------------------------------------------------------------------------------
# 3. Wake Word Detection Tests
# ------------------------------------------------------------------------------

def test_local_acoustic_wake_detector_trigger_and_debounce():
    """Verify local acoustic detector spots two-syllable pattern and enforces cooldown."""
    detector = LocalAcousticWakeDetector(sensitivity=0.8)
    assert detector.is_available() is True
    assert detector.wake_phrase == "Hey HAKS"

    # Simulate two acoustic bursts ('Hey' + 'HAKS') with high ZCR at the end
    # Syllable 1: 15 frames high energy
    for _ in range(15):
        frame = np.random.uniform(-0.25, 0.25, 512).astype(np.float32)
        detector.process_frame(frame)

    # Dip: 4 frames low energy
    for _ in range(4):
        detector.process_frame(np.zeros(512, dtype=np.float32))

    # Syllable 2: 12 frames high energy with high zero-crossing tail ('ks')
    triggered = False
    for i in range(12):
        if i >= 6:
            # Alternating signs create high ZCR for fricative 's'
            fricative = np.array([0.2, -0.2] * 256, dtype=np.float32)
            res = detector.process_frame(fricative)
        else:
            burst = np.random.uniform(-0.3, 0.3, 512).astype(np.float32)
            res = detector.process_frame(burst)
        if res:
            triggered = True

    assert triggered is True

    # Immediate subsequent frames must be rejected by cooldown debounce
    immediate_followup = detector.process_frame(np.random.uniform(-0.3, 0.3, 512).astype(np.float32))
    assert immediate_followup is False


def test_porcupine_wake_detector_status():
    """Verify Porcupine adapter reports clear diagnostic status when unconfigured."""
    detector = PorcupineWakeDetector(access_key="", keyword_path="missing.ppn")
    assert detector.is_available() is False
    status = detector.get_status_info()
    assert status["is_ready"] is False
    assert status["has_access_key"] is False
    assert "AccessKey is not configured" in status["error"]


# ------------------------------------------------------------------------------
# 4. Speech-to-Text (Faster-Whisper) Tests
# ------------------------------------------------------------------------------

def test_faster_whisper_lazy_loading(voice_config: AppConfig):
    """Verify FasterWhisperSTT does not load model until transcribe/load is called."""
    stt = FasterWhisperSTT(voice_config, model_size="tiny")
    assert stt.is_loaded is False

    # Mock WhisperModel internal to test lazy loading without downloading weights in test
    mock_model = MagicMock()
    mock_segment = MagicMock()
    mock_segment.text = "வணக்கம் HAKS"
    mock_segment.start = 0.0
    mock_segment.end = 1.2
    mock_segment.avg_logprob = -0.1
    mock_segment.no_speech_prob = 0.01

    mock_info = MagicMock()
    mock_info.language = "ta"
    mock_info.language_probability = 0.98

    mock_model.transcribe.return_value = ([mock_segment], mock_info)

    with patch("voice.stt.faster_whisper_stt.WhisperModel", return_value=mock_model):
        stt.load_model()
        assert stt.is_loaded is True

        dummy_audio = np.zeros(16000, dtype=np.float32)
        res = stt.transcribe(dummy_audio, language="mixed")

        assert isinstance(res, TranscriptionResult)
        assert res.text == "வணக்கம் HAKS"
        assert res.language == "ta"
        assert res.language_probability == 0.98

        stt.unload_model()
        assert stt.is_loaded is False


# ------------------------------------------------------------------------------
# 5. Text-to-Speech (TTS) & Language Router Tests
# ------------------------------------------------------------------------------

def test_tts_script_language_detection():
    """Verify language-aware script classifier identifies Tamil, English, and mixed sentences."""
    assert LanguageAwareTTSRouter.contains_tamil("வணக்கம் HAKS") is True
    assert LanguageAwareTTSRouter.contains_tamil("Hello world") is False
    assert LanguageAwareTTSRouter.contains_tamil("தமிழ்நாடு") is True

    router = LanguageAwareTTSRouter()
    assert router.detect_language("வணக்கம்") == "ta"
    assert router.detect_language("Good morning assistant") == "en"
    assert router.detect_language("வணக்கம், how are you?") == "mixed"


def test_edge_tts_voices_and_voice_resolution():
    """Verify EdgeTTSEngine lists authentic Tamil and English voices and resolves correctly."""
    edge = EdgeTTSEngine()
    voices = edge.list_voices()
    assert len(voices) >= 6

    tamil_voices = [v for v in voices if "ta-" in v.language]
    assert any("Pallavi" in v.name for v in tamil_voices)
    assert any("Valluvar" in v.name for v in tamil_voices)

    english_voices = [v for v in voices if "en-" in v.language]
    assert any("Neerja" in v.name for v in english_voices)

    # Resolution tests
    assert edge.resolve_voice_for_text("வணக்கம் உலகம்") == "ta-IN-PallaviNeural"
    assert edge.resolve_voice_for_text("Hello HAKS") == "en-IN-NeerjaNeural"
    assert edge.resolve_voice_for_text("Hello", requested_voice="en-US-JennyNeural") == "en-US-JennyNeural"


@pytest.mark.asyncio
async def test_sapi_tts_offline_synthesis(tmp_path: Path):
    """Verify SapiTTSEngine synthesizes WAV file using Windows SAPI."""
    sapi = SapiTTSEngine()
    if not sapi.is_available():
        pytest.skip("pyttsx3 not available in environment.")

    out_file = tmp_path / "test_sapi.wav"
    res_path = await sapi.synthesize_to_file("Testing offline voice", out_file)
    assert res_path.exists()
    assert res_path.stat().st_size > 0


# ------------------------------------------------------------------------------
# 6. Audio Player & Barge-in Interruption Tests
# ------------------------------------------------------------------------------

def test_audio_player_barge_in_stop():
    """Verify AudioPlayerService stops immediately when interrupted."""
    player = AudioPlayerService()
    interrupted_calls = []
    player.on_playback_interrupted = lambda: interrupted_calls.append(True)

    # Simulate active playback state
    player._is_playing = True
    assert player.is_playing is True

    player.stop()
    assert player.is_playing is False
    assert player._stop_event.is_set()


# ------------------------------------------------------------------------------
# 7. Voice Commands & Continuous State Machine Tests
# ------------------------------------------------------------------------------

def test_voice_command_parsing(voice_config: AppConfig):
    """Verify built-in control commands in English and Tamil are accurately parsed."""
    engine = ContinuousVoiceEngine(config=voice_config)

    assert engine._parse_voice_command("Stop HAKS") == VoiceCommandType.STOP
    assert engine._parse_voice_command("நிறுத்து") == VoiceCommandType.STOP
    assert engine._parse_voice_command("Sleep HAKS") == VoiceCommandType.SLEEP
    assert engine._parse_voice_command("தூங்கு") == VoiceCommandType.SLEEP
    assert engine._parse_voice_command("Wake up HAKS") == VoiceCommandType.WAKE
    assert engine._parse_voice_command("எழுந்திரு") == VoiceCommandType.WAKE
    assert engine._parse_voice_command("Cancel") == VoiceCommandType.CANCEL
    assert engine._parse_voice_command("Repeat") == VoiceCommandType.REPEAT
    assert engine._parse_voice_command("Stop speaking") == VoiceCommandType.STOP_SPEAKING
    assert engine._parse_voice_command("What is the weather today?") == VoiceCommandType.NONE


def test_continuous_conversation_state_transitions(
    temp_workspace: AppConfig, test_db_manager: DatabaseManager
):
    """Verify state machine transitions cleanly across states."""
    controller = AppController(config=temp_workspace, db_manager=test_db_manager)
    engine = controller.voice_engine

    assert engine.state == VoiceState.STOPPED

    engine._set_state(VoiceState.STANDBY)
    assert controller.get_voice_state() == "STANDBY"

    engine._handle_wake_word_detected()
    assert controller.get_voice_state() == "WAKE_DETECTED"

    engine._set_state(VoiceState.LISTENING)
    assert controller.get_voice_state() == "LISTENING"

    engine._set_state(VoiceState.PROCESSING)
    assert controller.get_voice_state() == "PROCESSING"

    engine._set_state(VoiceState.THINKING)
    assert controller.get_voice_state() == "THINKING"

    engine._set_state(VoiceState.SPEAKING)
    assert controller.get_voice_state() == "SPEAKING"

    # User barge-in during speaking
    engine._handle_barge_in_interruption()
    assert controller.get_voice_state() == "INTERRUPTED"
    engine.stop()


@pytest.mark.asyncio
async def test_speech_turn_integration_with_ai_and_memory(
    temp_workspace: AppConfig, test_db_manager: DatabaseManager
):
    """Verify full speech turn routes through AI engine and records in database."""
    controller = AppController(config=temp_workspace, db_manager=test_db_manager)
    engine = controller.voice_engine

    # Seed an approved personal memory
    controller.add_personal_memory(
        title="Spoken Preference",
        content="User loves listening to Tamil classical songs in the morning.",
        category="preference",
        tags="tamil, music",
        importance=4,
    )

    # Mock Faster-Whisper to return transcribed turn
    mock_transcription = TranscriptionResult(
        text="What are my morning preferences?",
        language="en",
        confidence=0.95,
        duration_seconds=2.0,
    )
    engine.stt.transcribe = MagicMock(return_value=mock_transcription)

    # Mock AI manager to return response with memory context
    controller.get_ai_manager().generate = AsyncMock(
        return_value=AIResponse(
            content="Your preference is Tamil classical songs.",
            model="phi3:mini",
        )
    )

    # Mock TTS synthesize to avoid network calls during unit test
    dummy_wav = temp_workspace.paths.data_dir / "dummy.wav"
    dummy_wav.write_bytes(b"RIFFdummywav")
    engine.tts.synthesize = AsyncMock(
        return_value=SpokenUtterance(
            text="Your preference is Tamil classical songs.",
            detected_language="en",
            voice_name="en-IN-NeerjaNeural",
            audio_path=str(dummy_wav),
        )
    )
    engine.player.play_file = MagicMock(return_value=True)

    dummy_audio = np.zeros(32000, dtype=np.float32)
    await engine._process_speech_turn(dummy_audio)

    # Verify conversation history updated
    conv_id = controller.get_active_conversation_id()
    messages = controller.load_conversation(conv_id, emit_event=False)
    assert any("What are my morning preferences?" in m["content"] for m in messages)
    assert any("Tamil classical songs" in m["content"] for m in messages)


def test_app_controller_voice_controls(
    temp_workspace: AppConfig, test_db_manager: DatabaseManager
):
    """Verify AppController voice helper methods interact cleanly with voice engine."""
    controller = AppController(config=temp_workspace, db_manager=test_db_manager)

    assert controller.get_voice_state() == "STOPPED"
    assert controller.is_mic_muted() is False

    controller.toggle_mic_mute()
    assert controller.is_mic_muted() is True
    assert controller.voice_engine.audio_capture.is_muted is True

    controller.toggle_mic_mute()
    assert controller.is_mic_muted() is False

    controller.trigger_push_to_talk()
    assert controller.get_voice_state() == "LISTENING"

    controller.interrupt_voice()
    assert controller.get_voice_state() == "INTERRUPTED"
    controller.stop_voice_assistant()
    assert controller.get_voice_state() == "STOPPED"
