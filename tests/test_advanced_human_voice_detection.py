"""Comprehensive Automated Verification Test Suite for HAKS Advanced Voice Detection.

Covers all 17 requirements:
1. Silence detection.
2. Normal human speech detection.
3. Quiet human speech detection.
4. Fan noise rejection.
5. Keyboard typing click rejection.
6. Traffic rumble noise rejection.
7. Background music rejection during playback.
8. HAKS speaker output leaking into the microphone (AEC suppression).
9. Genuine user interruption during HAKS speech (barge-in confirmation).
10. Tamil speech turn.
11. English speech turn.
12. Tanglish (code-switched) speech turn.
13. Multiple consecutive conversation turns.
14. Silero model initialization failure resilience.
15. WebRTC VAD fallback verification.
16. Audio device changes handling.
17. Gemini network interruption resilience.
"""

from functools import lru_cache
import os
from pathlib import Path
import tempfile
import time
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

try:
    import av
    import pyttsx3
except ImportError:
    av = None
    pyttsx3 = None

from config.settings import AppConfig
from voice.apm.webrtc_apm import WebRTCAPM, SpeakerReferenceTracker
from voice.audio_capture import AudioCaptureService
from voice.conversation_engine import ContinuousVoiceEngine
from voice.gemini_live.audio_streamer import GeminiAudioStreamer
from voice.gemini_live.client import GeminiLiveClient
from voice.gemini_live.pcm_player import PCMPlayer
from voice.interruption_controller import InterruptionController
from voice.models import VoiceState
from voice.vad.base import BaseVADEngine
from voice.vad.detector import VoiceActivityDetector
from voice.vad.silero_vad import SileroVAD
from voice.vad.webrtc_vad import WebRTCVADEngine


# ------------------------------------------------------------------------------
# Audio Fixture Synthesizers (16kHz Mono Float32)
# ------------------------------------------------------------------------------

def make_silence(duration_ms: int = 100, amp: float = 0.0) -> np.ndarray:
    """Generate absolute or near-zero silence."""
    num_samples = int(16000 * (duration_ms / 1000.0))
    if amp == 0.0:
        return np.zeros(num_samples, dtype=np.float32)
    return np.random.uniform(-amp, amp, num_samples).astype(np.float32)


@lru_cache(maxsize=4)
def get_cached_speech_fixture(phrase: str = "Hello HAKS how are you today") -> np.ndarray:
    """Generate or retrieve cached genuine human speech using offline TTS."""
    if pyttsx3 is not None and av is not None:
        try:
            engine = pyttsx3.init()
            tmp_wav = tempfile.mktemp(suffix=".wav")
            engine.save_to_file(phrase, tmp_wav)
            engine.runAndWait()

            container = av.open(tmp_wav)
            stream = next(s for s in container.streams if s.type == "audio")
            frames = [f.to_ndarray().flatten() for f in container.decode(stream)]
            container.close()
            try:
                os.remove(tmp_wav)
            except Exception:
                pass

            raw = np.concatenate(frames)
            if raw.dtype == np.int16 or np.max(np.abs(raw)) > 1.0:
                raw = raw.astype(np.float32) / 32768.0
            else:
                raw = raw.astype(np.float32)

            if stream.rate != 16000:
                indices = np.linspace(0, len(raw) - 1, int(len(raw) * 16000 / stream.rate))
                audio_16k = np.interp(indices, np.arange(len(raw)), raw).astype(np.float32)
            else:
                audio_16k = raw

            return audio_16k
        except Exception:
            pass

    # Fallback to harmonic glottal pulse train
    duration_ms = 1000
    num_samples = int(16000 * (duration_ms / 1000.0))
    t = np.linspace(0, duration_ms / 1000.0, num_samples, endpoint=False)
    sig = np.sin(2 * np.pi * 140 * t) + 0.5 * np.sin(2 * np.pi * 280 * t)
    return (sig * 0.3).astype(np.float32)


def make_fan_noise(duration_ms: int = 200, amplitude: float = 0.05) -> np.ndarray:
    """Generate broadband low-frequency fan noise with 60Hz and 120Hz motor hum."""
    num_samples = int(16000 * (duration_ms / 1000.0))
    t = np.linspace(0, duration_ms / 1000.0, num_samples, endpoint=False)
    white = np.random.uniform(-1.0, 1.0, num_samples)
    lowpass = np.convolve(white, np.ones(16) / 16.0, mode="same")
    hum = 0.4 * np.sin(2 * np.pi * 60 * t) + 0.2 * np.sin(2 * np.pi * 120 * t)
    fan = lowpass + hum
    rms = np.sqrt(np.mean(fan**2))
    return ((fan / (rms + 1e-6)) * amplitude).astype(np.float32)


def make_keyboard_clicks(duration_ms: int = 200, amplitude: float = 0.18) -> np.ndarray:
    """Generate sharp transient typing click impulses."""
    num_samples = int(16000 * (duration_ms / 1000.0))
    signal = np.zeros(num_samples, dtype=np.float32)
    click_len = 160  # 10ms
    for click_pos in (100, int(num_samples * 0.6)):
        if click_pos + click_len < num_samples:
            t_click = np.linspace(0, 0.01, click_len)
            impulse = np.exp(-t_click * 600.0) * np.sin(2 * np.pi * 3500.0 * t_click)
            signal[click_pos : click_pos + click_len] += impulse * amplitude
    return signal.astype(np.float32)


def make_traffic_noise(duration_ms: int = 200, amplitude: float = 0.06) -> np.ndarray:
    """Generate continuous low-frequency street rumble (40-100Hz)."""
    num_samples = int(16000 * (duration_ms / 1000.0))
    t = np.linspace(0, duration_ms / 1000.0, num_samples, endpoint=False)
    rumble = np.sin(2 * np.pi * 55.0 * t) + 0.7 * np.sin(2 * np.pi * 85.0 * t)
    noise = np.convolve(np.random.uniform(-1, 1, num_samples), np.ones(32) / 32.0, mode="same")
    traffic = rumble * 0.6 + noise * 0.4
    rms = np.sqrt(np.mean(traffic**2))
    return ((traffic / (rms + 1e-6)) * amplitude).astype(np.float32)


def make_background_music(duration_ms: int = 200, amplitude: float = 0.08) -> np.ndarray:
    """Generate polyphonic harmonic chord (C major)."""
    num_samples = int(16000 * (duration_ms / 1000.0))
    t = np.linspace(0, duration_ms / 1000.0, num_samples, endpoint=False)
    chord = (
        np.sin(2 * np.pi * 261.63 * t)  # C4
        + np.sin(2 * np.pi * 329.63 * t)  # E4
        + np.sin(2 * np.pi * 392.00 * t)  # G4
    )
    rms = np.sqrt(np.mean(chord**2))
    return ((chord / (rms + 1e-6)) * amplitude).astype(np.float32)


# ------------------------------------------------------------------------------
# Test 1: Silence Detection
# ------------------------------------------------------------------------------

def test_01_silence_detection():
    """Verify silence produces near-zero probability and no speech events."""
    vad = VoiceActivityDetector(vad_engine="silero", speech_threshold=0.55)
    started = []
    vad.on_speech_started = lambda: started.append(True)

    # Feed 20 frames of silence
    for _ in range(20):
        frame = make_silence(duration_ms=32, amp=0.0005)
        is_speech = vad.process_frame(frame)
        assert not is_speech
        assert vad.speech_probability < 0.20

    assert not vad.is_speech_active
    assert len(started) == 0


# ------------------------------------------------------------------------------
# Test 2: Normal Human Speech Detection
# ------------------------------------------------------------------------------

def test_02_normal_human_speech():
    """Verify normal human speech is detected with high confidence."""
    vad = VoiceActivityDetector(vad_engine="silero", speech_threshold=0.50)
    started = []
    ended = []
    vad.on_speech_started = lambda: started.append(True)
    vad.on_speech_ended = lambda audio: ended.append(audio)

    # 1. Warmup with 3 frames of silence
    for _ in range(3):
        vad.process_frame(make_silence(32))

    # 2. Feed genuine human speech fixture
    speech = get_cached_speech_fixture()
    assert len(speech) > 16000

    speech_frames_found = 0
    for i in range(0, len(speech) - 512, 512):
        chunk = speech[i : i + 512]
        is_speech = vad.process_frame(chunk)
        if is_speech:
            speech_frames_found += 1

    assert speech_frames_found > 0
    assert len(started) >= 1

    # 3. Feed trailing silence (20 frames = ~640ms > 450ms timeout)
    for _ in range(20):
        vad.process_frame(make_silence(32))

    assert vad.is_speech_active is False
    assert len(ended) >= 1
    assert len(ended[0]) > 0


# ------------------------------------------------------------------------------
# Test 3: Quiet Human Speech Detection
# ------------------------------------------------------------------------------

def test_03_quiet_human_speech():
    """Verify quiet human speech is detected when formant structure is preserved."""
    vad = VoiceActivityDetector(vad_engine="silero", speech_threshold=0.35)
    speech = get_cached_speech_fixture() * 0.15  # Scale down to whisper / quiet volume

    probs = []
    for i in range(0, len(speech) - 512, 512):
        chunk = speech[i : i + 512]
        prob = vad._active_engine.predict_probability(chunk)
        probs.append(prob)

    assert max(probs) >= 0.50


# ------------------------------------------------------------------------------
# Test 4: Fan Noise Rejection
# ------------------------------------------------------------------------------

def test_04_fan_noise_rejection():
    """Verify continuous low-frequency fan noise does not trigger false speech."""
    vad = VoiceActivityDetector(vad_engine="silero", speech_threshold=0.55)
    started = []
    vad.on_speech_started = lambda: started.append(True)

    # Feed 30 frames of fan noise
    for _ in range(30):
        frame = make_fan_noise(duration_ms=32, amplitude=0.08)
        vad.process_frame(frame)
        assert vad.speech_probability < 0.40

    assert len(started) == 0
    assert not vad.is_speech_active


# ------------------------------------------------------------------------------
# Test 5: Keyboard Typing Click Rejection
# ------------------------------------------------------------------------------

def test_05_keyboard_typing_rejection():
    """Verify transient keyboard clicks do not falsely trigger speech start."""
    vad = VoiceActivityDetector(vad_engine="silero", speech_threshold=0.55)
    started = []
    vad.on_speech_started = lambda: started.append(True)

    # Feed clicks alternating with ambient silence
    for _ in range(10):
        click_frame = make_keyboard_clicks(duration_ms=32, amplitude=0.20)
        silence_frame = make_silence(duration_ms=32)
        vad.process_frame(click_frame)
        vad.process_frame(silence_frame)

    assert len(started) == 0
    assert not vad.is_speech_active


# ------------------------------------------------------------------------------
# Test 6: Traffic Noise Rejection
# ------------------------------------------------------------------------------

def test_06_traffic_noise_rejection():
    """Verify low-frequency traffic rumble is ignored by VAD."""
    vad = VoiceActivityDetector(vad_engine="silero", speech_threshold=0.55)
    started = []
    vad.on_speech_started = lambda: started.append(True)

    for _ in range(25):
        frame = make_traffic_noise(duration_ms=32, amplitude=0.07)
        vad.process_frame(frame)
        assert vad.speech_probability < 0.35

    assert len(started) == 0
    assert not vad.is_speech_active


# ------------------------------------------------------------------------------
# Test 7: Background Music Rejection During Assistant Playback
# ------------------------------------------------------------------------------

def test_07_background_music_rejection():
    """Verify background music during SPEAKING does not cause false barge-in."""
    controller = InterruptionController(
        speech_probability_threshold=0.55,
        confirmation_duration_ms=150,
        min_speech_rms=0.02,
    )
    interrupted = []
    controller.on_interruption_confirmed = lambda p: interrupted.append(True)

    # Simulate SPEAKING state while ambient music plays
    music = make_background_music(duration_ms=32, amplitude=0.08)
    for _ in range(15):
        was_interrupted = controller.process_frame(
            clean_frame=music,
            speech_probability=0.25,
            current_state=VoiceState.SPEAKING,
        )
        assert not was_interrupted

    assert len(interrupted) == 0
    assert not controller.is_interrupted


# ------------------------------------------------------------------------------
# Test 8: HAKS Speaker Output Leaking into Microphone (AEC Suppression)
# ------------------------------------------------------------------------------

def test_08_speaker_leak_aec_suppression():
    """Verify WebRTC AEC suppresses HAKS speaker output leaking into the microphone."""
    apm = WebRTCAPM(enable_aec=True, enable_ns=True)

    # Generate synthetic speaker playback tone
    t = np.linspace(0, 0.5, 8000, endpoint=False)
    spk_audio = (0.35 * np.sin(2 * np.pi * 320.0 * t)).astype(np.float32)

    # Push to reference tracker
    apm.reference_tracker.push_samples(spk_audio, source_sample_rate=16000)

    # Simulate mic capture receiving 80% of speaker leak
    mic_leak = (spk_audio * 0.80).astype(np.float32)

    cleaned_frames = []
    for i in range(0, 4000, 512):
        chunk = mic_leak[i : i + 512]
        clean = apm.process_frame(chunk)
        cleaned_frames.append(clean)

    in_rms = np.sqrt(np.mean(mic_leak[:4000]**2))
    out_rms = np.sqrt(np.mean(np.concatenate(cleaned_frames)**2))
    assert out_rms < in_rms  # Verifies active attenuation of echo


# ------------------------------------------------------------------------------
# Test 9: User Interruption During HAKS Speech (Barge-in Confirmation)
# ------------------------------------------------------------------------------

def test_09_user_interruption_during_speech():
    """Verify genuine human speech during SPEAKING triggers barge-in after temporal confirmation."""
    controller = InterruptionController(
        speech_probability_threshold=0.55,
        confirmation_duration_ms=120,
        min_speech_rms=0.015,
        preroll_duration_ms=300,
    )
    confirmed_events = []
    controller.on_interruption_confirmed = lambda preroll: confirmed_events.append(preroll)

    # Use genuine speech chunk from active portion of utterance
    speech = get_cached_speech_fixture()
    speech_frame = speech[5 * 512 : 6 * 512]
    if len(speech_frame) < 512 or np.sqrt(np.mean(speech_frame**2)) < 0.02:
        speech_frame = make_human_speech(duration_ms=32, amplitude=0.25)

    # 1. First 2 frames of user speech (not yet confirmed due to temporal filter)
    for _ in range(2):
        res = controller.process_frame(speech_frame, speech_probability=0.85, current_state=VoiceState.SPEAKING)
        assert res is False

    # 2. Next 3 frames cross confirmation threshold (total 5 frames = 160ms >= 120ms)
    triggered = False
    for _ in range(3):
        if controller.process_frame(speech_frame, speech_probability=0.88, current_state=VoiceState.SPEAKING):
            triggered = True
            break

    assert triggered is True
    assert len(confirmed_events) == 1
    assert controller.is_interrupted is True
    assert len(confirmed_events[0]) > 0


# ------------------------------------------------------------------------------
# Test 10: Tamil Speech Turn
# ------------------------------------------------------------------------------

def test_10_tamil_speech_turn(temp_workspace: AppConfig):
    """Verify complete Tamil voice turn processing and routing."""
    engine = ContinuousVoiceEngine(config=temp_workspace)
    assert engine.config.voice.preferred_language in ("ta", "en", "mixed")

    mock_res = MagicMock()
    mock_res.text = "வணக்கம், ஹாக்ஸ் எப்படி இருக்கிறீர்கள்?"
    mock_res.language = "ta"
    engine.stt.transcribe = MagicMock(return_value=mock_res)

    audio_frame = get_cached_speech_fixture()[:9600]
    with patch.object(engine, "_generate_ai_response", return_value="நான் நன்றாக இருக்கிறேன்."):
        import asyncio
        loop = asyncio.new_event_loop()
        loop.run_until_complete(engine._process_speech_turn(audio_frame))
        loop.close()

    assert engine._last_user_transcript == "வணக்கம், ஹாக்ஸ் எப்படி இருக்கிறீர்கள்?"


# ------------------------------------------------------------------------------
# Test 11: English Speech Turn
# ------------------------------------------------------------------------------

def test_11_english_speech_turn(temp_workspace: AppConfig):
    """Verify complete English voice turn processing."""
    engine = ContinuousVoiceEngine(config=temp_workspace)
    mock_res = MagicMock()
    mock_res.text = "What is the weather today in Chennai?"
    mock_res.language = "en"
    engine.stt.transcribe = MagicMock(return_value=mock_res)

    audio_frame = get_cached_speech_fixture()[:9600]
    with patch.object(engine, "_generate_ai_response", return_value="It is sunny and 32 degrees."):
        import asyncio
        loop = asyncio.new_event_loop()
        loop.run_until_complete(engine._process_speech_turn(audio_frame))
        loop.close()

    assert engine._last_user_transcript == "What is the weather today in Chennai?"


# ------------------------------------------------------------------------------
# Test 12: Tanglish Speech Turn
# ------------------------------------------------------------------------------

def test_12_tanglish_speech_turn(temp_workspace: AppConfig):
    """Verify code-switched Tanglish speech turn."""
    engine = ContinuousVoiceEngine(config=temp_workspace)
    mock_res = MagicMock()
    mock_res.text = "HAKS next meeting enna time scheduled?"
    mock_res.language = "mixed"
    engine.stt.transcribe = MagicMock(return_value=mock_res)

    audio_frame = get_cached_speech_fixture()[:9600]
    with patch.object(engine, "_generate_ai_response", return_value="Your next meeting is at 3 PM."):
        import asyncio
        loop = asyncio.new_event_loop()
        loop.run_until_complete(engine._process_speech_turn(audio_frame))
        loop.close()

    assert "meeting" in engine._last_user_transcript


# ------------------------------------------------------------------------------
# Test 13: Multiple Consecutive Turns
# ------------------------------------------------------------------------------

def test_13_multiple_consecutive_turns(temp_workspace: AppConfig):
    """Verify continuous mode transitions across multiple turns without hanging."""
    temp_workspace.voice.continuous_mode = True
    engine = ContinuousVoiceEngine(config=temp_workspace)
    engine._conversation_turn_active = True
    engine._state = VoiceState.SPEAKING

    # 1. Turn 1 natural finish -> LISTENING
    engine._on_player_finished()
    assert engine.state == VoiceState.LISTENING

    # 2. Turn 2 user speaks -> PROCESSING
    speech_data = get_cached_speech_fixture()[:6400]
    engine._on_vad_speech_ended(speech_data)
    assert engine.state == VoiceState.PROCESSING

    # 3. Simulate turn 2 assistant finished speaking -> LISTENING
    engine._set_state(VoiceState.SPEAKING)
    engine._on_player_finished()
    assert engine.state == VoiceState.LISTENING


# ------------------------------------------------------------------------------
# Test 14: Silero Model Initialization Failure
# ------------------------------------------------------------------------------

def test_14_silero_init_failure_handling():
    """Verify system handles Silero load failure gracefully without crashing."""
    with patch("silero_vad.load_silero_vad", side_effect=RuntimeError("Model file missing")):
        vad = VoiceActivityDetector(vad_engine="silero")
        # On first frame, lazy loader encounters error and safely activates fallback
        res = vad.process_frame(make_silence(32))
        assert vad.is_fallback_active is True
        assert "Inference error" in (vad.fallback_reason or "") or "Silero VAD" in (vad.fallback_reason or "")
        assert isinstance(res, bool)


# ------------------------------------------------------------------------------
# Test 15: WebRTC Fallback Verification
# ------------------------------------------------------------------------------

def test_15_webrtc_fallback_operation():
    """Verify WebRTC VAD functions properly as explicit or fallback engine."""
    webrtc_engine = WebRTCVADEngine(speech_threshold=0.50)
    assert webrtc_engine.engine_name in ("webrtcvad", "energy-fallback")

    # Feed silence
    is_active, prob = webrtc_engine.process_frame(make_silence(32))
    assert not is_active
    assert prob < 0.30

    webrtc_engine.reset()
    assert not webrtc_engine.is_speech_active


# ------------------------------------------------------------------------------
# Test 16: Audio Device Changes
# ------------------------------------------------------------------------------

def test_16_audio_device_changes(temp_workspace: AppConfig):
    """Verify audio device switch calls reset and flushes buffers."""
    engine = ContinuousVoiceEngine(config=temp_workspace)

    # Push speaker reference data
    engine.apm.reference_tracker.push_samples(np.ones(1000, dtype=np.float32) * 0.2)
    assert len(engine.apm.reference_tracker._buffer) > 0

    # Device change event
    engine.apm.on_device_changed()
    assert len(engine.apm.reference_tracker._buffer) == 0


# ------------------------------------------------------------------------------
# Test 17: Gemini Network Interruption
# ------------------------------------------------------------------------------

def test_17_gemini_network_interruption(temp_workspace: AppConfig):
    """Verify Gemini Live client handles network interruption safely."""
    client = GeminiLiveClient(config=temp_workspace)
    client.connect()
    assert client.is_connected is True

    # Trigger network interruption
    client.on_network_interruption()
    # Queue is drained, stream player stopped, no crash
    assert client.audio_streamer._queue.empty()
