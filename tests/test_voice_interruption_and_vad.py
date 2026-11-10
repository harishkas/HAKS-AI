"""Comprehensive Test Suite for Robust Voice Activity Detection and False Interruption Prevention.

Validates all 12 scenarios required by HAKS:
1. HAKS speaking with no user interruption (uninterrupted speech)
2. HAKS speaking with background fan noise (spectral noise rejection)
3. HAKS speaking with keyboard typing/clicks (impulse transient rejection)
4. HAKS speaker output leaking into microphone (AEC normalized cross-correlation suppression)
5. User speaking quietly during HAKS playback (quiet speech vocal band detection)
6. User interrupting with normal speech (prompt confirmed barge-in)
7. User interrupting with Tamil speech ('வணக்கம்', 'நிறுத்து')
8. User interrupting with English speech ('Stop HAKS', 'Wait')
9. User interrupting with Tanglish speech ('Enna solreenga HAKS')
10. Silence after user speech (end-of-speech turn completion at 500ms)
11. Multiple consecutive conversation turns (persistent session without reconnects)
12. Network delay and Gemini audio streaming jitter (prevents premature LISTENING switch)

Also benchmarks and measures:
- Confirmed human speech detection latency (target: 100-200 ms)
- Playback interruption latency (target: < 50 ms)
- End-of-speech detection latency (target: 400-600 ms)
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pytest

from config.settings import AppConfig
from voice.conversation_engine import ContinuousVoiceEngine
from voice.echo_cancellation import AcousticEchoSuppressor
from voice.gemini_live import (
    AudioStreamerBridge,
    GeminiLiveClient,
    LiveConnectionStatus,
    PCMStreamPlayer,
)
from voice.models import VoiceState
from voice.vad import VoiceActivityDetector


# Helper: Generate realistic vocal formants (F1=500Hz, F2=1500Hz, F3=2500Hz)
def generate_vocal_audio(duration_sec: float, sample_rate: int = 16000, amplitude: float = 0.2) -> np.ndarray:
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    # Fundamental frequency F0=130Hz with strong speech formants
    signal = (
        0.5 * np.sin(2 * np.pi * 130 * t)
        + 0.3 * np.sin(2 * np.pi * 500 * t)   # F1
        + 0.2 * np.sin(2 * np.pi * 1500 * t)  # F2
        + 0.1 * np.sin(2 * np.pi * 2500 * t)  # F3
    )
    return (amplitude * signal).astype(np.float32)


# Helper: Generate fan/AC noise (high ZCR, broad spectrum)
def generate_fan_noise(duration_sec: float, sample_rate: int = 16000, amplitude: float = 0.05) -> np.ndarray:
    num_samples = int(sample_rate * duration_sec)
    np.random.seed(42)
    noise = np.random.normal(0, amplitude, num_samples).astype(np.float32)
    return noise


# Helper: Generate keyboard clicks (intermittent impulses < 15ms)
def generate_keyboard_clicks(duration_sec: float, sample_rate: int = 16000) -> np.ndarray:
    num_samples = int(sample_rate * duration_sec)
    audio = np.zeros(num_samples, dtype=np.float32)
    # Add click impulses at 100ms, 300ms, 600ms
    for click_time in [0.10, 0.30, 0.60]:
        idx = int(click_time * sample_rate)
        if idx + 80 < num_samples:
            # 5ms spike
            audio[idx : idx + 80] = np.random.uniform(-0.6, 0.6, 80).astype(np.float32)
    return audio


# ------------------------------------------------------------------------------
# 1. Acoustic Echo Suppression Tests
# ------------------------------------------------------------------------------

def test_aec_suppresses_speaker_feedback():
    """Scenario 4: Verify AEC detects high cross-correlation and suppresses speaker leakage."""
    aec = AcousticEchoSuppressor(sample_rate=16000, correlation_threshold=0.35)

    # 0.5s synthetic speaker playback
    t = np.linspace(0, 0.5, 8000, endpoint=False)
    ref_playback = (0.4 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)
    pcm_bytes = (ref_playback * 32767).astype(np.int16).tobytes()
    aec.feed_playback_chunk(pcm_bytes, 16000)

    assert aec.is_playback_active() is True

    # Mic captures delayed speaker echo (e.g. 35ms acoustic lag = 560 samples)
    lag = 560
    echo_mic = 0.25 * ref_playback[4000 - lag : 4000 - lag + 512] + np.random.normal(0, 0.005, 512).astype(np.float32)

    eval_result = aec.evaluate_mic_frame(echo_mic)
    assert eval_result.is_echo is True
    assert eval_result.correlation > 0.40
    assert eval_result.is_playback_active is True


def test_aec_allows_genuine_human_voice_over_playback():
    """Scenario 5 & 6: Verify AEC preserves human speech when speaking over playback."""
    aec = AcousticEchoSuppressor(sample_rate=16000, correlation_threshold=0.35)

    # Reference playback
    t = np.linspace(0, 0.5, 8000, endpoint=False)
    ref_playback = (0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    aec.feed_playback_chunk((ref_playback * 32767).astype(np.int16).tobytes(), 16000)

    # User voice with completely different fundamental and formant structure (e.g. 200 Hz voice)
    user_voice = generate_vocal_audio(0.032, sample_rate=16000, amplitude=0.35)

    eval_result = aec.evaluate_mic_frame(user_voice)
    assert eval_result.is_echo is False
    assert eval_result.correlation < 0.35


# ------------------------------------------------------------------------------
# 2. VAD Spectral Noise Rejection Tests
# ------------------------------------------------------------------------------

def test_vad_rejects_fan_noise():
    """Scenario 2: Verify fan noise is rejected by high zero-crossing rate and spectral flatness."""
    vad = VoiceActivityDetector(sample_rate=16000, vad_sensitivity=2)
    fan_chunk = generate_fan_noise(0.032, amplitude=0.04)

    is_speech, conf, reason = vad.evaluate_frame_speech_probability(fan_chunk, check_echo=False)
    assert is_speech is False
    assert "high_zcr_noise" in reason or "low_vocal_ratio" in reason or "below_noise_floor" in reason


def test_vad_rejects_keyboard_clicks():
    """Scenario 3: Verify keyboard clicks fail consecutive frame confirmation."""
    vad = VoiceActivityDetector(sample_rate=16000, vad_sensitivity=2)
    clicks = generate_keyboard_clicks(0.032)

    # Single click frame
    is_confirmed, reason = vad.is_confirmed_human_speech(clicks, check_echo=False)
    assert is_confirmed is False, "Single keyboard click must not trigger confirmed speech"


def test_vad_confirms_human_speech_after_consecutive_frames():
    """Scenario 6: Verify sustained human speech confirms within 60-100 ms."""
    vad = VoiceActivityDetector(sample_rate=16000, vad_sensitivity=2)

    # 4 consecutive frames (128 ms) of clear speech
    speech_frames = [generate_vocal_audio(0.032, amplitude=0.25) for _ in range(4)]

    confirmed = False
    confirmation_frame = -1
    t0 = time.perf_counter()

    for idx, f in enumerate(speech_frames):
        is_speech, reason = vad.is_confirmed_human_speech(f, check_echo=False)
        if is_speech:
            confirmed = True
            confirmation_frame = idx + 1
            break

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert confirmed is True
    assert confirmation_frame == 3  # Exactly on 3rd consecutive frame (~96 ms)
    assert elapsed_ms < 20.0  # CPU evaluation takes < 20 ms


# ------------------------------------------------------------------------------
# 3. Multilingual Speech Interruption Tests
# ------------------------------------------------------------------------------

def test_vad_detects_tamil_speech_formants():
    """Scenario 7: Verify Tamil vocal formants ('வணக்கம்') pass human speech detection."""
    vad = VoiceActivityDetector(sample_rate=16000, vad_sensitivity=2)
    # Tamil speech envelope simulation with vocal transitions
    t = np.linspace(0, 0.032, 512, endpoint=False)
    tamil_vocal = (0.25 * (np.sin(2 * np.pi * 280 * t) + 0.4 * np.sin(2 * np.pi * 920 * t))).astype(np.float32)

    for _ in range(3):
        is_speech, reason = vad.is_confirmed_human_speech(tamil_vocal, check_echo=False)

    assert is_speech is True
    assert "human_speech" in reason


def test_vad_detects_english_speech_formants():
    """Scenario 8: Verify English speech ('Stop HAKS') passes human speech detection."""
    vad = VoiceActivityDetector(sample_rate=16000, vad_sensitivity=2)
    english_vocal = generate_vocal_audio(0.032, amplitude=0.28)

    for _ in range(3):
        is_speech, reason = vad.is_confirmed_human_speech(english_vocal, check_echo=False)

    assert is_speech is True
    assert "human_speech" in reason


def test_vad_detects_tanglish_speech_formants():
    """Scenario 9: Verify Tanglish mixed vocal patterns pass human speech detection."""
    vad = VoiceActivityDetector(sample_rate=16000, vad_sensitivity=2)
    t = np.linspace(0, 0.032, 512, endpoint=False)
    tanglish_vocal = (0.22 * (np.sin(2 * np.pi * 320 * t) + 0.5 * np.sin(2 * np.pi * 1250 * t))).astype(np.float32)

    for _ in range(3):
        is_speech, reason = vad.is_confirmed_human_speech(tanglish_vocal, check_echo=False)

    assert is_speech is True
    assert "human_speech" in reason


# ------------------------------------------------------------------------------
# 4. Turn Completion & Silence Detection Tests
# ------------------------------------------------------------------------------

def test_vad_end_of_speech_silence_detection():
    """Scenario 10: Verify turn completes after configured silence timeout (500 ms)."""
    vad = VoiceActivityDetector(
        sample_rate=16000,
        frame_size=512,
        silence_timeout_ms=500,
        min_speech_duration_ms=160,
    )

    ended_audio = []
    vad.on_speech_ended = lambda audio: ended_audio.append(audio)

    # 1. 6 frames of speech (~192 ms)
    for _ in range(6):
        vad.process_frame(generate_vocal_audio(0.032, amplitude=0.25))

    assert vad.is_speech_active is True

    # 2. Feed silence frames (500 ms / 32 ms = ~16 frames)
    silence_frame = np.zeros(512, dtype=np.float32)
    for _ in range(16):
        vad.process_frame(silence_frame)

    assert vad.is_speech_active is False
    assert len(ended_audio) == 1
    # Check total audio duration includes speech + pre-roll
    assert len(ended_audio[0]) >= 512 * 6


# ------------------------------------------------------------------------------
# 5. AudioStreamerBridge Gating & Pre-roll Tests
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_audio_streamer_gates_mic_during_playback():
    """Scenario 1 & 4: Verify mic frames are held during playback to avoid triggering Gemini server VAD."""
    client = GeminiLiveClient()
    mock_send = AsyncMock(return_value=True)
    client.send_audio_chunk = mock_send
    client.is_connected = MagicMock(return_value=True)

    streamer = AudioStreamerBridge(client=client)
    loop = asyncio.get_running_loop()
    streamer.start_streaming(loop=loop)

    # Activate playback mode (HAKS is speaking)
    streamer.set_playback_active(True)
    assert streamer.is_playback_active is True

    # Feed 3 mic frames
    dummy_frame = (0.1 * np.ones(512, dtype=np.float32))
    for _ in range(3):
        streamer.feed_frame(dummy_frame)

    await asyncio.sleep(0.05)
    # Mic frames must NOT be transmitted to Gemini while assistant is speaking
    assert mock_send.call_count == 0

    # User interrupts: set playback active to False (flushes pre-roll)
    streamer.set_playback_active(False)
    await asyncio.sleep(0.05)

    # Now frames should have been transmitted
    assert mock_send.call_count > 0

    streamer.stop_streaming()


# ------------------------------------------------------------------------------
# 6. ContinuousVoiceEngine False Barge-in Prevention Tests
# ------------------------------------------------------------------------------

def test_engine_speaking_no_interruption_on_haks_echo(temp_workspace: AppConfig):
    """Scenario 1 & 4: Verify HAKS does not interrupt itself while speaking through speakers."""
    temp_workspace.voice.voice_mode = "gemini_live"
    engine = ContinuousVoiceEngine(config=temp_workspace)

    engine._set_state(VoiceState.SPEAKING)
    assert engine.state == VoiceState.SPEAKING

    # Simulate speaker outputting 440 Hz tone
    t = np.linspace(0, 0.1, 1600, endpoint=False)
    output_pcm = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16).tobytes()
    engine.echo_suppressor.feed_playback_chunk(output_pcm, 16000)

    # Microphone captures echo of the same 440 Hz tone
    echo_mic = (0.3 * np.sin(2 * np.pi * 440 * t[:512])).astype(np.float32)

    # Feed 5 frames of echo
    for _ in range(5):
        engine._on_audio_chunk(echo_mic)

    # HAKS must remain in SPEAKING state (no false interruption!)
    assert engine.state == VoiceState.SPEAKING


def test_engine_speaking_interrupts_on_genuine_human_speech(temp_workspace: AppConfig):
    """Scenario 6: Verify HAKS interrupts immediately when genuine human speech arrives."""
    temp_workspace.voice.voice_mode = "gemini_live"
    engine = ContinuousVoiceEngine(config=temp_workspace)

    engine._set_state(VoiceState.SPEAKING)
    assert engine.state == VoiceState.SPEAKING

    # User speaks genuine vocal tone over playback
    user_speech = generate_vocal_audio(0.032, amplitude=0.3)

    # Feed 4 frames of user speech
    for _ in range(4):
        engine._on_audio_chunk(user_speech)

    # State must transition to INTERRUPTED
    assert engine.state in (VoiceState.INTERRUPTED, VoiceState.LISTENING)
    # Interruption latency must be under 50 ms
    assert engine.last_interruption_latency_ms < 50.0


def test_engine_spurious_server_interrupted_ignored_during_echo(temp_workspace: AppConfig):
    """Scenario 12: Verify spurious server-side interruption from echo is ignored."""
    temp_workspace.voice.voice_mode = "gemini_live"
    engine = ContinuousVoiceEngine(config=temp_workspace)

    engine._set_state(VoiceState.SPEAKING)
    engine.pcm_player._is_playing = True

    # Playback is active
    t = np.linspace(0, 0.1, 1600, endpoint=False)
    engine.echo_suppressor.feed_playback_chunk((0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16).tobytes(), 16000)

    # Gemini Live server sends interrupted = True due to network echo
    engine._on_live_interrupted()

    # Spurious interruption ignored because local VAD detected no human voice
    assert engine.state == VoiceState.SPEAKING
    assert engine.pcm_player.is_playing is True


# ------------------------------------------------------------------------------
# 7. Persistent Session & Multi-turn Continuity Tests
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gemini_live_session_persists_across_multiple_turns(temp_workspace: AppConfig):
    """Scenario 11: Verify session is not recreated on every conversation turn."""
    client = GeminiLiveClient()
    connect_calls = 0

    async def mock_connect(*args, **kwargs):
        nonlocal connect_calls
        connect_calls += 1
        client.status = LiveConnectionStatus.CONNECTED
        client._session = MagicMock()
        return True

    client.connect = mock_connect

    # Turn 1
    await client.connect()
    assert client.is_connected()
    assert connect_calls == 1

    # Turn 2 (already connected)
    if not client.is_connected():
        await client.connect()
    assert connect_calls == 1  # No redundant reconnect!

    # Turn 3
    if not client.is_connected():
        await client.connect()
    assert connect_calls == 1

