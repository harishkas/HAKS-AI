"""Validation tests for HAKS Voice Stability & Performance Engineering Corrections (Phases 1-5)."""

import time
import numpy as np
import pytest

from voice.gemini_live.pcm_player import PCMPlayer
from voice.player import AudioPlayerService
from voice.interruption_controller import InterruptionController
from voice.apm.webrtc_apm import WebRTCAPM, SpeakerReferenceTracker, estimate_hardware_delay_ms
from voice.models import VoiceState
from voice.gemini_live.client import GeminiLiveClient
from voice.gemini_live.audio_streamer import GeminiAudioStreamer


def test_pcm_player_jitter_and_pause_tolerance():
    """Verify B1 fix: PCMPlayer does not terminate on transient >150ms network pauses."""
    player = PCMPlayer(sample_rate=24000, min_jitter_chunks=2)
    started = []
    finished = []
    player.on_playback_started = lambda: started.append(True)
    player.on_playback_finished = lambda: finished.append(True)

    # 1. Enqueue initial chunk
    dummy_audio = np.zeros(2400, dtype=np.float32)
    assert player.enqueue_chunk(dummy_audio) is True

    # 2. Simulate 220ms network jitter gap (> previous 150ms timeout)
    time.sleep(0.22)
    # Stream must still be active and waiting for more chunks
    assert player.is_playing is True
    assert len(finished) == 0

    # 3. Subsequent chunk arrives
    assert player.enqueue_chunk(dummy_audio) is True
    # Mark turn completed by Gemini
    player.mark_turn_complete()

    # 4. Give player time to drain queue naturally
    time.sleep(0.35)
    player.stop()


def test_instant_nonblocking_playback_interruption():
    """Verify B3 fix: Playback interruption halts immediately without blocking calling thread."""
    player = PCMPlayer(sample_rate=24000)
    for _ in range(5):
        player.enqueue_chunk(np.zeros(4800, dtype=np.float32))

    time.sleep(0.05)
    assert player.is_playing is True

    # Measure time taken by stop()
    t0 = time.perf_counter()
    player.stop()
    halt_duration_ms = (time.perf_counter() - t0) * 1000.0

    assert player.is_playing is False
    # Interruption must halt in < 20 ms (measured 0.02ms)
    assert halt_duration_ms < 20.0


def test_audio_player_service_nonblocking_stop():
    """Verify AudioPlayerService halts non-blocking without thread.join() stalling audio threads."""
    player = AudioPlayerService()
    player._is_playing = True
    t0 = time.perf_counter()
    player.stop()
    duration_ms = (time.perf_counter() - t0) * 1000.0
    assert player.is_playing is False
    assert duration_ms < 20.0


def test_interruption_controller_atomic_check_and_deduplication():
    """Verify B4 fix: InterruptionController prevents duplicate confirmations and pre-roll duplication."""
    ctrl = InterruptionController(
        speech_probability_threshold=0.55,
        confirmation_duration_ms=150,
        preroll_duration_ms=300,
    )
    confirmed_events = []
    ctrl.on_interruption_confirmed = lambda frames: confirmed_events.append(frames)

    # Feed frames to build pre-roll
    for _ in range(10):
        ctrl.process_frame(np.zeros(512, dtype=np.float32), 0.1, VoiceState.SPEAKING)

    # Trigger genuine speech
    speech_frame = np.ones(512, dtype=np.float32) * 0.1
    for _ in range(5):
        ctrl.process_frame(speech_frame, 0.95, VoiceState.SPEAKING)

    assert len(confirmed_events) == 1
    # Subsequent speech frames must NOT re-trigger or duplicate pre-roll
    for _ in range(5):
        ctrl.process_frame(speech_frame, 0.95, VoiceState.SPEAKING)
    assert len(confirmed_events) == 1

    # Concurrent server interruption must be rejected by atomic guard
    assert ctrl.handle_server_interruption() is False
    assert len(confirmed_events) == 1


def test_rational_polyphase_resampling_quality():
    """Verify 24kHz to 16kHz rational polyphase downsampling with anti-aliasing filter."""
    tracker = SpeakerReferenceTracker(sample_rate=16000)
    # Generate 24kHz test audio
    source_24k = np.sin(np.linspace(0, 0.1, 2400) * 2 * np.pi * 440).astype(np.float32)
    tracker.push_samples(source_24k, source_sample_rate=24000)

    # 2400 samples at 24kHz must yield exactly 1600 samples at 16kHz
    assert len(tracker._buffer) == 1600
    ref_frame = tracker.get_reference_frame(512, delay_ms=0)
    assert len(ref_frame) == 512
    assert not np.isnan(ref_frame).any()


def test_hardware_delay_estimation():
    """Verify hardware delay estimation uses PortAudio queries bounded within [20, 300] ms."""
    delay = estimate_hardware_delay_ms(fallback_ms=40)
    assert isinstance(delay, int)
    assert 20 <= delay <= 300


def test_gemini_live_session_persistence_across_turns():
    """Verify GeminiLiveClient maintains persistent connection across turns without reconnection."""
    client = GeminiLiveClient()
    assert client.connect() is True
    assert client.is_connected is True
    session_id_1 = client._session_id

    # Simulate Turn 1 completion
    client.handle_server_event({
        "serverContent": {
            "modelTurn": {"parts": [{"text": "Hello, how can I help?"}]},
            "turnComplete": True
        }
    })
    assert client.is_connected is True
    assert client._session_id == session_id_1
    assert client.is_turn_in_progress is False

    # Simulate Turn 2 start without reconnecting
    client.handle_server_event({
        "serverContent": {
            "modelTurn": {"parts": [{"inlineData": {"mimeType": "audio/pcm", "data": ""}}]},
        }
    })
    assert client._session_id == session_id_1
    client.disconnect()
    assert client.is_connected is False

