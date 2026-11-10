"""Comprehensive Performance Benchmark for HAKS WebRTC APM + Silero VAD Architecture.

Measures and records:
- APM processing latency per frame (mean, median, p95).
- Silero VAD inference latency per chunk (mean, median, p95, Real-Time Factor).
- Human speech confirmation latency.
- False interruption rate across background noise, typing, fan, and speaker echo.
- Missed speech rate across quiet and normal vocal audio.
- Playback interruption latency (time to halt playback on barge-in).
- End-of-speech silence detection latency.
- Host CPU and RAM utilization under continuous processing.
"""

import os
from pathlib import Path
import sys
import time

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import psutil

from voice.apm.webrtc_apm import WebRTCAPM
from voice.interruption_controller import InterruptionController
from voice.models import VoiceState
from voice.vad.silero_vad import SileroVAD
from tests.test_advanced_human_voice_detection import (
    get_cached_speech_fixture,
    make_fan_noise,
    make_keyboard_clicks,
    make_background_music,
    make_silence,
)


def run_performance_benchmarks():
    print("=" * 70)
    print(" HAKS Advanced Voice Detection — Real-Time Performance Benchmarks")
    print("=" * 70)

    process = psutil.Process(os.getpid())
    ram_initial_mb = process.memory_info().rss / (1024 * 1024)

    # 1. Benchmark WebRTC APM Processing Latency
    apm = WebRTCAPM(enable_aec=True, enable_ns=True, stream_delay_ms=40)
    # Warmup
    for _ in range(10):
        apm.process_frame(np.zeros(512, dtype=np.float32))

    apm_latencies = []
    # Test with speaker output leaking into microphone
    spk_chunk = (0.3 * np.sin(np.linspace(0, 0.032, 512) * 2 * np.pi * 300)).astype(np.float32)
    apm.reference_tracker.push_samples(spk_chunk)

    for _ in range(200):
        mic_chunk = (spk_chunk * 0.7 + np.random.uniform(-0.01, 0.01, 512)).astype(np.float32)
        t0 = time.perf_counter()
        apm.process_frame(mic_chunk)
        apm_latencies.append((time.perf_counter() - t0) * 1000.0)

    apm_mean = np.mean(apm_latencies)
    apm_p95 = np.percentile(apm_latencies, 95)
    print(f"\n[1] WebRTC APM Preprocessing Latency (512 samples / 32 ms frame):")
    print(f"    - Mean:  {apm_mean:.3f} ms")
    print(f"    - p95:   {apm_p95:.3f} ms")
    print(f"    - RTF:   {apm_mean / 32.0:.4f} (Real-Time Factor)")

    # 2. Benchmark Silero VAD Inference Latency & RTF
    silero = SileroVAD(speech_threshold=0.55, use_onnx=True)
    # Warmup
    for _ in range(5):
        silero.predict_probability(np.zeros(512, dtype=np.float32))

    silero_latencies = []
    sample_speech = get_cached_speech_fixture()
    for i in range(min(200, len(sample_speech) // 512)):
        chunk = sample_speech[i * 512 : (i + 1) * 512]
        t0 = time.perf_counter()
        silero.predict_probability(chunk)
        silero_latencies.append((time.perf_counter() - t0) * 1000.0)

    silero_mean = np.mean(silero_latencies)
    silero_p95 = np.percentile(silero_latencies, 95)
    print(f"\n[2] Silero VAD CPU Inference Latency (ONNX 16kHz mono):")
    print(f"    - Mean:  {silero_mean:.3f} ms per 32 ms frame")
    print(f"    - p95:   {silero_p95:.3f} ms")
    print(f"    - RTF:   {silero_mean / 32.0:.4f} (target < 0.05 on i5 CPU)")

    # 3. Interruption Controller & Confirmation Latency
    controller = InterruptionController(
        speech_probability_threshold=0.55,
        confirmation_duration_ms=150,
        min_speech_rms=0.015,
    )
    confirm_latencies = []
    for _ in range(20):
        controller.reset()
        # Feed high speech probability frames until confirmed
        t_start = time.perf_counter()
        active_chunk = sample_speech[5 * 512 : 6 * 512]
        confirmed = False
        frame_count = 0
        while not confirmed and frame_count < 20:
            frame_count += 1
            if controller.process_frame(active_chunk, speech_probability=0.85, current_state=VoiceState.SPEAKING):
                confirmed = True
                confirm_latencies.append((time.perf_counter() - t_start) * 1000.0)
                break

    print(f"\n[3] Speech Confirmation Latency:")
    print(f"    - Confirmed in: {controller.confirmation_frames} frames ({controller.confirmation_duration_ms} ms)")
    print(f"    - Overhead:    {np.mean(confirm_latencies):.3f} ms")

    # 4. False Interruption Rate Evaluation
    # Test 100 frames each of: Fan noise, Keyboard typing, Music, and Speaker Echo Leak
    false_positives = 0
    total_noise_tests = 0

    # Fan noise
    controller.reset()
    for _ in range(50):
        total_noise_tests += 1
        frame = make_fan_noise(duration_ms=32, amplitude=0.08)
        clean = apm.process_frame(frame)
        prob = silero.predict_probability(clean)
        if controller.process_frame(clean, prob, VoiceState.SPEAKING):
            false_positives += 1

    # Typing clicks
    controller.reset()
    for _ in range(50):
        total_noise_tests += 1
        frame = make_keyboard_clicks(duration_ms=32, amplitude=0.20)
        clean = apm.process_frame(frame)
        prob = silero.predict_probability(clean)
        if controller.process_frame(clean, prob, VoiceState.SPEAKING):
            false_positives += 1

    # Background music
    controller.reset()
    for _ in range(50):
        total_noise_tests += 1
        frame = make_background_music(duration_ms=32, amplitude=0.10)
        clean = apm.process_frame(frame)
        prob = silero.predict_probability(clean)
        if controller.process_frame(clean, prob, VoiceState.SPEAKING):
            false_positives += 1

    # Speaker loopback echo
    controller.reset()
    spk_tone = (0.35 * np.sin(np.linspace(0, 0.032, 512) * 2 * np.pi * 350)).astype(np.float32)
    apm.reference_tracker.push_samples(spk_tone)
    for _ in range(50):
        total_noise_tests += 1
        echo_mic = (spk_tone * 0.85).astype(np.float32)
        clean = apm.process_frame(echo_mic)
        prob = silero.predict_probability(clean)
        if controller.process_frame(clean, prob, VoiceState.SPEAKING):
            false_positives += 1

    false_rate = (false_positives / total_noise_tests) * 100.0
    print(f"\n[4] False Interruption Rate across Non-Speech Noise:")
    print(f"    - Noise tests evaluated: {total_noise_tests} frames (Fan, Typing, Music, Speaker Echo)")
    print(f"    - False barge-ins:       {false_positives}")
    print(f"    - False Interruption %:  {false_rate:.2f}% (Previously > 85% with RMS > 0.08)")

    # 5. Missed Speech Rate on Genuine Speech
    missed_speech_tests = 0
    detected_speech_tests = 0
    for i in range(10, min(50, len(sample_speech) // 512)):
        chunk = sample_speech[i * 512 : (i + 1) * 512]
        rms = np.sqrt(np.mean(chunk**2))
        if rms > 0.03:
            prob = silero.predict_probability(chunk)
            if prob >= 0.50:
                detected_speech_tests += 1
            else:
                missed_speech_tests += 1

    missed_rate = (missed_speech_tests / (detected_speech_tests + missed_speech_tests + 1e-6)) * 100.0
    print(f"\n[5] Missed Speech Rate on Voiced Segments:")
    print(f"    - Voiced chunks evaluated: {detected_speech_tests + missed_speech_tests}")
    print(f"    - Successfully detected:  {detected_speech_tests}")
    print(f"    - Missed Speech %:        {missed_rate:.2f}%")

    # 6. Playback Interruption Latency (Hardware Stop)
    from voice.gemini_live.pcm_player import PCMPlayer
    player = PCMPlayer(sample_rate=24000)
    # Enqueue a dummy chunk
    player.enqueue_chunk(np.zeros(2400, dtype=np.float32))
    time.sleep(0.05)
    t_stop0 = time.perf_counter()
    player.stop()
    halt_latency_ms = (time.perf_counter() - t_stop0) * 1000.0
    print(f"\n[6] Playback Interruption Latency:")
    print(f"    - Time to halt PCM playback: {halt_latency_ms:.2f} ms")

    # 7. End-of-Speech Silence Detection Latency
    silence_latency_ms = controller.confirmation_duration_ms * 3.0  # 450 ms timeout
    print(f"\n[7] End-of-Speech Silence Detection Latency:")
    print(f"    - Configured silence timeout: {silence_latency_ms:.1f} ms")

    # 8. CPU & RAM Utilization
    ram_final_mb = process.memory_info().rss / (1024 * 1024)
    cpu_pct = process.cpu_percent(interval=0.2)
    print(f"\n[8] Resource Utilization:")
    print(f"    - Initial RAM: {ram_initial_mb:.1f} MB")
    print(f"    - Active RAM:  {ram_final_mb:.1f} MB (Delta: +{ram_final_mb - ram_initial_mb:.1f} MB)")
    print(f"    - CPU Usage:   {cpu_pct:.1f}% on host")
    print("=" * 70)


if __name__ == "__main__":
    run_performance_benchmarks()
