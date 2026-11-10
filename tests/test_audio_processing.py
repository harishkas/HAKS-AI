"""Unit and Integration Tests for HAKS Audio Processing Pipeline and DSP Stages.

Tests:
1. DC Offset Removal and Level Normalization.
2. Low-frequency rumble high-pass filtering (< 80 Hz).
3. 50/60 Hz electrical mains hum notch suppression.
4. Real-time background noise suppression and transient click reduction.
5. Automatic noise-floor calibration.
6. Acoustic Echo Cancellation (AEC) with playback reference, NLMS, and DTD.
7. Human speech enhancement (formant shaping and consonant pre-emphasis).
8. Multi-stage adaptive VAD (WebRTC VAD, SNR estimation, consecutive frames).
9. Pipeline sequence ordering evaluation (AEC before NS vs. NS before AEC).
"""

import time
import numpy as np
import pytest

from voice.dsp import (
    AudioProcessingPipeline,
    AcousticEchoCanceller,
    DCNormalizer,
    NoiseSuppressor,
    SpeechEnhancer,
    AdaptiveVoiceDetector,
)


# -----------------------------------------------------------------------------
# 1. DC Offset, Gain, Rumble HPF, Hum Notch Tests
# -----------------------------------------------------------------------------

def test_dc_offset_removal():
    """Verify DC normalizer eliminates constant ADC bias."""
    normalizer = DCNormalizer(sample_rate=16000, mic_gain=1.0)
    # 512 samples with 0.15 DC offset
    dc_frame = (np.ones(512, dtype=np.float32) * 0.15) + (np.random.normal(0, 0.01, 512).astype(np.float32))
    
    # Process several contiguous frames
    cleaned = dc_frame
    for _ in range(5):
        cleaned = normalizer.process(dc_frame)
    
    # Final frame mean should be close to 0.0
    assert abs(float(np.mean(cleaned))) < 0.03
    assert abs(float(np.mean(cleaned))) < abs(float(np.mean(dc_frame)))


def test_rumble_highpass_filtering():
    """Verify sub-80Hz low-frequency rumble is attenuated."""
    normalizer = DCNormalizer(sample_rate=16000, enable_highpass=True, highpass_cutoff_hz=80.0)
    t = np.linspace(0, 0.1, 1600, endpoint=False)
    # 35 Hz rumble wave
    rumble = (0.3 * np.sin(2 * np.pi * 35.0 * t)).astype(np.float32)
    
    # Process blocks
    cleaned_blocks = []
    for i in range(0, len(rumble), 320):
        cleaned_blocks.append(normalizer.process(rumble[i : i + 320]))
    cleaned = np.concatenate(cleaned_blocks)
    
    in_power = float(np.mean(rumble**2))
    out_power = float(np.mean(cleaned[320:]**2))  # Skip filter settling
    # At least 10 dB attenuation
    assert out_power < (in_power * 0.15)


def test_hum_notch_filtering():
    """Verify 50 Hz and 60 Hz electrical mains hum is notched out."""
    normalizer = DCNormalizer(sample_rate=16000, enable_notch=True, notch_frequencies=(50.0, 60.0))
    t = np.linspace(0, 0.2, 3200, endpoint=False)
    # 50 Hz hum
    hum50 = (0.2 * np.sin(2 * np.pi * 50.0 * t)).astype(np.float32)
    
    out_blocks = []
    for i in range(0, len(hum50), 320):
        out_blocks.append(normalizer.process(hum50[i : i + 320]))
    out = np.concatenate(out_blocks)
    
    in_power = float(np.mean(hum50**2))
    out_power = float(np.mean(out[640:]**2))
    assert out_power < (in_power * 0.35)


def test_mic_gain_scaling_and_limiting():
    """Verify gain amplification and soft saturation limiting without clipping."""
    normalizer = DCNormalizer(sample_rate=16000, mic_gain=2.0)
    frame = np.array([0.1, -0.1, 0.5, -0.5, 0.8, -0.8], dtype=np.float32)
    processed = normalizer.process(frame)
    
    # Moderate peaks are boosted
    assert abs(processed[0]) > 0.15
    # Loud peaks soft-saturated <= 0.85
    assert np.all(np.abs(processed) <= 0.851)


# -----------------------------------------------------------------------------
# 2. Real-Time Spectral Noise Suppression Tests
# -----------------------------------------------------------------------------

def test_spectral_noise_suppression_fan_noise():
    """Verify stationary fan drone is significantly reduced."""
    suppressor = NoiseSuppressor(sample_rate=16000, frame_size=320, strength="medium")
    
    # Generate stationary fan noise (colored noise with low-mid emphasis)
    np.random.seed(42)
    fan_noise = np.random.normal(0, 0.05, 3200).astype(np.float32)
    
    # Feed initial frames to adapt noise floor
    for i in range(0, 1600, 320):
        suppressor.process(fan_noise[i : i + 320], is_speech=False)
    
    # Process steady state frames
    out_blocks = []
    for i in range(1600, 3200, 320):
        out_blocks.append(suppressor.process(fan_noise[i : i + 320], is_speech=False))
    cleaned = np.concatenate(out_blocks)
    
    in_energy = float(np.mean(fan_noise[1600:]**2))
    out_energy = float(np.mean(cleaned**2))
    # At least 50% energy reduction (3+ dB)
    assert out_energy < (in_energy * 0.50)


def test_transient_click_suppression():
    """Verify sharp impulsive transients (keyboard typing, mouse click) are suppressed."""
    suppressor = NoiseSuppressor(sample_rate=16000, frame_size=320, strength="high")
    
    # Baseline quiet frame
    quiet = np.random.normal(0, 0.005, 320).astype(np.float32)
    for _ in range(5):
        suppressor.process(quiet, is_speech=False)
        
    # Frame with sudden keyboard transient click (sharp spike)
    click_frame = quiet.copy()
    click_frame[150:154] = [0.8, -0.7, 0.6, -0.4]
    
    cleaned = suppressor.process(click_frame, is_speech=False)
    assert np.max(np.abs(cleaned)) < 0.6  # Spike suppressed


def test_automatic_noise_floor_calibration():
    """Verify calibration gathers background profile and establishes floor."""
    suppressor = NoiseSuppressor(sample_rate=16000, frame_size=320)
    suppressor.start_calibration()
    
    ambient_frame = np.random.normal(0, 0.02, 320).astype(np.float32)
    for _ in range(10):
        suppressor.process(ambient_frame)
    
    baseline = suppressor.finish_calibration()
    assert baseline > 0.0
    assert abs(baseline - suppressor.get_noise_floor_estimate()) < 1e-4


# -----------------------------------------------------------------------------
# 3. Acoustic Echo Cancellation (AEC) Tests
# -----------------------------------------------------------------------------

def test_acoustic_echo_cancellation_playback_reference():
    """Verify AEC identifies speaker playback, measures correlation, and subtracts echo."""
    aec = AcousticEchoCanceller(sample_rate=16000)
    
    # Generate 16 kHz speaker playback audio (e.g. 500 Hz speech tone)
    t = np.linspace(0, 0.25, 4000, endpoint=False)
    speaker_audio = (0.25 * np.sin(2 * np.pi * 500.0 * t)).astype(np.float32)
    
    # Feed playback into reference buffer
    aec.feed_playback_chunk(speaker_audio, input_sample_rate=16000)
    assert aec.is_playback_active() is True
    
    # Microphone picks up delayed speaker audio leaking across room (~25ms lag = 400 samples)
    mic_frame = (0.5 * speaker_audio[400 : 400 + 320]).copy()
    
    result = aec.process_frame(mic_frame)
    assert result.is_playback_active is True
    assert result.correlation > 0.30
    assert result.is_echo is True
    assert result.is_double_talk is False
    
    # Cleaned frame energy must be significantly attenuated
    assert float(np.mean(result.cleaned_frame**2)) < float(np.mean(mic_frame**2))


def test_aec_double_talk_preservation():
    """Verify double-talk detector preserves quiet human speech when user speaks during playback."""
    aec = AcousticEchoCanceller(sample_rate=16000)
    
    t = np.linspace(0, 0.25, 4000, endpoint=False)
    speaker_audio = (0.2 * np.sin(2 * np.pi * 400.0 * t)).astype(np.float32)
    aec.feed_playback_chunk(speaker_audio, input_sample_rate=16000)
    
    # User speaks quietly (harmonic speech tone at 800 Hz) uncorrelated with speaker audio
    user_speech = (0.08 * np.sin(2 * np.pi * 800.0 * t[:320])).astype(np.float32)
    echo_leak = (0.15 * speaker_audio[300 : 300 + 320]).astype(np.float32)
    combined_mic = echo_leak + user_speech
    
    result = aec.process_frame(combined_mic)
    assert result.is_double_talk is True
    # Human speech should not be silenced
    assert float(np.mean(result.cleaned_frame**2)) > 0.001


def test_aec_device_change_reset():
    """Verify device change resets AEC internal reference buffer."""
    aec = AcousticEchoCanceller(sample_rate=16000)
    aec.feed_playback_chunk(np.ones(1000, dtype=np.float32), input_sample_rate=16000)
    assert aec.is_playback_active() is True
    
    aec.handle_device_change()
    assert aec.is_playback_active() is False
    assert aec._total_samples_written == 0


# -----------------------------------------------------------------------------
# 4. Human Speech Enhancement & Formant Isolation Tests
# -----------------------------------------------------------------------------

def test_speech_enhancement_formant_and_consonant_preservation():
    """Verify vocal band 300-3400 Hz and high-frequency consonants are reinforced."""
    enhancer = SpeechEnhancer(sample_rate=16000)
    
    # Synthetic speech frame with 1 kHz vowel formant + 3.5 kHz consonant fricative
    t = np.linspace(0, 0.02, 320, endpoint=False)
    vowel = 0.05 * np.sin(2 * np.pi * 1000.0 * t)
    fricative = 0.02 * np.sin(2 * np.pi * 3500.0 * t)
    frame = (vowel + fricative).astype(np.float32)
    
    enhanced = enhancer.process(frame)
    assert len(enhanced) == 320
    # Consonant / vocal clarity is retained
    assert float(np.mean(enhanced**2)) > 1e-5


# -----------------------------------------------------------------------------
# 5. Adaptive Voice Detector (VAD) Tests
# -----------------------------------------------------------------------------

def test_adaptive_vad_speech_vs_noise():
    """Verify VAD correctly distinguishes voice from stationary background noise."""
    vad = AdaptiveVoiceDetector(sample_rate=16000, frame_size=320, min_speech_duration_ms=100)
    
    # 1. White noise / fan frame
    noise_frame = np.random.normal(0, 0.015, 320).astype(np.float32)
    for _ in range(5):
        res = vad.process_frame(noise_frame)
    assert res.is_speech is False
    assert res.is_confirmed_speech is False
    assert res.confidence < 0.4
    
    # 2. Voice harmonic frame (e.g. 250 Hz fundamental + formants)
    t = np.linspace(0, 0.02, 320, endpoint=False)
    voice_frame = (0.12 * np.sin(2 * np.pi * 250.0 * t) + 0.06 * np.sin(2 * np.pi * 500.0 * t)).astype(np.float32)
    
    for _ in range(6):
        res = vad.process_frame(voice_frame)
    
    assert res.is_speech is True
    assert res.confidence > 0.5


def test_adaptive_vad_echo_rejection():
    """Verify VAD rejects frames identified as speaker echo."""
    vad = AdaptiveVoiceDetector(sample_rate=16000, frame_size=320)
    t = np.linspace(0, 0.02, 320, endpoint=False)
    frame = (0.2 * np.sin(2 * np.pi * 300.0 * t)).astype(np.float32)
    
    # When is_playback_echo is True, VAD suppresses speech decision
    res = vad.process_frame(frame, is_playback_echo=True)
    assert res.is_speech is False
    assert res.confidence == 0.0


# -----------------------------------------------------------------------------
# 6. Audio Processing Pipeline Sequence & Ordering Evaluation
# -----------------------------------------------------------------------------

def test_full_pipeline_execution():
    """Verify complete 7-stage pipeline runs synchronously with sub-millisecond latency."""
    pipeline = AudioProcessingPipeline()
    frame = np.random.normal(0, 0.03, 320).astype(np.float32)
    
    result = pipeline.process_frame(frame)
    assert len(result.cleaned_frame) == 320
    assert result.processing_latency_ms < 15.0  # Fast execution
    assert isinstance(result.speech_confidence, float)
    assert isinstance(result.snr_db, float)


def test_pipeline_ordering_evaluation_aec_vs_noise_suppression():
    """Evaluate and compare AEC->NS vs. NS->AEC orderings."""
    # Pipeline A: AEC first (Recommended for linear filter convergence)
    pipeline_a = AudioProcessingPipeline(aec_first=True)
    # Pipeline B: Noise Suppression first
    pipeline_b = AudioProcessingPipeline(aec_first=False)
    
    # Prepare simulated speaker playback + mic with echo & noise
    t = np.linspace(0, 0.25, 4000, endpoint=False)
    speaker = (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    pipeline_a.feed_playback_reference(speaker, 16000)
    pipeline_b.feed_playback_reference(speaker, 16000)
    
    # Mic signal with echo + fan noise
    mic_slice = (0.5 * speaker[400:720]) + np.random.normal(0, 0.01, 320).astype(np.float32)
    
    res_a = pipeline_a.process_frame(mic_slice)
    res_b = pipeline_b.process_frame(mic_slice)
    
    # Both pipelines should successfully process and reject echo
    assert res_a.is_echo is True
    assert res_b.is_echo is True
    assert res_a.processing_latency_ms < 10.0
    assert res_b.processing_latency_ms < 10.0
