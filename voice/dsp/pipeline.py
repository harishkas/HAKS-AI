"""Unified Audio Processing Pipeline for HAKS.

Implements the complete audio-enhancement chain:
Microphone Capture (16 kHz mono)
  │
  ▼
[1] DC Offset Removal & Level Normalization (DCNormalizer)
  │
  ▼
[2] Low-Frequency Rumble HPF (< 80 Hz) & 50/60 Hz Hum Notch Filtering
  │
  ▼
[3] Acoustic Echo Cancellation (AcousticEchoCanceller)
  │
  ▼
[4] Real-Time Background Noise Suppression (NoiseSuppressor)
  │
  ▼
[5] Human Speech Enhancement (SpeechEnhancer)
  │
  ▼
[6] Multi-Stage Adaptive VAD (AdaptiveVoiceDetector)
  │
  ▼
[7] Speech Confirmation & Double-Talk Gate
  │
  ▼
Gemini Live API & Faster-Whisper STT

ORDERING EVALUATION:
AEC vs. Noise Suppression Sequence:
1. Linear Filter Condition: The NLMS adaptive echo canceller relies on a linear transfer
   function between speaker reference audio x(n) and mic feedback d(n). Non-linear spectral
   subtraction introduces phase and spectral distortions that disrupt adaptive filter convergence.
   Therefore, running AEC before non-linear spectral suppression provides superior echo cancellation.
2. Residual Cleaning: Running spectral noise suppression after AEC cleans both stationary
   environmental background noise (fans, air conditioners, traffic) AND any residual speaker echo.
The pipeline provides full support for either sequence via `aec_first` configuration.
"""

from dataclasses import dataclass
import logging
import threading
import time
from typing import Optional, Tuple
import numpy as np

from config.settings import AppConfig, get_settings
from voice.dsp.dc_normalizer import DCNormalizer
from voice.dsp.echo_canceller import AcousticEchoCanceller, AECEvaluationResult
from voice.dsp.noise_suppressor import NoiseSuppressor
from voice.dsp.speech_enhancer import SpeechEnhancer
from voice.dsp.vad_detector import AdaptiveVoiceDetector, VADFrameResult

logger = logging.getLogger("haks.voice.dsp.pipeline")


@dataclass
class ProcessedAudioFrame:
    """Complete diagnostic and audio output package from the DSP pipeline."""

    raw_frame: np.ndarray
    cleaned_frame: np.ndarray
    is_speech: bool
    is_confirmed_speech: bool
    speech_confidence: float
    snr_db: float
    rms_energy: float
    is_echo: bool
    is_double_talk: bool
    is_playback_active: bool
    processing_latency_ms: float


class AudioProcessingPipeline:
    """Master audio processing pipeline combining DC normalization, AEC, noise suppression,

    speech enhancement, and multi-stage adaptive VAD.
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        sample_rate: int = 16000,
        frame_size: int = 320,  # 20ms frame
        aec_first: bool = True,
    ) -> None:
        self.config = config or get_settings()
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.aec_first = aec_first

        voice_cfg = self.config.voice
        self.enable_noise_suppression: bool = getattr(voice_cfg, "enable_noise_suppression", True)
        self.enable_aec: bool = getattr(voice_cfg, "enable_echo_cancellation", True)

        # Stage 1 & 2: DC Normalization, gain, rumble highpass, and hum notch
        self.dc_normalizer = DCNormalizer(
            sample_rate=self.sample_rate,
            mic_gain=getattr(voice_cfg, "mic_input_gain", 1.0),
            enable_highpass=getattr(voice_cfg, "highpass_rumble", True),
            enable_notch=getattr(voice_cfg, "notch_electrical_hum", True),
        )

        # Stage 3: Acoustic Echo Cancellation
        self.echo_canceller = AcousticEchoCanceller(
            sample_rate=self.sample_rate,
        )

        # Stage 4: Real-time Background Noise Suppression
        strength = getattr(voice_cfg, "noise_suppression_strength", "medium")
        self.noise_suppressor = NoiseSuppressor(
            sample_rate=self.sample_rate,
            frame_size=self.frame_size,
            strength=strength,
        )

        # Stage 5: Human Speech Enhancement
        self.speech_enhancer = SpeechEnhancer(
            sample_rate=self.sample_rate,
        )

        # Stage 6 & 7: Multi-Stage Adaptive VAD & Speech Confirmation
        vad_mode = getattr(voice_cfg, "vad_sensitivity", 2)
        min_speech_ms = getattr(voice_cfg, "min_speech_duration_ms", 160)
        silence_timeout_ms = getattr(voice_cfg, "silence_timeout_ms", 500)
        sensitivity = getattr(voice_cfg, "human_speech_sensitivity", 0.65)
        self.vad_detector = AdaptiveVoiceDetector(
            sample_rate=self.sample_rate,
            frame_size=self.frame_size,
            vad_mode=vad_mode,
            min_speech_duration_ms=min_speech_ms,
            silence_timeout_ms=silence_timeout_ms,
            sensitivity=sensitivity,
        )

        self._lock = threading.Lock()
        self._last_metrics: Optional[ProcessedAudioFrame] = None

    def feed_playback_reference(
        self,
        pcm_data: bytes | np.ndarray,
        input_sample_rate: int = 24000,
    ) -> None:
        """Feed speaker playback audio into the AEC reference buffer."""
        if self.enable_aec:
            self.echo_canceller.feed_playback_chunk(pcm_data, input_sample_rate)

    def handle_device_change(self) -> None:
        """Handle hardware audio input or output device switch."""
        with self._lock:
            self.dc_normalizer.reset()
            self.echo_canceller.handle_device_change()
            self.noise_suppressor.reset()
            self.speech_enhancer.reset()
            self.vad_detector.reset()
        logger.info("AudioProcessingPipeline: Reset complete after audio device change.")

    def calibrate_noise_floor(self, duration_seconds: float = 1.0) -> None:
        """Trigger automatic noise-floor calibration."""
        self.noise_suppressor.start_calibration()

    def finish_noise_calibration(self) -> float:
        """Conclude noise-floor calibration."""
        return self.noise_suppressor.finish_calibration()

    def process_frame(self, raw_frame: np.ndarray) -> ProcessedAudioFrame:
        """Execute the complete 7-stage audio processing pipeline on a microphone frame.

        Args:
            raw_frame: 1D numpy array of 16 kHz float32 microphone samples.

        Returns:
            ProcessedAudioFrame containing enhanced audio, speech status, confidence, and metrics.
        """
        t0 = time.perf_counter()

        with self._lock:
            # Stage 1: DC offset removal, gain scaling, rumble high-pass, 50/60 Hz notch
            frame_dc = self.dc_normalizer.process(raw_frame)

            # Stage 2 & 3: AEC and Noise Suppression ordering
            is_echo = False
            is_double_talk = False
            is_playback_active = self.echo_canceller.is_active

            if self.aec_first:
                # Sequence A: AEC -> Noise Suppression
                if self.enable_aec:
                    aec_res: AECEvaluationResult = self.echo_canceller.process_frame(frame_dc)
                    frame_aec = aec_res.cleaned_frame
                    is_echo = bool(aec_res.is_echo)
                    is_double_talk = bool(aec_res.is_double_talk)
                else:
                    frame_aec = frame_dc

                if self.enable_noise_suppression:
                    frame_ns = self.noise_suppressor.process(frame_aec, is_speech=is_double_talk)
                else:
                    frame_ns = frame_aec
            else:
                # Sequence B: Noise Suppression -> AEC
                if self.enable_noise_suppression:
                    frame_ns_pre = self.noise_suppressor.process(frame_dc)
                else:
                    frame_ns_pre = frame_dc

                if self.enable_aec:
                    aec_res = self.echo_canceller.process_frame(frame_ns_pre)
                    frame_ns = aec_res.cleaned_frame
                    is_echo = bool(aec_res.is_echo)
                    is_double_talk = bool(aec_res.is_double_talk)
                else:
                    frame_ns = frame_ns_pre

            # Stage 4: Human Speech Enhancement (formants, consonant pre-emphasis)
            frame_enhanced = self.speech_enhancer.process(frame_ns)

            # Stage 5 & 6: Multi-Stage Adaptive VAD & Speech Confirmation
            vad_res: VADFrameResult = self.vad_detector.process_frame(
                frame_enhanced,
                is_playback_echo=is_echo,
            )

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        result = ProcessedAudioFrame(
            raw_frame=raw_frame,
            cleaned_frame=frame_enhanced,
            is_speech=vad_res.is_speech,
            is_confirmed_speech=vad_res.is_confirmed_speech,
            speech_confidence=vad_res.confidence,
            snr_db=vad_res.snr_db,
            rms_energy=vad_res.rms_energy,
            is_echo=is_echo,
            is_double_talk=is_double_talk,
            is_playback_active=is_playback_active,
            processing_latency_ms=latency_ms,
        )

        self._last_metrics = result
        return result
