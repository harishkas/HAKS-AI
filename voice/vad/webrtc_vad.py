"""WebRTC VAD Fallback Engine for HAKS.

Provides a fast, lightweight Gaussian Mixture Model (GMM) voice activity detector
as a resilient fallback when Silero neural VAD is unavailable or fails:
- Sub-frame analysis (10ms / 160 samples per window at 16kHz).
- Evaluates voiced frame proportions to produce a continuous speech probability.
- Configurable aggressiveness mode (0 to 3).
- Resilient software energy fallback if `webrtcvad` is not installed.
"""

import logging
import time
from typing import Optional
import numpy as np

try:
    import webrtcvad
    WEBRTC_VAD_AVAILABLE = True
except ImportError:
    webrtcvad = None
    WEBRTC_VAD_AVAILABLE = False

from voice.vad.base import BaseVADEngine

logger = logging.getLogger("haks.voice.vad.webrtc")


class WebRTCVADEngine(BaseVADEngine):
    """WebRTC VAD implementation with 10ms frame chunking and continuous probability estimation."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_size: int = 512,
        mode: int = 2,
        speech_threshold: float = 0.50,
        min_speech_duration_ms: int = 150,
        speech_confirmation_ms: int = 150,
        end_of_speech_silence_ms: int = 450,
        pre_roll_ms: int = 300,
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            frame_size=frame_size,
            speech_threshold=speech_threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            speech_confirmation_ms=speech_confirmation_ms,
            end_of_speech_silence_ms=end_of_speech_silence_ms,
            pre_roll_ms=pre_roll_ms,
        )
        self.mode = max(0, min(3, mode))
        self._vad: Optional[Any] = None
        # Noise floor tracking for software energy fallback
        self._noise_floor: float = 0.005

        self._init_vad()

    @property
    def engine_name(self) -> str:
        return "webrtcvad" if self._vad is not None else "energy-fallback"

    def _init_vad(self) -> None:
        """Initialize the underlying webrtcvad object."""
        if not WEBRTC_VAD_AVAILABLE:
            logger.warning("webrtcvad module unavailable; using adaptive energy fallback.")
            self._vad = None
            return

        try:
            self._vad = webrtcvad.Vad(self.mode)
            logger.info("WebRTC VAD initialized (mode=%d, sample_rate=%d).", self.mode, self.sample_rate)
        except Exception as exc:
            logger.warning("Failed to initialize webrtcvad: %s; using energy fallback.", exc)
            self._vad = None

    def predict_probability(self, frame: np.ndarray) -> float:
        """Estimate speech probability on a 16kHz frame.

        Splits 512-sample frame into 160-sample (10ms) sub-frames for WebRTC VAD.
        Returns the voiced sub-frame ratio blended with RMS energy.
        """
        t0 = time.perf_counter()

        # Fallback to adaptive energy if webrtcvad not present
        if self._vad is None:
            rms = float(np.sqrt(np.mean(frame**2))) if len(frame) > 0 else 0.0
            if not self._is_speech_active:
                self._noise_floor = 0.95 * self._noise_floor + 0.05 * rms
            threshold = max(0.008, self._noise_floor * 2.2)
            prob = min(1.0, max(0.0, rms / (threshold * 2.0)))
            self._last_inference_latency_ms = (time.perf_counter() - t0) * 1000.0
            return prob

        try:
            # Convert float32 [-1.0, 1.0] to int16 PCM bytes
            int16_samples = (np.clip(frame, -1.0, 1.0) * 32767.0).astype(np.int16)

            # WebRTC VAD requires exact 10ms (160 samples), 20ms (320), or 30ms (480) slices
            # With 512 samples, evaluate three 160-sample slices (total 480 samples)
            subframe_size = 160
            voiced_count = 0
            total_slices = 0

            for offset in (0, 160, 320):
                if offset + subframe_size <= len(int16_samples):
                    slice_bytes = int16_samples[offset : offset + subframe_size].tobytes()
                    is_voiced = self._vad.is_speech(slice_bytes, self.sample_rate)
                    if is_voiced:
                        voiced_count += 1
                    total_slices += 1

            if total_slices > 0:
                raw_prob = voiced_count / float(total_slices)
            else:
                raw_prob = 0.0

            # Compute RMS to penalize absolute silence
            rms = float(np.sqrt(np.mean(frame**2))) if len(frame) > 0 else 0.0
            if rms < 0.005:
                raw_prob *= 0.1  # Low energy suppresses false positive

            self._last_inference_latency_ms = (time.perf_counter() - t0) * 1000.0
            return float(raw_prob)

        except Exception as exc:
            logger.debug("Error in WebRTC VAD inference: %s", exc)
            self._last_inference_latency_ms = (time.perf_counter() - t0) * 1000.0
            return 0.0

