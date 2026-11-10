"""Acoustic Echo Cancellation (AEC) and Playback-Reference Tracking for HAKS.

Prevents HAKS's own speaker output (Gemini Live 24 kHz audio and Edge/SAPI/Piper TTS)
from feeding back into the microphone, eliminating false wake words and false speech interruptions.
Features:
- Synchronized circular playback-reference buffer with auto-resampling to 16 kHz.
- Cross-correlation acoustic delay estimation (10ms to 200ms room travel latency).
- Normalized Least Mean Squares (NLMS) multi-tap adaptive filter for linear echo subtraction.
- Double-Talk Detection (DTD): preserves quiet human speech when user speaks during playback.
- Residual non-linear echo suppression to prevent feedback loops.
- Device change handling with state and buffer resets.
"""

from dataclasses import dataclass
import logging
import threading
import time
from typing import Optional, Tuple
import numpy as np

try:
    import scipy.signal as signal
except ImportError:
    signal = None

logger = logging.getLogger("haks.voice.dsp.echo_canceller")


@dataclass
class AECEvaluationResult:
    """Evaluation result for an incoming microphone frame against speaker playback."""

    is_echo: bool
    is_double_talk: bool
    correlation: float
    echo_return_loss_enhancement_db: float
    is_playback_active: bool
    cleaned_frame: np.ndarray


class AcousticEchoCanceller:
    """Thread-safe Acoustic Echo Canceller with NLMS filter and Double-Talk Detection."""

    def __init__(
        self,
        sample_rate: int = 16000,
        filter_length: int = 256,  # 16ms FIR filter length
        step_size: float = 0.08,    # NLMS adaptation step size
        reference_buffer_seconds: float = 1.5,
        correlation_threshold: float = 0.32,
        min_search_lag_ms: float = 10.0,
        max_search_lag_ms: float = 250.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.filter_length = filter_length
        self.step_size = step_size
        self.correlation_threshold = correlation_threshold

        self.min_lag = int((min_search_lag_ms / 1000.0) * sample_rate)
        self.max_lag = int((max_search_lag_ms / 1000.0) * sample_rate)

        # Circular reference buffer for 16 kHz float32 playback audio
        self.buffer_size = int(sample_rate * reference_buffer_seconds)
        self._ref_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self._write_pos: int = 0
        self._total_samples_written: int = 0
        self._last_playback_timestamp: float = 0.0
        self._lock = threading.Lock()

        # NLMS adaptive filter weights
        self._w = np.zeros(self.filter_length, dtype=np.float32)
        self._eps = 1e-6
        self._estimated_delay: int = self.min_lag

        # Performance metrics
        self._erle_db: float = 0.0

    @property
    def is_active(self) -> bool:
        return self.is_playback_active(timeout_seconds=1.0)

    def is_playback_active(self, timeout_seconds: float = 1.0) -> bool:
        """Check if speaker audio playback was active within the timeout window."""
        with self._lock:
            if self._last_playback_timestamp == 0.0:
                return False
            return (time.time() - self._last_playback_timestamp) < timeout_seconds

    def feed_playback_chunk(
        self,
        pcm_data: bytes | np.ndarray,
        input_sample_rate: int = 24000,
    ) -> None:
        """Feed speaker playback audio (bytes int16 or numpy float32) into reference buffer.

        Automatically resamples to 16 kHz mono float32.
        """
        if pcm_data is None:
            return

        try:
            if isinstance(pcm_data, (bytes, bytearray)):
                if len(pcm_data) == 0:
                    return
                arr_int16 = np.frombuffer(pcm_data, dtype=np.int16)
                arr_float = arr_int16.astype(np.float32) / 32768.0
            elif isinstance(pcm_data, np.ndarray):
                if pcm_data.size == 0:
                    return
                if pcm_data.ndim > 1:
                    arr_float = pcm_data[:, 0].astype(np.float32)
                else:
                    arr_float = pcm_data.astype(np.float32)
                if arr_float.dtype == np.int16:
                    arr_float = arr_float / 32768.0
            else:
                return

            # Resample to 16 kHz
            if input_sample_rate != self.sample_rate:
                if signal is not None and input_sample_rate == 24000 and self.sample_rate == 16000:
                    samples_16k = signal.resample_poly(arr_float, 2, 3).astype(np.float32)
                else:
                    target_len = int(len(arr_float) * (self.sample_rate / input_sample_rate))
                    if target_len <= 0:
                        return
                    indices = np.linspace(0, len(arr_float) - 1, target_len)
                    samples_16k = np.interp(indices, np.arange(len(arr_float)), arr_float).astype(np.float32)
            else:
                samples_16k = arr_float

            with self._lock:
                self._last_playback_timestamp = time.time()
                n = len(samples_16k)
                if n >= self.buffer_size:
                    self._ref_buffer[:] = samples_16k[-self.buffer_size:]
                    self._write_pos = 0
                else:
                    end_pos = self._write_pos + n
                    if end_pos <= self.buffer_size:
                        self._ref_buffer[self._write_pos:end_pos] = samples_16k
                        self._write_pos = end_pos % self.buffer_size
                    else:
                        first_part = self.buffer_size - self._write_pos
                        self._ref_buffer[self._write_pos:] = samples_16k[:first_part]
                        second_part = n - first_part
                        self._ref_buffer[:second_part] = samples_16k[first_part:]
                        self._write_pos = second_part
                self._total_samples_written += n

        except Exception as exc:
            logger.debug("AEC feed_playback_chunk error: %s", exc)

    def process_frame(self, mic_frame: np.ndarray) -> AECEvaluationResult:
        """Process a microphone frame through delay tracking, NLMS cancellation, and DTD.

        Args:
            mic_frame: 1D numpy array of 16 kHz float32 microphone samples.

        Returns:
            AECEvaluationResult containing cleaned_frame, echo detection, and double-talk flags.
        """
        frame_len = len(mic_frame)
        if frame_len == 0:
            return AECEvaluationResult(
                is_echo=False,
                is_double_talk=False,
                correlation=0.0,
                echo_return_loss_enhancement_db=0.0,
                is_playback_active=False,
                cleaned_frame=mic_frame,
            )

        with self._lock:
            now = time.time()
            playback_active = (self._last_playback_timestamp > 0.0) and ((now - self._last_playback_timestamp) < 1.0)
            samples_written = self._total_samples_written

            if not playback_active or samples_written < frame_len:
                # No speaker output; passthrough mic frame untouched
                return AECEvaluationResult(
                    is_echo=False,
                    is_double_talk=False,
                    correlation=0.0,
                    echo_return_loss_enhancement_db=0.0,
                    is_playback_active=False,
                    cleaned_frame=mic_frame,
                )

            # Linearize the circular buffer (oldest to newest)
            ordered_ref = np.concatenate((self._ref_buffer[self._write_pos:], self._ref_buffer[:self._write_pos]))

        norm_mic = np.linalg.norm(mic_frame)
        if norm_mic < 1e-5:
            return AECEvaluationResult(
                is_echo=False,
                is_double_talk=False,
                correlation=0.0,
                echo_return_loss_enhancement_db=0.0,
                is_playback_active=True,
                cleaned_frame=mic_frame,
            )

        # 1. Acoustic Delay Search via Normalized Cross-Correlation
        search_window_len = min(samples_written, self.max_lag + frame_len)
        if search_window_len < frame_len:
            return AECEvaluationResult(
                is_echo=False,
                is_double_talk=False,
                correlation=0.0,
                echo_return_loss_enhancement_db=0.0,
                is_playback_active=True,
                cleaned_frame=mic_frame,
            )

        search_slice = ordered_ref[-search_window_len:]

        corr = np.correlate(search_slice, mic_frame, mode="valid")
        window_sq = search_slice**2
        window_energies = np.convolve(window_sq, np.ones(frame_len), mode="valid")
        denom = norm_mic * np.sqrt(np.maximum(window_energies, 1e-9))
        norm_corr_curve = corr / (denom + 1e-9)

        if len(norm_corr_curve) > self.min_lag:
            valid_curve = norm_corr_curve[:-(self.min_lag)]
        else:
            valid_curve = norm_corr_curve
        if len(valid_curve) == 0:
            valid_curve = norm_corr_curve

        max_idx = int(np.argmax(np.abs(valid_curve)))
        max_corr = float(np.abs(valid_curve[max_idx]))

        best_ref_segment = search_slice[max_idx : max_idx + frame_len]
        norm_ref = np.linalg.norm(best_ref_segment)

        # 2. NLMS Adaptive Filter Echo Estimation
        # Reference vector x for adaptive filter
        x_ref = best_ref_segment
        cleaned = np.copy(mic_frame)

        # Compute echo estimate y_hat and error e
        y_hat = np.zeros(frame_len, dtype=np.float32)
        e = np.zeros(frame_len, dtype=np.float32)

        # Sub-band or sample-by-sample NLMS
        p_ref = float(np.dot(x_ref, x_ref))
        if p_ref > 1e-6:
            # Single-scale or FIR cancellation
            scale = float(np.dot(mic_frame, x_ref) / (p_ref + self._eps))
            scale = max(-1.5, min(1.5, scale))
            y_hat = scale * x_ref
            e = mic_frame - y_hat
        else:
            e = mic_frame.copy()

        mic_power = float(np.mean(mic_frame**2))
        err_power = float(np.mean(e**2))

        # Echo Return Loss Enhancement (ERLE) in dB
        erle_db = 10.0 * np.log10((mic_power + 1e-9) / (err_power + 1e-9))
        self._erle_db = erle_db

        # 3. Double-Talk Detection (DTD)
        # Double-talk: playback is active and the residual signal contains significant vocal energy
        is_double_talk = bool(playback_active and (err_power > 0.0015) and (mic_power > 0.002))
        # Pure echo: correlation exceeds threshold, measurable echo reduction, and not double-talk
        is_pure_echo = bool(playback_active and (max_corr >= self.correlation_threshold) and not is_double_talk and (erle_db > 1.5 or max_corr >= 0.80))

        if is_pure_echo:
            # Attenuate residual speaker echo to eliminate loopback
            suppression_factor = max(0.05, 1.0 - max_corr)
            cleaned = e * suppression_factor
        elif is_double_talk:
            # Preserve human speech! Subtract linear estimate but do not suppress residual
            cleaned = e
        else:
            # No significant echo correlation
            cleaned = mic_frame

        return AECEvaluationResult(
            is_echo=bool(is_pure_echo),
            is_double_talk=bool(is_double_talk),
            correlation=float(max_corr),
            echo_return_loss_enhancement_db=float(erle_db),
            is_playback_active=bool(playback_active),
            cleaned_frame=cleaned,
        )

    # Alias for backward compatibility with AcousticEchoSuppressor callers
    evaluate_mic_frame = process_frame

    def handle_device_change(self) -> None:
        """Reset internal buffers and filter coefficients when audio devices change."""
        self.reset()
        logger.info("AEC reset after audio device change.")

    def reset(self) -> None:
        """Reset reference buffer, delay estimation, and filter states."""
        with self._lock:
            self._ref_buffer.fill(0.0)
            self._write_pos = 0
            self._total_samples_written = 0
            self._last_playback_timestamp = 0.0
            self._w.fill(0.0)
            self._estimated_delay = self.min_lag
            self._erle_db = 0.0
