"""Acoustic Echo Suppression and Playback-Reference Tracking for HAKS.

Prevents HAKS's own speaker output (TTS and Gemini Live 24 kHz audio) from
feeding back into the microphone and causing false speech interruptions.
Uses normalized cross-correlation and adaptive residual energy analysis.
"""

from dataclasses import dataclass
import logging
import threading
import time
from typing import Optional
import numpy as np

try:
    import scipy.signal as signal
except ImportError:
    signal = None

logger = logging.getLogger("haks.voice.echo_cancellation")

# Re-export upgraded AcousticEchoCanceller for backward compatibility
from voice.dsp.echo_canceller import AcousticEchoCanceller, AECEvaluationResult


@dataclass
class EchoEvaluationResult:
    """Result of analyzing a microphone frame against active speaker playback."""

    is_echo: bool
    correlation: float
    residual_energy: float
    is_playback_active: bool
    cleaned_frame: np.ndarray


class AcousticEchoSuppressor:
    """Thread-safe playback-reference buffer and echo correlation detector."""

    def __init__(
        self,
        sample_rate: int = 16000,
        reference_buffer_seconds: float = 1.0,
        correlation_threshold: float = 0.35,
        min_search_lag_ms: float = 10.0,
        max_search_lag_ms: float = 160.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.buffer_size = int(sample_rate * reference_buffer_seconds)
        self.correlation_threshold = correlation_threshold
        self.min_lag = int((min_search_lag_ms / 1000.0) * sample_rate)
        self.max_lag = int((max_search_lag_ms / 1000.0) * sample_rate)

        # Circular reference buffer for 16 kHz float32 playback audio
        self._ref_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self._write_pos: int = 0
        self._total_samples_written: int = 0
        self._last_playback_timestamp: float = 0.0
        self._lock = threading.Lock()

    def is_playback_active(self, timeout_seconds: float = 0.25) -> bool:
        """Check if audio playback was active within the timeout window."""
        with self._lock:
            if self._last_playback_timestamp == 0.0:
                return False
            return (time.time() - self._last_playback_timestamp) < timeout_seconds

    def feed_playback_chunk(self, pcm_bytes: bytes, input_sample_rate: int = 24000) -> None:
        """Feed an output audio chunk being sent to speakers (e.g. 24 kHz or 16 kHz int16)."""
        if not pcm_bytes:
            return

        try:
            # Convert int16 bytes to float32 [-1.0, 1.0]
            arr_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
            arr_float = (arr_int16.astype(np.float32) / 32768.0)

            # Resample to 16 kHz if necessary
            if input_sample_rate != self.sample_rate:
                if signal is not None and input_sample_rate == 24000 and self.sample_rate == 16000:
                    samples_16k = signal.resample_poly(arr_float, 2, 3).astype(np.float32)
                else:
                    target_len = int(len(arr_float) * (self.sample_rate / input_sample_rate))
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
            logger.debug("Error feeding playback chunk to echo suppressor: %s", exc)

    def evaluate_mic_frame(self, mic_frame: np.ndarray) -> EchoEvaluationResult:
        """Analyze a 16 kHz microphone frame against recent speaker playback."""
        if len(mic_frame) == 0:
            return EchoEvaluationResult(
                is_echo=False,
                correlation=0.0,
                residual_energy=0.0,
                is_playback_active=False,
                cleaned_frame=mic_frame,
            )

        with self._lock:
            now = time.time()
            playback_active = (self._last_playback_timestamp > 0.0) and ((now - self._last_playback_timestamp) < 0.25)
            if not playback_active or self._total_samples_written < len(mic_frame):
                return EchoEvaluationResult(
                    is_echo=False,
                    correlation=0.0,
                    residual_energy=float(np.mean(mic_frame**2)),
                    is_playback_active=False,
                    cleaned_frame=mic_frame,
                )

            # Extract linear reference buffer ordered from oldest to newest
            ordered_ref = np.concatenate((self._ref_buffer[self._write_pos:], self._ref_buffer[:self._write_pos]))

        frame_len = len(mic_frame)
        norm_mic = np.linalg.norm(mic_frame)
        if norm_mic < 1e-4:
            return EchoEvaluationResult(
                is_echo=False,
                correlation=0.0,
                residual_energy=0.0,
                is_playback_active=True,
                cleaned_frame=mic_frame,
            )

        # Search correlation in acoustic delay window [min_lag, max_lag]
        search_window_len = self.max_lag + frame_len
        if len(ordered_ref) < search_window_len:
            return EchoEvaluationResult(
                is_echo=False,
                correlation=0.0,
                residual_energy=float(np.mean(mic_frame**2)),
                is_playback_active=True,
                cleaned_frame=mic_frame,
            )

        search_slice = ordered_ref[-search_window_len:]
        corr = np.correlate(search_slice, mic_frame, mode="valid")

        # Compute sliding energy of the window for normalized cross-correlation
        window_sq = search_slice**2
        window_energies = np.convolve(window_sq, np.ones(frame_len), mode="valid")
        denom = norm_mic * np.sqrt(np.maximum(window_energies, 1e-9))
        norm_corr_curve = corr / (denom + 1e-9)

        # Restrict correlation search to valid physical acoustic lag
        valid_curve = norm_corr_curve[:-(self.min_lag)] if self.min_lag > 0 else norm_corr_curve
        if len(valid_curve) == 0:
            valid_curve = norm_corr_curve

        max_idx = int(np.argmax(np.abs(valid_curve)))
        max_corr = float(np.abs(valid_curve[max_idx]))

        # Best matching reference segment
        best_ref_segment = search_slice[max_idx : max_idx + frame_len]
        norm_ref = np.linalg.norm(best_ref_segment)

        # Scale factor for residual calculation (Wiener / LMS single-tap estimation)
        alpha = 0.0
        if norm_ref > 1e-4:
            alpha = float(np.dot(mic_frame, best_ref_segment) / (norm_ref**2 + 1e-9))
            alpha = max(0.0, min(1.5, alpha))

        # Compute echo-cancelled residual
        residual = mic_frame - (alpha * best_ref_segment)
        res_energy = float(np.mean(residual**2))
        mic_energy = float(np.mean(mic_frame**2))

        # If correlation is high and residual energy reduction is significant, it's echo
        is_echo = (max_corr >= self.correlation_threshold) and (mic_energy > 0) and ((res_energy / (mic_energy + 1e-9)) < 0.65)

        return EchoEvaluationResult(
            is_echo=is_echo,
            correlation=max_corr,
            residual_energy=res_energy,
            is_playback_active=True,
            cleaned_frame=residual if is_echo else mic_frame,
        )

    def clear(self) -> None:
        """Reset reference buffer and timestamps."""
        with self._lock:
            self._ref_buffer.fill(0.0)
            self._write_pos = 0
            self._total_samples_written = 0
            self._last_playback_timestamp = 0.0
