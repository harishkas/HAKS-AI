"""Real-time Multi-Band Spectral Noise Suppressor and Transient Click Filter.

Suppresses stationary and non-stationary background noise:
- Cooling fan drone
- Air conditioner air rushing
- Traffic and vehicle sounds
- Keyboard typing and mouse clicks
- Static hiss and environmental noise

Features:
- Dynamic noise floor tracking per frequency bin.
- Configurable suppression strength (Low, Medium, High, Ultra).
- Transient impulse suppression for keyboard/mouse clicks.
- Automatic noise-floor calibration routine.
- Highly optimized FFT-based spectral subtraction with 50% overlap-add.
"""

from enum import Enum
import logging
from typing import Optional
import numpy as np

logger = logging.getLogger("haks.voice.dsp.noise_suppressor")


class NoiseSuppressionStrength(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


class NoiseSuppressor:
    """Real-time spectral subtraction & transient click noise suppressor."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_size: int = 320,  # 20ms frame
        strength: str = "medium",
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.fft_size = 512  # Next power of 2 for fast FFT
        self.num_bins = (self.fft_size // 2) + 1

        # Window function (Hanning)
        self._window = np.hanning(self.fft_size).astype(np.float32)

        # Overlap-add buffer for continuous synthesis
        self._overlap_buffer = np.zeros(self.fft_size - self.frame_size, dtype=np.float32)
        self._input_history = np.zeros(self.fft_size - self.frame_size, dtype=np.float32)

        # Dynamic noise floor estimation across FFT bins
        self._noise_spectrum = np.full(self.num_bins, 1e-4, dtype=np.float32)
        self._noise_alpha = 0.05  # EMA tracking rate for stationary noise
        self._is_calibrating: bool = False
        self._calibration_frames: list[np.ndarray] = []

        # Click / Transient detector states
        self._prev_frame_energy: float = 1e-4

        # Configure strength parameters (over-subtraction alpha, spectral floor beta)
        self.set_strength(strength)

    def set_strength(self, strength: str) -> None:
        """Set noise suppression aggressiveness: low, medium, high, ultra."""
        norm_str = strength.lower().strip()
        if norm_str == "low":
            self._alpha = 1.2
            self._beta = 0.15  # -16 dB floor
            self._transient_factor = 2.5
        elif norm_str == "high":
            self._alpha = 2.4
            self._beta = 0.03  # -30 dB floor
            self._transient_factor = 1.6
        elif norm_str == "ultra":
            self._alpha = 3.2
            self._beta = 0.01  # -40 dB floor
            self._transient_factor = 1.3
        else:  # medium (default)
            self._alpha = 1.8
            self._beta = 0.06  # -24 dB floor
            self._transient_factor = 2.0
        self.strength_name = norm_str

    def start_calibration(self) -> None:
        """Begin accumulating ambient noise frames for automatic calibration."""
        self._is_calibrating = True
        self._calibration_frames.clear()
        logger.info("Started noise floor calibration.")

    def finish_calibration(self) -> float:
        """Complete calibration and calculate baseline noise floor."""
        self._is_calibrating = False
        if not self._calibration_frames:
            return float(np.mean(self._noise_spectrum))

        stacked = np.array(self._calibration_frames)
        # 75th percentile of spectral magnitude during calibration
        self._noise_spectrum = np.percentile(stacked, 75, axis=0).astype(np.float32)
        # Ensure minimum floor
        self._noise_spectrum = np.maximum(self._noise_spectrum, 1e-5)
        avg_floor = float(np.mean(self._noise_spectrum))
        logger.info("Noise floor calibration completed (avg magnitude: %.6f across %d frames).", avg_floor, len(self._calibration_frames))
        self._calibration_frames.clear()
        return avg_floor

    def get_noise_floor_estimate(self) -> float:
        """Return scalar average noise floor magnitude."""
        return float(np.mean(self._noise_spectrum))

    def reset(self) -> None:
        """Reset internal buffers and noise floor estimates."""
        self._overlap_buffer.fill(0.0)
        self._input_history.fill(0.0)
        self._noise_spectrum.fill(1e-4)
        self._prev_frame_energy = 1e-4

    def process(self, frame: np.ndarray, is_speech: bool = False) -> np.ndarray:
        """Process an audio frame to suppress background noise and transient clicks.

        Args:
            frame: 1D numpy array of float32 samples.
            is_speech: Optional hint from VAD; if False, updates noise floor estimation.

        Returns:
            Cleaned 1D numpy array of float32 samples with length equal to frame.
        """
        frame_len = len(frame)
        if frame_len == 0:
            return frame

        # If incoming frame doesn't match nominal frame_size, pad or slice gracefully
        if frame_len < self.frame_size:
            padded = np.zeros(self.frame_size, dtype=np.float32)
            padded[:frame_len] = frame
            cleaned_padded = self.process(padded, is_speech=is_speech)
            return cleaned_padded[:frame_len]

        # 1. Transient click / tap suppression (keyboard typing, mouse clicks)
        frame_energy = float(np.mean(frame**2))
        energy_ratio = frame_energy / (self._prev_frame_energy + 1e-9)

        cleaned_input = frame.copy()
        # Impulsive spikes (sharp transient onset without harmonic sustain)
        if energy_ratio > (self._transient_factor * 8.0) and not is_speech:
            # Soft threshold transient peak
            peak_val = np.max(np.abs(frame))
            thresh = np.std(frame) * 2.5
            if peak_val > thresh:
                cleaned_input = np.clip(frame, -thresh, thresh)

        self._prev_frame_energy = (0.85 * self._prev_frame_energy) + (0.15 * frame_energy)

        # 2. Overlap buffer construction
        # Combine historical samples with current frame to create fft_size block
        hist_len = len(self._input_history)
        fft_input = np.zeros(self.fft_size, dtype=np.float32)
        fft_input[:hist_len] = self._input_history
        fft_input[hist_len : hist_len + self.frame_size] = cleaned_input[:self.frame_size]

        # Update input history for next frame
        self._input_history = fft_input[-hist_len:].copy()

        # 3. Windowing & FFT
        windowed = fft_input * self._window
        spectrum = np.fft.rfft(windowed, n=self.fft_size)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        # 4. Calibration accumulation
        if self._is_calibrating:
            self._calibration_frames.append(magnitude.copy())

        # 5. Adaptive Noise Floor Tracking (when speech is absent or during low energy)
        if not is_speech and not self._is_calibrating:
            # Minimum statistics / recursive EMA update
            self._noise_spectrum = (1.0 - self._noise_alpha) * self._noise_spectrum + self._noise_alpha * np.minimum(magnitude, self._noise_spectrum * 1.5)
            self._noise_spectrum = np.maximum(self._noise_spectrum, 1e-6)

        # 6. Multi-Band Spectral Subtraction
        # Over-subtracted noise spectrum
        noise_sub = self._alpha * self._noise_spectrum
        mag_sub = magnitude - noise_sub
        # Apply spectral floor beta
        min_mag = self._beta * magnitude
        cleaned_mag = np.maximum(mag_sub, min_mag)

        # 7. Inverse FFT & Overlap-Add Synthesis
        reconstructed_spec = cleaned_mag * np.exp(1j * phase)
        time_domain = np.fft.irfft(reconstructed_spec, n=self.fft_size).real.astype(np.float32)

        # Synthesis windowing
        time_domain *= self._window

        # Overlap-add with previous frame's tail
        output_frame = np.zeros(self.frame_size, dtype=np.float32)
        overlap_len = len(self._overlap_buffer)
        output_frame[:overlap_len] = self._overlap_buffer + time_domain[:overlap_len]
        output_frame[overlap_len:] = time_domain[overlap_len : self.frame_size]

        # Store remainder in overlap buffer for next frame
        self._overlap_buffer = time_domain[self.frame_size : self.frame_size + overlap_len].copy()

        return output_frame

