"""DC Offset Removal, Level Normalization, Rumble High-Pass, and Hum Suppression.

Stage 1 & 2 of the HAKS Audio Processing Pipeline:
- Removes ADC DC offset with a high-pass single-pole DC blocker.
- Applies microphone input gain and soft saturation limiting.
- Suppresses low-frequency rumble (< 80 Hz) via Butterworth IIR filter.
- Suppresses 50 Hz and 60 Hz electrical mains hum and harmonics with notch filters.
- Preserves filter state across contiguous streaming audio frames.
"""

import logging
from typing import Optional, Tuple
import numpy as np

try:
    import scipy.signal as signal
except ImportError:
    signal = None

logger = logging.getLogger("haks.voice.dsp.dc_normalizer")


class DCNormalizer:
    """Removes DC bias, normalizes level, filters sub-audible rumble and mains hum."""

    def __init__(
        self,
        sample_rate: int = 16000,
        mic_gain: float = 1.0,
        enable_highpass: bool = True,
        highpass_cutoff_hz: float = 80.0,
        enable_notch: bool = True,
        notch_frequencies: Tuple[float, ...] = (50.0, 60.0),
    ) -> None:
        self.sample_rate = sample_rate
        self.mic_gain = max(0.1, min(5.0, mic_gain))
        self.enable_highpass = enable_highpass
        self.highpass_cutoff_hz = highpass_cutoff_hz
        self.enable_notch = enable_notch
        self.notch_frequencies = notch_frequencies

        # DC Blocker filter state: y[n] = x[n] - x[n-1] + R * y[n-1]
        self._dc_r: float = 0.995
        self._dc_b = np.array([1.0, -1.0], dtype=np.float32)
        self._dc_a = np.array([1.0, -0.995], dtype=np.float32)
        self._dc_zi = signal.lfilter_zi(self._dc_b, self._dc_a) * 0.0 if signal is not None else None

        # Filter designs and states
        self._hp_b: Optional[np.ndarray] = None
        self._hp_a: Optional[np.ndarray] = None
        self._hp_zi: Optional[np.ndarray] = None

        self._notch_filters: list[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        self._init_filters()

    def _init_filters(self) -> None:
        """Initialize IIR filter coefficients and initial conditions."""
        if signal is None:
            logger.warning("scipy.signal not available; basic DC blocking only.")
            return

        nyquist = 0.5 * self.sample_rate

        # 1. High-pass filter for rumble (< 80 Hz, 2nd order Butterworth)
        try:
            norm_cutoff = min(0.49, self.highpass_cutoff_hz / nyquist)
            self._hp_b, self._hp_a = signal.butter(2, norm_cutoff, btype="highpass")
            self._hp_zi = signal.lfilter_zi(self._hp_b, self._hp_a) * 0.0
        except Exception as exc:
            logger.error("Failed to construct rumble highpass filter: %s", exc)
            self._hp_b = self._hp_a = self._hp_zi = None

        # 2. Notch filters for 50 Hz and 60 Hz electrical hum
        self._notch_filters.clear()
        for f0 in self.notch_frequencies:
            if f0 < nyquist:
                try:
                    q = 30.0  # High quality factor for narrow notch
                    b, a = signal.iirnotch(f0, q, self.sample_rate)
                    zi = signal.lfilter_zi(b, a) * 0.0
                    self._notch_filters.append((b, a, zi))
                except Exception as exc:
                    logger.debug("Error creating notch filter for %.1f Hz: %s", f0, exc)

    def set_gain(self, gain: float) -> None:
        """Set microphone input gain (0.1x to 5.0x)."""
        self.mic_gain = max(0.1, min(5.0, gain))

    def reset(self) -> None:
        """Reset internal filter states."""
        if signal is not None and self._dc_b is not None:
            self._dc_zi = signal.lfilter_zi(self._dc_b, self._dc_a) * 0.0
        if self._hp_b is not None and self._hp_a is not None:
            self._hp_zi = signal.lfilter_zi(self._hp_b, self._hp_a) * 0.0
        new_notches = []
        for b, a, _ in self._notch_filters:
            zi = signal.lfilter_zi(b, a) * 0.0
            new_notches.append((b, a, zi))
        self._notch_filters = new_notches

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Process an audio frame in-place or return a cleaned float32 frame.

        Args:
            frame: 1D numpy array of float32 samples.

        Returns:
            Cleaned and normalized float32 array.
        """
        if len(frame) == 0:
            return frame

        out = frame.astype(np.float32, copy=True)

        # 1. DC offset removal (Single-pole IIR DC blocker via compiled lfilter)
        if signal is not None and self._dc_zi is not None:
            try:
                out, self._dc_zi = signal.lfilter(self._dc_b, self._dc_a, out, zi=self._dc_zi)
            except Exception as exc:
                logger.debug("Error in DC blocking: %s", exc)
        else:
            out = out - float(np.mean(out))

        # 2. Low-frequency rumble suppression (High-pass filter)
        if self.enable_highpass and self._hp_b is not None and signal is not None:
            try:
                out, self._hp_zi = signal.lfilter(self._hp_b, self._hp_a, out, zi=self._hp_zi)
            except Exception as exc:
                logger.debug("Error in highpass filtering: %s", exc)

        # 3. 50/60 Hz electrical hum notch suppression
        if self.enable_notch and signal is not None:
            updated_notches = []
            for b, a, zi in self._notch_filters:
                try:
                    out, new_zi = signal.lfilter(b, a, out, zi=zi)
                    updated_notches.append((b, a, new_zi))
                except Exception as exc:
                    logger.debug("Error in notch filtering: %s", exc)
                    updated_notches.append((b, a, zi))
            self._notch_filters = updated_notches

        # 4. Microphone Input Gain scaling
        if abs(self.mic_gain - 1.0) > 1e-3:
            out *= self.mic_gain

        # 5. Soft saturation limiter (tanh) to prevent digital clipping
        # Soft-knee compression when exceeding +/-0.85
        peak = np.max(np.abs(out))
        if peak > 0.85:
            # Smooth soft-knee saturation
            out = np.tanh(out / 0.85) * 0.85

        return out
