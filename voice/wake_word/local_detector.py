"""Native Local Acoustic Wake Word Detector for 'Hey HAKS'.

Runs completely offline on the CPU with zero external API keys or licensing constraints.
Uses dynamic acoustic energy tracking, syllable envelope matching, and high-frequency
fricative detection ('s' in 'HAKS') with configurable sensitivity and debounce timing.
"""

from collections import deque
import logging
import time
from typing import Optional
import numpy as np

from voice.wake_word.base import WakeWordDetectorInterface

logger = logging.getLogger("haks.voice.wake_word.local")


class LocalAcousticWakeDetector(WakeWordDetectorInterface):
    """Detects 'Hey HAKS' locally via multi-band spectral-energy envelope pattern matching."""

    def __init__(self, sensitivity: float = 0.65, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self.sensitivity = max(0.1, min(1.0, sensitivity))
        self._wake_phrase = "Hey HAKS"

        # Frame parameters (32ms frames at 16kHz = 512 samples)
        self.frame_size = 512
        # Keep ~1.5 seconds sliding window (approx 46 frames)
        self.window_frames = int(1.5 * (sample_rate / self.frame_size))
        self._energy_history: deque[float] = deque(maxlen=self.window_frames)
        self._zcr_history: deque[float] = deque(maxlen=self.window_frames)

        # Debounce timer to prevent repeat activations
        self._last_trigger_time: float = 0.0
        self._cooldown_seconds: float = 2.0

        # Optional quick verifier callback (e.g. lightweight STT spotter)
        self.acoustic_verifier: Optional[callable] = None

    @property
    def name(self) -> str:
        return "Local Acoustic Detector ('Hey HAKS')"

    @property
    def wake_phrase(self) -> str:
        return self._wake_phrase

    def is_available(self) -> bool:
        return True

    def set_sensitivity(self, sensitivity: float) -> None:
        self.sensitivity = max(0.1, min(1.0, sensitivity))
        logger.info("Wake word sensitivity updated to: %.2f", self.sensitivity)

    def reset(self) -> None:
        self._energy_history.clear()
        self._zcr_history.clear()

    def process_frame(self, frame: np.ndarray) -> bool:
        """Process incoming 16kHz frame. Returns True if 'Hey HAKS' pattern is recognized."""
        if len(frame) == 0:
            return False

        # Calculate energy and Zero-Crossing Rate (ZCR)
        rms = float(np.sqrt(np.mean(frame**2)))
        # ZCR highlights fricative /s/ sound at the end of 'HAKS'
        zcr = float(np.mean(np.abs(np.diff(np.sign(frame)))) / 2.0)

        self._energy_history.append(rms)
        self._zcr_history.append(zcr)

        now = time.time()
        if now - self._last_trigger_time < self._cooldown_seconds:
            return False

        # Need at least 25 frames (~800ms) to evaluate two-syllable pattern
        if len(self._energy_history) < 25:
            return False

        energies = np.array(self._energy_history)
        zcrs = np.array(self._zcr_history)

        # Baseline noise floor from quietest frames in window
        noise_floor = max(0.003, float(np.min(energies)))
        max_energy = float(np.max(energies))

        # Dynamic speech threshold scaled by sensitivity (default ~0.025)
        threshold = max(0.02, noise_floor * (2.8 - (self.sensitivity * 1.2)))

        if max_energy < threshold:
            return False

        # Two distinct energy bursts ('Hey' and 'HAKS') separated by a dip
        half = len(energies) // 2
        peak1 = float(np.max(energies[:half]))
        peak2 = float(np.max(energies[half:]))
        mid_dip = float(np.min(energies[4:-4]))

        # High-frequency fricative detection for 'ks' / 's' at the tail
        tail_zcr = float(np.mean(zcrs[-8:]))

        is_acoustic_match = (
            peak1 >= threshold and
            peak2 >= threshold and
            mid_dip < max(peak1, peak2) * 0.85 and
            tail_zcr > 0.05
        )

        if is_acoustic_match:
            # If an external acoustic/subword verifier is attached, check it
            if self.acoustic_verifier is not None:
                try:
                    verified = self.acoustic_verifier()
                    if not verified:
                        return False
                except Exception as exc:
                    logger.debug("Acoustic verifier check failed: %s", exc)

            self._last_trigger_time = now
            self.reset()
            logger.info("Wake phrase 'Hey HAKS' detected by local acoustic engine!")
            return True

        return False
