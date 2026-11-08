"""Picovoice Porcupine Wake Word Detector Adapter for HAKS.

Integrates Picovoice Porcupine for low-power wake word processing when an AccessKey
and custom keyword file (.ppn) for 'Hey HAKS' are configured.
"""

import logging
from pathlib import Path
from typing import Optional
import numpy as np

try:
    import pvporcupine
except ImportError:
    pvporcupine = None

from voice.wake_word.base import WakeWordDetectorInterface

logger = logging.getLogger("haks.voice.wake_word.porcupine")


class PorcupineWakeDetector(WakeWordDetectorInterface):
    """Adapter for Picovoice Porcupine custom keyword detection ('Hey HAKS')."""

    def __init__(
        self,
        access_key: str = "",
        keyword_path: str = "",
        sensitivity: float = 0.65,
    ) -> None:
        self.access_key = access_key.strip()
        self.keyword_path = keyword_path.strip()
        self.sensitivity = max(0.0, min(1.0, sensitivity))
        self._porcupine: Optional[Any] = None
        self._is_initialized: bool = False
        self._init_error: Optional[str] = None

        self._initialize()

    @property
    def name(self) -> str:
        return "Picovoice Porcupine"

    @property
    def wake_phrase(self) -> str:
        return "Hey HAKS"

    def is_available(self) -> bool:
        return self._is_initialized and self._porcupine is not None

    def get_status_info(self) -> dict:
        """Provide detailed configuration status for the UI/diagnostics."""
        return {
            "has_library": pvporcupine is not None,
            "has_access_key": bool(self.access_key),
            "keyword_path": self.keyword_path,
            "keyword_file_exists": Path(self.keyword_path).exists() if self.keyword_path else False,
            "is_ready": self.is_available(),
            "error": self._init_error,
        }

    def _initialize(self) -> bool:
        """Initialize Porcupine instance with credentials and keyword path."""
        if pvporcupine is None:
            self._init_error = "pvporcupine package is not installed."
            return False

        if not self.access_key:
            self._init_error = (
                "Porcupine AccessKey is not configured. Please supply an AccessKey in Settings "
                "or use the built-in Local Acoustic Wake Word Detector."
            )
            return False

        if not self.keyword_path or not Path(self.keyword_path).exists():
            self._init_error = (
                f"Custom keyword file for 'Hey HAKS' not found at: {self.keyword_path}. "
                "A .ppn file trained for Windows must be generated on the Picovoice Console."
            )
            return False

        try:
            self._porcupine = pvporcupine.create(
                access_key=self.access_key,
                keyword_paths=[self.keyword_path],
                sensitivities=[self.sensitivity],
            )
            self._is_initialized = True
            self._init_error = None
            logger.info("Porcupine initialized successfully with custom keyword: %s", self.keyword_path)
            return True
        except Exception as exc:
            self._init_error = f"Porcupine initialization failed: {exc}"
            logger.warning("Porcupine initialization error: %s", exc)
            self._is_initialized = False
            self._porcupine = None
            return False

    def set_sensitivity(self, sensitivity: float) -> None:
        self.sensitivity = max(0.0, min(1.0, sensitivity))
        if self._porcupine:
            self.reset()
            self._initialize()

    def process_frame(self, frame: np.ndarray) -> bool:
        """Feed 16-bit PCM integer frame to Porcupine. Returns True on keyword match."""
        if not self.is_available() or self._porcupine is None:
            return False

        try:
            # Porcupine requires int16 samples
            if frame.dtype != np.int16:
                pcm = (frame * 32767).astype(np.int16)
            else:
                pcm = frame

            # Porcupine expects frame_length (usually 512 samples)
            expected_len = self._porcupine.frame_length
            if len(pcm) >= expected_len:
                keyword_index = self._porcupine.process(pcm[:expected_len])
                return keyword_index >= 0
        except Exception as exc:
            logger.error("Error during Porcupine frame processing: %s", exc)

        return False

    def reset(self) -> None:
        """Release native Porcupine resources."""
        if self._porcupine is not None:
            try:
                self._porcupine.delete()
            except Exception:
                pass
            self._porcupine = None
        self._is_initialized = False
