"""Abstract interface for Wake Word Detection in HAKS."""

from abc import ABC, abstractmethod
import numpy as np


class WakeWordDetectorInterface(ABC):
    """Abstract base class for wake word engines."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the wake word engine."""
        pass

    @property
    @abstractmethod
    def wake_phrase(self) -> str:
        """Target wake phrase (e.g. 'Hey HAKS')."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Whether this detector engine is ready and configured."""
        pass

    @abstractmethod
    def set_sensitivity(self, sensitivity: float) -> None:
        """Adjust detection sensitivity between 0.0 (strict) and 1.0 (sensitive)."""
        pass

    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> bool:
        """Process an incoming audio frame (16kHz float32).

        Returns:
            bool: True if the wake phrase was detected.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal buffers and detection state."""
        pass
