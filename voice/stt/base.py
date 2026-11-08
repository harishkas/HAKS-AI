"""Abstract interface for Speech-to-Text (STT) Engines in HAKS."""

from abc import ABC, abstractmethod
import numpy as np
from voice.models import TranscriptionResult


class SpeechToTextInterface(ABC):
    """Abstract interface for offline and local Speech-to-Text engines."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Engine display name."""
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether model weights are loaded in memory."""
        pass

    @abstractmethod
    def load_model(self) -> bool:
        """Load model into memory (lazy initialization)."""
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """Unload model from memory to free system RAM."""
        pass

    @abstractmethod
    def transcribe(
        self,
        audio_data: np.ndarray,
        language: str = "auto",
        initial_prompt: str = "",
    ) -> TranscriptionResult:
        """Transcribe 16kHz float32 audio numpy array to text with language & confidence."""
        pass

    @abstractmethod
    def benchmark(self) -> dict:
        """Run quick inference benchmark on CPU and return latency metrics."""
        pass
