"""Abstract interface for Text-to-Speech (TTS) Engines in HAKS."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from voice.models import TTSVoiceInfo


class TextToSpeechInterface(ABC):
    """Abstract interface for voice synthesis engines."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Engine name."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Whether this TTS engine is functional and ready."""
        pass

    @abstractmethod
    def supports_tamil(self) -> bool:
        """Whether this engine supports genuine Tamil voice synthesis."""
        pass

    @abstractmethod
    def supports_english(self) -> bool:
        """Whether this engine supports English voice synthesis."""
        pass

    @abstractmethod
    def list_voices(self) -> list[TTSVoiceInfo]:
        """List available voices for this engine."""
        pass

    @abstractmethod
    async def synthesize_to_file(
        self,
        text: str,
        output_path: Path,
        voice: Optional[str] = None,
        speed: float = 1.0,
        volume: float = 1.0,
    ) -> Path:
        """Synthesize text and save audio output to file (WAV or MP3).

        Returns:
            Path: Filesystem path to the written audio file.
        """
        pass
