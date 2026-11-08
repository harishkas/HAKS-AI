"""Speech-to-Text (STT) package for HAKS."""

from voice.stt.base import SpeechToTextInterface
from voice.stt.faster_whisper_stt import FasterWhisperSTT

__all__ = [
    "SpeechToTextInterface",
    "FasterWhisperSTT",
]
