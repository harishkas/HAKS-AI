"""Wake word detection package for HAKS."""

from voice.wake_word.base import WakeWordDetectorInterface
from voice.wake_word.local_detector import LocalAcousticWakeDetector
from voice.wake_word.porcupine_detector import PorcupineWakeDetector

__all__ = [
    "WakeWordDetectorInterface",
    "LocalAcousticWakeDetector",
    "PorcupineWakeDetector",
]
