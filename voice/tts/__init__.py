"""Text-to-Speech (TTS) package for HAKS."""

from voice.tts.base import TextToSpeechInterface
from voice.tts.edge_tts_engine import EdgeTTSEngine
from voice.tts.piper_tts_engine import PiperTTSEngine
from voice.tts.router import LanguageAwareTTSRouter
from voice.tts.sapi_tts_engine import SapiTTSEngine

__all__ = [
    "TextToSpeechInterface",
    "EdgeTTSEngine",
    "SapiTTSEngine",
    "PiperTTSEngine",
    "LanguageAwareTTSRouter",
]
