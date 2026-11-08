"""Voice AI Assistant Package for HAKS.

Provides complete Phase 6 Advanced Voice capabilities:
- Real-time audio capture & device enumeration
- Voice activity detection with adaptive noise floor
- Wake word detection ('Hey HAKS' local acoustic + Porcupine)
- Faster-Whisper bilingual STT (Tamil, English, Mixed)
- Multi-engine bilingual TTS (Edge Neural, SAPI, Piper)
- Continuous conversation state machine & barge-in interruption
"""

from voice.audio_capture import AudioCaptureService
from voice.conversation_engine import ContinuousVoiceEngine
from voice.models import (
    AudioDevice,
    SpokenUtterance,
    TTSVoiceInfo,
    TranscriptionResult,
    VoiceCommandType,
    VoiceState,
)
from voice.player import AudioPlayerService
from voice.stt.faster_whisper_stt import FasterWhisperSTT
from voice.tts.edge_tts_engine import EdgeTTSEngine
from voice.tts.router import LanguageAwareTTSRouter
from voice.tts.sapi_tts_engine import SapiTTSEngine
from voice.vad import VoiceActivityDetector
from voice.wake_word.local_detector import LocalAcousticWakeDetector
from voice.wake_word.porcupine_detector import PorcupineWakeDetector

__all__ = [
    "AudioCaptureService",
    "AudioDevice",
    "AudioPlayerService",
    "ContinuousVoiceEngine",
    "EdgeTTSEngine",
    "FasterWhisperSTT",
    "LanguageAwareTTSRouter",
    "LocalAcousticWakeDetector",
    "PiperTTSEngine",
    "PorcupineWakeDetector",
    "SapiTTSEngine",
    "SpokenUtterance",
    "TTSVoiceInfo",
    "TranscriptionResult",
    "VoiceActivityDetector",
    "VoiceCommandType",
    "VoiceState",
]
