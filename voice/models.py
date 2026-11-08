"""Data models and state representations for HAKS Voice Architecture.

Defines the core state machine, transcription structures, audio device metadata,
and voice command classifications.
"""

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class VoiceState(str, Enum):
    """Voice assistant state machine states."""
    STANDBY = "STANDBY"
    WAKE_DETECTED = "WAKE_DETECTED"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    INTERRUPTED = "INTERRUPTED"
    SLEEP = "SLEEP"
    STOPPED = "STOPPED"


class VoiceCommandType(str, Enum):
    """Categorization of voice control commands."""
    STOP = "STOP"
    SLEEP = "SLEEP"
    WAKE = "WAKE"
    CANCEL = "CANCEL"
    REPEAT = "REPEAT"
    STOP_SPEAKING = "STOP_SPEAKING"
    NONE = "NONE"


class AudioDevice(BaseModel):
    """Audio input/output hardware device details."""
    index: int
    name: str
    host_api: str
    max_input_channels: int = 0
    max_output_channels: int = 0
    default_samplerate: float = 16000.0
    is_default_input: bool = False
    is_default_output: bool = False


class TranscriptionResult(BaseModel):
    """Speech-to-Text transcription output."""
    text: str
    language: str = "en"
    language_probability: float = 0.0
    confidence: float = 0.0
    duration_seconds: float = 0.0
    is_final: bool = True
    raw_segments: list[dict[str, Any]] = Field(default_factory=list)


class SpokenUtterance(BaseModel):
    """Text-to-speech spoken output payload."""
    text: str
    detected_language: str = "en"
    voice_name: str
    audio_path: Optional[str] = None
    duration_seconds: float = 0.0


class TTSVoiceInfo(BaseModel):
    """Voice metadata for text-to-speech synthesis."""
    id: str
    name: str
    language: str  # 'ta-IN', 'en-IN', 'en-US'
    gender: str = "Unknown"
    engine: str = "edge"  # 'edge', 'sapi', 'piper'
