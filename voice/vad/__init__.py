"""Voice Activity Detection (VAD) package for HAKS.

Includes:
- VoiceActivityDetector: Unified VAD coordinator with Silero primary and WebRTC fallback.
- BaseVADEngine: Abstract interface for streaming VAD implementations.
- SileroVAD: Primary high-accuracy neural VAD with CPU inference.
- WebRTCVADEngine: Fast lightweight GMM-based fallback.
"""

from voice.vad.base import BaseVADEngine
from voice.vad.detector import VoiceActivityDetector
from voice.vad.silero_vad import SileroVAD
from voice.vad.webrtc_vad import WebRTCVADEngine

__all__ = [
    "BaseVADEngine",
    "SileroVAD",
    "WebRTCVADEngine",
    "VoiceActivityDetector",
]
