"""Gemini Live Native Real-time Speech-to-Speech Subsystem for HAKS.

Provides bidirectional asynchronous WebSocket audio streaming using Google's
Gemini Live API (gemini-3.8-live), 24 kHz low-latency linear PCM playback,
and instant barge-in interruption.
"""

from voice.gemini_live.client import GeminiLiveClient, LiveConnectionStatus
from voice.gemini_live.pcm_player import PCMStreamPlayer
from voice.gemini_live.audio_streamer import AudioStreamerBridge

__all__ = [
    "GeminiLiveClient",
    "LiveConnectionStatus",
    "PCMStreamPlayer",
    "AudioStreamerBridge",
]

