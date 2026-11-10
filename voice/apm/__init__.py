"""WebRTC Audio Processing Module (APM) package for HAKS.

Provides Acoustic Echo Cancellation (AEC), Noise Suppression (NS),
Automatic Gain Control (AGC), and synchronized playback-reference audio tracking.
"""

from voice.apm.webrtc_apm import WebRTCAPM, SpeakerReferenceTracker, APMStatus

__all__ = ["WebRTCAPM", "SpeakerReferenceTracker", "APMStatus"]

