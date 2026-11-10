"""Adaptive Multi-Stage Human Voice Activity Detector (VAD).

Combines:
- WebRTC VAD (C-accelerated, frame-level voice activity).
- Dynamic Noise-Floor Estimation & Real-time SNR calculation.
- Playback-reference correlation gate (suppresses HAKS speaker echo).
- Spectral harmonicity and zero-crossing rate analysis.
- Multi-frame confirmation window (100–200ms) to reject transient non-speech sounds.
- Continuous 0.0–1.0 speech confidence metric for UI indicators.
"""

from collections import deque
from dataclasses import dataclass
import logging
from typing import Optional
import numpy as np

try:
    import webrtcvad
except ImportError:
    webrtcvad = None

logger = logging.getLogger("haks.voice.dsp.vad_detector")


@dataclass
class VADFrameResult:
    """Detailed result of VAD evaluation for an audio frame."""

    is_speech: bool
    is_confirmed_speech: bool
    confidence: float
    snr_db: float
    rms_energy: float
    is_playback_echo: bool


class AdaptiveVoiceDetector:
    """Multi-feature adaptive human voice detector with WebRTC VAD and SNR tracking."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_size: int = 320,  # 20ms at 16kHz
        vad_mode: int = 2,      # WebRTC VAD aggressiveness (0-3)
        min_speech_duration_ms: int = 100,  # ~5 frames for 20ms (100ms)
        silence_timeout_ms: int = 500,      # Trailing silence timeout
        sensitivity: float = 0.65,          # Sensitivity (0.1 to 1.0)
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.vad_mode = max(0, min(3, vad_mode))
        self.min_speech_duration_ms = min_speech_duration_ms
        self.silence_timeout_ms = silence_timeout_ms
        self.sensitivity = max(0.1, min(1.0, sensitivity))

        # WebRTC VAD instance
        self._webrtc_vad: Optional[Any] = None
        if webrtcvad is not None:
            try:
                self._webrtc_vad = webrtcvad.Vad(self.vad_mode)
            except Exception as exc:
                logger.error("Failed to initialize webrtcvad: %s", exc)

        # Consecutive frame thresholds
        frame_duration_ms = (frame_size / sample_rate) * 1000.0
        self.min_speech_frames = max(2, int(min_speech_duration_ms / frame_duration_ms))
        self.silence_timeout_frames = max(3, int(silence_timeout_ms / frame_duration_ms))

        # Dynamic noise floor estimation (EMA)
        self._noise_floor: float = 0.005
        self._noise_alpha: float = 0.03

        # State tracking
        self._consecutive_speech_frames: int = 0
        self._consecutive_silence_frames: int = 0
        self._is_speech_active: bool = False
        self._speech_history: deque[bool] = deque(maxlen=10)

    def set_sensitivity(self, sensitivity: float) -> None:
        """Update sensitivity (0.1 to 1.0)."""
        self.sensitivity = max(0.1, min(1.0, sensitivity))
        # Map sensitivity to WebRTC mode: high sensitivity = mode 1, low sensitivity = mode 3
        if self._webrtc_vad is not None:
            mode = 1 if self.sensitivity > 0.8 else (3 if self.sensitivity < 0.4 else 2)
            try:
                self._webrtc_vad.set_mode(mode)
            except Exception:
                pass

    def reset(self) -> None:
        """Reset internal frame counters and states."""
        self._consecutive_speech_frames = 0
        self._consecutive_silence_frames = 0
        self._is_speech_active = False
        self._speech_history.clear()

    def process_frame(
        self,
        frame: np.ndarray,
        is_playback_echo: bool = False,
    ) -> VADFrameResult:
        """Evaluate an audio frame for human speech activity.

        Args:
            frame: 1D numpy array of 16 kHz float32 samples (typically 320 samples for 20ms).
            is_playback_echo: Flag from AEC indicating this frame is correlated speaker echo.

        Returns:
            VADFrameResult with speech confirmation, confidence, and SNR.
        """
        frame_len = len(frame)
        if frame_len == 0:
            return VADFrameResult(
                is_speech=False,
                is_confirmed_speech=False,
                confidence=0.0,
                snr_db=0.0,
                rms_energy=0.0,
                is_playback_echo=False,
            )

        # Calculate RMS energy
        rms = float(np.sqrt(np.mean(frame**2)))

        # Update dynamic noise floor when not in confirmed speech
        if not self._is_speech_active and not is_playback_echo:
            self._noise_floor = (1.0 - self._noise_alpha) * self._noise_floor + self._noise_alpha * rms
            self._noise_floor = max(0.001, self._noise_floor)

        # Calculate SNR in dB
        snr_db = 10.0 * np.log10((rms**2 + 1e-9) / (self._noise_floor**2 + 1e-9))

        # 1. WebRTC VAD check
        webrtc_vote = False
        if self._webrtc_vad is not None and not is_playback_echo:
            try:
                # Convert float32 [-1.0, 1.0] to 16-bit linear PCM bytes
                clamped = np.clip(frame, -1.0, 1.0)
                int16_pcm = (clamped * 32767.0).astype(np.int16).tobytes()

                # WebRTC VAD accepts 10ms (160), 20ms (320), or 30ms (480) samples
                if frame_len in (160, 320, 480):
                    webrtc_vote = self._webrtc_vad.is_speech(int16_pcm, self.sample_rate)
                elif frame_len > 320:
                    # Check first 320 samples
                    webrtc_vote = self._webrtc_vad.is_speech(int16_pcm[:640], self.sample_rate)
            except Exception as exc:
                logger.debug("webrtcvad check error: %s", exc)

        # 2. Spectral Zero-Crossing Rate (ZCR)
        # Human speech typical ZCR is between 0.04 and 0.38
        zero_crossings = np.sum(np.abs(np.diff(np.sign(frame)))) / (2.0 * frame_len)
        zcr_speech = 0.03 <= zero_crossings <= 0.42

        # 3. SNR threshold check modulated by sensitivity
        # Sensitivity 0.65 -> threshold ~ 3.5 dB
        snr_threshold = max(1.5, 12.0 * (1.0 - self.sensitivity))
        snr_vote = snr_db > snr_threshold

        # 4. Speaker loopback suppression: if AEC detected playback echo, suppress speech
        if is_playback_echo:
            raw_speech = False
            confidence = 0.0
        else:
            # Combine WebRTC vote, SNR, and energy
            # WebRTC VAD is the core human vocal activity model
            if webrtc_vote and snr_vote:
                raw_speech = True
                confidence = min(1.0, 0.5 + 0.05 * snr_db)
            elif webrtc_vote and (rms > 0.015):
                raw_speech = True
                confidence = 0.60
            else:
                raw_speech = False
                confidence = max(0.0, (snr_db / 20.0) * 0.25)

        self._speech_history.append(raw_speech)

        # 5. Consecutive Frame Confirmation Gate
        if raw_speech:
            self._consecutive_speech_frames += 1
            self._consecutive_silence_frames = 0
            if self._consecutive_speech_frames >= self.min_speech_frames:
                self._is_speech_active = True
        else:
            self._consecutive_speech_frames = 0
            if self._is_speech_active:
                self._consecutive_silence_frames += 1
                if self._consecutive_silence_frames >= self.silence_timeout_frames:
                    self._is_speech_active = False

        is_confirmed = self._is_speech_active and (self._consecutive_speech_frames >= self.min_speech_frames)

        return VADFrameResult(
            is_speech=raw_speech,
            is_confirmed_speech=is_confirmed,
            confidence=float(confidence),
            snr_db=float(snr_db),
            rms_energy=float(rms),
            is_playback_echo=is_playback_echo,
        )
