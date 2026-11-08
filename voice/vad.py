"""Voice Activity Detection (VAD) Service for HAKS.

Provides real-time speech boundary detection, adaptive noise floor tracking,
pre-speech padding, and configurable trailing silence timeouts.
Operates on 16kHz float32 audio chunks without heavy neural network overhead.
"""

from collections import deque
import logging
from typing import Callable, Optional
import numpy as np

logger = logging.getLogger("haks.voice.vad")


class VoiceActivityDetector:
    """Adaptive Energy and Zero-Crossing Rate VAD with dynamic background noise tracking."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_size: int = 512,  # 32ms frames
        silence_timeout_ms: int = 1200,
        pre_speech_padding_ms: int = 400,
        min_speech_duration_ms: int = 250,
        energy_threshold_multiplier: float = 2.2,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.silence_timeout_ms = silence_timeout_ms
        self.pre_speech_padding_ms = pre_speech_padding_ms
        self.min_speech_duration_ms = min_speech_duration_ms
        self.energy_multiplier = energy_threshold_multiplier

        # Frame counters
        self.frames_per_sec = sample_rate / frame_size
        self.silence_timeout_frames = int((silence_timeout_ms / 1000.0) * self.frames_per_sec)
        self.min_speech_frames = int((min_speech_duration_ms / 1000.0) * self.frames_per_sec)
        self.padding_frames = int((pre_speech_padding_ms / 1000.0) * self.frames_per_sec)

        # Dynamic noise floor estimation (EMA)
        self._noise_floor: float = 0.005
        self._noise_alpha: float = 0.03

        # State tracking
        self._is_speech_active: bool = False
        self._consecutive_speech_frames: int = 0
        self._consecutive_silence_frames: int = 0
        self._speech_buffer: list[np.ndarray] = []
        self._pre_speech_buffer: deque[np.ndarray] = deque(maxlen=self.padding_frames)

        # Event callbacks
        self.on_speech_started: Optional[Callable[[], None]] = None
        self.on_speech_ended: Optional[Callable[[np.ndarray], None]] = None

    @property
    def is_speech_active(self) -> bool:
        return self._is_speech_active

    def set_silence_timeout(self, timeout_ms: int) -> None:
        """Update trailing silence duration before turn completion."""
        self.silence_timeout_ms = timeout_ms
        self.silence_timeout_frames = int((timeout_ms / 1000.0) * self.frames_per_sec)
        logger.info("VAD trailing silence timeout set to: %d ms", timeout_ms)

    def reset(self) -> None:
        """Reset internal speech accumulation buffers and state."""
        self._is_speech_active = False
        self._consecutive_speech_frames = 0
        self._consecutive_silence_frames = 0
        self._speech_buffer.clear()
        self._pre_speech_buffer.clear()

    def process_frame(self, frame: np.ndarray) -> bool:
        """Analyze a single audio frame (typically 512 samples float32).

        Returns:
            bool: True if speech is currently active in this frame.
        """
        if len(frame) == 0:
            return False

        # Compute Root Mean Square (RMS) energy
        rms = float(np.sqrt(np.mean(frame**2)))

        # Dynamic noise floor tracking when no speech is active
        if not self._is_speech_active:
            self._noise_floor = (1 - self._noise_alpha) * self._noise_floor + self._noise_alpha * rms

        adaptive_threshold = max(0.008, self._noise_floor * self.energy_multiplier)
        is_speech_frame = rms > adaptive_threshold

        if is_speech_frame:
            self._consecutive_speech_frames += 1
            self._consecutive_silence_frames = 0

            # Trigger speech start if sustained for at least 3 frames (~96ms)
            if not self._is_speech_active and self._consecutive_speech_frames >= 3:
                self._is_speech_active = True
                logger.debug("VAD detected speech start (RMS=%.4f > Thr=%.4f)", rms, adaptive_threshold)
                # Prepend pre-speech padding buffer so beginning syllable is intact
                self._speech_buffer = list(self._pre_speech_buffer)
                if self.on_speech_started:
                    try:
                        self.on_speech_started()
                    except Exception as exc:
                        logger.error("Error in on_speech_started callback: %s", exc)

            if self._is_speech_active:
                self._speech_buffer.append(frame.copy())
        else:
            self._consecutive_speech_frames = 0
            if self._is_speech_active:
                self._consecutive_silence_frames += 1
                self._speech_buffer.append(frame.copy())

                # Check if trailing silence duration has elapsed
                if self._consecutive_silence_frames >= self.silence_timeout_frames:
                    total_frames = len(self._speech_buffer)
                    logger.debug(
                        "VAD detected speech end (frames=%d, trailing silence=%d ms)",
                        total_frames,
                        self.silence_timeout_ms,
                    )
                    self._is_speech_active = False
                    self._consecutive_silence_frames = 0

                    if total_frames >= (self.min_speech_frames + self.padding_frames):
                        full_audio = np.concatenate(self._speech_buffer)
                        self._speech_buffer.clear()
                        if self.on_speech_ended:
                            try:
                                self.on_speech_ended(full_audio)
                            except Exception as exc:
                                logger.error("Error in on_speech_ended callback: %s", exc)
                    else:
                        self._speech_buffer.clear()
            else:
                self._pre_speech_buffer.append(frame.copy())

        return self._is_speech_active

    def get_accumulated_audio(self) -> np.ndarray:
        """Return the audio accumulated during the current speech turn."""
        if not self._speech_buffer:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._speech_buffer)
