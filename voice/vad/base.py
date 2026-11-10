"""Abstract Base VAD Engine Interface for HAKS.

Defines the standard contract for all Voice Activity Detection backends:
- Frame-by-frame streaming processing
- Continuous speech probability tracking
- Configurable pre-roll buffering
- Speech start confirmation and end-of-speech silence detection
- Clean state reset
"""

from abc import ABC, abstractmethod
from collections import deque
import logging
from typing import Callable, Optional
import numpy as np

logger = logging.getLogger("haks.voice.vad.base")


class BaseVADEngine(ABC):
    """Abstract interface representing a VAD engine in HAKS."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_size: int = 512,
        speech_threshold: float = 0.55,
        min_speech_duration_ms: int = 150,
        speech_confirmation_ms: int = 150,
        end_of_speech_silence_ms: int = 450,
        pre_roll_ms: int = 300,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.speech_threshold = speech_threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.speech_confirmation_ms = speech_confirmation_ms
        self.end_of_speech_silence_ms = end_of_speech_silence_ms
        self.pre_roll_ms = pre_roll_ms

        self.frames_per_sec = sample_rate / frame_size
        self.confirmation_frames = max(1, int((speech_confirmation_ms / 1000.0) * self.frames_per_sec))
        self.min_speech_frames = max(1, int((min_speech_duration_ms / 1000.0) * self.frames_per_sec))
        self.silence_timeout_frames = max(1, int((end_of_speech_silence_ms / 1000.0) * self.frames_per_sec))
        self.pre_roll_frames = max(1, int((pre_roll_ms / 1000.0) * self.frames_per_sec))

        # State tracking
        self._is_speech_active: bool = False
        self._current_speech_probability: float = 0.0
        self._consecutive_speech_frames: int = 0
        self._consecutive_silence_frames: int = 0
        self._last_inference_latency_ms: float = 0.0

        # Buffers
        self._pre_roll_buffer: deque[np.ndarray] = deque(maxlen=self.pre_roll_frames)
        self._speech_buffer: list[np.ndarray] = []

        # Callbacks
        self.on_speech_started: Optional[Callable[[], None]] = None
        self.on_speech_ended: Optional[Callable[[np.ndarray], None]] = None

    @property
    def is_speech_active(self) -> bool:
        """True if the user is currently engaged in an active speech turn."""
        return self._is_speech_active

    @property
    def speech_probability(self) -> float:
        """Latest evaluated speech probability (0.0 to 1.0)."""
        return self._current_speech_probability

    @property
    def last_inference_latency_ms(self) -> float:
        """Latency in milliseconds of the most recent inference pass."""
        return self._last_inference_latency_ms

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Identifier of this engine implementation."""
        pass

    @abstractmethod
    def predict_probability(self, frame: np.ndarray) -> float:
        """Run raw model/energy inference on a single 16kHz audio frame.

        Args:
            frame: 1D numpy array of 16kHz float32 audio.

        Returns:
            float: Speech probability between 0.0 and 1.0.
        """
        pass

    def process_frame(self, frame: np.ndarray) -> tuple[bool, float]:
        """Analyze a frame, update recurrent/temporal buffers, and emit boundary events.

        Args:
            frame: 1D numpy array (typically 512 samples float32 at 16kHz).

        Returns:
            tuple[bool, float]: (is_speech_active, speech_probability)
        """
        if len(frame) == 0:
            return self._is_speech_active, 0.0

        # Ensure float32 1D
        if frame.dtype != np.float32:
            frame = frame.astype(np.float32)
        if frame.ndim > 1:
            frame = frame.flatten()

        prob = self.predict_probability(frame)
        self._current_speech_probability = prob
        is_speech_frame = prob >= self.speech_threshold

        if is_speech_frame:
            self._consecutive_speech_frames += 1
            self._consecutive_silence_frames = 0

            # Confirm speech start after sustained confirmation frames
            if not self._is_speech_active and self._consecutive_speech_frames >= self.confirmation_frames:
                self._is_speech_active = True
                logger.debug(
                    "VAD [%s]: Speech start confirmed (prob=%.3f, threshold=%.3f)",
                    self.engine_name,
                    prob,
                    self.speech_threshold,
                )
                # Prepend pre-roll buffer
                self._speech_buffer = list(self._pre_roll_buffer)
                if self.on_speech_started:
                    try:
                        self.on_speech_started()
                    except Exception as exc:
                        logger.error("Error in on_speech_started: %s", exc)

            if self._is_speech_active:
                self._speech_buffer.append(frame.copy())
        else:
            self._consecutive_speech_frames = 0
            if self._is_speech_active:
                self._consecutive_silence_frames += 1
                self._speech_buffer.append(frame.copy())

                # End of speech detection
                if self._consecutive_silence_frames >= self.silence_timeout_frames:
                    total_frames = len(self._speech_buffer)
                    self._is_speech_active = False
                    self._consecutive_silence_frames = 0

                    # Validate minimum speech duration
                    if total_frames >= (self.min_speech_frames + self.pre_roll_frames):
                        full_audio = np.concatenate(self._speech_buffer)
                        self._speech_buffer.clear()
                        logger.debug(
                            "VAD [%s]: Speech turn completed (%.2fs)",
                            self.engine_name,
                            len(full_audio) / self.sample_rate,
                        )
                        if self.on_speech_ended:
                            try:
                                self.on_speech_ended(full_audio)
                            except Exception as exc:
                                logger.error("Error in on_speech_ended: %s", exc)
                    else:
                        logger.debug(
                            "VAD [%s]: Discarding speech fragment below min duration (%d frames)",
                            self.engine_name,
                            total_frames,
                        )
                        self._speech_buffer.clear()
            else:
                self._pre_roll_buffer.append(frame.copy())

        return self._is_speech_active, prob

    def get_accumulated_audio(self) -> np.ndarray:
        """Return accumulated audio in the current speech turn."""
        if not self._speech_buffer:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._speech_buffer)

    def reset(self) -> None:
        """Reset temporal state counters and audio buffers."""
        self._is_speech_active = False
        self._current_speech_probability = 0.0
        self._consecutive_speech_frames = 0
        self._consecutive_silence_frames = 0
        self._speech_buffer.clear()
        self._pre_roll_buffer.clear()

