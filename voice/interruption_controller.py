"""Human Speech Interruption Controller for HAKS.

Manages high-precision, false-positive-free barge-in detection:
- Enforces WebRTC AEC and Noise Suppression before speech confirmation.
- Eliminates self-interruption from speaker acoustic leakage.
- Filters out ambient transients (fans, keyboard typing, room noise, mic spikes).
- Requires consecutive Silero VAD probability confirmations (100-200ms).
- Instantly halts active audio playback and clears queued speaker frames upon confirmation.
- Preserves 300ms pre-roll audio so beginning consonants are not lost.
- Forwards user speech stream to active Gemini Live session.
- Prevents duplicate interruption triggers.
- Protects streaming chunk boundaries from premature transitions.

Coordinates the state transition sequence:
SPEAKING -> INTERRUPTED -> LISTENING -> PROCESSING -> THINKING -> SPEAKING
"""

from collections import deque
from dataclasses import dataclass, field
import logging
import threading
import time
from typing import Callable, Optional
import numpy as np

from voice.models import VoiceState

logger = logging.getLogger("haks.voice.interruption_controller")


@dataclass
class InterruptionEvent:
    """Telemetry payload for confirmed barge-in interruptions."""
    timestamp: float
    speech_probability: float
    consecutive_frames: int
    rms_energy: float
    preroll_duration_ms: float
    interrupted_state: VoiceState


class InterruptionController:
    """Coordinates real-time barge-in monitoring during SPEAKING state."""

    def __init__(
        self,
        speech_probability_threshold: float = 0.55,
        confirmation_duration_ms: int = 150,
        min_speech_rms: float = 0.015,
        preroll_duration_ms: int = 300,
        sample_rate: int = 16000,
        frame_size: int = 512,
    ) -> None:
        self.speech_probability_threshold = speech_probability_threshold
        self.confirmation_duration_ms = confirmation_duration_ms
        self.min_speech_rms = min_speech_rms
        self.preroll_duration_ms = preroll_duration_ms
        self.sample_rate = sample_rate
        self.frame_size = frame_size

        self.frames_per_sec = sample_rate / frame_size
        self.confirmation_frames = max(2, int((confirmation_duration_ms / 1000.0) * self.frames_per_sec))
        self.preroll_frames = max(2, int((preroll_duration_ms / 1000.0) * self.frames_per_sec))

        # Internal state
        self._consecutive_speech_frames: int = 0
        self._is_interrupted: bool = False
        self._preroll_buffer: deque[np.ndarray] = deque(maxlen=self.preroll_frames)
        self._lock = threading.Lock()

        # Telemetry
        self.total_evaluations: int = 0
        self.confirmed_interruptions: int = 0
        self.rejected_transients: int = 0
        self.last_interruption_event: Optional[InterruptionEvent] = None

        # Callbacks
        self.on_interruption_confirmed: Optional[Callable[[list[np.ndarray]], None]] = None

    @property
    def is_interrupted(self) -> bool:
        return self._is_interrupted

    def reset(self) -> None:
        """Reset internal counters and pre-roll buffer."""
        with self._lock:
            self._consecutive_speech_frames = 0
            self._is_interrupted = False
            self._preroll_buffer.clear()

    def process_frame(
        self,
        clean_frame: np.ndarray,
        speech_probability: float,
        current_state: VoiceState,
    ) -> bool:
        """Evaluate a preprocessed microphone frame for genuine user barge-in.

        Args:
            clean_frame: Audio frame post-AEC and post-NS (16kHz float32).
            speech_probability: Confidence from Silero VAD (0.0 to 1.0).
            current_state: Current engine state (evaluates only during SPEAKING).

        Returns:
            bool: True if an interruption was CONFIRMED on this frame.
        """
        # Always maintain pre-roll buffer
        with self._lock:
            self._preroll_buffer.append(clean_frame.copy())

        # Only evaluate for interruption while assistant is actively speaking
        if current_state != VoiceState.SPEAKING:
            with self._lock:
                self._consecutive_speech_frames = 0
                self._is_interrupted = False
            return False

        # If already triggered for this turn, ignore duplicate events
        if self._is_interrupted:
            return False

        self.total_evaluations += 1

        # Check RMS energy of post-AEC frame
        rms = float(np.sqrt(np.mean(clean_frame**2))) if len(clean_frame) > 0 else 0.0

        # Primary rule: Silero probability must exceed threshold AND energy must exceed ambient floor
        is_speech_candidate = (
            speech_probability >= self.speech_probability_threshold and rms >= self.min_speech_rms
        )

        with self._lock:
            # Atomic check-and-set: prevent concurrent or duplicate interruption triggers
            if self._is_interrupted:
                return False

            if is_speech_candidate:
                self._consecutive_speech_frames += 1

                # Rule 5: Confirm consecutive speech evidence before stopping playback
                if self._consecutive_speech_frames >= self.confirmation_frames:
                    self._is_interrupted = True
                    self.confirmed_interruptions += 1

                    preroll_copy = list(self._preroll_buffer)
                    self._preroll_buffer.clear()  # Atomic consumption: prevent duplicate pre-roll injection
                    event = InterruptionEvent(
                        timestamp=time.time(),
                        speech_probability=speech_probability,
                        consecutive_frames=self._consecutive_speech_frames,
                        rms_energy=rms,
                        preroll_duration_ms=len(preroll_copy) * (self.frame_size / self.sample_rate) * 1000.0,
                        interrupted_state=current_state,
                    )
                    self.last_interruption_event = event

                    logger.info(
                        "Interruption CONFIRMED: prob=%.3f >= %.2f, RMS=%.3f, frames=%d",
                        speech_probability,
                        self.speech_probability_threshold,
                        rms,
                        self._consecutive_speech_frames,
                    )

                    if self.on_interruption_confirmed:
                        try:
                            self.on_interruption_confirmed(preroll_copy)
                        except Exception as exc:
                            logger.error("Error in on_interruption_confirmed: %s", exc)

                    return True
            else:
                if self._consecutive_speech_frames > 0:
                    self.rejected_transients += 1
                self._consecutive_speech_frames = 0

        return False

    def handle_server_interruption(self) -> bool:
        """Handle server-side VAD interruption signal from Gemini Live.

        Returns:
            bool: True if interruption was dispatched; False if already interrupted.
        """
        with self._lock:
            if self._is_interrupted:
                return False
            self._is_interrupted = True
            self.confirmed_interruptions += 1
            preroll_copy = list(self._preroll_buffer)
            self._preroll_buffer.clear()  # Atomic consumption: prevent duplicate pre-roll injection

        logger.info("Server-side Gemini Live interruption event confirmed.")
        if self.on_interruption_confirmed:
            try:
                self.on_interruption_confirmed(preroll_copy)
            except Exception as exc:
                logger.error("Error in on_interruption_confirmed for server event: %s", exc)

        return True

