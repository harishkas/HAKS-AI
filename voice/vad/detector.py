"""Voice Activity Detection (VAD) Service for HAKS.

Coordinates advanced Voice Activity Detection engines:
- Silero VAD (primary deep-learning CPU detector with recurrent temporal modeling)
- WebRTC VAD (lightweight, zero-overhead GMM fallback)

Provides safe fallback handling, live speech probability metrics,
pre-roll speech buffering, and trailing silence completion.
Fully backward compatible with legacy VoiceActivityDetector interface.
"""

import logging
from typing import Callable, Optional
import numpy as np

from voice.vad.base import BaseVADEngine
from voice.vad.silero_vad import SileroVAD
from voice.vad.webrtc_vad import WebRTCVADEngine

logger = logging.getLogger("haks.voice.vad")


class VoiceActivityDetector:
    """Unified Voice Activity Detector with primary Silero VAD and resilient WebRTC VAD fallback."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_size: int = 512,  # 32ms frames at 16kHz
        silence_timeout_ms: int = 1200,
        pre_speech_padding_ms: int = 400,
        min_speech_duration_ms: int = 250,
        energy_threshold_multiplier: float = 2.2,
        vad_engine: str = "silero",
        speech_threshold: float = 0.55,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.silence_timeout_ms = silence_timeout_ms
        self.pre_speech_padding_ms = pre_speech_padding_ms
        self.min_speech_duration_ms = min_speech_duration_ms
        self.energy_multiplier = energy_threshold_multiplier
        self.preferred_engine = vad_engine.lower()
        self.speech_threshold = speech_threshold

        # Event callbacks
        self._on_speech_started: Optional[Callable[[], None]] = None
        self._on_speech_ended: Optional[Callable[[np.ndarray], None]] = None

        # Fallback state
        self._is_fallback_active: bool = False
        self._fallback_reason: Optional[str] = None

        # Instantiate fallback engine first so it is immediately ready
        self._fallback_engine = WebRTCVADEngine(
            sample_rate=self.sample_rate,
            frame_size=self.frame_size,
            speech_threshold=0.45,
            min_speech_duration_ms=self.min_speech_duration_ms,
            speech_confirmation_ms=120,
            end_of_speech_silence_ms=self.silence_timeout_ms,
            pre_roll_ms=self.pre_speech_padding_ms,
        )

        # Primary engine
        self._silero_engine: Optional[SileroVAD] = None
        self._active_engine: BaseVADEngine = self._fallback_engine

        self._init_primary_engine()

    def _init_primary_engine(self) -> None:
        """Initialize primary Silero VAD or configure fallback."""
        if self.preferred_engine == "silero":
            try:
                self._silero_engine = SileroVAD(
                    sample_rate=self.sample_rate,
                    frame_size=self.frame_size,
                    speech_threshold=self.speech_threshold,
                    min_speech_duration_ms=self.min_speech_duration_ms,
                    speech_confirmation_ms=150,
                    end_of_speech_silence_ms=self.silence_timeout_ms,
                    pre_roll_ms=self.pre_speech_padding_ms,
                    use_onnx=True,
                )
                self._active_engine = self._silero_engine
                self._is_fallback_active = False
                logger.info("VAD: Silero VAD configured as primary engine.")
            except Exception as exc:
                self._activate_fallback(f"Silero VAD init exception: {type(exc).__name__}")
        else:
            self._active_engine = self._fallback_engine
            self._is_fallback_active = False
            logger.info("VAD: WebRTC VAD configured as requested engine.")

        self._wire_engine_callbacks(self._active_engine)

    def _activate_fallback(self, reason: str) -> None:
        """Switch active engine to WebRTC VAD fallback safely."""
        self._is_fallback_active = True
        self._fallback_reason = reason
        self._active_engine = self._fallback_engine
        self._wire_engine_callbacks(self._active_engine)
        logger.warning(
            "VAD: Switched to WebRTC VAD fallback safely (reason=%s).",
            reason,
        )

    def _wire_engine_callbacks(self, engine: BaseVADEngine) -> None:
        """Attach external callbacks to underlying engine."""
        engine.on_speech_started = self._handle_speech_started
        engine.on_speech_ended = self._handle_speech_ended

    def _handle_speech_started(self) -> None:
        if self._on_speech_started:
            try:
                self._on_speech_started()
            except Exception as exc:
                logger.error("Error in on_speech_started: %s", exc)

    def _handle_speech_ended(self, audio: np.ndarray) -> None:
        if self._on_speech_ended:
            try:
                self._on_speech_ended(audio)
            except Exception as exc:
                logger.error("Error in on_speech_ended: %s", exc)

    # --------------------------------------------------------------------------
    # Public Properties & Callbacks
    # --------------------------------------------------------------------------

    @property
    def on_speech_started(self) -> Optional[Callable[[], None]]:
        return self._on_speech_started

    @on_speech_started.setter
    def on_speech_started(self, cb: Optional[Callable[[], None]]) -> None:
        self._on_speech_started = cb
        self._active_engine.on_speech_started = self._handle_speech_started

    @property
    def on_speech_ended(self) -> Optional[Callable[[np.ndarray], None]]:
        return self._on_speech_ended

    @on_speech_ended.setter
    def on_speech_ended(self, cb: Optional[Callable[[np.ndarray], None]]) -> None:
        self._on_speech_ended = cb
        self._active_engine.on_speech_ended = self._handle_speech_ended

    @property
    def is_speech_active(self) -> bool:
        return self._active_engine.is_speech_active

    @property
    def speech_probability(self) -> float:
        return self._active_engine.speech_probability

    @property
    def engine_name(self) -> str:
        return self._active_engine.engine_name

    @property
    def is_fallback_active(self) -> bool:
        return self._is_fallback_active

    @property
    def fallback_reason(self) -> Optional[str]:
        return self._fallback_reason

    @property
    def last_inference_latency_ms(self) -> float:
        return self._active_engine.last_inference_latency_ms

    def set_engine(self, engine_name: str) -> None:
        """Switch active VAD engine ('silero' or 'webrtc')."""
        self.preferred_engine = engine_name.lower()
        self._init_primary_engine()

    def set_silence_timeout(self, timeout_ms: int) -> None:
        """Update trailing silence duration before turn completion."""
        self.silence_timeout_ms = timeout_ms
        self._fallback_engine.end_of_speech_silence_ms = timeout_ms
        self._fallback_engine.silence_timeout_frames = max(
            1, int((timeout_ms / 1000.0) * self._fallback_engine.frames_per_sec)
        )
        if self._silero_engine:
            self._silero_engine.end_of_speech_silence_ms = timeout_ms
            self._silero_engine.silence_timeout_frames = max(
                1, int((timeout_ms / 1000.0) * self._silero_engine.frames_per_sec)
            )
        logger.info("VAD trailing silence timeout set to: %d ms", timeout_ms)

    def reset(self) -> None:
        """Reset internal speech accumulation buffers and recurrent model state."""
        self._active_engine.reset()

    def process_frame(self, frame: np.ndarray) -> bool:
        """Analyze a single audio frame (typically 512 samples float32).

        Returns:
            bool: True if speech is currently active in this frame.
        """
        if len(frame) == 0:
            return False

        try:
            is_active, _ = self._active_engine.process_frame(frame)
            return is_active
        except Exception as exc:
            # If primary Silero engine fails during inference, switch safely to fallback
            if self._active_engine != self._fallback_engine:
                self._activate_fallback(f"Inference error: {type(exc).__name__}")
                is_active, _ = self._fallback_engine.process_frame(frame)
                return is_active
            logger.error("VAD process_frame error: %s", exc)
            return False

    def get_accumulated_audio(self) -> np.ndarray:
        """Return the audio accumulated during the current speech turn."""
        return self._active_engine.get_accumulated_audio()
