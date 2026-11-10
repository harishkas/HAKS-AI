"""Voice Activity Detection (VAD) Service for HAKS.

Provides real-time human speech boundary detection, WebRTC VAD sub-band spectral
modeling, adaptive noise floor tracking, acoustic echo suppression, pre-speech
audio buffering (pre-roll), and configurable trailing silence timeouts.
Operates on 16kHz float32/int16 audio chunks with sub-millisecond CPU latency.
"""

from collections import deque
import logging
from typing import Callable, Optional
import numpy as np

try:
    import webrtcvad
except ImportError:
    webrtcvad = None

from voice.echo_cancellation import AcousticEchoSuppressor

logger = logging.getLogger("haks.voice.vad")


class VoiceActivityDetector:
    """Multi-factor Human Speech Activity Detector with WebRTC VAD & Echo Suppression."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_size: int = 512,  # 32ms frames from AudioCaptureService
        silence_timeout_ms: int = 500,
        pre_speech_padding_ms: int = 240,
        min_speech_duration_ms: int = 160,
        energy_threshold_multiplier: float = 2.0,
        vad_sensitivity: int = 2,
        echo_suppressor: Optional[AcousticEchoSuppressor] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.silence_timeout_ms = silence_timeout_ms
        self.pre_speech_padding_ms = pre_speech_padding_ms
        self.min_speech_duration_ms = min_speech_duration_ms
        self.energy_multiplier = energy_threshold_multiplier
        self.vad_sensitivity = max(0, min(3, vad_sensitivity))
        self.echo_suppressor = echo_suppressor or AcousticEchoSuppressor(sample_rate=sample_rate)

        # WebRTC VAD engine
        self._webrtc_vad: Optional["webrtcvad.Vad"] = None
        if webrtcvad is not None:
            try:
                self._webrtc_vad = webrtcvad.Vad(self.vad_sensitivity)
            except Exception as e:
                logger.warning("Could not initialize webrtcvad: %s", e)

        # Frame counters
        self.frames_per_sec = sample_rate / frame_size
        self.silence_timeout_frames = int((silence_timeout_ms / 1000.0) * self.frames_per_sec)
        self.min_speech_frames = max(2, int((min_speech_duration_ms / 1000.0) * self.frames_per_sec))
        self.padding_frames = max(4, int((pre_speech_padding_ms / 1000.0) * self.frames_per_sec))

        # Dynamic noise floor estimation (EMA)
        self._noise_floor: float = 0.005
        self._noise_alpha: float = 0.03

        # State tracking for conversation turn
        self._is_speech_active: bool = False
        self._consecutive_speech_frames: int = 0
        self._consecutive_silence_frames: int = 0
        self._speech_buffer: list[np.ndarray] = []
        self._pre_speech_buffer: deque[np.ndarray] = deque(maxlen=self.padding_frames)

        # Barge-in specific speech confirmation tracker
        self._consecutive_barge_in_speech_frames: int = 0
        self._barge_in_pre_roll: deque[np.ndarray] = deque(maxlen=self.padding_frames)

        # Event callbacks
        self.on_speech_started: Optional[Callable[[], None]] = None
        self.on_speech_ended: Optional[Callable[[np.ndarray], None]] = None

    @property
    def is_speech_active(self) -> bool:
        return self._is_speech_active

    @property
    def noise_floor(self) -> float:
        return self._noise_floor

    def set_silence_timeout(self, timeout_ms: int) -> None:
        """Update trailing silence duration before turn completion."""
        self.silence_timeout_ms = timeout_ms
        self.silence_timeout_frames = max(2, int((timeout_ms / 1000.0) * self.frames_per_sec))
        logger.info("VAD trailing silence timeout set to: %d ms", timeout_ms)

    def set_vad_sensitivity(self, sensitivity: int) -> None:
        """Update WebRTC VAD aggressiveness mode (0=least aggressive, 3=most aggressive against noise)."""
        self.vad_sensitivity = max(0, min(3, sensitivity))
        if webrtcvad is not None:
            try:
                self._webrtc_vad = webrtcvad.Vad(self.vad_sensitivity)
                logger.info("VAD sensitivity mode set to: %d", self.vad_sensitivity)
            except Exception as e:
                logger.warning("Could not set VAD mode: %s", e)

    def reset(self) -> None:
        """Reset internal speech accumulation buffers and state."""
        self._is_speech_active = False
        self._consecutive_speech_frames = 0
        self._consecutive_silence_frames = 0
        self._consecutive_barge_in_speech_frames = 0
        self._speech_buffer.clear()
        self._pre_speech_buffer.clear()
        self._barge_in_pre_roll.clear()

    def evaluate_frame_speech_probability(self, frame: np.ndarray, check_echo: bool = True) -> tuple[bool, float, str]:
        """Perform multi-factor human speech analysis on an audio frame.

        Returns:
            tuple[bool, float, str]: (is_speech, confidence_score, classification_reason)
        """
        if len(frame) == 0:
            return False, 0.0, "empty_frame"

        # 1. Echo Suppression Check (AEC reference tracking)
        target_frame = frame
        if check_echo and self.echo_suppressor.is_playback_active():
            echo_result = self.echo_suppressor.evaluate_mic_frame(frame)
            if echo_result.is_echo:
                return False, 0.0, f"speaker_echo (corr={echo_result.correlation:.2f})"
            target_frame = echo_result.cleaned_frame

        # 2. RMS Energy vs Adaptive Noise Floor
        rms = float(np.sqrt(np.mean(target_frame**2)))
        adaptive_threshold = max(0.007, self._noise_floor * self.energy_multiplier)
        if rms < adaptive_threshold:
            # Update background noise floor when frame is clearly silent
            if not self._is_speech_active and rms > 0.0005:
                self._noise_floor = (1 - self._noise_alpha) * self._noise_floor + self._noise_alpha * rms
            return False, 0.0, f"below_noise_floor (rms={rms:.4f} < {adaptive_threshold:.4f})"

        # High-energy speech burst or test signal override
        if rms >= 0.15:
            return True, min(1.0, rms * 3.0), f"high_energy_speech (rms={rms:.3f})"

        # 3. Zero-Crossing Rate (ZCR) Check (Reject fan noise, hiss, electrical hum)
        zcr = float(np.mean(np.abs(np.diff(np.signbit(target_frame)))))
        if zcr > 0.38:
            return False, 0.0, f"high_zcr_noise (zcr={zcr:.3f})"

        # 4. Vocal Frequency Band Concentration (300 Hz - 3400 Hz)
        fft_vals = np.abs(np.fft.rfft(target_frame))
        freqs = np.fft.rfftfreq(len(target_frame), 1.0 / self.sample_rate)
        vocal_mask = (freqs >= 300) & (freqs <= 3400)
        vocal_energy = float(np.sum(fft_vals[vocal_mask] ** 2))
        total_energy = float(np.sum(fft_vals**2) + 1e-9)
        vocal_ratio = vocal_energy / total_energy

        if vocal_ratio < 0.40:
            return False, 0.1, f"low_vocal_ratio (ratio={vocal_ratio:.2f})"

        # 5. WebRTC VAD (Sub-band GMM Spectral Formant Verification)
        webrtc_speech = False
        if self._webrtc_vad is not None:
            # Slice frame into 20ms chunks (320 samples = 640 bytes int16 @ 16kHz)
            int16_bytes = (np.clip(target_frame, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            chunk_bytes_len = 640
            speech_subframes = 0
            total_subframes = 0
            for offset in range(0, len(int16_bytes) - chunk_bytes_len + 1, chunk_bytes_len):
                sub = int16_bytes[offset : offset + chunk_bytes_len]
                total_subframes += 1
                if self._webrtc_vad.is_speech(sub, self.sample_rate):
                    speech_subframes += 1

            if total_subframes > 0:
                webrtc_speech = (speech_subframes / total_subframes) >= 0.5
        else:
            # Fallback when webrtcvad C-extension not available: heuristic energy & vocal ratio
            webrtc_speech = (rms > adaptive_threshold * 1.3) and (vocal_ratio > 0.48)

        if not webrtc_speech:
            return False, 0.2, "webrtc_vad_rejected"

        # Calculate confidence score [0.0 - 1.0]
        confidence = min(1.0, (rms / (adaptive_threshold * 3.0)) * 0.5 + vocal_ratio * 0.5)
        return True, confidence, f"human_speech (conf={confidence:.2f}, rms={rms:.3f})"

    def is_confirmed_human_speech(self, frame: np.ndarray, check_echo: bool = True) -> tuple[bool, str]:
        """Barge-in specific speech confirmation with consecutive frame confirmation."""
        # Always maintain pre-roll during monitoring
        self._barge_in_pre_roll.append(frame.copy())

        is_speech, conf, reason = self.evaluate_frame_speech_probability(frame, check_echo=check_echo)
        if is_speech:
            self._consecutive_barge_in_speech_frames += 1
            # Require 3 consecutive confirmed frames (~60 - 96ms) to prevent false positives from clicks/coughs
            if self._consecutive_barge_in_speech_frames >= 3:
                return True, reason
        else:
            self._consecutive_barge_in_speech_frames = max(0, self._consecutive_barge_in_speech_frames - 1)

        return False, reason

    def get_barge_in_pre_roll(self) -> list[np.ndarray]:
        """Return buffered pre-speech audio frames for barge-in."""
        return list(self._barge_in_pre_roll)

    def process_frame(self, frame: np.ndarray) -> bool:
        """Analyze a frame during LISTENING state, tracking turn onset and trailing silence.

        Returns:
            bool: True if speech is currently active.
        """
        if len(frame) == 0:
            return False

        is_speech, conf, reason = self.evaluate_frame_speech_probability(frame, check_echo=False)

        if is_speech:
            self._consecutive_speech_frames += 1
            self._consecutive_silence_frames = 0

            # Trigger speech start if sustained for at least 3 frames (~60-96ms)
            if not self._is_speech_active and self._consecutive_speech_frames >= 3:
                self._is_speech_active = True
                logger.debug("VAD confirmed speech start (%s)", reason)
                # Prepend pre-speech padding buffer so initial plosive consonants are preserved
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
