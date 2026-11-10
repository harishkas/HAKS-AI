"""Real-Time Microphone Audio Capture Service for HAKS.

Provides non-blocking audio capture, device selection, live RMS audio level monitoring,
thread-safe circular buffering, and mute controls using sounddevice.
Calibrated for 16,000 Hz mono float32 audio.
"""

from collections import deque
import logging
import threading
import time
from typing import Callable, Optional
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

from config.settings import AppConfig, get_settings
from voice.dsp import AudioProcessingPipeline, ProcessedAudioFrame
from voice.models import AudioDevice

logger = logging.getLogger("haks.voice.audio_capture")


class AudioCaptureService:
    """Manages real-time microphone stream, device enumeration, and ring buffering."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or get_settings()
        self.sample_rate: int = self.config.voice.input_sample_rate
        self.block_size: int = 512  # 32ms frames at 16kHz
        self.channels: int = 1

        self._stream: Optional[Any] = None
        self._is_capturing: bool = False
        self._is_muted: bool = False
        self._device_index: Optional[int] = self.config.voice.input_device_index

        # Unified Audio Processing Pipeline (DC removal, HPF, Hum notch, AEC, NS, Enhancer, VAD)
        self.pipeline = AudioProcessingPipeline(self.config, sample_rate=self.sample_rate)

        # Thread-safe ring buffer holding the most recent 10 seconds of audio frames
        self._max_buffer_frames = int((16000 / self.block_size) * 10)
        self._ring_buffer: deque[np.ndarray] = deque(maxlen=self._max_buffer_frames)
        self._buffer_lock = threading.Lock()

        # Listeners for real-time chunks, audio levels, and speech confidence
        self._chunk_callbacks: list[Callable[[np.ndarray], None]] = []
        self._level_callbacks: list[Callable[[float], None]] = []
        self._confidence_callbacks: list[Callable[[float], None]] = []

        self._current_audio_level: float = 0.0
        self._last_level_emit_time: float = 0.0

    @property
    def is_capturing(self) -> bool:
        return self._is_capturing

    @property
    def is_muted(self) -> bool:
        return self._is_muted

    def set_muted(self, muted: bool) -> None:
        """Toggle microphone mute."""
        self._is_muted = muted
        logger.info("Audio capture muted: %s", muted)

    def add_chunk_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Register a callback invoked with each incoming audio chunk (np.ndarray float32)."""
        if callback not in self._chunk_callbacks:
            self._chunk_callbacks.append(callback)

    def remove_chunk_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        if callback in self._chunk_callbacks:
            self._chunk_callbacks.remove(callback)

    def add_level_callback(self, callback: Callable[[float], None]) -> None:
        """Register a callback for live RMS audio level (0.0 to 1.0)."""
        if callback not in self._level_callbacks:
            self._level_callbacks.append(callback)

    def remove_level_callback(self, callback: Callable[[float], None]) -> None:
        if callback in self._level_callbacks:
            self._level_callbacks.remove(callback)

    def add_confidence_callback(self, callback: Callable[[float], None]) -> None:
        """Register a callback for live speech confidence (0.0 to 1.0)."""
        if callback not in self._confidence_callbacks:
            self._confidence_callbacks.append(callback)

    def remove_confidence_callback(self, callback: Callable[[float], None]) -> None:
        if callback in self._confidence_callbacks:
            self._confidence_callbacks.remove(callback)

    def calibrate_noise_floor(self) -> None:
        """Begin room ambient noise calibration."""
        self.pipeline.calibrate_noise_floor()

    def finish_noise_calibration(self) -> float:
        """Finish room ambient noise calibration and return baseline noise floor."""
        return self.pipeline.finish_noise_calibration()

    @staticmethod
    def list_input_devices() -> list[AudioDevice]:
        """Enumerate all available physical and virtual audio input devices."""
        devices: list[AudioDevice] = []
        if sd is None:
            return devices

        try:
            device_list = sd.query_devices()
            default_input_idx = sd.default.device[0] if sd.default.device else -1

            for idx, dev in enumerate(device_list):
                if dev.get("max_input_channels", 0) > 0:
                    api_info = sd.query_hostapis(dev.get("hostapi", 0))
                    devices.append(
                        AudioDevice(
                            index=idx,
                            name=dev.get("name", f"Device {idx}"),
                            host_api=api_info.get("name", "Unknown API"),
                            max_input_channels=dev.get("max_input_channels", 0),
                            max_output_channels=dev.get("max_output_channels", 0),
                            default_samplerate=dev.get("default_samplerate", 16000.0),
                            is_default_input=(idx == default_input_idx),
                        )
                    )
        except Exception as exc:
            logger.error("Error querying input audio devices: %s", exc)

        return devices

    def set_device_index(self, index: Optional[int]) -> None:
        """Set active input device index and restart stream if active."""
        if self._device_index == index:
            return
        self._device_index = index
        logger.info("Setting audio input device index to: %s", index)
        self.pipeline.handle_device_change()
        if self._is_capturing:
            self.stop()
            self.start()

    def start(self) -> bool:
        """Initialize and start real-time microphone capture stream."""
        if self._is_capturing:
            return True

        if sd is None:
            logger.error("sounddevice library not available; cannot start audio capture.")
            return False

        try:
            target_device = self._device_index
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                device=target_device,
                channels=self.channels,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
            self._is_capturing = True
            logger.info(
                "Audio capture started on device %s at %d Hz.",
                target_device if target_device is not None else "default",
                self.sample_rate,
            )
            return True
        except Exception as exc:
            logger.error("Failed to start audio input stream: %s", exc)
            self._is_capturing = False
            self._stream = None
            return False

    def stop(self) -> None:
        """Stop microphone capture and release hardware resources."""
        self._is_capturing = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:
                logger.warning("Error closing audio input stream: %s", exc)
            finally:
                self._stream = None
        self._current_audio_level = 0.0
        logger.info("Audio capture stopped.")

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """Internal callback executed in sounddevice's high-priority audio thread."""
        if status:
            logger.warning("Audio input stream status: %s", status)

        if not self._is_capturing:
            return

        # Handle mute: pass silence if muted
        if self._is_muted:
            chunk = np.zeros_like(indata[:, 0], dtype=np.float32)
            level = 0.0
            confidence = 0.0
        else:
            raw_chunk = indata[:, 0].copy()
            # Process through unified AudioProcessingPipeline
            # (DC removal, rumble HPF, 50/60Hz notch, AEC, noise suppression, speech enhancement, VAD)
            processed: ProcessedAudioFrame = self.pipeline.process_frame(raw_chunk)
            chunk = processed.cleaned_frame
            confidence = processed.speech_confidence

            # Calculate RMS audio level normalized between 0.0 and 1.0
            rms = float(np.sqrt(np.mean(chunk**2)))
            level = min(1.0, rms * 5.0)  # Scale up for responsive visual VU meter

        self._current_audio_level = level
        self._current_speech_confidence = confidence

        with self._buffer_lock:
            self._ring_buffer.append(chunk)

        # Dispatch to callbacks
        for cb in list(self._chunk_callbacks):
            try:
                cb(chunk)
            except Exception as e:
                logger.error("Error in audio chunk callback: %s", e)

        # Dispatch to level callbacks
        for lvl_cb in list(self._level_callbacks):
            try:
                lvl_cb(level)
            except Exception as e:
                logger.error("Error in audio level callback: %s", e)

        for conf_cb in list(self._confidence_callbacks):
            try:
                conf_cb(confidence)
            except Exception as e:
                logger.error("Error in audio confidence callback: %s", e)

    def get_buffered_audio(self, duration_seconds: float) -> np.ndarray:
        """Retrieve recent audio up to `duration_seconds` as a single 1D numpy array."""
        frames_needed = int((self.sample_rate * duration_seconds) / self.block_size)
        with self._buffer_lock:
            total_available = len(self._ring_buffer)
            frames_to_take = min(frames_needed, total_available)
            if frames_to_take == 0:
                return np.zeros(0, dtype=np.float32)

            chunks = list(self._ring_buffer)[-frames_to_take:]

        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    def clear_buffer(self) -> None:
        """Clear circular audio buffer."""
        with self._buffer_lock:
            self._ring_buffer.clear()
