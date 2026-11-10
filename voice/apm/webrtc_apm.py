"""WebRTC Audio Processing (APM) for HAKS Microphone Preprocessing.

Integrates real-time WebRTC Audio Processing Module (APM) capabilities:
- Acoustic Echo Cancellation (AEC): Eliminates speaker output loopback into microphone.
- Noise Suppression (NS): Filters steady background noise (fans, hums, traffic).
- Automatic Gain Control (AGC): Dynamically stabilizes speech amplitude when enabled.
- Synchronized speaker playback-reference tracking for true AEC.
- Thread-safe buffer management and device change handling.

Implementation details:
Uses `pywebrtc_audio` (WebRTC APM C++ bindings for Windows / x86_64).
If `pywebrtc_audio` is unavailable, gracefully falls back to a software pass-through
pipeline while logging the operational status.
"""

from collections import deque
from dataclasses import dataclass
import logging
import threading
import time
from typing import Optional
import numpy as np

try:
    import pywebrtc_audio
    PYWEBRTC_AVAILABLE = True
except ImportError:
    pywebrtc_audio = None
    PYWEBRTC_AVAILABLE = False

logger = logging.getLogger("haks.voice.apm")


def estimate_hardware_delay_ms(
    input_device_index: Optional[int] = None,
    output_device_index: Optional[int] = None,
    fallback_ms: int = 40,
) -> int:
    """Calculate measured hardware roundtrip acoustic latency using sounddevice PortAudio query."""
    try:
        import sounddevice as sd
        in_dev = sd.query_devices(input_device_index, kind="input") if input_device_index is not None else sd.query_devices(kind="input")
        out_dev = sd.query_devices(output_device_index, kind="output") if output_device_index is not None else sd.query_devices(kind="output")
        in_lat = float(in_dev.get("default_low_input_latency", 0.02) or 0.02)
        out_lat = float(out_dev.get("default_low_output_latency", 0.02) or 0.02)
        measured_ms = int(round((in_lat + out_lat) * 1000.0))
        return max(20, min(300, measured_ms))
    except Exception:
        return fallback_ms


@dataclass
class APMStatus:
    """Status telemetry for audio preprocessing."""
    is_available: bool
    aec_enabled: bool
    ns_enabled: bool
    agc_enabled: bool
    speech_probability: float = 0.0
    gain_db: float = 0.0
    last_process_latency_ms: float = 0.0
    frames_processed: int = 0
    echo_cancelled_frames: int = 0


class SpeakerReferenceTracker:
    """Thread-safe circular buffer tracking audio sent to speakers for AEC reference.

    Speakers play audio into the physical room where the microphone also sits.
    For WebRTC AEC to eliminate this acoustic leak, it requires the exact speaker
    render audio ('far-end' signal) aligned in time with the microphone capture ('near-end' signal).
    """

    def __init__(self, sample_rate: int = 16000, max_buffer_seconds: float = 2.0) -> None:
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_buffer_seconds)
        self._lock = threading.Lock()
        self._buffer: deque[float] = deque(maxlen=self.max_samples)
        self._last_push_time: float = 0.0
        self._is_active: bool = False

    @property
    def is_active(self) -> bool:
        """True if speaker audio was pushed within the last 500 ms."""
        with self._lock:
            return self._is_active and (time.time() - self._last_push_time < 0.5)

    def push_samples(self, samples: np.ndarray, source_sample_rate: int = 16000) -> None:
        """Enqueue speaker playback audio samples.

        Args:
            samples: 1D numpy array of float32 samples.
            source_sample_rate: Native sample rate of the audio (resampled if not 16kHz).
        """
        if len(samples) == 0:
            return

        # Flatten or downmix stereo to mono if necessary
        if samples.ndim > 1:
            samples = np.mean(samples, axis=1)

        # Normalize float32 range [-1.0, 1.0]
        if samples.dtype == np.int16:
            samples = samples.astype(np.float32) / 32768.0
        else:
            samples = samples.astype(np.float32)

        # Rational polyphase downsampling (24kHz to 16kHz) with anti-aliasing FIR filter
        if source_sample_rate == 24000 and self.sample_rate == 16000 and len(samples) > 24:
            num_taps = 24
            fc = 0.625
            n = np.arange(-num_taps // 2, num_taps // 2)
            h = np.sinc(n * fc) * np.blackman(num_taps)
            h /= np.sum(h)
            filtered = np.convolve(samples, h, mode="same")
            target_len = int(round(len(samples) * 2.0 / 3.0))
            if target_len > 0:
                indices = np.linspace(0, len(samples) - 1, target_len)
                samples = np.interp(indices, np.arange(len(samples)), filtered).astype(np.float32)
        elif source_sample_rate != self.sample_rate and len(samples) > 1:
            target_len = int(len(samples) * self.sample_rate / source_sample_rate)
            if target_len > 0:
                indices = np.linspace(0, len(samples) - 1, target_len)
                samples = np.interp(indices, np.arange(len(samples)), samples).astype(np.float32)

        with self._lock:
            self._buffer.extend(samples.tolist())
            self._last_push_time = time.time()
            self._is_active = True

    def get_reference_frame(self, num_samples: int = 512, delay_ms: int = 40) -> np.ndarray:
        """Extract a synchronized reference frame aligned for AEC processing.

        Args:
            num_samples: Number of samples needed (typically 512 for a 32ms frame).
            delay_ms: Estimated hardware acoustic loopback latency between speaker write
                      and microphone arrival.

        Returns:
            np.ndarray: Float32 1D array of length `num_samples`. If no audio was playing,
                        returns zeros (clean silence).
        """
        with self._lock:
            total_available = len(self._buffer)
            delay_samples = int((delay_ms / 1000.0) * self.sample_rate)

            # If buffer doesn't have enough history or playback has stopped
            if total_available < num_samples or (time.time() - self._last_push_time > 0.6):
                self._is_active = False
                return np.zeros(num_samples, dtype=np.float32)

            # Look back with delay offset
            start_idx = max(0, total_available - num_samples - delay_samples)
            end_idx = start_idx + num_samples

            # Slice from deque
            frame_list = [self._buffer[i] for i in range(start_idx, min(end_idx, total_available))]

            # Pad with silence if short
            if len(frame_list) < num_samples:
                frame_list.extend([0.0] * (num_samples - len(frame_list)))

            return np.array(frame_list, dtype=np.float32)

    def clear(self) -> None:
        """Flush reference buffer."""
        with self._lock:
            self._buffer.clear()
            self._is_active = False


class WebRTCAPM:
    """Wraps WebRTC Audio Processing Module (APM) for 16kHz mono microphone streams."""

    def __init__(
        self,
        sample_rate: int = 16000,
        enable_aec: bool = True,
        enable_ns: bool = True,
        enable_agc: bool = False,
        ns_level: int = 2,
        stream_delay_ms: int = 40,
    ) -> None:
        self.sample_rate = sample_rate
        self.enable_aec = enable_aec
        self.enable_ns = enable_ns
        self.enable_agc = enable_agc
        self.ns_level = max(0, min(3, ns_level))
        if stream_delay_ms == 40:
            self.stream_delay_ms = estimate_hardware_delay_ms(fallback_ms=40)
        else:
            self.stream_delay_ms = stream_delay_ms

        self.reference_tracker = SpeakerReferenceTracker(sample_rate=sample_rate)
        self._processor: Optional[Any] = None
        self._lock = threading.Lock()

        # Telemetry metrics
        self._frames_processed: int = 0
        self._echo_cancelled_frames: int = 0
        self._last_process_latency_ms: float = 0.0
        self._speech_probability: float = 0.0

        self._init_processor()

    def _init_processor(self) -> None:
        """Instantiate the underlying WebRTC AudioProcessor."""
        if not PYWEBRTC_AVAILABLE:
            logger.warning(
                "pywebrtc_audio is not installed or available on this system. "
                "WebRTC APM will operate in software pass-through mode."
            )
            self._processor = None
            return

        try:
            self._processor = pywebrtc_audio.AudioProcessor(
                sample_rate=self.sample_rate,
                num_channels=1,
                echo_cancellation=self.enable_aec,
                noise_suppression=self.enable_ns,
                high_pass_filter=True,
                auto_gain_control=self.enable_agc,
                ns_level=self.ns_level,
                stream_delay_ms=self.stream_delay_ms,
            )
            logger.info(
                "WebRTC APM initialized: AEC=%s, NS=%s (level=%d), AGC=%s, delay=%dms",
                self.enable_aec,
                self.enable_ns,
                self.ns_level,
                self.enable_agc,
                self.stream_delay_ms,
            )
        except Exception as exc:
            logger.error("Failed to initialize pywebrtc_audio AudioProcessor: %s", exc)
            self._processor = None

    @property
    def is_available(self) -> bool:
        """True if the underlying WebRTC C++ processor is successfully loaded."""
        return self._processor is not None

    @property
    def speech_probability(self) -> float:
        """Estimated speech probability from APM noise suppressor."""
        return self._speech_probability

    def get_status(self) -> APMStatus:
        """Return snapshot of APM telemetry."""
        with self._lock:
            gain = 0.0
            if self.enable_agc and self._processor:
                try:
                    gain = float(self._processor.gain_db)
                except Exception:
                    pass
            return APMStatus(
                is_available=self.is_available,
                aec_enabled=self.enable_aec,
                ns_enabled=self.enable_ns,
                agc_enabled=self.enable_agc,
                speech_probability=self._speech_probability,
                gain_db=gain,
                last_process_latency_ms=self._last_process_latency_ms,
                frames_processed=self._frames_processed,
                echo_cancelled_frames=self._echo_cancelled_frames,
            )

    def process_frame(
        self,
        near_frame: np.ndarray,
        far_frame: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Process a microphone frame through AEC, NS, and AGC.

        Args:
            near_frame: 1D numpy array of 16kHz float32 or int16 samples from microphone.
            far_frame: Optional explicit reference speaker frame. If None, fetched
                       automatically from `reference_tracker`.

        Returns:
            np.ndarray: Cleaned audio frame with identical dtype and length.
        """
        if len(near_frame) == 0:
            return near_frame

        t0 = time.perf_counter()

        # If APM is not available, pass through
        if self._processor is None:
            self._last_process_latency_ms = (time.perf_counter() - t0) * 1000.0
            return near_frame

        with self._lock:
            # Obtain speaker reference frame for AEC
            if far_frame is None:
                if self.enable_aec:
                    far_frame = self.reference_tracker.get_reference_frame(
                        num_samples=len(near_frame),
                        delay_ms=self.stream_delay_ms,
                    )
                else:
                    far_frame = np.zeros_like(near_frame)

            # WebRTC APM requires a matching far-end array if AEC is enabled
            if self.enable_aec and far_frame is None:
                far_frame = np.zeros_like(near_frame)

            try:
                # Ensure input types match
                is_float = near_frame.dtype == np.float32 or np.issubdtype(near_frame.dtype, np.floating)
                if is_float:
                    near_in = near_frame.astype(np.float32)
                    far_in = far_frame.astype(np.float32) if far_frame is not None else np.zeros_like(near_in)
                else:
                    near_in = near_frame.astype(np.int16)
                    far_in = far_frame.astype(np.int16) if far_frame is not None else np.zeros_like(near_in)

                # Execute WebRTC C++ APM process
                clean_frame = self._processor.process(near_in, far_in)

                # Extract speech probability metric if exposed
                if hasattr(self._processor, "speech_probability"):
                    try:
                        self._speech_probability = float(self._processor.speech_probability)
                    except Exception:
                        pass

                self._frames_processed += 1
                if self.reference_tracker.is_active:
                    self._echo_cancelled_frames += 1

                self._last_process_latency_ms = (time.perf_counter() - t0) * 1000.0
                return clean_frame

            except Exception as exc:
                logger.debug("Error during WebRTC APM process_frame: %s", exc)
                self._last_process_latency_ms = (time.perf_counter() - t0) * 1000.0
                return near_frame

    def reset(self) -> None:
        """Reset internal APM state and clear reference buffer (e.g. on device change)."""
        with self._lock:
            if self._processor and hasattr(self._processor, "reset"):
                try:
                    self._processor.reset()
                except Exception as exc:
                    logger.debug("Error resetting pywebrtc AudioProcessor: %s", exc)
            self.reference_tracker.clear()
            self._speech_probability = 0.0
            logger.info("WebRTC APM state reset completed.")

    def on_device_changed(
        self,
        input_device_index: Optional[int] = None,
        output_device_index: Optional[int] = None,
    ) -> None:
        """Handle audio input/output device switch and recalibrate hardware delay."""
        measured_delay = estimate_hardware_delay_ms(input_device_index, output_device_index, fallback_ms=40)
        self.stream_delay_ms = measured_delay
        logger.info(
            "Audio device changed; recalibrated hardware delay to %d ms and resetting APM filters.",
            measured_delay,
        )
        self.reset()
