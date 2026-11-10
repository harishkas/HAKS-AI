"""Audio Streamer for HAKS Gemini Live Integration.

Manages formatted, ordered transmission of preprocessed microphone audio frames:
- Formats 16kHz float32 audio to 16-bit linear PCM mono bytes.
- Supports pre-roll audio frame injection on speech turn initiation.
- Bounded asynchronous queue preventing memory buildup during network jitter.
- Thread-safe non-blocking audio transfer.
"""

from collections import deque
import logging
from queue import Empty, Full, Queue
import threading
from typing import Callable, Optional
import numpy as np

logger = logging.getLogger("haks.voice.gemini_live.audio_streamer")


class GeminiAudioStreamer:
    """Buffers, formats, and transmits clean 16kHz audio frames to Gemini Live."""

    def __init__(
        self,
        sample_rate: int = 16000,
        max_queue_frames: int = 300,  # ~9.6 seconds buffer to absorb network jitter
    ) -> None:
        self.sample_rate = sample_rate
        self.max_queue_frames = max_queue_frames
        self.dropped_frames: int = 0

        self._queue: Queue[bytes] = Queue(maxsize=max_queue_frames)
        self._is_active: bool = False
        self._lock = threading.Lock()
        self._send_callback: Optional[Callable[[bytes], None]] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @property
    def is_active(self) -> bool:
        return self._is_active

    def set_send_callback(self, callback: Optional[Callable[[bytes], None]]) -> None:
        """Register the transmission callback (e.g. WebSocket or API stream)."""
        self._send_callback = callback

    def start(self) -> None:
        """Start audio streaming worker."""
        if self._is_active:
            return
        self._is_active = True
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._stream_worker,
            name="gemini_audio_streamer",
            daemon=True,
        )
        self._worker_thread.start()
        logger.info("Gemini audio streamer started.")

    def stop(self) -> None:
        """Halt streaming and drain queue."""
        self._is_active = False
        self._stop_event.set()
        self.clear()
        if self._worker_thread and self._worker_thread.is_alive() and threading.current_thread() != self._worker_thread:
            self._worker_thread.join(timeout=0.5)
        logger.info("Gemini audio streamer stopped.")

    def clear(self) -> None:
        """Flush queued audio frames."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Empty:
                break

    def inject_preroll(self, preroll_frames: list[np.ndarray]) -> None:
        """Inject preserved pre-roll audio frames ahead of real-time stream.

        Args:
            preroll_frames: Ordered sequence of 16kHz audio chunks.
        """
        for frame in preroll_frames:
            self.send_frame(frame)

    def send_frame(self, frame: np.ndarray) -> bool:
        """Format 16kHz float32 frame to int16 PCM bytes and enqueue for transmission.

        Args:
            frame: 1D numpy array of 16kHz float32 audio.
        """
        if not self._is_active or len(frame) == 0:
            return False

        # Convert float32 [-1.0, 1.0] to 16-bit linear PCM little-endian
        clipped = np.clip(frame, -1.0, 1.0)
        pcm_bytes = (clipped * 32767.0).astype(np.int16).tobytes()

        try:
            self._queue.put_nowait(pcm_bytes)
            return True
        except Full:
            logger.debug("Audio streamer queue full; dropping oldest frame.")
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(pcm_bytes)
                return True
            except (Empty, Full):
                return False

    def _stream_worker(self) -> None:
        """Background thread executing formatted delivery."""
        while not self._stop_event.is_set():
            try:
                pcm_data = self._queue.get(timeout=0.1)
            except Empty:
                continue

            if self._send_callback:
                try:
                    self._send_callback(pcm_data)
                except Exception as exc:
                    logger.error("Error transmitting audio chunk to Gemini Live: %s", exc)

