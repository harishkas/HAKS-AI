"""Real-time Streaming PCM Audio Player for Gemini Live 24 kHz Output.

Plays linear PCM (24,000 Hz, 16-bit mono, little-endian) streamed over WebSockets
using sounddevice.RawOutputStream with sub-20ms instant barge-in interruption.
"""

import logging
import queue
import threading
import time
from typing import Callable, Optional
import numpy as np

from voice.echo_cancellation import AcousticEchoSuppressor

try:
    import sounddevice as sd
except ImportError:
    sd = None

logger = logging.getLogger("haks.voice.gemini_live.pcm_player")


class PCMStreamPlayer:
    """Thread-safe streaming audio player for 24 kHz 16-bit linear PCM."""

    def __init__(
        self,
        sample_rate: int = 24000,
        channels: int = 1,
        output_device_index: Optional[int] = None,
        echo_suppressor: Optional[AcousticEchoSuppressor] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.output_device_index = output_device_index
        self.echo_suppressor = echo_suppressor
        self._volume: float = 1.0

        self._queue: queue.Queue[Optional[bytes]] = queue.Queue()
        self._is_playing: bool = False
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Callbacks
        self.on_playback_started: Optional[Callable[[], None]] = None
        self.on_playback_finished: Optional[Callable[[], None]] = None
        self.on_playback_interrupted: Optional[Callable[[], None]] = None

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @property
    def volume(self) -> float:
        return self._volume

    @volume.setter
    def volume(self, val: float) -> None:
        self._volume = max(0.0, min(1.0, val))

    def write_chunk(self, pcm_bytes: bytes) -> None:
        """Feed a 24 kHz 16-bit PCM chunk to the playback stream."""
        if not pcm_bytes:
            return

        with self._lock:
            self._queue.put(pcm_bytes)
            if not self._is_playing or self._worker_thread is None or not self._worker_thread.is_alive():
                self._stop_event.clear()
                self._is_playing = True
                self._worker_thread = threading.Thread(
                    target=self._playback_worker,
                    name="pcm_stream_worker",
                    daemon=True,
                )
                self._worker_thread.start()

    def mark_turn_complete(self) -> None:
        """Signal that the current assistant turn has ended."""
        with self._lock:
            if self._is_playing:
                # Sentinel None signals end of current stream
                self._queue.put(None)

    def stop(self) -> None:
        """Instantly halt active playback and flush pending audio (barge-in)."""
        with self._lock:
            if not self._is_playing:
                return
            logger.info("PCMStreamPlayer: Instant interruption signal received (barge-in).")
            self._stop_event.set()

            # Drain queue immediately
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=0.3)

        self._is_playing = False
        if self.on_playback_interrupted:
            try:
                self.on_playback_interrupted()
            except Exception as e:
                logger.error("Error in on_playback_interrupted callback: %s", e)

    def _playback_worker(self) -> None:
        """Worker thread streaming PCM chunks to sounddevice."""
        if sd is None:
            logger.error("sounddevice is not installed; PCM streaming disabled.")
            self._is_playing = False
            return

        interrupted = False
        if self.on_playback_started:
            try:
                self.on_playback_started()
            except Exception as e:
                logger.error("Error in on_playback_started callback: %s", e)

        stream = None
        try:
            stream = sd.RawOutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                device=self.output_device_index,
                dtype="int16",
                blocksize=1024,
            )
            stream.start()

            while not self._stop_event.is_set():
                try:
                    chunk = self._queue.get(timeout=0.2)
                except queue.Empty:
                    # If queue empty and turn complete, finish playback
                    continue

                if chunk is None:
                    # End of turn sentinel received and buffer drained
                    if not interrupted:
                        time.sleep(0.06)  # Allow final hardware buffer drain
                    break

                if self._stop_event.is_set():
                    interrupted = True
                    break

                # Feed active chunk to echo suppressor reference buffer
                if self.echo_suppressor is not None:
                    self.echo_suppressor.feed_playback_chunk(chunk, self.sample_rate)

                # Apply volume scaling if needed
                if self._volume < 0.99:
                    arr = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
                    arr = (arr * self._volume).astype(np.int16)
                    processed_bytes = arr.tobytes()
                else:
                    processed_bytes = chunk

                # Write out in small chunks to allow responsive interruption
                block_bytes = 2048  # 1024 samples int16 = ~42ms
                offset = 0
                total_len = len(processed_bytes)

                while offset < total_len:
                    if self._stop_event.is_set():
                        interrupted = True
                        break
                    end = min(offset + block_bytes, total_len)
                    sub_chunk = processed_bytes[offset:end]
                    stream.write(sub_chunk)
                    offset = end

                if interrupted:
                    break

        except Exception as exc:
            logger.error("PCM playback worker error: %s", exc)
        finally:
            if stream is not None:
                try:
                    if interrupted:
                        stream.abort()
                    else:
                        stream.stop()
                    stream.close()
                except Exception:
                    pass

            self._is_playing = False

            if interrupted:
                logger.info("PCM playback stopped by interruption.")
                if self.on_playback_interrupted:
                    try:
                        self.on_playback_interrupted()
                    except Exception as e:
                        logger.error("Error in on_playback_interrupted: %s", e)
            else:
                if self.on_playback_finished:
                    try:
                        self.on_playback_finished()
                    except Exception as e:
                        logger.error("Error in on_playback_finished: %s", e)

