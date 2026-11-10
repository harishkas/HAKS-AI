"""Real-time Streaming PCM Player for HAKS Gemini Live.

Plays incoming streaming PCM audio chunks (16kHz or 24kHz mono) with low latency.
Features:
- Instant thread-safe barge-in interruption and queue clearing without blocking audio threads.
- Robust jitter buffering: does not abort playback on transient network chunk delays.
- Response completion tied directly to Gemini `turnComplete` signals.
- Feeds every rendered speaker audio frame to `SpeakerReferenceTracker` for Acoustic Echo Cancellation.
- Hardware output device selection.
"""

from collections import deque
import logging
from queue import Empty, Queue
import threading
import time
from typing import Callable, Optional
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

from voice.apm.webrtc_apm import SpeakerReferenceTracker

logger = logging.getLogger("haks.voice.gemini_live.pcm_player")


class PCMPlayer:
    """Streams live incoming PCM audio chunks through sounddevice with instant barge-in halt."""

    def __init__(
        self,
        sample_rate: int = 24000,
        channels: int = 1,
        output_device_index: Optional[int] = None,
        reference_tracker: Optional[SpeakerReferenceTracker] = None,
        min_jitter_chunks: int = 2,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.output_device_index = output_device_index
        self.reference_tracker = reference_tracker
        self.min_jitter_chunks = min_jitter_chunks

        self._queue: Queue[np.ndarray] = Queue(maxsize=300)
        self._is_playing: bool = False
        self._is_turn_complete: bool = True
        self._last_chunk_time: float = 0.0
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._active_stream: Optional[Any] = None
        self._volume: float = 1.0

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

    def mark_turn_complete(self) -> None:
        """Signal from Gemini Live that the model has completed generating audio for this turn."""
        self._is_turn_complete = True
        logger.debug("PCMPlayer marked turn complete. Output stream will drain and finalize.")

    def set_output_device(self, device_index: Optional[int]) -> None:
        """Switch audio output device and halt active stream."""
        if self.output_device_index != device_index:
            self.stop()
            self.output_device_index = device_index
            logger.info("PCMPlayer output device set to %s", device_index)

    def enqueue_chunk(self, chunk: bytes | np.ndarray, source_sample_rate: Optional[int] = None) -> bool:
        """Enqueue incoming PCM chunk for immediate playback.

        Args:
            chunk: Raw PCM bytes (int16 16kHz/24kHz) or numpy array.
            source_sample_rate: Sample rate if different from player default.
        """
        sr = source_sample_rate or self.sample_rate

        # Convert bytes to numpy float32
        if isinstance(chunk, bytes):
            audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        elif isinstance(chunk, np.ndarray):
            if chunk.dtype == np.int16:
                audio = chunk.astype(np.float32) / 32768.0
            else:
                audio = chunk.astype(np.float32)
        else:
            return False

        if len(audio) == 0:
            return False

        # Resample to player sample rate if needed
        if sr != self.sample_rate and len(audio) > 1:
            target_len = int(len(audio) * self.sample_rate / sr)
            if target_len > 0:
                indices = np.linspace(0, len(audio) - 1, target_len)
                audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

        try:
            self._is_turn_complete = False
            self._last_chunk_time = time.time()
            self._queue.put_nowait(audio)
            self._ensure_worker_running()
            return True
        except Exception:
            logger.warning("PCMPlayer queue full; dropping frame.")
            return False

    def _ensure_worker_running(self) -> None:
        """Start background worker thread if not active."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            if not self._stop_event.is_set():
                return
            # Worker is terminating from a previous stop call; join briefly
            self._worker_thread.join(timeout=0.08)

        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._play_worker,
            name="pcm_player_worker",
            daemon=True,
        )
        self._worker_thread.start()

    def stop(self) -> None:
        """Instantly interrupt and halt playback, clear queue, and flush AEC reference non-blocking."""
        self._stop_event.set()
        self._is_turn_complete = True

        # Non-blocking stream abort (instantly halts hardware audio output)
        stream = self._active_stream
        if stream is not None:
            try:
                stream.abort()
            except Exception:
                pass

        # Drain queue immediately
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Empty:
                break

        if self.reference_tracker:
            try:
                self.reference_tracker.clear()
            except Exception:
                pass

        self._is_playing = False

    def _play_worker(self) -> None:
        """Streaming worker pulling chunks and writing to sounddevice."""
        if sd is None:
            logger.error("sounddevice library not available; cannot play PCM.")
            return

        self._is_playing = True
        interrupted = False

        if self.on_playback_started:
            try:
                self.on_playback_started()
            except Exception as e:
                logger.error("Error in on_playback_started: %s", e)

        stream = None
        try:
            # Jitter buffer cushion: wait briefly for initial chunks unless turn completed early
            wait_start = time.time()
            while (
                self._queue.qsize() < self.min_jitter_chunks
                and not self._is_turn_complete
                and not self._stop_event.is_set()
                and (time.time() - wait_start < 0.15)
            ):
                time.sleep(0.01)

            block_size = 512
            stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                device=self.output_device_index,
                dtype="float32",
                blocksize=block_size,
            )
            stream.start()
            self._active_stream = stream

            last_chunk_time = time.time()

            # Process queue with adaptive jitter-tolerant timeout
            while not self._stop_event.is_set():
                try:
                    chunk = self._queue.get(timeout=0.05)
                    last_chunk_time = time.time()
                except Empty:
                    # If queue is empty AND Gemini has signaled turnComplete, playback is done!
                    if self._is_turn_complete:
                        logger.debug("PCMPlayer finished playing all chunks for completed turn.")
                        break

                    # If turnComplete is not yet received, tolerate transient token-generation delays
                    # Only time out if idle for > 5.0 seconds
                    if time.time() - last_chunk_time > 5.0:
                        logger.warning("PCMPlayer idle timeout (5.0s) waiting for subsequent Gemini chunks.")
                        break
                    continue

                if self._stop_event.is_set():
                    interrupted = True
                    break

                # Apply volume scaling
                if self._volume < 0.99:
                    chunk = chunk * self._volume

                # Feed rendered speaker frame to SpeakerReferenceTracker for AEC
                if self.reference_tracker:
                    self.reference_tracker.push_samples(chunk, source_sample_rate=self.sample_rate)

                # Write out in block_size chunks
                idx = 0
                total = len(chunk)
                while idx < total:
                    if self._stop_event.is_set():
                        interrupted = True
                        break
                    sub_end = min(idx + block_size, total)
                    sub_chunk = chunk[idx:sub_end]
                    try:
                        stream.write(sub_chunk)
                    except Exception as wr_err:
                        # Aborted or closed
                        interrupted = True
                        break
                    idx = sub_end

                if interrupted:
                    break

        except Exception as exc:
            logger.error("PCM streaming playback error: %s", exc)
        finally:
            self._active_stream = None
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass

            self._is_playing = False
            if interrupted or self._stop_event.is_set():
                logger.info("PCM playback interrupted.")
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
