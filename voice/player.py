"""Thread-safe Audio Player with Instant Barge-in Interruption for HAKS.

Decodes MP3, WAV, and audio formats using PyAV (`av`) and streams through
`sounddevice.OutputStream` in responsive 40ms blocks with instant barge-in halt.
"""

import logging
from pathlib import Path
import threading
import time
from typing import Callable, Optional
import numpy as np

try:
    import av
    import sounddevice as sd
except ImportError:
    av = None
    sd = None

logger = logging.getLogger("haks.voice.player")


class AudioPlayerService:
    """Streams audio playback with volume scaling and instant thread-safe interruption."""

    def __init__(self, output_device_index: Optional[int] = None) -> None:
        self.output_device_index = output_device_index
        self._is_playing: bool = False
        self._stop_event = threading.Event()
        self._playback_thread: Optional[threading.Thread] = None
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

    def stop(self) -> None:
        """Instantly interrupt and halt active audio playback."""
        if self._is_playing:
            logger.info("Interrupting audio playback (barge-in signal)...")
            self._stop_event.set()
            if self._playback_thread and self._playback_thread.is_alive():
                self._playback_thread.join(timeout=1.0)
            self._is_playing = False

    def play_file(self, file_path: str | Path, block: bool = False) -> bool:
        """Play an audio file (MP3/WAV) asynchronously or blocking."""
        self.stop()  # Stop any active playback first

        path_obj = Path(file_path)
        if not path_obj.exists():
            logger.error("Audio playback file not found: %s", file_path)
            return False

        self._stop_event.clear()

        if block:
            self._play_worker(path_obj)
            return True
        else:
            self._playback_thread = threading.Thread(
                target=self._play_worker,
                args=(path_obj,),
                name="audio_player_worker",
                daemon=True,
            )
            self._playback_thread.start()
            return True

    def _play_worker(self, path: Path) -> None:
        """Worker thread executing streaming playback."""
        if av is None or sd is None:
            logger.error("av or sounddevice missing; cannot play audio.")
            return

        self._is_playing = True
        interrupted = False

        if self.on_playback_started:
            try:
                self.on_playback_started()
            except Exception as e:
                logger.error("Error in on_playback_started: %e", e)

        try:
            # Open and decode container with PyAV
            container = av.open(str(path))
            audio_stream = next((s for s in container.streams if s.type == "audio"), None)
            if audio_stream is None:
                logger.error("No audio stream found in container: %s", path)
                return

            sample_rate = audio_stream.rate or 24000
            channels = audio_stream.channels or 1

            # Configure high-responsiveness output stream (1024 frames blocksize ~40ms)
            block_size = 1024
            stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=channels,
                device=self.output_device_index,
                dtype="float32",
                blocksize=block_size,
            )
            stream.start()

            try:
                for frame in container.decode(audio_stream):
                    if self._stop_event.is_set():
                        interrupted = True
                        break

                    # Convert PyAV AudioFrame to numpy float32
                    pcm_array = frame.to_ndarray()

                    # Transpose if necessary: sounddevice expects (frames, channels)
                    if pcm_array.ndim == 2:
                        # PyAV can be (channels, frames)
                        if pcm_array.shape[0] == channels:
                            pcm_array = pcm_array.T

                    # Normalize if integer
                    if pcm_array.dtype == np.int16:
                        audio_floats = pcm_array.astype(np.float32) / 32768.0
                    else:
                        audio_floats = pcm_array.astype(np.float32)

                    # Apply volume scaling
                    if self._volume < 0.99:
                        audio_floats *= self._volume

                    # Write out block-by-block while checking stop event
                    idx = 0
                    total_samples = len(audio_floats)
                    while idx < total_samples:
                        if self._stop_event.is_set():
                            interrupted = True
                            break
                        chunk_end = min(idx + block_size, total_samples)
                        chunk = audio_floats[idx:chunk_end]
                        stream.write(chunk)
                        idx = chunk_end

                    if interrupted:
                        break
            finally:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
                container.close()

        except Exception as exc:
            logger.error("Audio playback error: %s", exc)
        finally:
            self._is_playing = False
            if interrupted:
                logger.info("Playback interrupted.")
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
