"""Real-time Microphone Audio Streamer for Gemini Live.

Bridges 16 kHz audio capture frames from AudioCaptureService to GeminiLiveClient
via an asynchronous worker queue.
"""

import asyncio
import logging
from typing import Optional
import numpy as np

from voice.gemini_live.client import GeminiLiveClient

from collections import deque
import logging
from typing import Optional
import numpy as np

from voice.gemini_live.client import GeminiLiveClient

logger = logging.getLogger("haks.voice.gemini_live.audio_streamer")


class AudioStreamerBridge:
    """Asynchronously streams microphone PCM frames to an active Gemini Live session."""

    def __init__(
        self,
        client: GeminiLiveClient,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        self.client = client
        self.loop = loop
        self._is_streaming: bool = False
        self._playback_active: bool = False
        self._held_pre_roll: deque[bytes] = deque(maxlen=8)  # ~256ms of 32ms frames
        self._queue: Optional[asyncio.Queue[bytes]] = None
        self._stream_task: Optional[asyncio.Task] = None

    @property
    def is_streaming(self) -> bool:
        return self._is_streaming

    @property
    def is_playback_active(self) -> bool:
        return self._playback_active

    def set_playback_active(self, active: bool) -> None:
        """Inform streamer whether assistant speaker output is currently active."""
        self._playback_active = active
        if not active:
            self.flush_held_frames()

    def flush_held_frames(self) -> None:
        """Push held pre-roll frames into active transmission queue upon barge-in."""
        if self._queue is not None and self.loop is not None:
            def _drain():
                while self._held_pre_roll:
                    try:
                        chunk = self._held_pre_roll.popleft()
                        if self._queue:
                            self._queue.put_nowait(chunk)
                    except (IndexError, asyncio.QueueFull):
                        break
            if self.loop.is_running():
                self.loop.call_soon_threadsafe(_drain)

    def start_streaming(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Begin queueing and transmitting microphone audio to Gemini Live."""
        if self._is_streaming:
            return

        if loop:
            self.loop = loop
        elif self.loop is None:
            try:
                self.loop = asyncio.get_running_loop()
            except RuntimeError:
                logger.error("No active event loop found for AudioStreamerBridge.")
                return

        self._is_streaming = True
        self._queue = asyncio.Queue(maxsize=100)
        self._stream_task = self.loop.create_task(self._streaming_worker())
        logger.info("AudioStreamerBridge: Started streaming microphone frames.")

    def stop_streaming(self) -> None:
        """Stop transmitting audio to Gemini Live and drain the buffer."""
        if not self._is_streaming:
            return

        self._is_streaming = False
        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
            self._stream_task = None

        if self._queue:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except Exception:
                    break
        logger.info("AudioStreamerBridge: Stopped streaming microphone frames.")

    def feed_frame(self, float_frame: np.ndarray) -> None:
        """Feed a float32 audio capture frame (from AudioCaptureService) into the streamer."""
        if not self._is_streaming or self._queue is None or self.loop is None:
            return

        try:
            # Convert float32 [-1.0, 1.0] to 16-bit linear PCM little-endian
            clamped = np.clip(float_frame, -1.0, 1.0)
            int16_pcm = (clamped * 32767.0).astype(np.int16).tobytes()

            if self.loop.is_running():
                self.loop.call_soon_threadsafe(self._put_chunk, int16_pcm)
        except Exception as exc:
            logger.debug("Error feeding frame to audio streamer: %s", exc)

    def _put_chunk(self, chunk: bytes) -> None:
        if self._queue and self._is_streaming:
            if self._playback_active:
                # Hold frames in pre-roll buffer during playback to prevent speaker echo reaching Gemini
                self._held_pre_roll.append(chunk)
                return

            try:
                self._queue.put_nowait(chunk)
            except asyncio.QueueFull:
                # Discard oldest chunk to prevent unbounded buffer delay
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait(chunk)
                except Exception:
                    pass

    async def _streaming_worker(self) -> None:
        """Worker task transmitting audio chunks across the WebSocket."""
        try:
            while self._is_streaming and self._queue is not None:
                chunk = await self._queue.get()
                if not self._is_streaming:
                    break

                if self.client.is_connected():
                    await self.client.send_audio_chunk(chunk)
                else:
                    await asyncio.sleep(0.05)

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("Audio streamer worker failed: %s", exc)
        finally:
            self._is_streaming = False

