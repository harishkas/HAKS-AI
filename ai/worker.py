"""PySide6 Background Worker for Non-Blocking AI Streaming Inference.

Runs asyncio inference loops on a dedicated QThread so that the Qt main UI
loop stays responsive and unblocked on low-spec hardware (Intel Core i5-7200U).
"""

import asyncio
from typing import Optional
from PySide6.QtCore import QObject, QThread, Signal

from ai.exceptions import GenerationCancelledError
from ai.manager import AIEngineManager, get_ai_manager
from ai.models import AIMessage
from core.logger import get_logger

logger = get_logger("ai.worker")


class AIWorkerSignals(QObject):
    """Signals emitted by AIInferenceWorker."""
    started = Signal()
    chunk_received = Signal(str)  # delta text
    finished = Signal(str, dict)  # full_text, metadata dict
    error = Signal(str)  # error description
    cancelled = Signal()


class AIInferenceWorker(QThread):
    """Dedicated background thread for streaming AI responses."""

    def __init__(
        self,
        messages: list[AIMessage],
        engine_type: Optional[str] = None,
        manager: Optional[AIEngineManager] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(None)  # Decouple thread lifecycle from widget destruction
        self.messages = messages
        self.engine_type = engine_type
        self.manager = manager or get_ai_manager()
        self.signals = AIWorkerSignals()
        self._is_cancelled = False

    def cancel(self) -> None:
        """Request immediate cancellation of inference."""
        self._is_cancelled = True
        self.manager.cancel_generation()

    def run(self) -> None:
        """Run the streaming async generator inside a dedicated thread asyncio loop."""
        self.signals.started.emit()

        async def _execute_stream() -> None:
            accumulated_text = []
            final_usage = {}
            finish_reason = "stop"

            try:
                stream_gen = self.manager.stream(
                    messages=self.messages,
                    engine_type=self.engine_type,
                )

                async for chunk in stream_gen:
                    if self._is_cancelled:
                        break

                    if chunk.delta_text:
                        accumulated_text.append(chunk.delta_text)
                        self.signals.chunk_received.emit(chunk.delta_text)

                    if chunk.usage:
                        final_usage = chunk.usage.model_dump()
                    if chunk.finish_reason:
                        finish_reason = chunk.finish_reason

                full_text = "".join(accumulated_text)
                engine = self.manager.get_engine(self.engine_type)
                metadata = {
                    "engine": engine.engine_name,
                    "model": getattr(engine, "current_model", "unknown"),
                    "finish_reason": finish_reason,
                    "usage": final_usage,
                }
                self.signals.finished.emit(full_text, metadata)

            except GenerationCancelledError:
                try:
                    logger.info("Worker caught GenerationCancelledError.")
                except Exception:
                    pass
                self.signals.cancelled.emit()
            except Exception as exc:
                try:
                    logger.error("AI inference error in worker: %s", exc)
                except Exception:
                    pass
                self.signals.error.emit(str(exc))

        # Run within new asyncio event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_execute_stream())
        except Exception as exc:
            try:
                self.signals.error.emit(str(exc))
            except Exception:
                pass
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            finally:
                loop.close()


class ModelDownloadSignals(QObject):
    """Signals emitted by ModelDownloadWorker."""
    started = Signal(str)  # model_name
    progress = Signal(str, int, int, float)  # status_msg, completed_bytes, total_bytes, percent
    finished = Signal(str)  # model_name
    error = Signal(str)  # error description
    cancelled = Signal()


class ModelDownloadWorker(QThread):
    """Dedicated background thread for pulling local Ollama models with live progress."""

    def __init__(
        self,
        model_name: str,
        manager: Optional[AIEngineManager] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(None)  # Decouple from widget lifecycle
        self.model_name = model_name
        self.manager = manager or get_ai_manager()
        self.signals = ModelDownloadSignals()
        self._is_cancelled = False

    def cancel(self) -> None:
        """Request immediate cancellation of model download."""
        self._is_cancelled = True

    def run(self) -> None:
        """Run model pull inside dedicated thread event loop."""
        self.signals.started.emit(self.model_name)

        async def _download() -> None:
            try:
                local_engine = self.manager.local_engine
                async for chunk in local_engine.pull_model(self.model_name):
                    if self._is_cancelled:
                        self.signals.cancelled.emit()
                        return

                    status = chunk.get("status", "")
                    completed = chunk.get("completed", 0)
                    total = chunk.get("total", 0)
                    percent = chunk.get("percent", 0.0)

                    self.signals.progress.emit(status, completed, total, percent)

                if not self._is_cancelled:
                    self.signals.finished.emit(self.model_name)
            except Exception as exc:
                try:
                    logger.error("Model download error for '%s': %s", self.model_name, exc)
                except Exception:
                    pass
                self.signals.error.emit(str(exc))

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_download())
        except Exception as exc:
            try:
                self.signals.error.emit(str(exc))
            except Exception:
                pass
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            finally:
                loop.close()
