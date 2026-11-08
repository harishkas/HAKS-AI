"""Central AI Engine Manager for HAKS.

Coordinates Local (Ollama) and Cloud AI engines behind a consistent interface.
Provides lifecycle management, active engine switching, non-blocking asynchronous
generation, streaming, cancellation control, and health monitoring.
"""

import asyncio
from typing import AsyncIterator, Optional

from ai.cloud_engine import CloudAIEngine
from ai.exceptions import (
    AIEngineError,
    GenerationCancelledError,
)
from ai.interface import AIEngineInterface
from ai.local_engine import LocalAIEngine
from ai.models import (
    AIEngineConfig,
    AIEngineHealth,
    AIMessage,
    AIResponse,
    AIStreamChunk,
    EngineCapabilities,
    EngineType,
)
from config.settings import AppConfig, get_settings
from core.logger import get_logger

logger = get_logger("ai.manager")


class AIEngineManager:
    """Unified coordinator managing Local and Cloud AI engines."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        local_engine: Optional[LocalAIEngine] = None,
        cloud_engine: Optional[CloudAIEngine] = None,
    ) -> None:
        self.app_config = config or get_settings()

        # Instantiate local and cloud engines
        local_cfg = AIEngineConfig(
            engine_type=EngineType.LOCAL,
            provider="ollama",
            model=self.app_config.ai.local_model,
            base_url=self.app_config.ai.ollama_base_url,
            timeout_seconds=self.app_config.ai.request_timeout_seconds,
        )
        self.local_engine = local_engine or LocalAIEngine(local_cfg)

        cloud_cfg = AIEngineConfig(
            engine_type=EngineType.CLOUD,
            provider=self.app_config.ai.cloud_provider,
            model="gemini-2.5-flash",
            timeout_seconds=self.app_config.ai.request_timeout_seconds,
        )
        self.cloud_engine = cloud_engine or CloudAIEngine(cloud_cfg)

        # Active engine setting (default "local")
        self._active_engine_type = (
            EngineType.LOCAL
            if self.app_config.ai.default_engine == "local"
            else EngineType.CLOUD
        )

        # Active cancellation control
        self._current_cancel_event: Optional[asyncio.Event] = None
        self._is_generating: bool = False

        logger.info("AIEngineManager initialized. Default active engine: %s", self._active_engine_type.value)

    @property
    def active_engine_type(self) -> EngineType:
        return self._active_engine_type

    @property
    def is_generating(self) -> bool:
        return self._is_generating

    def set_active_engine(self, engine_type: str | EngineType) -> None:
        """Switch active engine between 'local' and 'cloud'."""
        if isinstance(engine_type, str):
            engine_type = EngineType(engine_type.lower().strip())
        self._active_engine_type = engine_type
        logger.info("Active engine switched to: %s", self._active_engine_type.value)

    def get_engine(self, engine_type: Optional[str | EngineType] = None) -> AIEngineInterface:
        """Retrieve specified engine instance, or the active engine if None."""
        if engine_type is None:
            engine_type = self._active_engine_type
        elif isinstance(engine_type, str):
            engine_type = EngineType(engine_type.lower().strip())

        if engine_type == EngineType.LOCAL:
            return self.local_engine
        elif engine_type == EngineType.CLOUD:
            return self.cloud_engine
        raise AIEngineError(f"Unknown engine type: {engine_type}")

    def get_active_engine(self) -> AIEngineInterface:
        """Return the currently selected active engine."""
        return self.get_engine(self._active_engine_type)

    async def check_all_health(self) -> dict[str, AIEngineHealth]:
        """Perform concurrent health check across both local and cloud engines."""
        local_health, cloud_health = await asyncio.gather(
            self.local_engine.check_health(),
            self.cloud_engine.check_health(),
            return_exceptions=False,
        )
        return {
            "local": local_health,
            "cloud": cloud_health,
        }

    def get_engine_health(self) -> dict[str, AIEngineHealth]:
        """Return engine health states synchronously with caching."""
        if hasattr(self, "_last_health_cache") and self._last_health_cache:
            return self._last_health_cache
        try:
            loop = asyncio.new_event_loop()
            res = loop.run_until_complete(self.check_all_health())
            loop.close()
            self._last_health_cache = res
            return res
        except Exception as exc:
            logger.warning("Could not probe health synchronously: %s", exc)
            return {
                "local": AIEngineHealth(is_healthy=True, engine_name="Local AI", status_message="Assuming available"),
                "cloud": AIEngineHealth(is_healthy=True, engine_name="Cloud AI", status_message="Assuming available"),
            }

    async def check_health(self, engine_type: Optional[str | EngineType] = None) -> AIEngineHealth:
        """Check health of the designated engine (or active engine)."""
        engine = self.get_engine(engine_type)
        return await engine.check_health()

    def get_capabilities(self, engine_type: Optional[str | EngineType] = None) -> EngineCapabilities:
        """Retrieve capabilities of the designated engine."""
        engine = self.get_engine(engine_type)
        return engine.get_capabilities()

    def cancel_generation(self) -> None:
        """Signal cancellation for the ongoing generation task."""
        if self._current_cancel_event and not self._current_cancel_event.is_set():
            logger.info("Cancelling ongoing AI generation request...")
            self._current_cancel_event.set()
        self._is_generating = False

    async def generate(
        self,
        messages: list[AIMessage],
        engine_type: Optional[str | EngineType] = None,
        **kwargs,
    ) -> AIResponse:
        """Generate response asynchronously through designated or active engine."""
        engine = self.get_engine(engine_type)
        self._is_generating = True
        cancel_event = asyncio.Event()
        self._current_cancel_event = cancel_event

        try:
            task = asyncio.create_task(engine.generate(messages, **kwargs))

            # Monitor cancel event alongside task
            while not task.done():
                if cancel_event.is_set():
                    task.cancel()
                    raise GenerationCancelledError("Inference generation was cancelled by user.", engine=engine.engine_name)
                await asyncio.sleep(0.05)

            return await task
        finally:
            self._is_generating = False
            self._current_cancel_event = None

    async def stream(
        self,
        messages: list[AIMessage],
        engine_type: Optional[str | EngineType] = None,
        **kwargs,
    ) -> AsyncIterator[AIStreamChunk]:
        """Stream response tokens asynchronously through designated or active engine."""
        engine = self.get_engine(engine_type)
        self._is_generating = True
        cancel_event = asyncio.Event()
        self._current_cancel_event = cancel_event

        try:
            stream_gen = engine.stream(messages, **kwargs)
            async for chunk in stream_gen:
                if cancel_event.is_set():
                    logger.info("Streaming aborted due to cancellation request.")
                    raise GenerationCancelledError(
                        "Streaming was stopped by user.",
                        engine=engine.engine_name,
                    )
                yield chunk
        finally:
            self._is_generating = False
            self._current_cancel_event = None

    async def shutdown(self) -> None:
        """Clean up active resources and unload local models if necessary."""
        self.cancel_generation()
        logger.info("AIEngineManager shutdown complete.")


_manager_instance: Optional[AIEngineManager] = None


def get_ai_manager(config: Optional[AppConfig] = None) -> AIEngineManager:
    """Return singleton AIEngineManager instance."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = AIEngineManager(config)
    return _manager_instance
