"""Local AI Engine implementation using Ollama for HAKS.

Implements the common AIEngineInterface, wrapping OllamaClient with hardware-aware
safeguards for low-RAM machines (Intel i5-7200U, 8 GB RAM, 2 GB GPU).
"""

import time
from typing import Any, AsyncIterator, Optional

from ai.exceptions import (
    AIEngineConnectionError,
    AIEngineError,
    ModelNotFoundError,
)
from ai.interface import AIEngineInterface
from ai.models import (
    AIEngineConfig,
    AIEngineHealth,
    AIMessage,
    AIResponse,
    AIStreamChunk,
    EngineCapabilities,
    EngineType,
)
from ai.ollama_client import OllamaClient
from core.logger import get_logger

logger = get_logger("ai.local_engine")

# Recommended lightweight models optimized for dual-core CPU and 8 GB RAM
RECOMMENDED_LIGHTWEIGHT_MODELS = [
    {
        "name": "llama3.2:1b",
        "description": "Llama 3.2 1B (Q4_K_M ~1.3 GB RAM) - Extremely fast CPU inference, ideal default",
        "recommended_min_ram_gb": 4,
    },
    {
        "name": "qwen2.5:1.5b",
        "description": "Qwen 2.5 1.5B (Q4_K_M ~1.6 GB RAM) - Strong multilingual & reasoning",
        "recommended_min_ram_gb": 4,
    },
    {
        "name": "llama3.2:3b",
        "description": "Llama 3.2 3B (Q4_K_M ~2.5 GB RAM) - Balanced intelligence for 8 GB RAM systems",
        "recommended_min_ram_gb": 8,
    },
]

# Warning threshold: models larger than ~3.5 GB file size will strain 8 GB RAM
MAX_SAFE_MODEL_BYTES = 3_800_000_000  # ~3.8 GB


class LocalAIEngine(AIEngineInterface):
    """Local inference engine connecting to Ollama with resource safeguards."""

    def __init__(
        self,
        config: Optional[AIEngineConfig] = None,
        client: Optional[OllamaClient] = None,
    ) -> None:
        self.config = config or AIEngineConfig(
            engine_type=EngineType.LOCAL,
            provider="ollama",
            model="llama3.2:1b",
            base_url="http://127.0.0.1:11434",
            timeout_seconds=60.0,
        )
        self.client = client or OllamaClient(
            base_url=self.config.base_url,
            timeout_seconds=self.config.timeout_seconds,
        )
        self._installed_models_cache: list[dict] = []
        self._last_models_fetch: float = 0.0

    @property
    def engine_name(self) -> str:
        return "Local AI (Ollama)"

    @property
    def current_model(self) -> str:
        return self.config.model

    def set_model(self, model_name: str) -> None:
        """Update active local model."""
        self.config.model = model_name
        logger.info("Local AI active model changed to: %s", model_name)

    async def get_installed_models(self, force_refresh: bool = False) -> list[dict]:
        """Retrieve installed models, caching for 10 seconds."""
        now = time.monotonic()
        if force_refresh or (now - self._last_models_fetch > 10.0) or not self._installed_models_cache:
            try:
                self._installed_models_cache = await self.client.list_models()
                self._last_models_fetch = now
            except Exception as exc:
                logger.debug("Failed refreshing model list: %s", exc)
                return []
        return self._installed_models_cache

    def check_hardware_compatibility(self, model_size_bytes: int) -> tuple[bool, str]:
        """Evaluate whether a model is safe to run on an 8 GB RAM system."""
        if model_size_bytes > MAX_SAFE_MODEL_BYTES:
            size_gb = model_size_bytes / (1024**3)
            return (
                False,
                f"Model size ({size_gb:.1f} GB) exceeds safe 8 GB system budget (<3.5 GB). "
                f"May cause OS paging and system slowdown.",
            )
        return True, "Safe for 8 GB RAM system."

    async def check_health(self) -> AIEngineHealth:
        """Probe Ollama daemon health, version, and installed model availability."""
        start_time = time.perf_counter()
        is_running = await self.client.is_server_running()

        if not is_running:
            is_installed = self.client.is_ollama_installed()
            msg = (
                "Ollama is installed but not running."
                if is_installed
                else "Ollama is not installed or not found in system PATH."
            )
            return AIEngineHealth(
                is_healthy=False,
                engine_name=self.engine_name,
                status_message=msg,
                available_models=[],
                latency_ms=None,
                details={
                    "installed": is_installed,
                    "running": False,
                    "base_url": self.config.base_url,
                },
            )

        # Ollama is running, query models and version
        version = await self.client.get_version()
        models = await self.get_installed_models(force_refresh=True)
        model_names = [m["name"] for m in models]
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        model_status = (
            f"Active: '{self.config.model}' ({len(model_names)} installed)"
            if model_names
            else "No models installed. Run 'ollama pull llama3.2:1b'"
        )

        return AIEngineHealth(
            is_healthy=True,
            engine_name=self.engine_name,
            status_message=f"Connected to Ollama v{version or 'unknown'}. {model_status}",
            available_models=model_names,
            latency_ms=latency_ms,
            details={
                "version": version,
                "base_url": self.config.base_url,
                "model_count": len(model_names),
                "active_model": self.config.model,
            },
        )

    def get_capabilities(self) -> EngineCapabilities:
        """Return engine capabilities."""
        ctx_tokens = getattr(self.config, "num_ctx", 2048)
        return EngineCapabilities(
            streaming=True,
            is_local=True,
            context_window_tokens=ctx_tokens,
            supports_system_prompt=True,
            supports_vision=False,
            supports_tools=False,
        )

    def update_hardware_limits(
        self,
        num_ctx: Optional[int] = None,
        num_predict: Optional[int] = None,
        temperature: Optional[float] = None,
        cpu_threads: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
    ) -> None:
        """Update configurable local hardware limits."""
        if num_ctx is not None:
            self.config.num_ctx = int(num_ctx)
        if num_predict is not None:
            self.config.num_predict = int(num_predict)
        if temperature is not None:
            self.config.temperature = float(temperature)
        if cpu_threads is not None:
            self.config.cpu_threads = int(cpu_threads)
        if repeat_penalty is not None:
            self.config.repeat_penalty = float(repeat_penalty)
        logger.info(
            "Updated local hardware limits: num_ctx=%d, num_predict=%d, temp=%.2f, threads=%d",
            self.config.num_ctx,
            self.config.num_predict,
            self.config.temperature,
            self.config.cpu_threads,
        )

    async def generate(self, messages: list[AIMessage], **kwargs) -> AIResponse:
        """Generate response via Ollama with hardware-aware token limits."""
        model = kwargs.get("model", self.config.model)
        temp = kwargs.get("temperature", self.config.temperature)
        system_prompt = kwargs.get("system_prompt", self.config.system_prompt)
        timeout = kwargs.get("timeout", self.config.timeout_seconds)
        num_ctx = kwargs.get("num_ctx", getattr(self.config, "num_ctx", 2048))
        num_predict = kwargs.get("num_predict", getattr(self.config, "num_predict", 128))
        cpu_threads = kwargs.get("cpu_threads", getattr(self.config, "cpu_threads", 2))
        repeat_penalty = kwargs.get("repeat_penalty", getattr(self.config, "repeat_penalty", 1.15))
        options = kwargs.get("options")

        return await self.client.chat(
            model=model,
            messages=messages,
            temperature=temp,
            system_prompt=system_prompt,
            timeout=timeout,
            num_ctx=num_ctx,
            num_predict=num_predict,
            cpu_threads=cpu_threads,
            repeat_penalty=repeat_penalty,
            options=options,
        )

    async def stream(self, messages: list[AIMessage], **kwargs) -> AsyncIterator[AIStreamChunk]:
        """Stream response tokens via Ollama with hardware-aware token limits."""
        model = kwargs.get("model", self.config.model)
        temp = kwargs.get("temperature", self.config.temperature)
        system_prompt = kwargs.get("system_prompt", self.config.system_prompt)
        timeout = kwargs.get("timeout", self.config.timeout_seconds)
        num_ctx = kwargs.get("num_ctx", getattr(self.config, "num_ctx", 2048))
        num_predict = kwargs.get("num_predict", getattr(self.config, "num_predict", 128))
        cpu_threads = kwargs.get("cpu_threads", getattr(self.config, "cpu_threads", 2))
        repeat_penalty = kwargs.get("repeat_penalty", getattr(self.config, "repeat_penalty", 1.15))
        options = kwargs.get("options")

        async for chunk in self.client.chat_stream(
            model=model,
            messages=messages,
            temperature=temp,
            system_prompt=system_prompt,
            timeout=timeout,
            num_ctx=num_ctx,
            num_predict=num_predict,
            cpu_threads=cpu_threads,
            repeat_penalty=repeat_penalty,
            options=options,
        ):
            yield chunk


    async def unload_active_model(self) -> bool:
        """Unload active model from memory."""
        return await self.client.unload_model(self.config.model)

    async def pull_model(self, model: str, timeout: float = 600.0) -> AsyncIterator[dict[str, Any]]:
        """Download and install a model via Ollama with progress reporting."""
        async for progress in self.client.pull_model(model, timeout=timeout):
            yield progress
        # Refresh model cache after completion
        await self.get_installed_models(force_refresh=True)

    async def delete_model(self, model: str) -> bool:
        """Delete an installed model to free disk space."""
        success = await self.client.delete_model(model)
        if success:
            await self.get_installed_models(force_refresh=True)
        return success

    async def show_model_info(self, model: str) -> dict[str, Any]:
        """Fetch model architecture, parameters, and quantization details."""
        return await self.client.show_model_info(model)
