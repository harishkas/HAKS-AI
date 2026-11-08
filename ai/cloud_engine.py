"""Provider-Independent Cloud AI Engine for HAKS.

Coordinates interchangeable cloud adapters (Gemini, OpenAI, Anthropic),
secure OS credential retrieval, asynchronous execution, and privacy safeguards.
"""

from typing import AsyncIterator, Optional

from ai.credentials import CredentialManager, get_credential_manager
from ai.exceptions import (
    AIEngineError,
    ProviderNotConfiguredError,
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
from ai.providers.anthropic import AnthropicProviderAdapter
from ai.providers.base import CloudProviderAdapter
from ai.providers.gemini import GeminiProviderAdapter
from ai.providers.openai_compatible import OpenAICompatibleAdapter
from core.logger import get_logger

logger = get_logger("ai.cloud_engine")


class CloudAIEngine(AIEngineInterface):
    """Unified cloud engine supporting multiple upstream providers."""

    def __init__(
        self,
        config: Optional[AIEngineConfig] = None,
        credentials: Optional[CredentialManager] = None,
    ) -> None:
        self.credentials = credentials or get_credential_manager()

        # Registered adapters
        self._providers: dict[str, CloudProviderAdapter] = {
            "gemini": GeminiProviderAdapter(),
            "openai": OpenAICompatibleAdapter(
                provider_id="openai",
                display_name="OpenAI",
                base_url="https://api.openai.com/v1",
                default_model="gpt-4o-mini",
                supported_models=["gpt-4o-mini", "gpt-4o", "o3-mini", "gpt-4-turbo"],
            ),
            "groq": OpenAICompatibleAdapter(
                provider_id="groq",
                display_name="Groq Cloud",
                base_url="https://api.groq.com/openai/v1",
                default_model="llama-3.1-8b-instant",
                supported_models=["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"],
            ),
            "openrouter": OpenAICompatibleAdapter(
                provider_id="openrouter",
                display_name="OpenRouter",
                base_url="https://openrouter.ai/api/v1",
                default_model="google/gemini-2.0-flash-001",
                supported_models=[
                    "google/gemini-2.0-flash-001",
                    "meta-llama/llama-3.3-70b-instruct",
                    "anthropic/claude-3.5-haiku",
                    "openai/gpt-4o-mini",
                ],
            ),
            "anthropic": AnthropicProviderAdapter(),
            "custom": OpenAICompatibleAdapter(
                provider_id="custom",
                display_name="Custom OpenAI Endpoint",
                base_url="http://localhost:8000/v1",
                default_model="default",
                supported_models=["default"],
            ),
        }

        # Initialize config
        self.config = config or AIEngineConfig(
            engine_type=EngineType.CLOUD,
            provider="gemini",
            model="gemini-2.5-flash",
            timeout_seconds=60.0,
        )

        # Set default model for active provider if needed
        active_adapter = self.get_active_adapter()
        if active_adapter and self.config.model == "llama3.2:1b":
            self.config.model = active_adapter.default_model

    @property
    def engine_name(self) -> str:
        adapter = self.get_active_adapter()
        name = adapter.display_name if adapter else "Cloud AI"
        return f"Cloud AI ({name})"

    @property
    def provider_name(self) -> str:
        adapter = self.get_active_adapter()
        return adapter.display_name if adapter else "Cloud AI"

    @property
    def active_provider_id(self) -> str:
        return self.config.provider.lower().strip()

    @property
    def current_model(self) -> str:
        return self.config.model

    def get_active_adapter(self) -> Optional[CloudProviderAdapter]:
        """Return the active provider adapter instance."""
        return self._providers.get(self.active_provider_id)

    def set_provider(self, provider_id: str, model: Optional[str] = None) -> None:
        """Switch active cloud provider and optionally set model."""
        norm_id = provider_id.lower().strip()
        if norm_id not in self._providers:
            raise AIEngineError(f"Unsupported cloud provider: '{provider_id}'", engine="cloud")

        self.config.provider = norm_id
        adapter = self._providers[norm_id]
        self.config.model = model or adapter.default_model
        logger.info("Cloud provider switched to: %s (model: %s)", norm_id, self.config.model)

    def set_model(self, model: str) -> None:
        """Update active cloud model."""
        self.config.model = model
        logger.info("Cloud model set to: %s", model)

    def set_provider_base_url(self, provider_id: str, base_url: str) -> None:
        """Update base URL for custom or configurable cloud provider."""
        norm_id = provider_id.lower().strip()
        adapter = self._providers.get(norm_id)
        if adapter and hasattr(adapter, "set_base_url"):
            adapter.set_base_url(base_url)
            logger.info("Updated base URL for '%s' to: %s", norm_id, base_url)

    def get_supported_providers(self) -> list[dict[str, str]]:
        """List all supported cloud providers with status."""
        result = []
        for pid, adapter in self._providers.items():
            has_key = self.credentials.has_api_key(pid)
            result.append({
                "id": pid,
                "name": adapter.display_name,
                "has_key": has_key,
                "default_model": adapter.default_model,
            })
        return result

    def _require_api_key(self) -> str:
        """Fetch API key or raise structured exception if missing."""
        key = self.credentials.get_api_key(self.active_provider_id)
        if not key or not key.strip():
            raise ProviderNotConfiguredError(
                f"Cloud AI provider '{self.active_provider_id.upper()}' is not configured with an API key. "
                f"Please add your API key in Settings -> Cloud AI or via Windows Credential Manager.",
                engine="cloud",
                details={"provider": self.active_provider_id},
            )
        return key

    async def check_health(self) -> AIEngineHealth:
        """Check cloud adapter reachability and API key status."""
        adapter = self.get_active_adapter()
        if not adapter:
            return AIEngineHealth(
                is_healthy=False,
                engine_name=self.engine_name,
                status_message=f"Provider '{self.active_provider_id}' is not supported.",
                available_models=[],
            )

        api_key = self.credentials.get_api_key(self.active_provider_id)
        return await adapter.check_health(api_key, self.config.model)

    def get_capabilities(self) -> EngineCapabilities:
        """Return capabilities of the active provider adapter."""
        adapter = self.get_active_adapter()
        if adapter:
            return adapter.get_capabilities()
        return EngineCapabilities(streaming=True, is_local=False)

    async def generate(self, messages: list[AIMessage], **kwargs) -> AIResponse:
        """Generate cloud response using the active provider."""
        adapter = self.get_active_adapter()
        if not adapter:
            raise AIEngineError(f"No active adapter for provider '{self.active_provider_id}'", engine="cloud")

        api_key = self._require_api_key()
        model = kwargs.get("model", self.config.model)
        temp = kwargs.get("temperature", self.config.temperature)
        system_prompt = kwargs.get("system_prompt", self.config.system_prompt)
        timeout = kwargs.get("timeout", self.config.timeout_seconds)

        return await adapter.generate(
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=temp,
            system_prompt=system_prompt,
            timeout=timeout,
        )

    async def stream(self, messages: list[AIMessage], **kwargs) -> AsyncIterator[AIStreamChunk]:
        """Stream cloud response tokens using active provider."""
        adapter = self.get_active_adapter()
        if not adapter:
            raise AIEngineError(f"No active adapter for provider '{self.active_provider_id}'", engine="cloud")

        api_key = self._require_api_key()
        model = kwargs.get("model", self.config.model)
        temp = kwargs.get("temperature", self.config.temperature)
        system_prompt = kwargs.get("system_prompt", self.config.system_prompt)
        timeout = kwargs.get("timeout", self.config.timeout_seconds)

        async for chunk in adapter.stream(
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=temp,
            system_prompt=system_prompt,
            timeout=timeout,
        ):
            yield chunk
