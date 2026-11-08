"""Base abstract class for provider-independent cloud AI adapters."""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

import httpx

from ai.models import (
    AIMessage,
    AIResponse,
    AIStreamChunk,
    AIEngineHealth,
    EngineCapabilities,
)


class CloudProviderAdapter(ABC):
    """Abstract interface defining required provider operations."""

    def __init__(self, transport: Optional[httpx.AsyncBaseTransport] = None) -> None:
        self.transport = transport

    def _get_client(self, timeout: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=timeout, transport=self.transport)

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Machine identifier (e.g. 'gemini', 'openai', 'anthropic')."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable provider label."""
        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default recommended model."""
        pass

    @abstractmethod
    def get_supported_models(self) -> list[str]:
        """List of standard models supported by this provider."""
        pass

    @abstractmethod
    def get_capabilities(self) -> EngineCapabilities:
        """Return engine capabilities for this provider."""
        pass

    @abstractmethod
    async def check_health(self, api_key: Optional[str], model: str) -> AIEngineHealth:
        """Validate API key and connection health."""
        pass

    @abstractmethod
    async def generate(
        self,
        api_key: str,
        model: str,
        messages: list[AIMessage],
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        timeout: float = 60.0,
    ) -> AIResponse:
        """Asynchronously generate complete response."""
        pass

    @abstractmethod
    async def stream(
        self,
        api_key: str,
        model: str,
        messages: list[AIMessage],
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        timeout: float = 60.0,
    ) -> AsyncIterator[AIStreamChunk]:
        """Asynchronously stream tokens."""
        pass
