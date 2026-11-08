"""Common interface definition for all HAKS AI engines."""

from abc import ABC, abstractmethod
from typing import AsyncIterator

from ai.models import (
    AIMessage,
    AIResponse,
    AIStreamChunk,
    AIEngineHealth,
    EngineCapabilities,
)


class AIEngineInterface(ABC):
    """Abstract base class establishing the common async interface for HAKS AI engines."""

    @abstractmethod
    async def generate(self, messages: list[AIMessage], **kwargs) -> AIResponse:
        """Asynchronously generate a complete text completion."""
        pass

    @abstractmethod
    async def stream(self, messages: list[AIMessage], **kwargs) -> AsyncIterator[AIStreamChunk]:
        """Asynchronously stream response tokens as they are produced."""
        pass

    @abstractmethod
    async def check_health(self) -> AIEngineHealth:
        """Perform health and connectivity checks on the engine."""
        pass

    @abstractmethod
    def get_capabilities(self) -> EngineCapabilities:
        """Return declared engine capabilities."""
        pass

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Human-readable name of the engine."""
        pass
