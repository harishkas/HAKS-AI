"""HAKS AI Engine Package.

Provides a unified interface for Local AI (Ollama) and Cloud AI providers,
supporting asynchronous inference, streaming, health monitoring, and credential security.
"""

from ai.models import (
    AIMessage,
    AIResponse,
    AIStreamChunk,
    AIEngineConfig,
    AIEngineHealth,
    EngineCapabilities,
    EngineType,
)
from ai.interface import AIEngineInterface
from ai.local_engine import LocalAIEngine
from ai.cloud_engine import CloudAIEngine
from ai.ollama_client import OllamaClient
from ai.manager import AIEngineManager, get_ai_manager

__all__ = [
    "AIMessage",
    "AIResponse",
    "AIStreamChunk",
    "AIEngineConfig",
    "AIEngineHealth",
    "EngineCapabilities",
    "EngineType",
    "AIEngineInterface",
    "LocalAIEngine",
    "CloudAIEngine",
    "OllamaClient",
    "AIEngineManager",
    "get_ai_manager",
]
