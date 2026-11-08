"""Unit tests for AI Engine models, interface conformity, and capabilities."""

import pytest
from ai.interface import AIEngineInterface
from ai.local_engine import LocalAIEngine
from ai.cloud_engine import CloudAIEngine
from ai.models import (
    AIMessage,
    AIResponse,
    AIStreamChunk,
    AIEngineConfig,
    AIEngineHealth,
    EngineCapabilities,
    EngineType,
    TokenUsage,
)


def test_ai_message_and_token_usage():
    """Verify AIMessage attributes, dictionary serialization, and defaults."""
    msg = AIMessage(role="user", content="Hello HAKS", tokens=5)
    assert msg.role == "user"
    assert msg.content == "Hello HAKS"
    assert msg.tokens == 5
    assert msg.timestamp is not None
    assert msg.to_dict() == {"role": "user", "content": "Hello HAKS"}

    usage = TokenUsage(prompt_tokens=10, completion_tokens=25, total_tokens=35)
    assert usage.total_tokens == 35


def test_ai_stream_chunk_and_response():
    """Verify stream chunk and response model attributes."""
    chunk = AIStreamChunk(delta_text="Hello", is_done=False)
    assert chunk.delta_text == "Hello"
    assert chunk.is_done is False
    assert chunk.finish_reason is None

    resp = AIResponse(
        content="Final answer",
        model="llama3.2:1b",
        finish_reason="stop",
        usage=TokenUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
        latency_ms=120.5,
    )
    assert resp.content == "Final answer"
    assert resp.model == "llama3.2:1b"
    assert resp.latency_ms == 120.5


def test_ai_engine_config_and_health():
    """Verify configuration options and health model."""
    cfg = AIEngineConfig(
        engine_type=EngineType.LOCAL,
        provider="ollama",
        model="llama3.2:1b",
        timeout_seconds=30.0,
    )
    assert cfg.engine_type == EngineType.LOCAL
    assert cfg.model == "llama3.2:1b"
    assert cfg.timeout_seconds == 30.0

    health = AIEngineHealth(
        is_healthy=True,
        engine_name="Local AI (Ollama)",
        status_message="Connected",
        available_models=["llama3.2:1b"],
        latency_ms=15.2,
    )
    assert health.is_healthy is True
    assert "llama3.2:1b" in health.available_models


def test_common_interface_conformity():
    """Verify that both LocalAIEngine and CloudAIEngine implement AIEngineInterface."""
    assert issubclass(LocalAIEngine, AIEngineInterface)
    assert issubclass(CloudAIEngine, AIEngineInterface)

    local = LocalAIEngine()
    cloud = CloudAIEngine()

    assert isinstance(local, AIEngineInterface)
    assert isinstance(cloud, AIEngineInterface)

    assert hasattr(local, "generate")
    assert hasattr(local, "stream")
    assert hasattr(local, "check_health")
    assert hasattr(local, "get_capabilities")

    assert hasattr(cloud, "generate")
    assert hasattr(cloud, "stream")
    assert hasattr(cloud, "check_health")
    assert hasattr(cloud, "get_capabilities")

    # Verify declared capabilities
    local_caps = local.get_capabilities()
    cloud_caps = cloud.get_capabilities()

    assert isinstance(local_caps, EngineCapabilities)
    assert isinstance(cloud_caps, EngineCapabilities)
    assert local_caps.is_local is True
    assert cloud_caps.is_local is False
