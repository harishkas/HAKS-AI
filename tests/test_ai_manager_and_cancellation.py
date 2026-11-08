"""Tests for AIEngineManager, engine switching, cancellation, and privacy boundaries."""

import asyncio
from typing import AsyncIterator
import pytest

from ai.exceptions import GenerationCancelledError, ProviderNotConfiguredError
from ai.interface import AIEngineInterface
from ai.manager import AIEngineManager
from ai.models import (
    AIEngineHealth,
    AIMessage,
    AIResponse,
    AIStreamChunk,
    EngineCapabilities,
    EngineType,
    TokenUsage,
)


class MockTestEngine(AIEngineInterface):
    """Simple mock engine for lifecycle and cancellation testing."""

    def __init__(self, name: str, is_local: bool = True):
        self._name = name
        self._is_local = is_local
        self.current_model = "test-model"

    @property
    def engine_name(self) -> str:
        return self._name

    async def generate(self, messages: list[AIMessage], **kwargs) -> AIResponse:
        # Simulate small delay
        await asyncio.sleep(0.02)
        return AIResponse(
            content=f"Response from {self._name}",
            model=self.current_model,
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
        )

    async def stream(self, messages: list[AIMessage], **kwargs) -> AsyncIterator[AIStreamChunk]:
        for i in range(5):
            await asyncio.sleep(0.03)
            yield AIStreamChunk(delta_text=f"token_{i} ", is_done=(i == 4))

    async def check_health(self) -> AIEngineHealth:
        return AIEngineHealth(
            is_healthy=True,
            engine_name=self._name,
            status_message="Healthy mock",
            available_models=["test-model"],
        )

    def get_capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(streaming=True, is_local=self._is_local)


def test_ai_manager_switching():
    """Verify switching active engine between local and cloud."""
    local_mock = MockTestEngine("Local Mock", is_local=True)
    cloud_mock = MockTestEngine("Cloud Mock", is_local=False)

    mgr = AIEngineManager(local_engine=local_mock, cloud_engine=cloud_mock)

    assert mgr.active_engine_type == EngineType.LOCAL
    assert mgr.get_active_engine().engine_name == "Local Mock"

    mgr.set_active_engine("cloud")
    assert mgr.active_engine_type == EngineType.CLOUD
    assert mgr.get_active_engine().engine_name == "Cloud Mock"

    mgr.set_active_engine("local")
    assert mgr.active_engine_type == EngineType.LOCAL


@pytest.mark.asyncio
async def test_ai_manager_concurrent_health():
    """Verify concurrent check_all_health checks both engines."""
    local_mock = MockTestEngine("Local Mock", is_local=True)
    cloud_mock = MockTestEngine("Cloud Mock", is_local=False)

    mgr = AIEngineManager(local_engine=local_mock, cloud_engine=cloud_mock)
    health_dict = await mgr.check_all_health()

    assert "local" in health_dict
    assert "cloud" in health_dict
    assert health_dict["local"].is_healthy is True
    assert health_dict["cloud"].is_healthy is True


@pytest.mark.asyncio
async def test_ai_manager_cancellation():
    """Verify cancel_generation cleanly aborts an active streaming session."""
    local_mock = MockTestEngine("Local Mock", is_local=True)
    cloud_mock = MockTestEngine("Cloud Mock", is_local=False)
    mgr = AIEngineManager(local_engine=local_mock, cloud_engine=cloud_mock)

    messages = [AIMessage(role="user", content="Stream and cancel")]

    received = []

    async def _consumer():
        async for chunk in mgr.stream(messages):
            received.append(chunk.delta_text)
            if len(received) >= 2:
                # Cancel after 2 chunks
                mgr.cancel_generation()

    with pytest.raises(GenerationCancelledError):
        await _consumer()

    assert len(received) == 2
    assert mgr.is_generating is False


@pytest.mark.asyncio
async def test_no_silent_cloud_fallback(monkeypatch):
    """Verify privacy rule: when Cloud AI fails, HAKS must NOT silently fallback to Local AI."""
    from ai.cloud_engine import CloudAIEngine
    from ai.credentials import CredentialManager

    cm = CredentialManager(service_name="HAKS_EMPTY_TEST")
    cloud_engine = CloudAIEngine(credentials=cm)
    local_mock = MockTestEngine("Local Mock", is_local=True)

    mgr = AIEngineManager(local_engine=local_mock, cloud_engine=cloud_engine)
    mgr.set_active_engine("cloud")

    # Cloud has no key; it MUST raise ProviderNotConfiguredError, NOT silently execute Local Mock
    messages = [AIMessage(role="user", content="Classify confidential code")]

    with pytest.raises(ProviderNotConfiguredError):
        await mgr.generate(messages)
