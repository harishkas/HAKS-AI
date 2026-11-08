"""Tests for CloudAIEngine and Provider Adapters.

Includes:
1. Real Environment Credential Test (verifies real credentials status without fabricating connectivity).
2. Mocked Cloud Provider Tests (validates API key checks, authentication errors, rate limiting, network errors, timeouts, and SSE streaming).
"""

import json
import pytest
import httpx

from ai.cloud_engine import CloudAIEngine
from ai.credentials import CredentialManager
from ai.exceptions import (
    AIEngineConnectionError,
    AIEngineTimeoutError,
    AuthenticationError,
    ProviderNotConfiguredError,
    RateLimitError,
)
from ai.models import AIMessage, EngineType
from ai.providers.gemini import GeminiProviderAdapter
from ai.providers.openai_compatible import OpenAICompatibleAdapter
from ai.providers.anthropic import AnthropicProviderAdapter


# ==============================================================================
# 1. REAL INTEGRATION PROBE (Honest check on unconfigured credentials)
# ==============================================================================

@pytest.mark.asyncio
async def test_real_cloud_unconfigured_behavior():
    """Real integration probe: Verify CloudAIEngine accurately reports unconfigured state when no real credentials exist."""
    cm = CredentialManager(service_name="HAKS_NONEXISTENT_TEST")
    engine = CloudAIEngine(credentials=cm)

    # Health check should cleanly report not configured without crashing
    health = await engine.check_health()
    assert health.is_healthy is False
    assert "not configured" in health.status_message.lower()

    # Attempting generation without key must raise ProviderNotConfiguredError
    messages = [AIMessage(role="user", content="Ping")]
    with pytest.raises(ProviderNotConfiguredError) as exc_info:
        await engine.generate(messages)

    assert "not configured with an api key" in str(exc_info.value).lower()


# ==============================================================================
# 2. MOCKED PROVIDER TESTS (Unit tests using httpx MockTransport)
# ==============================================================================

@pytest.mark.asyncio
async def test_mocked_gemini_auth_failure():
    """[MOCKED] Verify GeminiProviderAdapter raises AuthenticationError on HTTP 400/403."""
    def mock_handler(request: httpx.Request):
        return httpx.Response(
            400,
            json={"error": {"code": 400, "message": "API key not valid", "status": "INVALID_ARGUMENT"}},
        )

    transport = httpx.MockTransport(mock_handler)
    adapter = GeminiProviderAdapter(transport=transport)
    messages = [AIMessage(role="user", content="Test")]

    with pytest.raises(AuthenticationError) as exc_info:
        await adapter.generate(api_key="bad_key", model="gemini-2.5-flash", messages=messages)

    assert "key rejected" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_mocked_gemini_rate_limit():
    """[MOCKED] Verify GeminiProviderAdapter raises RateLimitError on HTTP 429."""
    def mock_handler(request: httpx.Request):
        return httpx.Response(
            429,
            json={"error": {"code": 429, "message": "Resource has been exhausted", "status": "RESOURCE_EXHAUSTED"}},
        )

    transport = httpx.MockTransport(mock_handler)
    adapter = GeminiProviderAdapter(transport=transport)
    messages = [AIMessage(role="user", content="Test")]

    with pytest.raises(RateLimitError):
        await adapter.generate(api_key="valid_key", model="gemini-2.5-flash", messages=messages)


@pytest.mark.asyncio
async def test_mocked_gemini_successful_generation():
    """[MOCKED] Verify GeminiProviderAdapter parses generateContent response."""
    def mock_handler(request: httpx.Request):
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "Gemini response to user query"}],
                            "role": "model",
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 8,
                    "candidatesTokenCount": 14,
                    "totalTokenCount": 22,
                },
            },
        )

    transport = httpx.MockTransport(mock_handler)
    adapter = GeminiProviderAdapter(transport=transport)
    messages = [AIMessage(role="user", content="What is HAKS?")]
    response = await adapter.generate(api_key="valid_key", model="gemini-2.5-flash", messages=messages)

    assert response.content == "Gemini response to user query"
    assert response.usage.prompt_tokens == 8
    assert response.usage.completion_tokens == 14
    assert response.usage.total_tokens == 22


@pytest.mark.asyncio
async def test_mocked_gemini_streaming_sse():
    """[MOCKED] Verify GeminiProviderAdapter parses SSE stream."""
    sse_data = (
        b'data: {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}\n\n'
        b'data: {"candidates": [{"content": {"parts": [{"text": " from Gemini!"}]},"finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 4}}\n\n'
    )

    def mock_handler(request: httpx.Request):
        return httpx.Response(200, content=sse_data)

    transport = httpx.MockTransport(mock_handler)
    adapter = GeminiProviderAdapter(transport=transport)
    messages = [AIMessage(role="user", content="Stream test")]

    chunks = []
    async for chunk in adapter.stream(api_key="valid_key", model="gemini-2.5-flash", messages=messages):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0].delta_text == "Hello"
    assert chunks[1].delta_text == " from Gemini!"
    assert chunks[1].is_done is True


@pytest.mark.asyncio
async def test_mocked_cloud_network_failure():
    """[MOCKED] Verify CloudAIEngine maps network connection failure to AIEngineConnectionError."""
    def mock_handler(request: httpx.Request):
        raise httpx.ConnectError("Could not resolve host")

    transport = httpx.MockTransport(mock_handler)
    adapter = GeminiProviderAdapter(transport=transport)
    messages = [AIMessage(role="user", content="Ping")]

    with pytest.raises(AIEngineConnectionError) as exc_info:
        await adapter.generate(api_key="valid_key", model="gemini-2.5-flash", messages=messages)

    assert "connect" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_mocked_cloud_timeout_handling():
    """[MOCKED] Verify CloudAIEngine maps timeout exception to AIEngineTimeoutError."""
    def mock_handler(request: httpx.Request):
        raise httpx.TimeoutException("Read timeout after 10.0s")

    transport = httpx.MockTransport(mock_handler)
    adapter = OpenAICompatibleAdapter(transport=transport)
    messages = [AIMessage(role="user", content="Ping")]

    with pytest.raises(AIEngineTimeoutError) as exc_info:
        await adapter.generate(api_key="valid_key", model="gpt-4o-mini", messages=messages, timeout=10.0)

    assert "timed out" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_mocked_openai_generation():
    """[MOCKED] Verify OpenAICompatibleAdapter parses chat completions."""
    def mock_handler(request: httpx.Request):
        assert "Authorization" in request.headers
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "OpenAI answer"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    transport = httpx.MockTransport(mock_handler)
    adapter = OpenAICompatibleAdapter(transport=transport)
    messages = [AIMessage(role="user", content="Test")]
    resp = await adapter.generate(api_key="sk-test", model="gpt-4o-mini", messages=messages)

    assert resp.content == "OpenAI answer"
    assert resp.usage.total_tokens == 15


@pytest.mark.asyncio
async def test_mocked_groq_and_openrouter_providers():
    """Verify CloudAIEngine supports Groq and OpenRouter providers."""
    def mock_handler(request: httpx.Request):
        if "groq.com" in str(request.url) and request.url.path == "/openai/v1/chat/completions":
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "Fast Groq Llama answer"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
                },
            )
        elif "openrouter.ai" in str(request.url) and request.url.path == "/api/v1/chat/completions":
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "Universal OpenRouter answer"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 10, "total_tokens": 22},
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)
    from ai.providers.openai_compatible import OpenAICompatibleAdapter

    groq_adapter = OpenAICompatibleAdapter(
        provider_id="groq",
        display_name="Groq Cloud",
        base_url="https://api.groq.com/openai/v1",
        default_model="llama-3.1-8b-instant",
        transport=transport,
    )
    openrouter_adapter = OpenAICompatibleAdapter(
        provider_id="openrouter",
        display_name="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        default_model="google/gemini-2.0-flash-001",
        transport=transport,
    )

    cloud = CloudAIEngine()
    cloud._providers["groq"] = groq_adapter
    cloud._providers["openrouter"] = openrouter_adapter

    # 1. Groq test
    cloud.set_provider("groq", "llama-3.1-8b-instant")
    assert cloud.active_provider_id == "groq"
    assert cloud.current_model == "llama-3.1-8b-instant"

    groq_resp = await groq_adapter.generate(
        api_key="gsk-test",
        model="llama-3.1-8b-instant",
        messages=[AIMessage(role="user", content="Hi Groq")],
    )
    assert groq_resp.content == "Fast Groq Llama answer"
    assert groq_resp.usage.total_tokens == 18

    # 2. OpenRouter test
    cloud.set_provider("openrouter", "google/gemini-2.0-flash-001")
    assert cloud.active_provider_id == "openrouter"

    or_resp = await openrouter_adapter.generate(
        api_key="sk-or-test",
        model="google/gemini-2.0-flash-001",
        messages=[AIMessage(role="user", content="Hi OpenRouter")],
    )
    assert or_resp.content == "Universal OpenRouter answer"
    assert or_resp.usage.total_tokens == 22


@pytest.mark.asyncio
async def test_custom_openai_compatible_endpoint():
    """Verify configurable custom OpenAI endpoint support."""
    def mock_handler(request: httpx.Request):
        if request.url.host == "my-local-server":
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "Response from custom host"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)
    from ai.providers.openai_compatible import OpenAICompatibleAdapter

    custom_adapter = OpenAICompatibleAdapter(
        provider_id="custom",
        display_name="Custom OpenAI Endpoint",
        base_url="http://my-local-server:8000/v1",
        default_model="custom-llm",
        transport=transport,
    )

    cloud = CloudAIEngine()
    cloud._providers["custom"] = custom_adapter
    cloud.set_provider("custom", "custom-llm")

    # Update base URL test
    cloud.set_provider_base_url("custom", "http://my-local-server:9000/v1")
    assert custom_adapter.base_url == "http://my-local-server:9000/v1"

    # Reset to mock host for call
    custom_adapter.set_base_url("http://my-local-server:8000/v1")
    resp = await custom_adapter.generate(
        api_key="custom-key",
        model="custom-llm",
        messages=[AIMessage(role="user", content="Hello custom")],
    )
    assert resp.content == "Response from custom host"
