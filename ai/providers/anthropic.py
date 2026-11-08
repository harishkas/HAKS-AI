"""Anthropic Claude Cloud AI Provider Adapter for HAKS."""

import json
import time
from typing import AsyncIterator, Optional
import httpx

from ai.exceptions import (
    AIEngineConnectionError,
    AIEngineError,
    AIEngineTimeoutError,
    AuthenticationError,
    RateLimitError,
)
from ai.models import (
    AIMessage,
    AIResponse,
    AIStreamChunk,
    AIEngineHealth,
    EngineCapabilities,
    TokenUsage,
)
from ai.providers.base import CloudProviderAdapter
from core.logger import get_logger

logger = get_logger("ai.providers.anthropic")

BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProviderAdapter(CloudProviderAdapter):
    """Adapter for Anthropic Claude Messages API."""

    @property
    def provider_id(self) -> str:
        return "anthropic"

    @property
    def display_name(self) -> str:
        return "Anthropic Claude API"

    @property
    def default_model(self) -> str:
        return "claude-3-5-haiku-20241022"

    def get_supported_models(self) -> list[str]:
        return [
            "claude-3-5-haiku-20241022",
            "claude-3-5-sonnet-20241022",
        ]

    def get_capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            streaming=True,
            is_local=False,
            context_window_tokens=200_000,
            supports_system_prompt=True,
            supports_vision=True,
            supports_tools=True,
        )

    async def check_health(self, api_key: Optional[str], model: str) -> AIEngineHealth:
        """Verify API key validity with a lightweight messages call or header check."""
        if not api_key:
            return AIEngineHealth(
                is_healthy=False,
                engine_name=self.display_name,
                status_message="API Key is not configured in Windows Credential Manager or environment.",
                available_models=self.get_supported_models(),
                latency_ms=None,
                details={"configured": False},
            )

        start_time = time.perf_counter()
        headers = {
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        # Lightweight check with max_tokens=1
        body = {
            "model": model,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "ping"}],
        }

        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.post(f"{BASE_URL}/messages", headers=headers, json=body)
                latency_ms = (time.perf_counter() - start_time) * 1000.0

                if resp.status_code == 200:
                    return AIEngineHealth(
                        is_healthy=True,
                        engine_name=self.display_name,
                        status_message=f"Connected to Anthropic API. Model '{model}' active.",
                        available_models=self.get_supported_models(),
                        latency_ms=latency_ms,
                        details={"configured": True, "active_model": model},
                    )

                if resp.status_code in (401, 403):
                    return AIEngineHealth(
                        is_healthy=False,
                        engine_name=self.display_name,
                        status_message="Authentication failed: Invalid Anthropic API key.",
                        available_models=self.get_supported_models(),
                        latency_ms=latency_ms,
                        details={"http_status": resp.status_code},
                    )

                return AIEngineHealth(
                    is_healthy=False,
                    engine_name=self.display_name,
                    status_message=f"Anthropic returned HTTP {resp.status_code}",
                    available_models=self.get_supported_models(),
                    latency_ms=latency_ms,
                    details={"http_status": resp.status_code},
                )

        except httpx.ConnectError:
            return AIEngineHealth(
                is_healthy=False,
                engine_name=self.display_name,
                status_message=f"Network error: Could not reach {BASE_URL}.",
                available_models=self.get_supported_models(),
            )
        except httpx.TimeoutException:
            return AIEngineHealth(
                is_healthy=False,
                engine_name=self.display_name,
                status_message=f"Connection timed out reaching {BASE_URL}.",
                available_models=self.get_supported_models(),
            )

    async def generate(
        self,
        api_key: str,
        model: str,
        messages: list[AIMessage],
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        timeout: float = 60.0,
    ) -> AIResponse:
        """Non-streaming message completion."""
        url = f"{BASE_URL}/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

        req_messages = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        body = {
            "model": model,
            "max_tokens": 4096,
            "messages": req_messages,
            "temperature": temperature,
        }
        if system_prompt:
            body["system"] = system_prompt

        start_time = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, headers=headers, json=body)

                if resp.status_code in (401, 403):
                    raise AuthenticationError("Invalid Anthropic API key.", engine="anthropic")
                if resp.status_code == 429:
                    raise RateLimitError("Anthropic rate limit exceeded.", engine="anthropic")
                if resp.status_code != 200:
                    raise AIEngineError(
                        f"Anthropic API returned HTTP {resp.status_code}: {resp.text}",
                        engine="anthropic",
                    )

                data = resp.json()
                latency_ms = (time.perf_counter() - start_time) * 1000.0

                content_blocks = data.get("content", [])
                text = "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")

                usage_data = data.get("usage", {})
                usage = TokenUsage(
                    prompt_tokens=usage_data.get("input_tokens", 0),
                    completion_tokens=usage_data.get("output_tokens", 0),
                    total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0),
                )

                return AIResponse(
                    content=text,
                    model=model,
                    finish_reason=data.get("stop_reason", "stop"),
                    usage=usage,
                    latency_ms=latency_ms,
                    raw=data,
                )

        except httpx.ConnectError as exc:
            raise AIEngineConnectionError("Failed to connect to Anthropic API.", engine="anthropic") from exc
        except httpx.TimeoutException as exc:
            raise AIEngineTimeoutError(f"Anthropic request timed out after {timeout:.1f}s", engine="anthropic") from exc

    async def stream(
        self,
        api_key: str,
        model: str,
        messages: list[AIMessage],
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        timeout: float = 60.0,
    ) -> AsyncIterator[AIStreamChunk]:
        """Streaming message completion via SSE."""
        url = f"{BASE_URL}/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

        req_messages = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        body = {
            "model": model,
            "max_tokens": 4096,
            "messages": req_messages,
            "temperature": temperature,
            "stream": True,
        }
        if system_prompt:
            body["system"] = system_prompt

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", url, headers=headers, json=body) as resp:
                    if resp.status_code in (401, 403):
                        err_bytes = await resp.aread()
                        raise AuthenticationError(
                            f"Invalid Anthropic API key: {err_bytes.decode('utf-8', errors='ignore')}",
                            engine="anthropic",
                        )
                    if resp.status_code == 429:
                        raise RateLimitError("Anthropic rate limit exceeded.", engine="anthropic")
                    if resp.status_code != 200:
                        err_bytes = await resp.aread()
                        raise AIEngineError(
                            f"Anthropic stream returned HTTP {resp.status_code}: {err_bytes.decode('utf-8', errors='ignore')}",
                            engine="anthropic",
                        )

                    async for line in resp.aiter_lines():
                        if not line or not line.strip():
                            continue

                        if line.startswith("data: "):
                            raw_json = line[6:].strip()
                            try:
                                item = json.loads(raw_json)
                            except json.JSONDecodeError:
                                continue

                            event_type = item.get("type", "")
                            delta = ""
                            done = False

                            if event_type == "content_block_delta":
                                delta = item.get("delta", {}).get("text", "")
                            elif event_type == "message_stop":
                                done = True

                            yield AIStreamChunk(
                                delta_text=delta,
                                is_done=done,
                            )

        except httpx.ConnectError as exc:
            raise AIEngineConnectionError("Failed to connect to Anthropic API.", engine="anthropic") from exc
        except httpx.TimeoutException as exc:
            raise AIEngineTimeoutError(f"Anthropic streaming timed out after {timeout:.1f}s", engine="anthropic") from exc
