"""OpenAI-Compatible Cloud AI Provider Adapter for HAKS.

Supports OpenAI, Groq, OpenRouter, and standard OpenAI-compatible REST endpoints.
"""

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

logger = get_logger("ai.providers.openai_compatible")


class OpenAICompatibleAdapter(CloudProviderAdapter):
    """Adapter for OpenAI and OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        provider_id: str = "openai",
        display_name: str = "OpenAI",
        base_url: str = "https://api.openai.com/v1",
        default_model: str = "gpt-4o-mini",
        supported_models: Optional[list[str]] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        super().__init__(transport=transport)
        self._provider_id = provider_id
        self._display_name = display_name
        self.base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._supported_models = supported_models or [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-3.5-turbo",
        ]

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def display_name(self) -> str:
        return self._display_name

    @property
    def default_model(self) -> str:
        return self._default_model

    def get_supported_models(self) -> list[str]:
        return list(self._supported_models)

    def set_base_url(self, url: str) -> None:
        """Update base URL for custom OpenAI-compatible endpoints."""
        self.base_url = url.rstrip("/")

    def set_supported_models(self, models: list[str]) -> None:
        """Update list of models supported by this endpoint."""
        if models:
            self._supported_models = list(models)
            self._default_model = models[0]

    def get_capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            streaming=True,
            is_local=False,
            context_window_tokens=128_000,
            supports_system_prompt=True,
            supports_vision=True,
            supports_tools=True,
        )

    async def check_health(self, api_key: Optional[str], model: str) -> AIEngineHealth:
        """Verify API key validity against models endpoint."""
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
        headers = {"Authorization": f"Bearer {api_key}"}
        url = f"{self.base_url}/models"

        try:
            async with self._get_client(8.0) as client:
                resp = await client.get(url, headers=headers)
                latency_ms = (time.perf_counter() - start_time) * 1000.0

                if resp.status_code == 200:
                    data = resp.json()
                    all_models = [m["id"] for m in data.get("data", []) if "id" in m]
                    if self._provider_id == "openai":
                        models = [m for m in all_models if "gpt" in m or "o1" in m or "o3" in m]
                    else:
                        models = all_models[:15]
                    return AIEngineHealth(
                        is_healthy=True,
                        engine_name=self.display_name,
                        status_message=f"Connected to {self.display_name}.",
                        available_models=models or self.get_supported_models(),
                        latency_ms=latency_ms,
                        details={"configured": True, "active_model": model},
                    )

                if resp.status_code == 401:
                    return AIEngineHealth(
                        is_healthy=False,
                        engine_name=self.display_name,
                        status_message=f"Authentication failed: Invalid {self.display_name} API key.",
                        available_models=self.get_supported_models(),
                        latency_ms=latency_ms,
                        details={"http_status": 401},
                    )

                return AIEngineHealth(
                    is_healthy=False,
                    engine_name=self.display_name,
                    status_message=f"API returned HTTP {resp.status_code}",
                    available_models=self.get_supported_models(),
                    latency_ms=latency_ms,
                    details={"http_status": resp.status_code},
                )

        except httpx.ConnectError:
            return AIEngineHealth(
                is_healthy=False,
                engine_name=self.display_name,
                status_message=f"Network error: Could not reach {self.base_url}.",
                available_models=self.get_supported_models(),
            )
        except httpx.TimeoutException:
            return AIEngineHealth(
                is_healthy=False,
                engine_name=self.display_name,
                status_message=f"Connection timed out reaching {self.base_url}.",
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
        """Non-streaming completion."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        req_messages = []
        if system_prompt:
            req_messages.append({"role": "system", "content": system_prompt})
        for m in messages:
            req_messages.append({"role": m.role, "content": m.content})

        body = {
            "model": model,
            "messages": req_messages,
            "temperature": temperature,
            "stream": False,
        }

        start_time = time.perf_counter()
        try:
            async with self._get_client(timeout) as client:
                resp = await client.post(url, headers=headers, json=body)

                if resp.status_code == 401:
                    raise AuthenticationError(f"Invalid {self.display_name} API key.", engine=self.provider_id)
                if resp.status_code == 429:
                    raise RateLimitError(f"{self.display_name} API rate limit exceeded.", engine=self.provider_id)
                if resp.status_code != 200:
                    raise AIEngineError(
                        f"{self.display_name} API returned HTTP {resp.status_code}: {resp.text}",
                        engine=self.provider_id,
                    )

                data = resp.json()
                latency_ms = (time.perf_counter() - start_time) * 1000.0

                choices = data.get("choices", [])
                content = choices[0]["message"]["content"] if choices else ""
                finish_reason = choices[0].get("finish_reason", "stop") if choices else "stop"

                usage_data = data.get("usage", {})
                usage = TokenUsage(
                    prompt_tokens=usage_data.get("prompt_tokens", 0),
                    completion_tokens=usage_data.get("completion_tokens", 0),
                    total_tokens=usage_data.get("total_tokens", 0),
                )

                return AIResponse(
                    content=content,
                    model=model,
                    finish_reason=finish_reason,
                    usage=usage,
                    latency_ms=latency_ms,
                    raw=data,
                )

        except httpx.ConnectError as exc:
            raise AIEngineConnectionError(
                f"Cannot connect to {self.base_url}. Check network connectivity.",
                engine=self.provider_id,
            ) from exc
        except httpx.TimeoutException as exc:
            raise AIEngineTimeoutError(
                f"{self.display_name} request timed out after {timeout:.1f}s",
                engine=self.provider_id,
            ) from exc

    async def stream(
        self,
        api_key: str,
        model: str,
        messages: list[AIMessage],
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        timeout: float = 60.0,
    ) -> AsyncIterator[AIStreamChunk]:
        """Streaming chat completion using SSE."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        req_messages = []
        if system_prompt:
            req_messages.append({"role": "system", "content": system_prompt})
        for m in messages:
            req_messages.append({"role": m.role, "content": m.content})

        body = {
            "model": model,
            "messages": req_messages,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        try:
            async with self._get_client(timeout) as client:
                async with client.stream("POST", url, headers=headers, json=body) as resp:
                    if resp.status_code == 401:
                        err_bytes = await resp.aread()
                        raise AuthenticationError(
                            f"Invalid {self.display_name} API key: {err_bytes.decode('utf-8', errors='ignore')}",
                            engine=self.provider_id,
                        )
                    if resp.status_code == 429:
                        raise RateLimitError(f"{self.display_name} API rate limit exceeded.", engine=self.provider_id)
                    if resp.status_code != 200:
                        err_bytes = await resp.aread()
                        raise AIEngineError(
                            f"{self.display_name} stream returned HTTP {resp.status_code}: {err_bytes.decode('utf-8', errors='ignore')}",
                            engine=self.provider_id,
                        )

                    async for line in resp.aiter_lines():
                        if not line or not line.strip():
                            continue

                        if line.startswith("data: "):
                            raw_json = line[6:].strip()
                            if raw_json == "[DONE]":
                                break

                            try:
                                item = json.loads(raw_json)
                            except json.JSONDecodeError:
                                continue

                            choices = item.get("choices", [])
                            delta = ""
                            finish_reason = None
                            done = False

                            if choices:
                                delta = choices[0].get("delta", {}).get("content", "") or ""
                                finish_reason = choices[0].get("finish_reason")
                                if finish_reason is not None:
                                    done = True

                            usage = None
                            if "usage" in item and item["usage"] is not None:
                                u = item["usage"]
                                usage = TokenUsage(
                                    prompt_tokens=u.get("prompt_tokens", 0),
                                    completion_tokens=u.get("completion_tokens", 0),
                                    total_tokens=u.get("total_tokens", 0),
                                )

                            yield AIStreamChunk(
                                delta_text=delta,
                                is_done=done,
                                finish_reason=finish_reason,
                                usage=usage,
                            )

        except httpx.ConnectError as exc:
            raise AIEngineConnectionError(
                f"Cannot connect to {self.base_url}.",
                engine=self.provider_id,
            ) from exc
        except httpx.TimeoutException as exc:
            raise AIEngineTimeoutError(
                f"{self.display_name} streaming connection timed out after {timeout:.1f}s",
                engine=self.provider_id,
            ) from exc
