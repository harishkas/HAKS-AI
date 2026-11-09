"""Google Gemini Cloud AI Provider Adapter for HAKS."""

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

logger = get_logger("ai.providers.gemini")

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiProviderAdapter(CloudProviderAdapter):
    """Adapter for Google Gemini REST and SSE API."""

    @property
    def provider_id(self) -> str:
        return "gemini"

    @property
    def display_name(self) -> str:
        return "Google Gemini API"

    @property
    def default_model(self) -> str:
        return "gemini-2.5-flash"

    def get_supported_models(self) -> list[str]:
        return [
            "gemini-2.5-flash",
            "gemini-3.5-flash",
            "gemini-3.8-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
        ]


    def get_capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            streaming=True,
            is_local=False,
            context_window_tokens=1_000_000,
            supports_system_prompt=True,
            supports_vision=True,
            supports_tools=True,
        )

    def _convert_messages(
        self, messages: list[AIMessage], system_prompt: Optional[str]
    ) -> tuple[Optional[dict], list[dict]]:
        """Format messages for Gemini API."""
        sys_inst = None
        if system_prompt:
            sys_inst = {"parts": [{"text": system_prompt}]}

        contents = []
        for msg in messages:
            role = "model" if msg.role == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg.content}]})

        return sys_inst, contents

    async def check_health(self, api_key: Optional[str], model: str) -> AIEngineHealth:
        """Verify API key validity against Gemini models endpoint."""
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
        url = f"{BASE_URL}/models?key={api_key}"
        try:
            async with self._get_client(8.0) as client:
                resp = await client.get(url)
                latency_ms = (time.perf_counter() - start_time) * 1000.0

                if resp.status_code == 200:
                    data = resp.json()
                    models = [
                        m["name"].replace("models/", "")
                        for m in data.get("models", [])
                        if "generateContent" in m.get("supportedGenerationMethods", [])
                    ]
                    return AIEngineHealth(
                        is_healthy=True,
                        engine_name=self.display_name,
                        status_message=f"Connected to Google Gemini API. {len(models)} models available.",
                        available_models=models or self.get_supported_models(),
                        latency_ms=latency_ms,
                        details={"configured": True, "active_model": model},
                    )

                if resp.status_code in (400, 401, 403):
                    return AIEngineHealth(
                        is_healthy=False,
                        engine_name=self.display_name,
                        status_message="Authentication failed: Invalid Gemini API key.",
                        available_models=self.get_supported_models(),
                        latency_ms=latency_ms,
                        details={"http_status": resp.status_code},
                    )

                return AIEngineHealth(
                    is_healthy=False,
                    engine_name=self.display_name,
                    status_message=f"Gemini API returned HTTP {resp.status_code}",
                    available_models=self.get_supported_models(),
                    latency_ms=latency_ms,
                    details={"http_status": resp.status_code},
                )

        except httpx.ConnectError:
            return AIEngineHealth(
                is_healthy=False,
                engine_name=self.display_name,
                status_message="Network error: Could not reach Google Gemini API.",
                available_models=self.get_supported_models(),
            )
        except httpx.TimeoutException:
            return AIEngineHealth(
                is_healthy=False,
                engine_name=self.display_name,
                status_message="Connection timed out reaching Google Gemini API.",
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
        """Execute non-streaming completion."""
        sys_inst, contents = self._convert_messages(messages, system_prompt)
        payload = {
            "contents": contents,
            "generationConfig": {"temperature": temperature},
        }
        if sys_inst:
            payload["systemInstruction"] = sys_inst

        target_model = model
        url = f"{BASE_URL}/models/{target_model}:generateContent?key={api_key}"
        start_time = time.perf_counter()

        try:
            async with self._get_client(timeout) as client:
                resp = await client.post(url, json=payload)

                # Dynamic fallback if gemini-2.5-flash is no longer available to new users
                if resp.status_code == 404 and "gemini-2.5" in target_model and "no longer available" in resp.text:
                    logger.info("gemini-2.5-flash deprecated by upstream API, falling back to gemini-3.5-flash-lite...")
                    target_model = "gemini-3.5-flash-lite"
                    fallback_url = f"{BASE_URL}/models/{target_model}:generateContent?key={api_key}"
                    resp = await client.post(fallback_url, json=payload)

                # If upstream model experiences temporary 503 demand, retry once with fast flash-lite
                if resp.status_code == 503 and target_model != "gemini-3.5-flash-lite":
                    logger.info("Upstream returned 503 for %s, retrying with gemini-3.5-flash-lite...", target_model)
                    target_model = "gemini-3.5-flash-lite"
                    fallback_url = f"{BASE_URL}/models/{target_model}:generateContent?key={api_key}"
                    resp = await client.post(fallback_url, json=payload)

                if resp.status_code in (400, 401, 403):
                    raise AuthenticationError(
                        f"Gemini API key rejected (HTTP {resp.status_code}). Please verify your Gemini API key in Windows Credential Manager.",
                        engine="gemini",
                    )
                if resp.status_code == 429:
                    raise RateLimitError(
                        "Gemini API quota or rate limit exceeded (HTTP 429). Please check your Google AI Studio plan or retry shortly.",
                        engine="gemini",
                    )
                if resp.status_code == 503:
                    raise AIEngineError(
                        "Google Gemini API is temporarily experiencing high demand (HTTP 503). Spikes are temporary; please retry in a moment.",
                        engine="gemini",
                    )
                if resp.status_code != 200:
                    raise AIEngineError(
                        f"Gemini API returned HTTP {resp.status_code}: {resp.text}",
                        engine="gemini",
                    )

                data = resp.json()
                latency_ms = (time.perf_counter() - start_time) * 1000.0

                candidates = data.get("candidates", [])
                text = ""
                finish_reason = "stop"
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    text = "".join(p.get("text", "") for p in parts)
                    finish_reason = candidates[0].get("finishReason", "stop")

                usage_meta = data.get("usageMetadata", {})
                p_tok = usage_meta.get("promptTokenCount", 0)
                c_tok = usage_meta.get("candidatesTokenCount", 0)

                return AIResponse(
                    content=text,
                    model=target_model,
                    finish_reason=finish_reason,
                    usage=TokenUsage(
                        prompt_tokens=p_tok,
                        completion_tokens=c_tok,
                        total_tokens=p_tok + c_tok,
                    ),
                    latency_ms=latency_ms,
                    raw=data,
                )

        except httpx.ConnectError as exc:
            raise AIEngineConnectionError(
                "Failed to connect to Google Gemini API. Please check your internet connection.",
                engine="gemini",
            ) from exc
        except httpx.TimeoutException as exc:
            raise AIEngineTimeoutError(
                f"Gemini request timed out after {timeout:.1f}s. Please click Retry.",
                engine="gemini",
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
        """Stream tokens via SSE endpoint."""
        sys_inst, contents = self._convert_messages(messages, system_prompt)
        payload = {
            "contents": contents,
            "generationConfig": {"temperature": temperature},
        }
        if sys_inst:
            payload["systemInstruction"] = sys_inst

        target_model = model
        url = f"{BASE_URL}/models/{target_model}:streamGenerateContent?alt=sse&key={api_key}"

        try:
            async with self._get_client(timeout) as client:
                async with client.stream("POST", url, json=payload) as resp:
                    # Dynamic fallback if gemini-2.5-flash stream is discontinued
                    if resp.status_code == 404 and "gemini-2.5" in target_model:
                        err_bytes = await resp.aread()
                        err_str = err_bytes.decode("utf-8", errors="ignore")
                        if "no longer available" in err_str:
                            logger.info("gemini-2.5-flash stream deprecated, retrying with gemini-3.5-flash-lite...")
                            target_model = "gemini-3.5-flash-lite"
                            fallback_url = f"{BASE_URL}/models/{target_model}:streamGenerateContent?alt=sse&key={api_key}"
                            async with client.stream("POST", fallback_url, json=payload) as fb_resp:
                                if fb_resp.status_code != 200:
                                    fb_err = await fb_resp.aread()
                                    raise AIEngineError(f"Gemini stream returned HTTP {fb_resp.status_code}: {fb_err.decode('utf-8', errors='ignore')}", engine="gemini")
                                async for line in fb_resp.aiter_lines():
                                    if not line or not line.strip() or not line.startswith("data: "):
                                        continue
                                    raw_json = line[6:].strip()
                                    if not raw_json or raw_json == "[DONE]":
                                        break
                                    try:
                                        item = json.loads(raw_json)
                                    except json.JSONDecodeError:
                                        continue
                                    candidates = item.get("candidates", [])
                                    delta = ""
                                    done = False
                                    finish_reason = None
                                    if candidates:
                                        parts = candidates[0].get("content", {}).get("parts", [])
                                        delta = "".join(p.get("text", "") for p in parts)
                                        finish_reason = candidates[0].get("finishReason")
                                        if finish_reason in ("STOP", "MAX_TOKENS"):
                                            done = True
                                    yield AIStreamChunk(delta_text=delta, is_done=done, finish_reason=finish_reason)
                            return

                    if resp.status_code in (400, 401, 403):
                        err_bytes = await resp.aread()
                        raise AuthenticationError(
                            f"Gemini API key rejected (HTTP {resp.status_code}): {err_bytes.decode('utf-8', errors='ignore')}",
                            engine="gemini",
                        )
                    if resp.status_code == 429:
                        raise RateLimitError("Gemini API rate limit or quota exceeded (HTTP 429).", engine="gemini")
                    if resp.status_code == 503:
                        raise AIEngineError("Gemini API is temporarily experiencing high demand (HTTP 503). Please retry in a moment.", engine="gemini")
                    if resp.status_code != 200:
                        err_bytes = await resp.aread()
                        raise AIEngineError(
                            f"Gemini stream returned HTTP {resp.status_code}: {err_bytes.decode('utf-8', errors='ignore')}",
                            engine="gemini",
                        )


                    async for line in resp.aiter_lines():
                        if not line or not line.strip():
                            continue

                        # Handle SSE data: lines
                        if line.startswith("data: "):
                            raw_json = line[6:].strip()
                            if not raw_json or raw_json == "[DONE]":
                                break

                            try:
                                item = json.loads(raw_json)
                            except json.JSONDecodeError:
                                continue

                            candidates = item.get("candidates", [])
                            delta = ""
                            done = False
                            finish_reason = None
                            if candidates:
                                parts = candidates[0].get("content", {}).get("parts", [])
                                delta = "".join(p.get("text", "") for p in parts)
                                finish_reason = candidates[0].get("finishReason")
                                if finish_reason in ("STOP", "MAX_TOKENS"):
                                    done = True

                            usage = None
                            if "usageMetadata" in item:
                                u = item["usageMetadata"]
                                p_tok = u.get("promptTokenCount", 0)
                                c_tok = u.get("candidatesTokenCount", 0)
                                usage = TokenUsage(
                                    prompt_tokens=p_tok,
                                    completion_tokens=c_tok,
                                    total_tokens=p_tok + c_tok,
                                )

                            yield AIStreamChunk(
                                delta_text=delta,
                                is_done=done,
                                finish_reason=finish_reason,
                                usage=usage,
                            )

        except httpx.ConnectError as exc:
            raise AIEngineConnectionError(
                "Failed to connect to Google Gemini API. Check your internet connection.",
                engine="gemini",
            ) from exc
        except httpx.TimeoutException as exc:
            raise AIEngineTimeoutError(
                f"Gemini streaming connection timed out after {timeout:.1f}s",
                engine="gemini",
            ) from exc
