"""Asynchronous Ollama API Client for HAKS Local AI.

Communicates with the local Ollama daemon via its REST API (http://127.0.0.1:11434).
Supports model discovery, non-blocking asynchronous chat, streaming NDJSON tokens,
explicit unload controls for RAM management, and detailed error mapping.
"""

import json
import os
import shutil
import time
from typing import Any, AsyncIterator, Optional
import httpx

from ai.exceptions import (
    AIEngineConnectionError,
    AIEngineError,
    AIEngineTimeoutError,
    ContextOverflowError,
    EmptyResponseError,
    ModelNotFoundError,
    TokenLimitError,
)
from ai.models import (
    AIMessage,
    AIResponse,
    AIStreamChunk,
    TokenUsage,
    estimate_messages_tokens,
)
from core.logger import get_logger

logger = get_logger("ai.ollama_client")


class OllamaClient:
    """Async client interacting directly with the Ollama REST engine."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 60.0,
        keep_alive: str = "5m",
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.keep_alive = keep_alive  # e.g., '5m' or '0' (unload immediately)
        self.transport = transport

    def _get_client(self, timeout: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=timeout, transport=self.transport)

    @staticmethod
    def is_ollama_installed() -> bool:
        """Check if ollama binary is installed and discoverable on PATH or standard Windows paths."""
        if shutil.which("ollama") is not None:
            return True
        # Standard Windows installation paths
        win_paths = [
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
            os.path.expandvars(r"%ProgramFiles%\Ollama\ollama.exe"),
            os.path.expandvars(r"%ProgramFiles(x86)%\Ollama\ollama.exe"),
            os.path.expandvars(r"%USERPROFILE%\AppData\Local\Programs\Ollama\ollama.exe"),
        ]
        return any(os.path.isfile(p) for p in win_paths)

    async def is_server_running(self) -> bool:
        """Probe Ollama endpoint to verify if the server daemon is active."""
        try:
            async with self._get_client(3.0) as client:
                resp = await client.get(f"{self.base_url}/api/version")
                return resp.status_code == 200
        except Exception:
            return False

    async def get_version(self) -> Optional[str]:
        """Fetch Ollama daemon version string."""
        try:
            async with self._get_client(3.0) as client:
                resp = await client.get(f"{self.base_url}/api/version")
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("version")
        except Exception as exc:
            logger.debug("Failed to retrieve Ollama version: %s", exc)
        return None

    async def list_models(self) -> list[dict[str, Any]]:
        """List all models currently installed locally in Ollama.

        Returns list of model descriptors:
        [{"name": "llama3.2:1b", "size_bytes": 1320000000, "modified_at": "...", "details": {...}}]
        """
        try:
            async with self._get_client(5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                if resp.status_code == 200:
                    data = resp.json()
                    models = []
                    for item in data.get("models", []):
                        models.append({
                            "name": item.get("name", ""),
                            "model": item.get("model", ""),
                            "size_bytes": item.get("size", 0),
                            "modified_at": item.get("modified_at", ""),
                            "details": item.get("details", {}),
                        })
                    return models
                raise AIEngineError(
                    f"Ollama returned HTTP {resp.status_code} while listing models: {resp.text}",
                    engine="ollama",
                )
        except httpx.ConnectError as exc:
            raise AIEngineConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. Is Ollama running?",
                engine="ollama",
            ) from exc
        except httpx.TimeoutException as exc:
            raise AIEngineTimeoutError(
                f"Timeout connecting to Ollama at {self.base_url}",
                engine="ollama",
            ) from exc

    def _handle_error_response(self, status_code: int, raw_text: str, model: str) -> None:
        """Parse Ollama error response and map to domain-specific exceptions."""
        if status_code == 404:
            raise ModelNotFoundError(
                f"Local model '{model}' not found in Ollama. Pull it with: ollama pull {model}",
                engine="ollama",
                details={"model": model},
            )

        err_msg = raw_text
        try:
            err_data = json.loads(raw_text)
            if isinstance(err_data, dict):
                err_msg = err_data.get("error", raw_text)
        except Exception:
            pass

        lower_err = err_msg.lower()
        if "token repeat limit reached" in lower_err:
            raise TokenLimitError(
                f"Ollama prediction aborted (token repeat limit reached): {err_msg}",
                engine="ollama",
                details={"model": model, "status_code": status_code, "error": err_msg, "type": "repeat_limit"},
            )
        elif "token limit reached" in lower_err or "prediction aborted" in lower_err:
            raise TokenLimitError(
                f"Ollama prediction aborted (token limit reached): {err_msg}",
                engine="ollama",
                details={"model": model, "status_code": status_code, "error": err_msg, "type": "token_limit"},
            )
        elif "context window" in lower_err or "context length" in lower_err or "too many tokens" in lower_err or "exceeds context" in lower_err:
            raise ContextOverflowError(
                f"Ollama context window exceeded: {err_msg}",
                engine="ollama",
                details={"model": model, "status_code": status_code, "error": err_msg, "type": "context_overflow"},
            )

        raise AIEngineError(
            f"Ollama returned HTTP {status_code}: {err_msg}",
            engine="ollama",
            details={"model": model, "status_code": status_code, "error": err_msg},
        )

    def _build_payload_messages(
        self, messages: list[AIMessage], system_prompt: Optional[str] = None
    ) -> list[dict[str, str]]:
        """Construct payload messages, deduplicating consecutive system instructions."""
        payload_messages: list[dict[str, str]] = []
        if system_prompt and system_prompt.strip():
            payload_messages.append({"role": "system", "content": system_prompt.strip()})

        for msg in messages:
            content = msg.content.strip() if msg.content else ""
            if not content:
                continue
            if msg.role == "system" and payload_messages and payload_messages[-1]["role"] == "system":
                # Merge into existing single system instruction
                payload_messages[-1]["content"] += "\n\n" + content
                continue
            payload_messages.append({"role": msg.role, "content": content})

        return payload_messages

    async def chat(
        self,
        model: str,
        messages: list[AIMessage],
        temperature: float = 0.3,
        system_prompt: Optional[str] = None,
        timeout: Optional[float] = None,
        num_ctx: int = 2048,
        num_predict: int = 128,
        cpu_threads: int = 2,
        repeat_penalty: float = 1.15,
        options: Optional[dict[str, Any]] = None,
    ) -> AIResponse:
        """Send a chat completion request to Ollama and return the full response."""
        req_timeout = timeout or self.timeout_seconds
        start_time = time.perf_counter()

        payload_messages = self._build_payload_messages(messages, system_prompt)

        req_options: dict[str, Any] = {
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "temperature": temperature,
            "num_thread": cpu_threads,
            "repeat_penalty": repeat_penalty,
        }
        if options:
            req_options.update(options)

        est_tokens = estimate_messages_tokens(messages, system_prompt)
        logger.info(
            "Ollama chat request: model=%s, est_tokens=%d, num_ctx=%d, num_predict=%d, temp=%.2f, threads=%d",
            model,
            est_tokens,
            req_options.get("num_ctx", 2048),
            req_options.get("num_predict", 128),
            req_options.get("temperature", 0.3),
            req_options.get("num_thread", 2),
        )

        body = {
            "model": model,
            "messages": payload_messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": req_options,
        }

        try:
            async with self._get_client(req_timeout) as client:
                resp = await client.post(f"{self.base_url}/api/chat", json=body)

                if resp.status_code != 200:
                    self._handle_error_response(resp.status_code, resp.text, model)

                data = resp.json()
                latency_ms = (time.perf_counter() - start_time) * 1000.0

                content = data.get("message", {}).get("content", "")
                prompt_eval_count = data.get("prompt_eval_count", 0)
                eval_count = data.get("eval_count", 0)
                finish_reason = data.get("done_reason", "stop")

                logger.info(
                    "Ollama chat response: model=%s, status=200, finish_reason=%s, prompt_tokens=%d, eval_tokens=%d, latency=%.1fms",
                    model, finish_reason, prompt_eval_count, eval_count, latency_ms,
                )

                if not content and len(payload_messages) > 0:
                    logger.warning(
                        "Ollama returned empty content for model=%s, finish_reason=%s",
                        model, finish_reason,
                    )

                return AIResponse(
                    content=content,
                    model=model,
                    finish_reason=finish_reason,
                    usage=TokenUsage(
                        prompt_tokens=prompt_eval_count,
                        completion_tokens=eval_count,
                        total_tokens=prompt_eval_count + eval_count,
                    ),
                    latency_ms=latency_ms,
                    raw=data,
                )

        except httpx.ConnectError as exc:
            raise AIEngineConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. Ensure Ollama daemon is running.",
                engine="ollama",
            ) from exc
        except httpx.TimeoutException as exc:
            raise AIEngineTimeoutError(
                f"Ollama generation timed out after {req_timeout:.1f}s",
                engine="ollama",
            ) from exc

    async def chat_stream(
        self,
        model: str,
        messages: list[AIMessage],
        temperature: float = 0.3,
        system_prompt: Optional[str] = None,
        timeout: Optional[float] = None,
        num_ctx: int = 2048,
        num_predict: int = 128,
        cpu_threads: int = 2,
        repeat_penalty: float = 1.15,
        options: Optional[dict[str, Any]] = None,
    ) -> AsyncIterator[AIStreamChunk]:
        """Stream chat completion tokens asynchronously using Ollama's NDJSON endpoint."""
        req_timeout = timeout or self.timeout_seconds

        payload_messages = self._build_payload_messages(messages, system_prompt)

        req_options: dict[str, Any] = {
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "temperature": temperature,
            "num_thread": cpu_threads,
            "repeat_penalty": repeat_penalty,
        }
        if options:
            req_options.update(options)

        est_tokens = estimate_messages_tokens(messages, system_prompt)
        logger.info(
            "Ollama stream request: model=%s, est_tokens=%d, num_ctx=%d, num_predict=%d, temp=%.2f, threads=%d",
            model,
            est_tokens,
            req_options.get("num_ctx", 2048),
            req_options.get("num_predict", 128),
            req_options.get("temperature", 0.3),
            req_options.get("num_thread", 2),
        )

        body = {
            "model": model,
            "messages": payload_messages,
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": req_options,
        }

        try:
            async with self._get_client(req_timeout) as client:
                async with client.stream("POST", f"{self.base_url}/api/chat", json=body) as resp:
                    if resp.status_code != 200:
                        error_text = await resp.aread()
                        self._handle_error_response(
                            resp.status_code, error_text.decode("utf-8", errors="ignore"), model
                        )

                    async for line in resp.aiter_lines():
                        if not line or not line.strip():
                            continue

                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        delta = item.get("message", {}).get("content", "")
                        done = item.get("done", False)

                        usage = None
                        if done:
                            p_tok = item.get("prompt_eval_count", 0)
                            c_tok = item.get("eval_count", 0)
                            usage = TokenUsage(
                                prompt_tokens=p_tok,
                                completion_tokens=c_tok,
                                total_tokens=p_tok + c_tok,
                            )
                            logger.info(
                                "Ollama stream completed: model=%s, finish_reason=%s, prompt_tokens=%d, eval_tokens=%d",
                                model, item.get("done_reason"), p_tok, c_tok,
                            )

                        yield AIStreamChunk(
                            delta_text=delta,
                            is_done=done,
                            finish_reason=item.get("done_reason") if done else None,
                            usage=usage,
                        )


        except httpx.ConnectError as exc:
            raise AIEngineConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. Ensure Ollama daemon is running.",
                engine="ollama",
            ) from exc
        except httpx.TimeoutException as exc:
            raise AIEngineTimeoutError(
                f"Ollama streaming connection timed out after {req_timeout:.1f}s",
                engine="ollama",
            ) from exc

    async def unload_model(self, model: str) -> bool:
        """Explicitly unload a model from memory to reclaim RAM immediately.

        Ollama unloads a model when a chat request is sent with keep_alive: 0.
        """
        try:
            async with self._get_client(5.0) as client:
                resp = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": model,
                        "messages": [],
                        "keep_alive": 0,
                    },
                )
                logger.info("Unloaded model '%s' from RAM (HTTP %s)", model, resp.status_code)
                return resp.status_code == 200
        except Exception as exc:
            logger.warning("Failed to unload model '%s': %s", model, exc)
            return False

    async def pull_model(
        self,
        model: str,
        timeout: float = 600.0,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream progress while pulling and installing a local model via Ollama.

        Yields dictionaries with format:
        {
            "status": "downloading 0123...",
            "completed": 1234567,
            "total": 5432100,
            "percent": 22.7,
            "is_done": False
        }
        """
        try:
            async with self._get_client(timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/pull",
                    json={"name": model, "stream": True},
                ) as resp:
                    if resp.status_code != 200:
                        err_bytes = await resp.aread()
                        raise AIEngineError(
                            f"Ollama pull returned HTTP {resp.status_code}: {err_bytes.decode('utf-8', errors='ignore')}",
                            engine="ollama",
                        )

                    async for line in resp.aiter_lines():
                        if not line or not line.strip():
                            continue
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        status = item.get("status", "")
                        completed = item.get("completed", 0)
                        total = item.get("total", 0)
                        percent = round((completed / total) * 100.0, 1) if (total and total > 0 and completed) else 0.0
                        is_done = status == "success" or item.get("done", False)

                        yield {
                            "status": status,
                            "completed": completed,
                            "total": total,
                            "percent": percent,
                            "is_done": is_done,
                            "raw": item,
                        }
        except httpx.ConnectError as exc:
            raise AIEngineConnectionError(
                f"Cannot connect to Ollama at {self.base_url} to pull model '{model}'.",
                engine="ollama",
            ) from exc
        except httpx.TimeoutException as exc:
            raise AIEngineTimeoutError(
                f"Model pull timed out after {timeout:.0f}s for '{model}'.",
                engine="ollama",
            ) from exc

    async def delete_model(self, model: str) -> bool:
        """Delete an installed model to reclaim disk storage."""
        try:
            async with self._get_client(10.0) as client:
                # httpx DELETE with json body
                req = client.build_request("DELETE", f"{self.base_url}/api/delete", json={"name": model})
                resp = await client.send(req)
                if resp.status_code == 200:
                    logger.info("Deleted local model '%s'", model)
                    return True
                logger.warning("Failed to delete model '%s' (HTTP %s): %s", model, resp.status_code, resp.text)
                return False
        except Exception as exc:
            logger.warning("Error deleting model '%s': %s", model, exc)
            return False

    async def show_model_info(self, model: str) -> dict[str, Any]:
        """Fetch model parameters, quantization, and metadata from Ollama."""
        try:
            async with self._get_client(10.0) as client:
                resp = await client.post(f"{self.base_url}/api/show", json={"name": model})
                if resp.status_code == 200:
                    return resp.json()
                return {}
        except Exception as exc:
            logger.debug("Failed to retrieve model info for '%s': %s", model, exc)
            return {}
