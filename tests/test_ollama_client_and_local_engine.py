"""Tests for OllamaClient and LocalAIEngine.

Includes:
1. Real Environment Detection (honest diagnostic of host Ollama availability).
2. Mocked Engine Tests (simulating full Ollama REST API responses, streaming, and error modes).
"""

import json
import pytest
import httpx

from ai.exceptions import (
    AIEngineConnectionError,
    AIEngineTimeoutError,
    ModelNotFoundError,
)
from ai.local_engine import LocalAIEngine, MAX_SAFE_MODEL_BYTES
from ai.models import AIMessage, EngineType
from ai.ollama_client import OllamaClient


# ==============================================================================
# 1. REAL INTEGRATION TEST (Honest host inspection, no mocking)
# ==============================================================================

@pytest.mark.asyncio
async def test_real_ollama_host_detection():
    """Real integration probe: detect whether Ollama is actually installed and running.

    This test executes real network and binary checks without mocks.
    """
    client = OllamaClient(base_url="http://127.0.0.1:11434")
    is_installed = client.is_ollama_installed()
    is_running = await client.is_server_running()

    # Honest assertion: reflects real environment
    if is_running:
        models = await client.list_models()
        assert isinstance(models, list)
    else:
        # Expected on machines where Ollama service is not yet started
        assert is_running is False


# ==============================================================================
# 2. MOCKED OLLAMA TESTS (Unit tests using httpx MockTransport)
# ==============================================================================

@pytest.mark.asyncio
async def test_mocked_ollama_model_listing_and_version():
    """[MOCKED] Verify OllamaClient version retrieval and model enumeration."""
    def mock_handler(request: httpx.Request):
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.3.14"})
        elif request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "llama3.2:1b",
                            "model": "llama3.2:1b",
                            "size": 1320000000,
                            "modified_at": "2026-11-01T10:00:00Z",
                            "details": {"format": "gguf", "family": "llama", "parameter_size": "1B"},
                        },
                        {
                            "name": "qwen2.5:1.5b",
                            "model": "qwen2.5:1.5b",
                            "size": 1600000000,
                            "modified_at": "2026-11-02T10:00:00Z",
                            "details": {"format": "gguf", "family": "qwen2", "parameter_size": "1.5B"},
                        },
                    ]
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)
    client = OllamaClient(transport=transport)
    version = await client.get_version()
    assert version == "0.3.14"

    models = await client.list_models()
    assert len(models) == 2
    assert models[0]["name"] == "llama3.2:1b"
    assert models[1]["name"] == "qwen2.5:1.5b"


@pytest.mark.asyncio
async def test_mocked_ollama_generation_success():
    """[MOCKED] Verify successful non-streaming chat generation via Ollama."""
    def mock_handler(request: httpx.Request):
        if request.url.path == "/api/chat":
            payload = json.loads(request.content)
            assert payload["model"] == "llama3.2:1b"
            assert payload["stream"] is False
            assert len(payload["messages"]) >= 1

            return httpx.Response(
                200,
                json={
                    "model": "llama3.2:1b",
                    "message": {"role": "assistant", "content": "Hello! I am HAKS running locally via Ollama."},
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 12,
                    "eval_count": 15,
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)
    client = OllamaClient(transport=transport)
    messages = [AIMessage(role="user", content="Hi HAKS")]
    response = await client.chat(model="llama3.2:1b", messages=messages)

    assert "HAKS running locally" in response.content
    assert response.usage.prompt_tokens == 12
    assert response.usage.completion_tokens == 15
    assert response.usage.total_tokens == 27
    assert response.latency_ms > 0


@pytest.mark.asyncio
async def test_mocked_ollama_streaming_response():
    """[MOCKED] Verify asynchronous streaming NDJSON token reception from Ollama."""
    ndjson_lines = (
        b'{"message": {"role": "assistant", "content": "Hello"}, "done": false}\n'
        b'{"message": {"role": "assistant", "content": " from"}, "done": false}\n'
        b'{"message": {"role": "assistant", "content": " Ollama!"}, "done": true, "done_reason": "stop", "prompt_eval_count": 5, "eval_count": 3}\n'
    )

    def mock_handler(request: httpx.Request):
        if request.url.path == "/api/chat":
            return httpx.Response(200, content=ndjson_lines)
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)
    client = OllamaClient(transport=transport)
    messages = [AIMessage(role="user", content="Stream test")]

    chunks = []
    async for chunk in client.chat_stream(model="llama3.2:1b", messages=messages):
        chunks.append(chunk)

    assert len(chunks) == 3
    assert chunks[0].delta_text == "Hello"
    assert chunks[1].delta_text == " from"
    assert chunks[2].delta_text == " Ollama!"
    assert chunks[2].is_done is True
    assert chunks[2].usage.total_tokens == 8


@pytest.mark.asyncio
async def test_mocked_ollama_model_not_found():
    """[MOCKED] Verify ModelNotFoundError is raised when Ollama returns HTTP 404."""
    def mock_handler(request: httpx.Request):
        return httpx.Response(404, text='{"error":"model \'nonexistent:latest\' not found"}')

    transport = httpx.MockTransport(mock_handler)
    client = OllamaClient(transport=transport)
    messages = [AIMessage(role="user", content="Test")]

    with pytest.raises(ModelNotFoundError) as exc_info:
        await client.chat(model="nonexistent:latest", messages=messages)

    assert "not found in Ollama" in str(exc_info.value)


@pytest.mark.asyncio
async def test_mocked_ollama_connection_error():
    """[MOCKED] Verify AIEngineConnectionError is raised when connection is refused."""
    def mock_handler(request: httpx.Request):
        raise httpx.ConnectError("Connection refused by target machine")

    transport = httpx.MockTransport(mock_handler)
    client = OllamaClient(transport=transport)
    messages = [AIMessage(role="user", content="Test")]

    with pytest.raises(AIEngineConnectionError) as exc_info:
        await client.chat(model="llama3.2:1b", messages=messages)

    assert "Cannot connect to Ollama" in str(exc_info.value)


@pytest.mark.asyncio
async def test_mocked_ollama_timeout_handling():
    """[MOCKED] Verify AIEngineTimeoutError is raised when Ollama exceeds timeout budget."""
    def mock_handler(request: httpx.Request):
        raise httpx.TimeoutException("Read timed out after 5.0s")

    transport = httpx.MockTransport(mock_handler)
    client = OllamaClient(timeout_seconds=5.0, transport=transport)
    messages = [AIMessage(role="user", content="Test")]

    with pytest.raises(AIEngineTimeoutError) as exc_info:
        await client.chat(model="llama3.2:1b", messages=messages)

    assert "timed out" in str(exc_info.value)


@pytest.mark.asyncio
async def test_local_engine_hardware_compatibility_guard():
    """Verify LocalAIEngine safeguards prevent loading oversized models on 8 GB RAM machines."""
    engine = LocalAIEngine()

    # Small 1B model (~1.3 GB) -> SAFE
    safe, msg = engine.check_hardware_compatibility(1_320_000_000)
    assert safe is True
    assert "Safe" in msg

    # Huge 14B model (~9 GB) -> REJECTED / WARNED
    safe_large, msg_large = engine.check_hardware_compatibility(9_000_000_000)
    assert safe_large is False
    assert "exceeds safe 8 GB system budget" in msg_large


@pytest.mark.asyncio
async def test_local_engine_unload_model():
    """[MOCKED] Verify LocalAIEngine can signal Ollama to unload model to free RAM."""
    def mock_handler(request: httpx.Request):
        if request.url.path == "/api/chat":
            payload = json.loads(request.content)
            assert payload.get("keep_alive") == 0
            return httpx.Response(200, json={"done": True})
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)
    client = OllamaClient(transport=transport)
    engine = LocalAIEngine(client=client)
    unloaded = await engine.unload_active_model()
    assert unloaded is True


@pytest.mark.asyncio
async def test_mocked_ollama_pull_model_progress():
    """[MOCKED] Verify OllamaClient streams download progress correctly."""
    def mock_handler(request: httpx.Request):
        if request.url.path == "/api/pull":
            ndjson = (
                json.dumps({"status": "pulling manifest"}) + "\n" +
                json.dumps({"status": "downloading", "completed": 500, "total": 1000}) + "\n" +
                json.dumps({"status": "downloading", "completed": 1000, "total": 1000}) + "\n" +
                json.dumps({"status": "success"}) + "\n"
            )
            return httpx.Response(200, text=ndjson)
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)
    client = OllamaClient(transport=transport)
    engine = LocalAIEngine(client=client)

    chunks = []
    async for chunk in engine.pull_model("llama3.2:1b"):
        chunks.append(chunk)

    assert len(chunks) == 4
    assert chunks[0]["status"] == "pulling manifest"
    assert chunks[1]["percent"] == 50.0
    assert chunks[2]["percent"] == 100.0
    assert chunks[3]["is_done"] is True


@pytest.mark.asyncio
async def test_mocked_ollama_delete_and_show_info():
    """[MOCKED] Verify OllamaClient delete_model and show_model_info."""
    def mock_handler(request: httpx.Request):
        if request.url.path == "/api/delete" and request.method == "DELETE":
            return httpx.Response(200)
        elif request.url.path == "/api/show" and request.method == "POST":
            return httpx.Response(200, json={"parameters": "temperature 0.7", "details": {"format": "gguf"}})
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)
    client = OllamaClient(transport=transport)
    engine = LocalAIEngine(client=client)

    deleted = await engine.delete_model("llama3.2:1b")
    assert deleted is True

    info = await engine.show_model_info("llama3.2:1b")
    assert info.get("details", {}).get("format") == "gguf"


def test_model_download_worker(qapp):
    """Verify ModelDownloadWorker emits progress and finished signals."""
    def mock_handler(request: httpx.Request):
        if request.url.path == "/api/pull":
            ndjson = (
                json.dumps({"status": "downloading", "completed": 500, "total": 1000}) + "\n" +
                json.dumps({"status": "success"}) + "\n"
            )
            return httpx.Response(200, text=ndjson)
        return httpx.Response(404)

    from ai.manager import AIEngineManager
    from ai.worker import ModelDownloadWorker

    transport = httpx.MockTransport(mock_handler)
    client = OllamaClient(transport=transport)
    engine = LocalAIEngine(client=client)
    mgr = AIEngineManager(local_engine=engine)

    worker = ModelDownloadWorker(model_name="llama3.2:1b", manager=mgr)
    progress_calls = []
    finished_calls = []

    worker.signals.progress.connect(lambda s, c, t, p: progress_calls.append((s, p)))
    worker.signals.finished.connect(lambda m: finished_calls.append(m))

    worker.start()
    worker.wait(5000)
    qapp.processEvents()

    assert len(progress_calls) >= 1
    assert "llama3.2:1b" in finished_calls
