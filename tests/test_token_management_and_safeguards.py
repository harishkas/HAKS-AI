"""Tests for Hardware-Aware Token Management, Context Safeguards, and Windows Action Integration.

Covers:
1. Hardware-aware local model limits (num_ctx=2048, num_predict=128, temperature=0.3, threads=2)
2. Token estimation functions
3. Conversation context management and safe tool output truncation
4. Safeguards against duplicate system prompts and current user request preservation
5. Response error mapping: TokenLimitError, ContextOverflowError, EmptyResponseError
6. Controlled 1-step retry with reduced context
7. Windows action execution verification, concise AI prompt, and failure fallback
8. Live integration with actual llama3.2:1b model running on local Ollama
"""

import asyncio
import json
import pytest
import httpx

from ai.exceptions import (
    ContextOverflowError,
    EmptyResponseError,
    GenerationCancelledError,
    TokenLimitError,
)
from ai.local_engine import LocalAIEngine
from ai.manager import AIEngineManager
from ai.models import (
    AIEngineConfig,
    AIMessage,
    AIResponse,
    AIStreamChunk,
    EngineType,
    estimate_messages_tokens,
    estimate_tokens,
)
from ai.ollama_client import OllamaClient
from config.settings import AppConfig
from core.app_controller import AppController
from memory.context_builder import MemoryContextBuilder
from memory.personal_memory import PersonalMemoryManager
from memory.work_memory import WorkMemoryManager
from storage.database import DatabaseManager
from windows_control.models import ActionPlan, CommandResult, ExecutionStatus


# ==============================================================================
# 1. HARDWARE-AWARE SETTINGS & TOKEN ESTIMATION
# ==============================================================================

def test_hardware_aware_config_defaults(temp_workspace: AppConfig):
    """Verify hardware-aware default limits tuned for 8 GB RAM / dual-core CPU."""
    ai_cfg = temp_workspace.ai
    assert ai_cfg.local_num_ctx == 2048
    assert ai_cfg.local_num_predict == 128
    assert ai_cfg.local_temperature == 0.3
    assert ai_cfg.local_cpu_threads == 2
    assert ai_cfg.local_repeat_penalty == 1.15


def test_app_controller_local_model_settings_persistence(
    temp_workspace: AppConfig, test_db_manager: DatabaseManager
):
    """Verify AppController gets, sets, and persists local model limits into settings.db."""
    controller = AppController(config=temp_workspace, db_manager=test_db_manager)

    # Initial settings should match defaults
    initial = controller.get_local_model_settings()
    assert initial["num_ctx"] == 2048
    assert initial["num_predict"] == 128
    assert initial["temperature"] == 0.3
    assert initial["cpu_threads"] == 2

    # Update settings
    controller.set_local_model_settings(
        num_ctx=4096,
        num_predict=256,
        temperature=0.5,
        cpu_threads=3,
        repeat_penalty=1.2,
    )

    updated = controller.get_local_model_settings()
    assert updated["num_ctx"] == 4096
    assert updated["num_predict"] == 256
    assert updated["temperature"] == 0.5
    assert updated["cpu_threads"] == 3
    assert updated["repeat_penalty"] == 1.2

    # Verify synchronization with local engine in ai_manager
    ai_mgr = controller.get_ai_manager()
    assert ai_mgr.local_engine.config.num_ctx == 4096
    assert ai_mgr.local_engine.config.num_predict == 256
    assert ai_mgr.local_engine.config.temperature == 0.5
    assert ai_mgr.local_engine.config.cpu_threads == 3


def test_token_estimation_accuracy():
    """Verify estimate_tokens and estimate_messages_tokens functions."""
    assert estimate_tokens("") == 0
    short_text = "Hello world"
    est = estimate_tokens(short_text)
    assert 2 <= est <= 5

    long_text = "A" * 350
    assert 90 <= estimate_tokens(long_text) <= 110

    messages = [
        AIMessage(role="user", content="What is Python?"),
        AIMessage(role="assistant", content="Python is a programming language."),
    ]
    total_est = estimate_messages_tokens(messages, system_prompt="You are HAKS.")
    assert total_est > 0


# ==============================================================================
# 2. CONVERSATION CONTEXT MANAGEMENT & SAFEGUARDS
# ==============================================================================

def test_conversation_context_preserves_current_user_request(
    temp_workspace: AppConfig, test_db_manager: DatabaseManager
):
    """Verify current user request is NEVER silently discarded, even under strict token budget."""
    controller = AppController(config=temp_workspace, db_manager=test_db_manager)

    # Populate multiple messages
    for i in range(10):
        controller.send_message(f"Historic turn {i} with some background text.", role="user")
        controller.send_message(f"Assistant response {i} explaining details.", role="assistant")

    current_prompt = "What was my very first question?"
    controller.send_message(current_prompt, role="user")

    # Fetch with very tight token budget
    ai_msgs = controller.get_conversation_ai_messages(limit=20, max_total_tokens=60)
    assert len(ai_msgs) >= 1
    # Latest message must be the user's current prompt
    assert ai_msgs[-1].role == "user"
    assert current_prompt in ai_msgs[-1].content


def test_conversation_context_truncates_oversized_tool_output(
    temp_workspace: AppConfig, test_db_manager: DatabaseManager
):
    """Verify oversized tool/command outputs in history are safely truncated."""
    controller = AppController(config=temp_workspace, db_manager=test_db_manager)

    controller.send_message("Run dir on system drive", role="user")

    # Enormous tool output
    huge_stdout = "Directory listing:\n" + "\n".join([f"file_{i}.txt" for i in range(200)])
    body = f"✅ **Run command** — Completed successfully\n\nOutput:\n```\n{huge_stdout}\n```"
    controller.send_message(body, role="assistant", metadata={"action_plan": "dir"})

    controller.send_message("Now summarize what you found", role="user")

    ai_msgs = controller.get_conversation_ai_messages(limit=10)
    assistant_msg = [m for m in ai_msgs if m.role == "assistant"][0]

    assert "[... output truncated" in assistant_msg.content
    assert len(assistant_msg.content) < len(body)


def test_ollama_client_deduplicates_system_instructions():
    """Verify OllamaClient merges multiple system instructions into a single unified system prompt."""
    client = OllamaClient()
    messages = [
        AIMessage(role="system", content="Memory Context: User prefers dark theme."),
        AIMessage(role="user", content="Hello"),
    ]
    payload = client._build_payload_messages(
        messages, system_prompt="You are HAKS, a helpful AI assistant."
    )

    # Must contain only ONE system message at the beginning
    system_entries = [p for p in payload if p["role"] == "system"]
    assert len(system_entries) == 1
    assert "You are HAKS" in system_entries[0]["content"]
    assert "User prefers dark theme" in system_entries[0]["content"]


# ==============================================================================
# 3. RESPONSE SAFEGUARDS & ERROR MAPPING
# ==============================================================================

@pytest.mark.asyncio
async def test_ollama_client_token_limit_error_mapping():
    """Verify Ollama HTTP 500 token limit errors are mapped to TokenLimitError."""
    def mock_handler(request: httpx.Request):
        if request.url.path == "/api/chat":
            return httpx.Response(
                500,
                text=json.dumps({"error": "prediction aborted, token limit reached"}),
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)
    client = OllamaClient(transport=transport)
    messages = [AIMessage(role="user", content="Test prompt")]

    with pytest.raises(TokenLimitError) as exc_info:
        await client.chat(model="llama3.2:1b", messages=messages)

    assert "token limit reached" in str(exc_info.value).lower()
    assert exc_info.value.engine == "ollama"


@pytest.mark.asyncio
async def test_ollama_client_token_repeat_limit_error_mapping():
    """Verify Ollama HTTP 500 token repeat limit errors are mapped to TokenLimitError."""
    def mock_handler(request: httpx.Request):
        if request.url.path == "/api/chat":
            return httpx.Response(
                500,
                text=json.dumps({"error": "prediction aborted, token repeat limit reached"}),
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)
    client = OllamaClient(transport=transport)
    messages = [AIMessage(role="user", content="Test prompt")]

    with pytest.raises(TokenLimitError) as exc_info:
        await client.chat(model="llama3.2:1b", messages=messages)

    assert "token repeat limit reached" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_ollama_client_context_overflow_error_mapping():
    """Verify Ollama context overflow errors are mapped to ContextOverflowError."""
    def mock_handler(request: httpx.Request):
        if request.url.path == "/api/chat":
            return httpx.Response(
                500,
                text=json.dumps({"error": "model context window of 2048 exceeded by input"}),
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)
    client = OllamaClient(transport=transport)
    messages = [AIMessage(role="user", content="Test prompt")]

    with pytest.raises(ContextOverflowError) as exc_info:
        await client.chat(model="llama3.2:1b", messages=messages)

    assert "context window" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_ai_manager_controlled_retry_on_token_limit(temp_workspace: AppConfig):
    """Verify AIEngineManager executes exactly ONE controlled retry with reduced context on TokenLimitError."""
    call_count = 0
    received_lengths = []

    def mock_handler(request: httpx.Request):
        nonlocal call_count
        call_count += 1
        if request.url.path == "/api/chat":
            body = json.loads(request.content)
            received_lengths.append(len(body.get("messages", [])))

            # Fail the first attempt with token limit
            if call_count == 1:
                return httpx.Response(
                    500,
                    text=json.dumps({"error": "prediction aborted, token limit reached"}),
                )
            # Second attempt succeeds
            return httpx.Response(
                200,
                json={
                    "model": "llama3.2:1b",
                    "message": {"role": "assistant", "content": "Recovery succeeded with reduced context."},
                    "done": True,
                    "done_reason": "stop",
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)
    client = OllamaClient(transport=transport)
    local_engine = LocalAIEngine(client=client)
    manager = AIEngineManager(config=temp_workspace, local_engine=local_engine)

    messages = [
        AIMessage(role="system", content="System instruction"),
        AIMessage(role="user", content="Turn 1"),
        AIMessage(role="assistant", content="Turn 1 reply"),
        AIMessage(role="user", content="Turn 2 current request"),
    ]

    response = await manager.generate(messages, engine_type="local")
    assert response.content == "Recovery succeeded with reduced context."
    assert call_count == 2
    # Verify the retry sent reduced context
    assert received_lengths[1] < received_lengths[0]


@pytest.mark.asyncio
async def test_ai_manager_cancellation_preservation(temp_workspace: AppConfig):
    """Verify user cancellation is preserved and properly handled."""
    async def delayed_handler(request: httpx.Request):
        await asyncio.sleep(2.0)
        return httpx.Response(200, json={"message": {"content": "Late"}})

    transport = httpx.MockTransport(delayed_handler)
    client = OllamaClient(transport=transport)
    local_engine = LocalAIEngine(client=client)
    manager = AIEngineManager(config=temp_workspace, local_engine=local_engine)

    async def _cancel_soon():
        await asyncio.sleep(0.1)
        manager.cancel_generation()

    asyncio.create_task(_cancel_soon())

    with pytest.raises(GenerationCancelledError):
        await manager.generate([AIMessage(role="user", content="test")], engine_type="local")


# ==============================================================================
# 4. WINDOWS ACTION INTEGRATION & VERIFICATION
# ==============================================================================

def test_action_verification_logic():
    """Verify action execution is strictly verified before claiming success."""
    # Succeeded result
    success_res = CommandResult(
        action_id="act_1",
        command="notepad",
        exit_code=0,
        stdout="Process 1234 started",
        stderr="",
        status=ExecutionStatus.COMPLETED,
    )
    assert success_res.success is True
    assert (success_res.exit_code == 0 or success_res.exit_code is None)

    # Failed result
    failed_res = CommandResult(
        action_id="act_2",
        command="nonexistent_bin",
        exit_code=1,
        stdout="",
        stderr="File not found",
        status=ExecutionStatus.FAILED,
    )
    assert failed_res.success is False
    assert failed_res.exit_code != 0


# ==============================================================================
# 5. REAL LOCAL OLLAMA INTEGRATION TESTS (llama3.2:1b)
# ==============================================================================

@pytest.mark.asyncio
async def test_real_ollama_llama32_generation():
    """Real integration test with actual llama3.2:1b through Ollama.

    Verifies that with hardware-aware settings (num_ctx=2048, num_predict=128,
    temperature=0.3, cpu_threads=2), llama3.2:1b generates cleanly without token limit abort.
    """
    client = OllamaClient(base_url="http://127.0.0.1:11434", timeout_seconds=45.0)
    if not await client.is_server_running():
        pytest.skip("Local Ollama daemon is not running.")

    models = await client.list_models()
    model_names = [m["name"] for m in models]
    if not any("llama3.2:1b" in name for name in model_names):
        pytest.skip("llama3.2:1b is not pulled in local Ollama.")

    # 1. Short text completion
    messages = [AIMessage(role="user", content="Give me a one-sentence greeting.")]
    resp = await client.chat(
        model="llama3.2:1b",
        messages=messages,
        num_ctx=2048,
        num_predict=128,
        temperature=0.3,
        cpu_threads=2,
    )

    assert resp.content.strip() != ""
    assert resp.finish_reason in ("stop", "length")
    assert resp.usage.total_tokens > 0


@pytest.mark.asyncio
async def test_real_ollama_llama32_streaming():
    """Real integration test: verify token streaming with actual llama3.2:1b."""
    client = OllamaClient(base_url="http://127.0.0.1:11434", timeout_seconds=45.0)
    if not await client.is_server_running():
        pytest.skip("Local Ollama daemon is not running.")

    models = await client.list_models()
    model_names = [m["name"] for m in models]
    if not any("llama3.2:1b" in name for name in model_names):
        pytest.skip("llama3.2:1b is not pulled in local Ollama.")

    messages = [AIMessage(role="user", content="Count from 1 to 3.")]
    chunks = []
    async for chunk in client.chat_stream(
        model="llama3.2:1b",
        messages=messages,
        num_ctx=2048,
        num_predict=128,
        temperature=0.3,
        cpu_threads=2,
    ):
        chunks.append(chunk)

    assert len(chunks) > 0
    full_text = "".join(c.delta_text for c in chunks)
    assert full_text.strip() != ""
    assert chunks[-1].is_done is True


@pytest.mark.asyncio
async def test_real_ollama_llama32_action_confirmation():
    """Real integration test: simulate Windows action completion and concise AI confirmation generation."""
    client = OllamaClient(base_url="http://127.0.0.1:11434", timeout_seconds=45.0)
    if not await client.is_server_running():
        pytest.skip("Local Ollama daemon is not running.")

    models = await client.list_models()
    model_names = [m["name"] for m in models]
    if not any("llama3.2:1b" in name for name in model_names):
        pytest.skip("llama3.2:1b is not pulled in local Ollama.")

    # Concise action summary passed to AI
    action_result = "Action 'Open Notepad' was executed successfully. Process started with PID 4321."
    messages = [
        AIMessage(
            role="user",
            content=f"Summarize this completed Windows action in one brief confirmation sentence for the user:\n{action_result}",
        )
    ]

    resp = await client.chat(
        model="llama3.2:1b",
        messages=messages,
        num_ctx=2048,
        num_predict=64,
        temperature=0.3,
        cpu_threads=2,
    )

    assert resp.content.strip() != ""
    assert resp.finish_reason in ("stop", "length")

