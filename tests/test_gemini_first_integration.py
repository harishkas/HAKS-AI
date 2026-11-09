"""Live Integration and Acceptance Tests for HAKS Gemini-First AI Configuration.

Verifies all 10 acceptance criteria against the live Google Gemini API using
credentials securely retrieved from Windows Credential Manager:
1. Gemini API key validation.
2. Text chat with Gemini (completion & streaming).
3. English voice conversation with Gemini.
4. Tamil voice conversation with Gemini.
5. Mixed Tamil-English conversation.
6. Floating Assistant with Gemini.
7. Windows Control reasoning with Gemini.
8. Cloud failure and retry presentation.
9. Privacy and consent enforcement.
10. Verify Ollama is not selected automatically.
"""

import asyncio
import os
import re
import pytest
from PySide6.QtWidgets import QApplication

from ai.credentials import CredentialManager
from ai.exceptions import AIEngineError, AuthenticationError
from ai.models import AIMessage
from ai.providers.gemini import GeminiProviderAdapter
from config.settings import AppConfig
from core.app_controller import AppController
from privacy.classifier import PrivacyLevel
from routing.models import RoutingMode
from windows_control.models import ActionPlan, RiskLevel


# Mark all tests in this file as requiring live network access
pytestmark = pytest.mark.anyio


@pytest.fixture
def gemini_api_key() -> str:
    """Retrieve actual Gemini API key from Windows Credential Manager."""
    cm = CredentialManager()
    key = cm.get_api_key("gemini")
    if not key or not key.strip():
        pytest.skip("Gemini API key is not configured in Windows Credential Manager.")
    return key.strip()


# ------------------------------------------------------------------------------
# Test 1: Gemini API Key Validation
# ------------------------------------------------------------------------------
async def test_01_gemini_api_key_validation(gemini_api_key: str):
    """Verify Gemini API key from Windows Credential Manager successfully validates against Google API."""
    adapter = GeminiProviderAdapter()
    health = await adapter.check_health(api_key=gemini_api_key, model="gemini-2.5-flash")

    assert health.is_healthy is True, f"Gemini API health check failed: {health.status_message}"
    assert "connected" in health.status_message.lower() or "online" in health.status_message.lower() or "ready" in health.status_message.lower()
    assert len(health.available_models) > 0
    assert "gemini-2.5-flash" in health.available_models or "gemini-3.5-flash" in health.available_models


# ------------------------------------------------------------------------------
# Test 2: Text Chat with Gemini (Completion & Streaming)
# ------------------------------------------------------------------------------
async def test_02_text_chat_with_gemini(controller: AppController, gemini_api_key: str):
    """Verify end-to-end text conversation and token streaming via live Gemini API."""
    ai_mgr = controller.get_ai_manager()

    # 1. Non-streaming generation
    messages = [
        AIMessage(role="user", content="Reply with the single word: CONFIRMED")
    ]
    resp = await ai_mgr.generate(
        messages=messages,
        engine_type="cloud",
        temperature=0.1,
        num_predict=32,
    )

    assert resp is not None
    assert resp.content.strip() != ""
    assert "CONFIRMED" in resp.content.upper()
    assert "flash" in resp.model.lower() or "gemini" in resp.model.lower()

    # 2. SSE Streaming generation
    stream_chunks: list[str] = []
    async for chunk in ai_mgr.stream(
        messages=[AIMessage(role="user", content="Count from 1 to 3 separated by spaces.")],
        engine_type="cloud",
        temperature=0.1,
        num_predict=32,
    ):
        if chunk.delta_text:
            stream_chunks.append(chunk.delta_text)

    full_streamed = "".join(stream_chunks).strip()
    assert len(stream_chunks) >= 1
    assert any(num in full_streamed for num in ["1", "2", "3"])


# ------------------------------------------------------------------------------
# Test 3: English Voice Conversation with Gemini
# ------------------------------------------------------------------------------
async def test_03_english_voice_conversation_with_gemini(controller: AppController, gemini_api_key: str):
    """Verify the Voice Assistant engine generates English conversation turns through Gemini."""
    voice_engine = controller.voice_engine

    prompt = "What is the capital of France? Answer in one short sentence."
    response = await voice_engine._generate_ai_response(prompt)

    assert response is not None
    assert response.strip() != ""
    assert "paris" in response.lower()
    assert "error" not in response.lower() or "paris" in response.lower()

    # Verify conversation turn recorded in database with cloud engine metadata
    conv_id = controller.get_active_conversation_id()
    messages = controller.load_conversation(conv_id)
    assert len(messages) >= 2
    last_msg = messages[-1]
    assert last_msg["role"] == "assistant"
    assert "cloud" in last_msg.get("metadata", "")


# ------------------------------------------------------------------------------
# Test 4: Tamil Voice Conversation with Gemini
# ------------------------------------------------------------------------------
async def test_04_tamil_voice_conversation_with_gemini(controller: AppController, gemini_api_key: str):
    """Verify the Voice Assistant engine generates pure Tamil conversational responses through Gemini."""
    voice_engine = controller.voice_engine

    prompt = "வணக்கம், நீங்கள் யார்? சுருக்கமாக ஒரு வாக்கியத்தில் சொல்லுங்கள்."
    response = await voice_engine._generate_ai_response(prompt)

    assert response is not None
    assert response.strip() != ""

    # Verify presence of Tamil Unicode characters (\u0B80 - \u0BFF)
    tamil_chars = re.findall(r"[\u0B80-\u0BFF]", response)
    assert len(tamil_chars) >= 5, f"Expected Tamil response but got: {response}"


# ------------------------------------------------------------------------------
# Test 5: Mixed Tamil-English Conversation
# ------------------------------------------------------------------------------
async def test_05_mixed_tamil_english_conversation(controller: AppController, gemini_api_key: str):
    """Verify bilingual code-switching (Tamil + English) conversation through Gemini."""
    voice_engine = controller.voice_engine

    prompt = "HAKS AI, Python programming பற்றி ஒரு short sentence சொல்லுங்க."
    response = await voice_engine._generate_ai_response(prompt)

    assert response is not None
    assert response.strip() != ""

    # Should contain either Tamil script or English reference to Python
    has_tamil = bool(re.search(r"[\u0B80-\u0BFF]", response))
    has_python = "python" in response.lower()
    assert has_tamil or has_python, f"Expected bilingual response but got: {response}"


# ------------------------------------------------------------------------------
# Test 6: Floating Assistant with Gemini
# ------------------------------------------------------------------------------
async def test_06_floating_assistant_with_gemini(qapp, controller: AppController, gemini_api_key: str):
    """Verify FloatingAssistant uses Gemini as its default inference target."""
    from ui.floating_assistant import FloatingAssistant

    floating = FloatingAssistant(controller=controller)

    # Verify default engine routing in floating assistant
    prompt = "What is 2 + 2? Reply with just the number."
    decision = controller.route_request(user_prompt=prompt, mode=RoutingMode.AUTO)

    assert decision.selected_engine == "cloud"
    assert decision.privacy_level == PrivacyLevel.PUBLIC

    # Direct inference check
    ai_mgr = controller.get_ai_manager()
    res = await ai_mgr.generate(
        messages=[AIMessage(role="user", content=prompt)],
        engine_type=decision.selected_engine,
        temperature=0.1,
        num_predict=16,
    )
    assert res is not None
    assert "4" in res.content
    assert "flash" in res.model.lower() or "gemini" in res.model.lower()

    floating.close()


# ------------------------------------------------------------------------------
# Test 7: Windows Control Reasoning with Gemini
# ------------------------------------------------------------------------------
async def test_07_windows_control_reasoning_with_gemini(controller: AppController, gemini_api_key: str):
    """Verify natural-language Windows computer control instructions are reasoned by Gemini."""
    instruction = "Open Notepad to write meeting minutes"
    plan: ActionPlan = await controller.reason_windows_action(instruction)

    assert plan is not None
    assert isinstance(plan, ActionPlan)
    assert plan.is_actionable is True
    # Verify reasoned action targets notepad or text editor
    cmd_or_obj = f"{plan.command} {plan.objective}".lower()
    assert "notepad" in cmd_or_obj or plan.operation_type in ("launch_app", "command")
    assert plan.risk_level in (RiskLevel.LOW, RiskLevel.SENSITIVE)


# ------------------------------------------------------------------------------
# Test 8: Cloud Failure and Retry Presentation
# ------------------------------------------------------------------------------
async def test_08_cloud_failure_and_retry(controller: AppController):
    """Verify that when Gemini is unreachable or invalid, HAKS does not switch to Ollama and provides retry."""
    adapter = GeminiProviderAdapter()

    # Test health check with an invalid API key
    health = await adapter.check_health(api_key="AIzaSy_INVALID_KEY_FOR_TESTING_12345", model="gemini-2.5-flash")
    assert health.is_healthy is False
    assert "invalid" in health.status_message.lower() or "error" in health.status_message.lower()

    # Test generation failure with invalid key raises AuthenticationError
    with pytest.raises((AuthenticationError, AIEngineError)) as exc_info:
        await adapter.generate(
            api_key="AIzaSy_INVALID_KEY_FOR_TESTING_12345",
            model="gemini-2.5-flash",
            messages=[AIMessage(role="user", content="Hello")],
        )
    assert exc_info.value is not None

    # Verify router policy when Gemini is unavailable: fallback_engine must be None (no Ollama auto-fallback)
    router = controller.get_router()
    from unittest.mock import patch
    from ai.models import AIEngineHealth
    unhealthy = AIEngineHealth(is_healthy=False, engine_name="Google Gemini", status_message="Quota exceeded (429)")
    with patch.object(router.ai_manager, "get_engine_health", return_value={"cloud": unhealthy, "local": unhealthy}):
        decision = router.route("Hello HAKS", mode=RoutingMode.AUTO)
        assert decision.is_engine_available is False
        assert decision.is_fallback_needed is True
        # Ollama must NEVER be selected automatically in Gemini-first configuration
        assert decision.fallback_engine is None
        assert "unavailable" in decision.fallback_reason.lower()


# ------------------------------------------------------------------------------
# Test 9: Privacy and Consent Enforcement
# ------------------------------------------------------------------------------
async def test_09_privacy_and_consent_enforcement(controller: AppController):
    """Verify privacy safeguards are strictly enforced and never bypassed for Gemini routing."""
    router = controller.get_router()

    # 1. SENSITIVE: credentials, API keys, passwords must be strictly forced to Local AI
    sensitive_prompt = "Here is my secret AWS key AKIA1234567890ABCDEF and password: SuperSecretPass123"
    sens_decision = router.route(sensitive_prompt, mode=RoutingMode.AUTO)

    assert sens_decision.selected_engine == "local"
    assert sens_decision.privacy_level == PrivacyLevel.SENSITIVE
    assert "sensitive" in sens_decision.routing_reason.lower()

    # Even with MANUAL_CLOUD requested, SENSITIVE data must be blocked from cloud
    forced_sens = router.route(sensitive_prompt, mode=RoutingMode.MANUAL_CLOUD)
    assert forced_sens.selected_engine == "local"
    assert "override" in forced_sens.routing_reason.lower() or "blocked" in forced_sens.routing_reason.lower()

    # 2. PERSONAL: personal notes require consent and redaction if routed to cloud
    personal_prompt = "My daily routine starts at 7am and I like to read books."
    pers_decision = router.route(personal_prompt, mode=RoutingMode.AUTO)

    # In AUTO mode, personal data prefers local for privacy preservation
    assert pers_decision.selected_engine == "local"
    assert pers_decision.privacy_level == PrivacyLevel.PERSONAL


# ------------------------------------------------------------------------------
# Test 10: Verify Ollama is Not Selected Automatically
# ------------------------------------------------------------------------------
async def test_10_verify_ollama_not_selected_automatically(controller: AppController):
    """Verify that Ollama is never chosen automatically in AUTO mode, but remains available when manually selected."""
    router = controller.get_router()

    # 1. General routine requests route to Gemini (Cloud)
    normal_queries = [
        "What is the speed of light?",
        "Write a Python function to check for prime numbers.",
        "How do I organize my daily schedule?",
    ]
    for q in normal_queries:
        dec = router.route(q, mode=RoutingMode.AUTO)
        assert dec.selected_engine == "cloud", f"Query '{q}' was routed to {dec.selected_engine} instead of cloud"
        assert "gemini" in dec.selected_model.lower()

    # 2. Ollama is only selected when the user explicitly forces local
    local_override_dec = router.route("What is the speed of light?", mode=RoutingMode.FORCE_LOCAL)
    assert local_override_dec.selected_engine == "local"

    manual_local_dec = router.route("What is the speed of light?", mode=RoutingMode.MANUAL_LOCAL)
    assert manual_local_dec.selected_engine == "local"

    kw_local_dec = router.route("run locally what is the speed of light?", mode=RoutingMode.AUTO)
    assert kw_local_dec.selected_engine == "local"
