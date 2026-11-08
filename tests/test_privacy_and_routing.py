"""Comprehensive unit and integration tests for HAKS Phase 5.

Covers:
- Privacy classification (PUBLIC, PERSONAL, SENSITIVE)
- Complexity classification (9 categories, offline deterministic heuristics)
- Data redaction (emails, phones, API keys, passwords, credit card Luhn check)
- User consent lifecycle (single-use, non-blanket, cancellation, consumption)
- AI Engine Router decision matrix (forced local, privacy preservation, non-silent fallback)
- Memory context privacy filtering (strictly omitting SENSITIVE items for cloud)
- UI Dialog verification (FallbackApprovalDialog and ConsentDialog)
"""

import pytest
from PySide6.QtWidgets import QApplication

from config.settings import AppConfig
from core.app_controller import AppController
from memory.context_builder import MemoryContextBuilder
from privacy.classifier import (
    PrivacyClassifier,
    PrivacyLevel,
    luhn_checksum,
    calculate_shannon_entropy,
)
from privacy.consent_manager import ConsentDecision, ConsentManager, ConsentRequest
from privacy.redactor import DataRedactor
from routing.complexity import ComplexityCategory, ComplexityClassifier
from routing.models import RoutingDecision, RoutingMode
from routing.router import AIEngineRouter
from storage.database import DatabaseManager
from ui.dialogs.consent_dialog import ConsentDialog, FallbackApprovalDialog


# ------------------------------------------------------------------------------
# 1. Privacy Classifier Tests
# ------------------------------------------------------------------------------

def test_privacy_classifier_public():
    """Verify general knowledge and non-confidential queries are classified as PUBLIC."""
    classifier = PrivacyClassifier()

    queries = [
        "What is the speed of light in vacuum?",
        "How do I sort a list of tuples in Python by the second element?",
        "Explain the difference between TCP and UDP protocols.",
        "Write a quicksort implementation in C++",
    ]
    for q in queries:
        res = classifier.classify(q)
        assert res.level == PrivacyLevel.PUBLIC, f"Expected PUBLIC for '{q}', got {res.level}"
        assert res.confidence >= 0.9


def test_privacy_classifier_personal():
    """Verify personal preferences, habits, routines, and family contexts are PERSONAL."""
    classifier = PrivacyClassifier()

    personal_queries = [
        "My daily routine starts at 6:30 AM with a 5km run.",
        "I prefer dark mode in all my IDEs and drink black coffee.",
        "My wife and daughter are traveling to London next Friday.",
        "My personal goal for this year is to publish a tech blog.",
        "Call me at +1 555-0199 if you have any questions.",
    ]
    for q in personal_queries:
        res = classifier.classify(q)
        assert res.level == PrivacyLevel.PERSONAL, f"Expected PERSONAL for '{q}', got {res.level}"
        assert len(res.detected_categories) > 0


def test_privacy_classifier_sensitive_credentials():
    """Verify API keys, tokens, passwords, and connection strings are classified as SENSITIVE."""
    classifier = PrivacyClassifier()

    sensitive_queries = [
        "Here is my OpenAI key sk-1234567890abcdef1234567890abcdef for testing",
        "Gemini key AIzaSyA1B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6Q",
        "Connect using postgres://admin:SuperSecretPass123@prod.db.internal:5432/core",
        "password := 'UltraSecretMasterKey987!'",
        "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.doNotShareThisSignature",
        "SSN: 000-12-3456",
        "Patient was diagnosed with cancer and prescribed chemotherapy last week",
        "This contract is attorney-client privileged and strictly confidential under NDA",
    ]
    for q in sensitive_queries:
        res = classifier.classify(q)
        assert res.level == PrivacyLevel.SENSITIVE, f"Expected SENSITIVE for '{q}', got {res.level}"


def test_luhn_checksum_and_credit_card_detection():
    """Verify Luhn algorithm validation distinguishes valid and invalid card numbers."""
    # Standard valid Visa test number
    valid_card = "4532015112830366"
    assert luhn_checksum(valid_card) is True

    # Invalid card number (checksum fails)
    invalid_card = "4532015112830367"
    assert luhn_checksum(invalid_card) is False

    classifier = PrivacyClassifier()
    res_valid = classifier.classify(f"Please charge card {valid_card}")
    assert res_valid.level == PrivacyLevel.SENSITIVE
    assert any("credit_card" in cat for cat in res_valid.detected_categories)


def test_shannon_entropy_calculation():
    """Verify high-entropy strings trigger secret detection."""
    low_entropy = "aaaaaaaaaaaaaaaa"
    high_entropy = "9xK#mP2$vL8!qW5*zR1&A9bC3d"

    assert calculate_shannon_entropy(low_entropy) < 2.0
    assert calculate_shannon_entropy(high_entropy) > 3.5

    classifier = PrivacyClassifier()
    res = classifier.classify(f"Use random secret key {high_entropy}")
    assert res.level == PrivacyLevel.SENSITIVE


# ------------------------------------------------------------------------------
# 2. Complexity Classifier Tests
# ------------------------------------------------------------------------------

def test_complexity_simple_queries():
    """Verify short conversational queries are classified as SIMPLE."""
    classifier = ComplexityClassifier()

    simple_prompts = [
        "hi",
        "hello",
        "what is the time?",
        "thank you",
        "good morning HAKS",
    ]
    for p in simple_prompts:
        res = classifier.classify(p)
        assert res.category == ComplexityCategory.SIMPLE
        assert res.is_complex is False


def test_complexity_programming_and_architecture():
    """Verify coding and architecture requests are detected as complex tasks."""
    classifier = ComplexityClassifier()

    code_prompt = """
    def merge_sort(arr):
        if len(arr) <= 1:
            return arr
        mid = len(arr) // 2
        left = merge_sort(arr[:mid])
        right = merge_sort(arr[mid:])
        return merge(left, right)
    """
    res_code = classifier.classify(code_prompt)
    assert res_code.category in (ComplexityCategory.PROGRAMMING, ComplexityCategory.CODE_ANALYSIS)

    arch_prompt = "Design a high-availability microservices architecture with Kafka event streaming and Redis caching."
    res_arch = classifier.classify(arch_prompt)
    assert res_arch.is_complex is True
    assert res_arch.category == ComplexityCategory.TECHNICAL_ARCHITECTURE


def test_complexity_stack_trace_code_analysis():
    """Verify stack traces trigger CODE_ANALYSIS classification."""
    classifier = ComplexityClassifier()

    stack_trace = """
    Traceback (most recent call last):
      File "app/service.py", line 42, in process_event
        val = payload["missing_key"]
    KeyError: 'missing_key'
    """
    res = classifier.classify(stack_trace)
    assert res.is_complex is True
    assert res.category == ComplexityCategory.CODE_ANALYSIS


def test_complexity_multistep_problem_solving():
    """Verify enumerated step prompts trigger MULTISTEP_PROBLEM_SOLVING."""
    classifier = ComplexityClassifier()

    multistep = "Step 1: Parse incoming CSV. Step 2: Validate email formats. Step 3: Insert into SQLite database."
    res = classifier.classify(multistep)
    assert res.is_complex is True
    assert res.category == ComplexityCategory.MULTISTEP_PROBLEM_SOLVING


# ------------------------------------------------------------------------------
# 3. Data Redactor Tests
# ------------------------------------------------------------------------------

def test_data_redactor_masks_identifiers():
    """Verify emails, phones, API keys, credentials, and cards are masked while preserving structure."""
    redactor = DataRedactor()

    text = (
        "Send the report to alice@example.com and call +1 555-0199. "
        "Use API key sk-abcdef12345678901234567890 and password: 'SecretPassword99'."
    )
    result = redactor.redact(text)

    assert result.has_redactions is True
    assert result.redactions_count >= 3
    assert "[EMAIL_REDACTED]" in result.sanitized_text
    assert "[OPENAI_KEY_REDACTED]" in result.sanitized_text or "[API_KEY_REDACTED]" in result.sanitized_text
    assert "[PASSWORD_REDACTED]" in result.sanitized_text
    assert "alice@example.com" not in result.sanitized_text
    assert "sk-abcdef12345678901234567890" not in result.sanitized_text
    assert "SecretPassword99" not in result.sanitized_text


def test_data_redactor_clean_text_unchanged():
    """Verify clean public text has zero redactions."""
    redactor = DataRedactor()
    text = "Explain the difference between mutable and immutable data structures in Python."
    result = redactor.redact(text)

    assert result.has_redactions is False
    assert result.sanitized_text == text
    assert len(result.items) == 0


# ------------------------------------------------------------------------------
# 4. User Consent Manager Tests
# ------------------------------------------------------------------------------

def test_consent_manager_lifecycle():
    """Verify consent request creation, recording decisions, and single-use consumption."""
    manager = ConsentManager()

    req = manager.create_consent_request(
        destination_provider="Google Gemini",
        selected_model="gemini-2.5-flash",
        request_summary="Explain personal diet routine",
        privacy_level=PrivacyLevel.PERSONAL,
        detected_categories=["personal_preference"],
        original_data="My personal diet is low carb",
        transmitted_data="My personal diet is low carb",
        is_redacted=False,
    )

    assert req.decision == ConsentDecision.PENDING
    assert req.is_consumed is False

    # Retrieve active
    retrieved = manager.get_consent_request(req.id)
    assert retrieved is not None
    assert retrieved.id == req.id

    # Record approval
    manager.record_decision(req.id, ConsentDecision.APPROVED)
    assert manager.is_approved(req.id) is True

    # Single-use consumption
    assert manager.consume_consent(req.id) is True
    # Second consumption must fail (non-blanket, single-use only)
    assert manager.consume_consent(req.id) is False
    assert manager.is_approved(req.id) is False


def test_consent_manager_cancellation():
    """Verify cancelled or 'use local' consent decisions."""
    manager = ConsentManager()

    req = manager.create_consent_request(
        destination_provider="OpenAI",
        selected_model="gpt-4o",
        request_summary="Financial summary",
        privacy_level=PrivacyLevel.SENSITIVE,
        original_data="Sensitive text",
        transmitted_data="Sensitive text",
    )

    manager.record_decision(req.id, ConsentDecision.CANCELLED)
    assert manager.is_approved(req.id) is False

    # USE_LOCAL decision
    req2 = manager.create_consent_request(
        destination_provider="Anthropic",
        selected_model="claude-3-7-sonnet",
        request_summary="Work notes",
        privacy_level=PrivacyLevel.PERSONAL,
        original_data="Work notes",
        transmitted_data="Work notes",
    )
    manager.record_decision(req2.id, ConsentDecision.USE_LOCAL)
    assert manager.is_approved(req2.id) is False


# ------------------------------------------------------------------------------
# 5. AI Engine Router Policy Matrix Tests
# ------------------------------------------------------------------------------

def test_router_sensitive_forces_local(controller: AppController):
    """Verify sensitive content is strictly forced to Local AI regardless of mode."""
    router = controller.get_router()

    prompt = "Here is my secret sk-1234567890abcdef1234567890abcdef please analyze it"

    # Even if MANUAL_CLOUD is requested, router must force local
    decision = router.route(prompt, mode=RoutingMode.MANUAL_CLOUD)
    assert decision.selected_engine == "local"
    assert decision.privacy_level == PrivacyLevel.SENSITIVE
    assert "sensitive" in decision.reason.lower()


def test_router_personal_prefers_local(controller: AppController):
    """Verify personal requests route to Local AI in AUTO mode."""
    router = controller.get_router()

    prompt = "My daily routine starts at 7am and I like to read books."
    decision = router.route(prompt, mode=RoutingMode.AUTO)

    assert decision.selected_engine == "local"
    assert decision.privacy_level == PrivacyLevel.PERSONAL


def test_router_public_complex_routes_to_cloud(controller: AppController):
    """Verify complex public reasoning routes to Cloud AI in AUTO mode."""
    router = controller.get_router()

    prompt = "Design a distributed event-driven microservice architecture with Kafka and Elasticsearch."
    decision = router.route(prompt, mode=RoutingMode.AUTO)

    assert decision.selected_engine == "cloud"
    assert decision.privacy_level == PrivacyLevel.PUBLIC
    assert decision.complexity.is_complex is True


def test_router_public_simple_routes_to_local(controller: AppController):
    """Verify simple public queries route to Local AI in AUTO mode."""
    router = controller.get_router()

    prompt = "Hello HAKS"
    decision = router.route(prompt, mode=RoutingMode.AUTO)

    assert decision.selected_engine == "local"
    assert decision.privacy_level == PrivacyLevel.PUBLIC
    assert decision.complexity.category == ComplexityCategory.SIMPLE


def test_router_explicit_local_override(controller: AppController):
    """Verify explicit local instructions force Local AI."""
    router = controller.get_router()

    prompt = "offline only please design a high-throughput microservices architecture"
    decision = router.route(prompt, mode=RoutingMode.AUTO)

    assert decision.selected_engine == "local"
    assert "local" in decision.reason.lower()


def test_router_explicit_cloud_with_consent(controller: AppController):
    """Verify explicit cloud queries require consent and apply redaction preview."""
    router = controller.get_router()

    prompt = "ask gemini to review this personal email: my email is dev@haks.internal"
    decision = router.route(prompt, mode=RoutingMode.AUTO)

    assert decision.selected_engine == "cloud"
    assert decision.requires_consent is True
    assert decision.requires_redaction is True
    assert decision.consent_request is not None
    assert "[EMAIL_REDACTED]" in decision.consent_request.transmitted_data


def test_router_non_silent_fallback(controller: AppController):
    """Verify router flags unavailable engines explicitly without silent switching."""
    router = controller.get_router()

    # When Ollama is not running in test environment, targeting local triggers is_fallback_needed
    prompt = "Hello simple task"
    decision = router.route(prompt, mode=RoutingMode.AUTO)

    # Ollama is not running locally during tests
    if not decision.is_fallback_needed:
        # If ollama happens to be running on the machine, verify consistency
        assert decision.selected_engine == "local"
    else:
        assert decision.is_fallback_needed is True
        assert decision.fallback_engine == "cloud"
        assert decision.fallback_reason is not None


# ------------------------------------------------------------------------------
# 6. Memory Context Privacy Tests
# ------------------------------------------------------------------------------

def test_context_builder_privacy_omits_sensitive(controller: AppController):
    """Verify MemoryContextBuilder strictly excludes SENSITIVE memories when preparing cloud prompts."""
    # Seed personal and work memories
    controller.personal_memory.add_memory(
        title="Chess Hobby",
        content="Normal preference: I enjoy playing chess on weekends.",
        category="preference",
        tags=["hobby"],
    )
    controller.personal_memory.add_memory(
        title="Health Diagnosis",
        content="Confidential health note: diagnosed with hypertension and taking lisinopril.",
        category="health",
        tags=["medical"],
    )
    controller.work_memory.add_memory(
        title="Cloud Access Key",
        content="API secret: aws_access_key AKIA1234567890ABCDEF for staging.",
        project_name="infrastructure",
        tags=["aws"],
    )

    # 1. Build context for local target: should include personal and sensitive
    local_ctx_chess = controller.context_builder.build_context(
        query="chess",
        is_cloud_target=False,
    )
    assert "chess" in local_ctx_chess.lower()

    local_ctx_health = controller.context_builder.build_context(
        query="hypertension",
        is_cloud_target=False,
    )
    assert "hypertension" in local_ctx_health.lower()

    # 2. Build context for cloud target: MUST strictly omit SENSITIVE memories
    cloud_ctx = controller.context_builder.build_context(
        query="hypertension lisinopril AKIA1234567890ABCDEF",
        is_cloud_target=True,
        redact_for_cloud=True,
    )
    assert "hypertension" not in cloud_ctx.lower()
    assert "lisinopril" not in cloud_ctx.lower()
    assert "akia1234567890abcdef" not in cloud_ctx.lower()


# ------------------------------------------------------------------------------
# 7. UI Dialogs Verification
# ------------------------------------------------------------------------------

def test_fallback_approval_dialog_ui(qapp):
    """Verify FallbackApprovalDialog components and explicit approval workflow."""
    dialog = FallbackApprovalDialog(
        target_engine="local",
        fallback_engine="cloud",
        reason="Ollama service is not running on 127.0.0.1:11434",
    )

    assert dialog.target_engine == "local"
    assert dialog.fallback_engine == "cloud"
    assert dialog.approved_fallback is False

    # Simulate user clicking Switch Engine
    dialog.btn_switch.click()
    assert dialog.approved_fallback is True


def test_consent_dialog_ui(qapp):
    """Verify ConsentDialog renders provider, model, original vs sanitized tabs, and handles actions."""
    consent_req = ConsentRequest(
        destination_provider="Google Gemini",
        selected_model="gemini-2.5-flash",
        request_summary="Explain system architecture",
        privacy_level=PrivacyLevel.PERSONAL,
        detected_categories=["personal_contact"],
        original_data="Please review my email: admin@haks.internal",
        transmitted_data="Please review my email: [EMAIL_REDACTED]",
        is_redacted=True,
    )

    dialog = ConsentDialog(consent_req)
    assert dialog.consent_request.id == consent_req.id
    assert dialog.decision == ConsentDecision.PENDING

    # Test "Use Local AI Instead" button
    dialog.btn_use_local.click()
    assert dialog.decision == ConsentDecision.USE_LOCAL

    # Test "Approve This Request" button
    dialog2 = ConsentDialog(consent_req)
    dialog2.btn_approve.click()
    assert dialog2.decision == ConsentDecision.APPROVED

    # Test "Cancel Transmission" button
    dialog3 = ConsentDialog(consent_req)
    dialog3.btn_cancel.click()
    assert dialog3.decision == ConsentDecision.CANCELLED
