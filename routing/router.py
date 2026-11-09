"""AI Engine Router for HAKS.

Intelligently routes requests to Local AI (Ollama) or Cloud AI based on:
1. Privacy classification (PUBLIC, PERSONAL, SENSITIVE).
2. Request complexity (Simple, Programming, Architecture, Research).
3. User instructions (explicit local-only / cloud requests).
4. Engine availability and health checks.
5. User consent management.

Guarantees:
- Never silently switches between engines.
- Sensitive requests are forced to Local AI by default.
- Cloud requests with personal or sensitive data require explicit approval and redaction preview.
"""

import re
from typing import Optional

from ai.manager import AIEngineManager
from core.logger import get_logger
from privacy.classifier import PrivacyClassifier, PrivacyLevel, get_privacy_classifier
from privacy.consent_manager import ConsentDecision, ConsentManager, get_consent_manager
from privacy.redactor import DataRedactor, get_data_redactor
from routing.complexity import ComplexityCategory, ComplexityClassifier, get_complexity_classifier
from routing.models import RoutingDecision, RoutingMode

logger = get_logger("routing.router")

LOCAL_INSTRUCTION_PATTERN = re.compile(
    r"(?i)\b(offline only|local only|use local|don't use cloud|do not use cloud|run locally|strictly local|keep this local)\b"
)
CLOUD_INSTRUCTION_PATTERN = re.compile(
    r"(?i)\b(use cloud|ask gemini|ask openai|ask claude|send to cloud|use cloud ai)\b"
)


class AIEngineRouter:
    """Intelligent router orchestrating privacy, complexity, and engine selection."""

    def __init__(
        self,
        ai_manager: AIEngineManager,
        privacy_classifier: Optional[PrivacyClassifier] = None,
        complexity_classifier: Optional[ComplexityClassifier] = None,
        redactor: Optional[DataRedactor] = None,
        consent_manager: Optional[ConsentManager] = None,
        config: Optional[Any] = None,
    ) -> None:
        self.ai_manager = ai_manager
        self.config = config or getattr(ai_manager, "app_config", None)
        self.privacy_classifier = privacy_classifier or get_privacy_classifier()
        self.complexity_classifier = complexity_classifier or get_complexity_classifier()
        self.redactor = redactor or get_data_redactor()
        self.consent_manager = consent_manager or get_consent_manager()

    def route(
        self,
        user_prompt: str,
        mode: RoutingMode = RoutingMode.AUTO,
        retrieved_memory_context: str = "",
        is_online: Optional[bool] = None,
    ) -> RoutingDecision:
        """Evaluate prompt and memory context to produce an authoritative routing decision."""
        # 1. Classify prompt and combined context privacy
        combined_text = f"{user_prompt}\n{retrieved_memory_context}".strip()
        privacy_res = self.privacy_classifier.classify(combined_text)

        # 2. Classify reasoning complexity
        complexity_res = self.complexity_classifier.classify(user_prompt)

        # 3. Check for explicit local or cloud keywords in prompt
        has_local_kw = bool(LOCAL_INSTRUCTION_PATTERN.search(user_prompt))
        has_cloud_kw = bool(CLOUD_INSTRUCTION_PATTERN.search(user_prompt))

        # Determine target engine candidate
        target_engine = "local"
        reason = ""
        requires_consent = False
        requires_redaction = False

        # RULE 1: Explicit Local instruction or FORCE_LOCAL mode
        if mode == RoutingMode.FORCE_LOCAL or has_local_kw or mode == RoutingMode.MANUAL_LOCAL:
            target_engine = "local"
            reason = "Local AI selected due to user override or explicit local-only instruction."

        # RULE 2: SENSITIVE privacy level forces Local AI
        elif privacy_res.level == PrivacyLevel.SENSITIVE:
            target_engine = "local"
            if mode == RoutingMode.MANUAL_CLOUD or has_cloud_kw:
                reason = "Privacy Policy Override: Query contains sensitive credentials/secrets. Cloud transmission is blocked; routed to Local AI."
            else:
                reason = "Sensitive information detected (credentials, keys, or financial identifiers). Routed to Local AI for zero-leakage security."

        # RULE 3: PERSONAL privacy level prefers Local AI
        elif privacy_res.level == PrivacyLevel.PERSONAL:
            if mode == RoutingMode.MANUAL_CLOUD or has_cloud_kw:
                target_engine = "cloud"
                requires_consent = True
                requires_redaction = True
                reason = "Personal context detected with explicit cloud request. User consent and redaction required before transmission."
            else:
                target_engine = "local"
                reason = "Personal preference or history detected. Preferring Local AI to protect personal privacy."

        # RULE 4: PUBLIC privacy level
        else:
            if mode == RoutingMode.MANUAL_CLOUD or has_cloud_kw:
                target_engine = "cloud"
                reason = "Cloud AI selected via user request for public query."
            elif mode == RoutingMode.AUTO:
                if is_online is False:
                    target_engine = "local"
                    reason = "System is offline. Routed to Local AI (Ollama) for offline operation."
                else:
                    # Gemini-first configuration: default to Google Gemini for normal requests
                    app_cfg = self.config or getattr(self.ai_manager, "app_config", None) or getattr(self.ai_manager, "config", None)
                    ai_cfg = getattr(app_cfg, "ai", None) if app_cfg else None
                    default_eng = getattr(ai_cfg, "default_engine", None) or getattr(app_cfg, "default_engine", "cloud")
                    if default_eng == "local":
                        if complexity_res.is_complex:
                            target_engine = "cloud"
                            reason = f"Complex task ({complexity_res.category.value}) routed to Cloud AI for high-capacity reasoning."
                        else:
                            target_engine = "local"
                            reason = f"Routine query ({complexity_res.category.value}) routed to Local AI for privacy, speed, and zero cloud token usage."
                    else:
                        target_engine = "cloud"
                        reason = f"Gemini-first policy: Query ({complexity_res.category.value}) routed to Google Gemini by default."

        # Redaction check for cloud targets
        redaction_preview = None
        if target_engine == "cloud":
            redaction_res = self.redactor.redact(user_prompt)
            if redaction_res.has_redactions:
                requires_redaction = True
                requires_consent = True
                redaction_preview = redaction_res
                reason += f" Masked {redaction_res.redactions_count} sensitive identifier(s)."

        # Determine model
        if target_engine == "local":
            model_name = getattr(self.ai_manager.local_engine, "current_model", "llama3.2:1b")
        else:
            model_name = getattr(self.ai_manager.cloud_engine, "current_model", "gemini-2.5-flash")

        # Engine Availability & Fallback Protection (NEVER silent, NEVER auto-fallback to Ollama in Gemini-first)
        health_info = self.ai_manager.get_engine_health()
        target_health = health_info.get(target_engine)
        is_available = target_health.is_healthy if target_health else False
        avail_msg = target_health.status_message if target_health else "Engine not configured"

        is_fallback_needed = False
        fallback_engine = None
        fallback_reason = None

        if target_engine == "cloud" and is_online is False:
            is_available = False
            avail_msg = "Internet connection is offline. Cloud requests cannot be sent."
            is_fallback_needed = True
            fallback_engine = "local"
            fallback_reason = "System is offline. Cloud AI is unavailable without internet connectivity."
        elif not is_available:
            is_fallback_needed = True
            if target_engine == "cloud":
                # Do not silently or automatically switch to Ollama when Gemini fails
                fallback_engine = None
                fallback_reason = f"Cloud AI (Gemini) is unavailable: {avail_msg}. Please verify API key, network, or quota, and click Retry."
            else:
                fallback_engine = "cloud"
                fallback_reason = f"Target engine '{target_engine}' is unavailable: {avail_msg}."

        # Prepare consent packet if consent is required
        consent_packet = None
        if requires_consent:
            transmitted = redaction_preview.sanitized_text if redaction_preview else user_prompt
            consent_packet = self.consent_manager.create_consent_request(
                destination_provider=self.ai_manager.cloud_engine.provider_name if target_engine == "cloud" else "Local Ollama",
                selected_model=model_name,
                request_summary=user_prompt[:80] + ("..." if len(user_prompt) > 80 else ""),
                privacy_level=privacy_res.level,
                detected_categories=privacy_res.detected_categories,
                original_data=user_prompt,
                transmitted_data=transmitted,
                is_redacted=bool(redaction_preview and redaction_preview.has_redactions),
            )

        decision = RoutingDecision(
            selected_engine=target_engine,
            selected_model=model_name,
            routing_mode=mode,
            privacy=privacy_res,
            complexity=complexity_res,
            requires_consent=requires_consent,
            consent_request=consent_packet,
            requires_redaction=requires_redaction,
            redaction_preview=redaction_preview,
            routing_reason=reason,
            is_engine_available=is_available,
            availability_message=avail_msg,
            is_fallback_needed=is_fallback_needed,
            fallback_engine=fallback_engine,
            fallback_reason=fallback_reason,
        )

        logger.info(
            "Routing Decision: %s -> %s (%s) [Privacy: %s, Complexity: %s]",
            mode.value,
            decision.selected_engine,
            decision.selected_model,
            decision.privacy.level.value,
            decision.complexity.category.value,
        )
        return decision
