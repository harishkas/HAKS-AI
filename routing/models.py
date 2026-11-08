"""Models for AI Engine Routing and Policy Enforcement."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

from privacy.classifier import PrivacyClassificationResult, PrivacyLevel
from privacy.consent_manager import ConsentRequest
from privacy.redactor import RedactionResult
from routing.complexity import ComplexityCategory, ComplexityResult


class RoutingMode(str, Enum):
    """User-selected routing strategy."""

    AUTO = "AUTO"                # Intelligent routing based on privacy & complexity
    MANUAL_LOCAL = "LOCAL"       # User manually selected Local AI
    MANUAL_CLOUD = "CLOUD"       # User manually selected Cloud AI
    FORCE_LOCAL = "FORCE_LOCAL"  # Strict offline / local-only mode enforced


class RoutingDecision(BaseModel):
    """Result of intelligent routing decision."""

    selected_engine: str  # "local" or "cloud"
    selected_model: str   # Model identifier
    routing_mode: RoutingMode

    privacy: PrivacyClassificationResult
    complexity: ComplexityResult

    requires_consent: bool = False
    consent_request: Optional[ConsentRequest] = None

    requires_redaction: bool = False
    redaction_preview: Optional[RedactionResult] = None

    routing_reason: str
    is_engine_available: bool = True
    availability_message: str = "Ready"

    # Fallback protection (NEVER silent)
    is_fallback_needed: bool = False
    fallback_engine: Optional[str] = None
    fallback_reason: Optional[str] = None

    @property
    def privacy_level(self) -> PrivacyLevel:
        """Alias for privacy.level."""
        return self.privacy.level

    @property
    def reason(self) -> str:
        """Alias for routing_reason."""
        return self.routing_reason

    @property
    def display_badge(self) -> str:
        """Formatted text for UI status bar."""
        eng = "Local AI" if self.selected_engine == "local" else "Cloud AI"
        return f"[{eng} ({self.selected_model}) | {self.privacy.level.value} | {self.complexity.category.value}]"
