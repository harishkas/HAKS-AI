"""User Consent Manager for HAKS.

Manages per-request explicit user consent for cloud transmissions:
- Specific to each individual request (never permanent or blanket)
- Tracks destination provider, model, detected information categories, and data payload
- Records explicit approval, cancellation, or switch to Local AI
- Invalidates used or cancelled consent immediately
"""

from datetime import datetime, timezone
from enum import Enum
import logging
from typing import Optional
import uuid
from pydantic import BaseModel, Field

from privacy.classifier import PrivacyLevel

logger = logging.getLogger("haks.privacy.consent")


class ConsentDecision(str, Enum):
    """User response to a consent request."""

    PENDING = "PENDING"
    APPROVED = "APPROVED"
    CANCELLED = "CANCELLED"
    USE_LOCAL = "USE_LOCAL"


class ConsentRequest(BaseModel):
    """Information packet presented to user for explicit cloud transmission consent."""

    id: str = Field(default_factory=lambda: f"req_{uuid.uuid4().hex[:10]}")
    destination_provider: str
    selected_model: str
    request_summary: str
    privacy_level: PrivacyLevel
    detected_categories: list[str] = Field(default_factory=list)
    original_data: str
    transmitted_data: str
    is_redacted: bool = False
    decision: ConsentDecision = ConsentDecision.PENDING
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    decided_at: Optional[str] = None
    is_consumed: bool = False


class ConsentManager:
    """Tracks and evaluates request-specific consent."""

    def __init__(self) -> None:
        self._active_requests: dict[str, ConsentRequest] = {}

    def create_consent_request(
        self,
        destination_provider: str,
        selected_model: str,
        request_summary: str,
        privacy_level: PrivacyLevel,
        original_data: str,
        transmitted_data: str,
        detected_categories: Optional[list[str]] = None,
        is_redacted: bool = False,
    ) -> ConsentRequest:
        """Create a new pending consent request."""
        req = ConsentRequest(
            destination_provider=destination_provider,
            selected_model=selected_model,
            request_summary=request_summary,
            privacy_level=privacy_level,
            detected_categories=detected_categories or [],
            original_data=original_data,
            transmitted_data=transmitted_data,
            is_redacted=is_redacted,
            decision=ConsentDecision.PENDING,
        )
        self._active_requests[req.id] = req
        logger.info(
            "Created ConsentRequest %s for provider '%s' (privacy: %s, redacted: %s)",
            req.id,
            destination_provider,
            privacy_level.value,
            is_redacted,
        )
        return req

    def get_request(self, request_id: str) -> Optional[ConsentRequest]:
        """Retrieve request by ID."""
        return self._active_requests.get(request_id)

    def get_consent_request(self, request_id: str) -> Optional[ConsentRequest]:
        """Retrieve request by ID (alias for get_request)."""
        return self.get_request(request_id)

    def record_decision(self, request_id: str, decision: ConsentDecision) -> ConsentRequest:
        """Record user's decision on a pending consent request."""
        req = self._active_requests.get(request_id)
        if not req:
            raise ValueError(f"Consent request '{request_id}' not found.")
        if req.is_consumed:
            raise ValueError(f"Consent request '{request_id}' has already been consumed.")

        req.decision = decision
        req.decided_at = datetime.now(timezone.utc).isoformat()
        logger.info("Recorded consent decision for %s: %s", request_id, decision.value)
        return req

    def is_approved(self, request_id: str) -> bool:
        """Check if request has been approved and not yet consumed."""
        req = self._active_requests.get(request_id)
        if not req:
            return False
        return req.decision == ConsentDecision.APPROVED and not req.is_consumed

    def is_consent_granted(self, request_id: str) -> bool:
        """Alias for is_approved."""
        return self.is_approved(request_id)

    def consume_consent(self, request_id: str) -> bool:
        """Verify and mark consent as consumed for a single execution."""
        req = self._active_requests.get(request_id)
        if not req:
            logger.warning("Consent consume failed: request %s not found.", request_id)
            return False
        if req.decision != ConsentDecision.APPROVED:
            logger.warning("Consent consume failed: request %s is not approved (status: %s).", request_id, req.decision.value)
            return False
        if req.is_consumed:
            logger.warning("Consent consume failed: request %s was already consumed.", request_id)
            return False

        # Mark consumed so it can never be reused
        req.is_consumed = True
        logger.info("Consent request %s consumed successfully.", request_id)
        return True

    def clear(self) -> None:
        """Clear all consent requests."""
        self._active_requests.clear()


_consent_manager_instance: Optional[ConsentManager] = None


def get_consent_manager() -> ConsentManager:
    """Return singleton ConsentManager instance."""
    global _consent_manager_instance
    if _consent_manager_instance is None:
        _consent_manager_instance = ConsentManager()
    return _consent_manager_instance
