"""Privacy Classification Engine for HAKS.

Implements deterministic rule-based privacy classification across three levels:
- PUBLIC: General knowledge, public facts, standard programming syntax.
- PERSONAL: Personal preferences, history, goals, routines, and family context.
- SENSITIVE: Passwords, API keys, bearer tokens, financial IDs, health/medical records,
  and confidential credentials.

Never relies on an LLM for privacy detection. Operates 100% offline with
sub-millisecond latency.
"""

from enum import Enum
import math
import re
from typing import Optional
from pydantic import BaseModel, Field

from core.logger import get_logger

logger = get_logger("privacy.classifier")


class PrivacyLevel(str, Enum):
    """Categorization of data privacy and confidentiality."""

    PUBLIC = "PUBLIC"
    PERSONAL = "PERSONAL"
    SENSITIVE = "SENSITIVE"


class PrivacyClassificationResult(BaseModel):
    """Result of privacy classification on a given text."""

    level: PrivacyLevel
    detected_categories: list[str] = Field(default_factory=list)
    matched_patterns: list[str] = Field(default_factory=list)
    explanation: str
    confidence: float = 1.0


def luhn_checksum(card_number_str: str) -> bool:
    """Validate a credit card number using Luhn algorithm."""
    digits = [int(c) for c in card_number_str if c.isdigit()]
    if len(digits) not in (13, 14, 15, 16, 19):
        return False
    checksum = 0
    reverse_digits = digits[::-1]
    for i, d in enumerate(reverse_digits):
        if i % 2 == 1:
            doubled = d * 2
            checksum += doubled - 9 if doubled > 9 else doubled
        else:
            checksum += d
    return checksum % 10 == 0


def calculate_shannon_entropy(token: str) -> float:
    """Calculate Shannon entropy to detect high-entropy random secrets/tokens."""
    if not token or len(token) < 16:
        return 0.0
    probabilities = [token.count(c) / len(token) for c in set(token)]
    return -sum(p * math.log2(p) for p in probabilities)


# ------------------------------------------------------------------------------
# Pattern Definitions
# ------------------------------------------------------------------------------

SENSITIVE_PATTERNS = [
    # API Keys & Secrets
    ("openai_key", re.compile(r"\bsk-[a-zA-Z0-9_\-]{20,}\b")),
    ("gemini_key", re.compile(r"\bAIzaSy[a-zA-Z0-9_\-]{33}\b")),
    ("anthropic_key", re.compile(r"\bsk-ant-[a-zA-Z0-9_\-]{20,}\b")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("generic_api_key", re.compile(r"(?i)\b(api[_\-]?key|secret[_\-]?key|access[_\-]?token)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{16,})['\"]?")),
    ("bearer_token", re.compile(r"(?i)\bBearer\s+([a-zA-Z0-9_\-\.]{20,})\b")),
    ("jwt_token", re.compile(r"\beyJ[a-zA-Z0-9_\-]{10,}\.eyJ[a-zA-Z0-9_\-]{10,}\.[a-zA-Z0-9_\-]{10,}\b")),
    ("private_key", re.compile(r"-----BEGIN (?:RSA|EC|DSA|OPENSSH|PGP|ENCRYPTED|PRIVATE) KEY-----")),
    ("password_assignment", re.compile(r"(?i)\b(password|passwd|pwd)\s*(?::=|=|:)\s*['\"]?([^\s'\"]{4,})['\"]?")),
    ("db_connection_uri", re.compile(r"(?i)(postgres|mysql|mongodb|redis|mssql):\/\/[^\s]+:[^\s]+@[^\s]+")),

    # Financial Identifiers
    ("ssn_number", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("iban_number", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")),
    ("pan_card", re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")),
    ("financial_terms", re.compile(r"(?i)\b(bank account number|routing number|credit card cvv|card security code|wire transfer pin)\b")),

    # Medical & Health Data
    ("medical_health", re.compile(r"(?i)\b(diagnosed with|prescribed medication|prescription for|medical diagnosis|blood test results|biopsy results|chemotherapy|insulin dosage|patient medical history|psychiatric evaluation|hiv status|cancer diagnosis)\b")),

    # Highly Confidential Work Information
    ("confidential_legal", re.compile(r"(?i)\b(attorney-client privileged|strictly confidential|trade secret|proprietary and confidential|under nda|non-disclosure agreement)\b")),
]

PERSONAL_PATTERNS = [
    # Preferences and Habits
    ("personal_preference", re.compile(r"(?i)\b(i prefer|my favorite|i like to|i dislike|i love|i hate|my hobby is|my habits are)\b")),
    ("daily_routine", re.compile(r"(?i)\b(my daily routine|every morning i|every evening i|i wake up at|i go to sleep at)\b")),

    # Family & Relationships
    ("family_details", re.compile(r"(?i)\b(my wife|my husband|my son|my daughter|my mother|my father|my sister|my brother|my parents|my spouse|my kids)\b")),

    # Personal Bio & Life Events
    ("personal_history", re.compile(r"(?i)\b(i was born in|my birthday is|my anniversary is|i graduated from|i live in|my home address is|my hometown is)\b")),
    ("personal_goals", re.compile(r"(?i)\b(my personal goal|my dream is to|i am planning to buy a house|i want to lose weight|my personal budget)\b")),

    # Personal Communication details
    ("personal_contact", re.compile(r"(?i)\b(call me at|my personal email is|my phone number is|my mobile number is)\b")),
    ("email_address", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,24}\b")),
]


class PrivacyClassifier:
    """Deterministic, rule-based privacy classifier."""

    def classify(self, text: str) -> PrivacyClassificationResult:
        """Classify incoming text into PUBLIC, PERSONAL, or SENSITIVE.

        Returns:
            PrivacyClassificationResult with privacy level, categories, and explanation.
        """
        if not text or not text.strip():
            return PrivacyClassificationResult(
                level=PrivacyLevel.PUBLIC,
                detected_categories=["empty"],
                explanation="Empty text classified as PUBLIC.",
                confidence=1.0,
            )

        clean_text = text.strip()
        detected_categories: list[str] = []
        matched_patterns: list[str] = []

        # ----------------------------------------------------------------------
        # 1. Check for SENSITIVE patterns first (highest priority)
        # ----------------------------------------------------------------------
        for category, regex in SENSITIVE_PATTERNS:
            match = regex.search(clean_text)
            if match:
                detected_categories.append(category)
                matched_patterns.append(match.group(0)[:30] + ("..." if len(match.group(0)) > 30 else ""))

        # Check for credit card numbers with Luhn check
        cc_candidates = re.findall(r"\b(?:\d[ -]*?){13,19}\b", clean_text)
        for cand in cc_candidates:
            clean_digits = re.sub(r"\D", "", cand)
            if luhn_checksum(clean_digits):
                detected_categories.append("credit_card_number")
                matched_patterns.append(f"Card ending in {clean_digits[-4:]}")
                break

        # Check for high-entropy tokens that look like raw secrets
        tokens = clean_text.split()
        for tok in tokens:
            stripped = tok.strip("'\":;,()[]{}")
            if (
                len(stripped) >= 20
                and calculate_shannon_entropy(stripped) > 3.8
                and "@" not in stripped
                and "/" not in stripped
                and not stripped.startswith(("http://", "https://", "data:"))
            ):
                detected_categories.append("high_entropy_secret")
                matched_patterns.append(f"High-entropy token ({len(stripped)} chars)")
                break

        if detected_categories:
            explanation = (
                f"Classified as SENSITIVE: detected {len(detected_categories)} sensitive item(s) "
                f"({', '.join(detected_categories)}). Local-only processing enforced."
            )
            return PrivacyClassificationResult(
                level=PrivacyLevel.SENSITIVE,
                detected_categories=detected_categories,
                matched_patterns=matched_patterns,
                explanation=explanation,
                confidence=0.98,
            )

        # ----------------------------------------------------------------------
        # 2. Check for PERSONAL patterns
        # ----------------------------------------------------------------------
        for category, regex in PERSONAL_PATTERNS:
            match = regex.search(clean_text)
            if match:
                detected_categories.append(category)
                matched_patterns.append(match.group(0))

        if detected_categories:
            explanation = (
                f"Classified as PERSONAL: detected {len(detected_categories)} personal attribute(s) "
                f"({', '.join(detected_categories)}). Prefer Local AI or require consent for cloud."
            )
            return PrivacyClassificationResult(
                level=PrivacyLevel.PERSONAL,
                detected_categories=detected_categories,
                matched_patterns=matched_patterns,
                explanation=explanation,
                confidence=0.90,
            )

        # ----------------------------------------------------------------------
        # 3. Default to PUBLIC
        # ----------------------------------------------------------------------
        return PrivacyClassificationResult(
            level=PrivacyLevel.PUBLIC,
            detected_categories=["general"],
            matched_patterns=[],
            explanation="Classified as PUBLIC: no sensitive credentials, financial data, or personal details detected.",
            confidence=0.95,
        )


_classifier_instance: Optional[PrivacyClassifier] = None


def get_privacy_classifier() -> PrivacyClassifier:
    """Return singleton instance of PrivacyClassifier."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = PrivacyClassifier()
    return _classifier_instance
