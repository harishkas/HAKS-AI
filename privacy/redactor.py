"""Data Redaction Engine for HAKS.

Sanitizes text by detecting and masking sensitive identifiers before cloud transmission:
- Email addresses
- Phone numbers
- API keys & secrets
- Passwords & credentials
- Bearer & JWT tokens
- Credit cards & SSNs
- Database URIs & private keys

Preserves the technical meaning, code syntax, and context of the text.
"""

from typing import Optional
import re
from pydantic import BaseModel, Field

from core.logger import get_logger
from privacy.classifier import luhn_checksum

logger = get_logger("privacy.redactor")


class RedactedItem(BaseModel):
    """Details of an individual redaction made."""

    category: str
    placeholder: str
    original_snippet: str


class RedactionResult(BaseModel):
    """Result of text sanitization and redaction."""

    original_text: str
    sanitized_text: str
    redactions_count: int = 0
    categories_redacted: list[str] = Field(default_factory=list)
    items: list[RedactedItem] = Field(default_factory=list)

    @property
    def has_redactions(self) -> bool:
        return self.redactions_count > 0


class DataRedactor:
    """Configurable, deterministic text sanitization engine."""

    def __init__(self) -> None:
        # Ordered redaction rules: (Category, Regex, Replacement, Snippet Extraction)
        self.rules: list[tuple[str, re.Pattern, str]] = [
            # 1. Private keys
            ("private_key", re.compile(r"-----BEGIN (?:RSA|EC|DSA|OPENSSH|PGP|ENCRYPTED|PRIVATE) KEY-----[\s\S]*?-----END (?:RSA|EC|DSA|OPENSSH|PGP|ENCRYPTED|PRIVATE) KEY-----"), "[PRIVATE_KEY_REDACTED]"),

            # 2. Database URIs with credentials
            ("db_uri", re.compile(r"(?i)(?:postgres|mysql|mongodb|redis|mssql):\/\/[^\s:@]+:[^\s:@]+@[^\s]+"), "[DB_URI_REDACTED]"),

            # 3. Known API Keys
            ("api_key", re.compile(r"\bAIzaSy[a-zA-Z0-9_\-]{33}\b"), "[GEMINI_KEY_REDACTED]"),
            ("api_key", re.compile(r"\bsk-ant-[a-zA-Z0-9_\-]{20,}\b"), "[ANTHROPIC_KEY_REDACTED]"),
            ("api_key", re.compile(r"\bsk-[a-zA-Z0-9_\-]{20,}\b"), "[OPENAI_KEY_REDACTED]"),
            ("api_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[AWS_KEY_REDACTED]"),

            # 4. Bearer & JWT Tokens
            ("auth_token", re.compile(r"(?i)\bBearer\s+[a-zA-Z0-9_\-\.]{20,}\b"), "Bearer [AUTH_TOKEN_REDACTED]"),
            ("auth_token", re.compile(r"\beyJ[a-zA-Z0-9_\-]{10,}\.eyJ[a-zA-Z0-9_\-]{10,}\.[a-zA-Z0-9_\-]{10,}\b"), "[JWT_TOKEN_REDACTED]"),

            # 5. Passwords in key-value formats
            ("password", re.compile(r"(?i)\b(password|passwd|pwd|secret)\s*([:=])\s*['\"]?([^\s'\"]{4,})['\"]?"), r"\1\2'[PASSWORD_REDACTED]'"),

            # 6. Emails
            ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,24}\b"), "[EMAIL_REDACTED]"),

            # 7. SSN
            ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN_REDACTED]"),

            # 8. International / Indian / US Phone Numbers
            ("phone", re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE_REDACTED]"),
            ("phone", re.compile(r"\b(?:\+91[\-\s]?)?[6789]\d{9}\b"), "[PHONE_REDACTED]"),
        ]

    def redact(self, text: str) -> RedactionResult:
        """Sanitize text by replacing sensitive patterns with structured placeholders.

        Preserves technical syntax, code structure, and natural language meaning.
        """
        if not text:
            return RedactionResult(original_text="", sanitized_text="", redactions_count=0)

        sanitized = text
        redacted_items: list[RedactedItem] = []
        categories_set: set[str] = set()

        # 1. Apply regex rules
        for category, pattern, replacement in self.rules:
            matches = list(pattern.finditer(sanitized))
            if matches:
                for match in matches:
                    snippet = match.group(0)
                    # Mask snippet for security in report
                    safe_snippet = snippet[:4] + "***" + snippet[-4:] if len(snippet) > 8 else "***"
                    redacted_items.append(
                        RedactedItem(
                            category=category,
                            placeholder=replacement,
                            original_snippet=safe_snippet,
                        )
                    )
                    categories_set.add(category)
                sanitized = pattern.sub(replacement, sanitized)

        # 2. Check and redact credit card numbers with Luhn check
        cc_candidates = re.finditer(r"\b(?:\d[ -]*?){13,19}\b", sanitized)
        for match in list(cc_candidates):
            candidate_str = match.group(0)
            clean_digits = re.sub(r"\D", "", candidate_str)
            if luhn_checksum(clean_digits):
                sanitized = sanitized.replace(candidate_str, "[CREDIT_CARD_REDACTED]")
                redacted_items.append(
                    RedactedItem(
                        category="credit_card",
                        placeholder="[CREDIT_CARD_REDACTED]",
                        original_snippet=f"Card ending in {clean_digits[-4:]}",
                    )
                )
                categories_set.add("credit_card")

        logger.debug("Redaction complete. Total masked: %d items across categories %s", len(redacted_items), list(categories_set))

        return RedactionResult(
            original_text=text,
            sanitized_text=sanitized,
            redactions_count=len(redacted_items),
            categories_redacted=sorted(list(categories_set)),
            items=redacted_items,
        )

    def generate_preview(self, text: str) -> RedactionResult:
        """Generate a preview of the redaction without modifying underlying state."""
        return self.redact(text)


_redactor_instance: Optional[DataRedactor] = None


def get_data_redactor() -> DataRedactor:
    """Return singleton DataRedactor instance."""
    global _redactor_instance
    if _redactor_instance is None:
        _redactor_instance = DataRedactor()
    return _redactor_instance
