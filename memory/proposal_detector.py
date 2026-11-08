"""Automatic Memory Proposal Detector for HAKS.

Analyzes conversation turns for memorable personal facts or work decisions.
Enforces the mandatory architectural rule:
"AI suggests; user approves. Never automatically save permanent memories without user approval."
"""

import re
from typing import Optional
import uuid
from datetime import datetime, timezone

from memory.models import MemoryDomain, MemoryProposal, MemoryProposalStatus
from core.logger import get_logger

logger = get_logger("memory.proposal_detector")

# Patterns indicating Personal Memory
PERSONAL_PATTERNS = [
    (r"\b(i prefer|i like|i love|my favorite|i always want|i hate|i dislike)\b", "preference", 3),
    (r"\b(my name is|i am from|i live in|i was born|my birthday is|my wife|my husband|my son|my daughter|my brother|my sister)\b", "history", 4),
    (r"\b(my goal is|i want to become|i am planning to|i aim to|my dream)\b", "goal", 4),
    (r"\b(i speak|my mother tongue is|speak to me in|i understand tamil|i understand english)\b", "preference", 4),
    (r"\b(always remember that i|remember that i|keep in mind that i)\b", "general", 5),
    (r"\b(i usually|my habit is|every morning i|every evening i)\b", "experience", 2),
]

# Patterns indicating Work Memory
WORK_PATTERNS = [
    (r"\b(we decided to|the architecture is|architectural decision|design pattern|we are using)\b", "architecture", 4),
    (r"\b(the algorithm for|implementation approach|data structure|time complexity)\b", "algorithm", 3),
    (r"\b(for project|in project|in repository|the codebase|git branch)\b", "task", 3),
    (r"\b(the bug was|the root cause was|fixed the issue with|error resolution)\b", "decision", 3),
    (r"\b(the schema has|database table|foreign key constraint|sqlite pragma)\b", "architecture", 4),
    (r"\b(hardware requirement|i5-7200u|memory limit|ram budget|quantized model)\b", "architecture", 5),
    (r"\b(always remember for this project|critical rule for|mandatory guideline)\b", "decision", 5),
]


class MemoryProposalDetector:
    """Detects potential memory candidates from user and assistant text."""

    def detect_proposal(
        self,
        user_text: str,
        source_conversation_id: Optional[str] = None,
        assistant_text: Optional[str] = None,
    ) -> Optional[MemoryProposal]:
        """Analyze text and produce a MemoryProposal if significant information is found."""
        clean_user = user_text.strip()
        if len(clean_user) < 12:
            return None

        lower_user = clean_user.lower()

        # Check for explicit exclusions
        if any(neg in lower_user for neg in ["don't remember", "do not save", "forget this", "temporary", "ignore this"]):
            return None

        # 1. Check Work Patterns first
        for pattern, category, base_importance in WORK_PATTERNS:
            if re.search(pattern, lower_user):
                title = self._generate_title(clean_user, "Work Note")
                project = self._extract_project_name(clean_user)
                tags = ["work", category]
                if project != "default":
                    tags.append(project.lower())

                importance = base_importance
                if any(k in lower_user for k in ["critical", "mandatory", "always", "crucial"]):
                    importance = min(5, importance + 1)

                proposal = MemoryProposal(
                    id=f"prop_{uuid.uuid4().hex[:10]}",
                    domain=MemoryDomain.WORK,
                    source_conversation_id=source_conversation_id,
                    proposal_text=clean_user,
                    suggested_title=title,
                    suggested_category=category,
                    suggested_tags=tags,
                    suggested_importance=importance,
                    suggested_project=project,
                    confidence=0.85,
                    status=MemoryProposalStatus.PENDING,
                    created_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                )
                logger.info("Detected Work Memory proposal: '%s' (Confidence: 0.85)", title)
                return proposal

        # 2. Check Personal Patterns
        for pattern, category, base_importance in PERSONAL_PATTERNS:
            if re.search(pattern, lower_user):
                title = self._generate_title(clean_user, "Personal Note")
                tags = ["personal", category]
                importance = base_importance
                if any(k in lower_user for k in ["always", "never forget", "crucial", "remember"]):
                    importance = min(5, importance + 1)

                proposal = MemoryProposal(
                    id=f"prop_{uuid.uuid4().hex[:10]}",
                    domain=MemoryDomain.PERSONAL,
                    source_conversation_id=source_conversation_id,
                    proposal_text=clean_user,
                    suggested_title=title,
                    suggested_category=category,
                    suggested_tags=tags,
                    suggested_importance=importance,
                    confidence=0.88,
                    status=MemoryProposalStatus.PENDING,
                    created_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                )
                logger.info("Detected Personal Memory proposal: '%s' (Confidence: 0.88)", title)
                return proposal

        return None

    def _generate_title(self, text: str, fallback: str) -> str:
        """Create a clean 3-7 word summary title."""
        # Remove common introductory phrases
        cleaned = re.sub(
            r"^(please\s+)?(always\s+)?(remember\s+(that\s+)?|i\s+prefer\s+|my\s+name\s+is\s+|i\s+like\s+|we\s+decided\s+to\s+)",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        words = cleaned.split()
        if len(words) >= 3:
            candidate = " ".join(words[:6])
            return candidate[0].upper() + candidate[1:]
        return fallback

    def _extract_project_name(self, text: str) -> str:
        """Attempt to extract project identifier from text."""
        match = re.search(r"(?:for|in)\s+project\s+([a-zA-Z0-9_\-]+)", text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        if "haks" in text.lower():
            return "HAKS"
        return "default"
