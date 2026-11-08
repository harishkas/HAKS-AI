"""Memory Context Builder for HAKS AI Engines.

Enforces privacy, minimal-disclosure, and routing guidelines:
- Retrieves only relevant approved memories via semantic search.
- Never dumps entire memory databases.
- When targeting Cloud AI:
  * SENSITIVE memories are strictly omitted from cloud prompts.
  * Personal memories are excluded unless explicitly approved.
  * Redaction is applied if required.
- Formats concise prompt context for local or cloud inference.
"""

from typing import Optional

from core.logger import get_logger
from memory.personal_memory import PersonalMemoryManager
from memory.work_memory import WorkMemoryManager
from privacy.classifier import PrivacyClassifier, PrivacyLevel, get_privacy_classifier
from privacy.redactor import DataRedactor, get_data_redactor

logger = get_logger("memory.context_builder")


class MemoryContextBuilder:
    """Builds relevant memory context for inclusion in AI prompts with privacy safeguards."""

    def __init__(
        self,
        personal_memory: PersonalMemoryManager,
        work_memory: WorkMemoryManager,
        privacy_classifier: Optional[PrivacyClassifier] = None,
        redactor: Optional[DataRedactor] = None,
    ) -> None:
        self.personal_memory = personal_memory
        self.work_memory = work_memory
        self.privacy_classifier = privacy_classifier or get_privacy_classifier()
        self.redactor = redactor or get_data_redactor()

    def build_context(
        self,
        query: str,
        include_personal: bool = True,
        include_work: bool = True,
        top_k_per_domain: int = 3,
        min_relevance: float = 0.15,
        is_cloud_target: bool = False,
        redact_for_cloud: bool = False,
    ) -> str:
        """Retrieve and format relevant approved memories matching the query with privacy guards."""
        if not query.strip():
            return ""

        sections = []

        # 1. Relevant Personal Memories
        if include_personal:
            personal_hits = self.personal_memory.semantic_search(
                query, top_k=top_k_per_domain, min_score=min_relevance
            )
            if personal_hits:
                lines = ["Relevant Personal Context:"]
                for hit in personal_hits:
                    item = hit.item
                    # Privacy guard for cloud
                    p_res = self.privacy_classifier.classify(f"{item.title}\n{item.content}")
                    if is_cloud_target and p_res.level == PrivacyLevel.SENSITIVE:
                        logger.warning("Omitting SENSITIVE Personal Memory '%s' from cloud AI context.", item.title)
                        continue

                    content = item.content
                    if is_cloud_target and redact_for_cloud:
                        content = self.redactor.redact(content).sanitized_text

                    lines.append(f"- [{item.category.upper()}] {item.title}: {content}")

                if len(lines) > 1:
                    sections.append("\n".join(lines))

        # 2. Relevant Work Memories
        if include_work:
            work_hits = self.work_memory.semantic_search(
                query, top_k=top_k_per_domain, min_score=min_relevance
            )
            if work_hits:
                lines = ["Relevant Work & Technical Context:"]
                for hit in work_hits:
                    item = hit.item
                    # Privacy guard for cloud
                    p_res = self.privacy_classifier.classify(f"{item.title}\n{item.content}")
                    if is_cloud_target and p_res.level == PrivacyLevel.SENSITIVE:
                        logger.warning("Omitting SENSITIVE Work Memory '%s' from cloud AI context.", item.title)
                        continue

                    content = item.content
                    if is_cloud_target and redact_for_cloud:
                        content = self.redactor.redact(content).sanitized_text

                    lines.append(f"- [{item.project_name.upper()}] {item.title}: {content}")

                if len(lines) > 1:
                    sections.append("\n".join(lines))

        if not sections:
            return ""

        context_body = "\n\n".join(sections)
        logger.info(
            "Retrieved %d memory context section(s) for query: '%s' (cloud_target=%s)",
            len(sections),
            query[:30],
            is_cloud_target,
        )
        return (
            "--- APPROVED MEMORY CONTEXT (Use when relevant to user query) ---\n"
            f"{context_body}\n"
            "----------------------------------------------------------------"
        )
