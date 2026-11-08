"""Memory Context Builder for HAKS AI Engines.

Enforces privacy and minimal-disclosure guidelines:
- Retrieves only relevant approved memories via semantic search.
- Never dumps entire memory databases.
- Formats concise prompt context for local or cloud inference.
"""

from typing import Optional

from memory.models import MemoryDomain
from memory.personal_memory import PersonalMemoryManager
from memory.work_memory import WorkMemoryManager
from core.logger import get_logger

logger = get_logger("memory.context_builder")


class MemoryContextBuilder:
    """Builds relevant memory context for inclusion in AI prompts."""

    def __init__(
        self,
        personal_memory: PersonalMemoryManager,
        work_memory: WorkMemoryManager,
    ) -> None:
        self.personal_memory = personal_memory
        self.work_memory = work_memory

    def build_context(
        self,
        query: str,
        include_personal: bool = True,
        include_work: bool = True,
        top_k_per_domain: int = 3,
        min_relevance: float = 0.25,
    ) -> str:
        """Retrieve and format relevant approved memories matching the query."""
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
                    lines.append(f"- [{item.category.upper()}] {item.title}: {item.content}")
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
                    lines.append(f"- [{item.project_name.upper()}] {item.title}: {item.content}")
                sections.append("\n".join(lines))

        if not sections:
            return ""

        context_body = "\n\n".join(sections)
        logger.info("Retrieved %d memory context section(s) for query: '%s'", len(sections), query[:30])
        return (
            "--- APPROVED MEMORY CONTEXT (Use when relevant to user query) ---\n"
            f"{context_body}\n"
            "----------------------------------------------------------------"
        )
