"""Memory Proposal Workflow Manager for HAKS.

Manages memory proposals stored in memory_proposals.db.
Enforces the mandatory architectural rule:
"AI suggests; user approves. Never automatically save permanent memories without user approval."
"""

from datetime import datetime, timezone
from typing import Optional

from config.settings import AppConfig, get_settings
from core.logger import get_logger
from memory.models import MemoryDomain, MemoryProposal, MemoryProposalStatus
from memory.personal_memory import PersonalMemoryManager
from memory.work_memory import WorkMemoryManager
from storage.database import DatabaseManager, get_db_manager

logger = get_logger("memory.proposal_manager")


class MemoryProposalManager:
    """Manages the proposal lifecycle, user approval/rejection, and persistence dispatch."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        db_manager: Optional[DatabaseManager] = None,
        personal_memory: Optional[PersonalMemoryManager] = None,
        work_memory: Optional[WorkMemoryManager] = None,
    ) -> None:
        self.config = config or get_settings()
        self.db = db_manager or get_db_manager(self.config)
        self.personal_memory = personal_memory or PersonalMemoryManager(self.config, self.db)
        self.work_memory = work_memory or WorkMemoryManager(self.config, self.db)

    def create_proposal(self, proposal: MemoryProposal) -> str:
        """Store a detected proposal into memory_proposals.db in PENDING status."""
        tags_str = ", ".join(proposal.suggested_tags)
        with self.db.session("memory_proposals") as conn:
            conn.execute(
                """
                INSERT INTO memory_proposals (
                    id, domain, source_conversation_id, proposal_text, suggested_title,
                    suggested_category, suggested_tags, confidence, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    proposal.id,
                    proposal.domain.value,
                    proposal.source_conversation_id,
                    proposal.proposal_text,
                    proposal.suggested_title,
                    proposal.suggested_category,
                    tags_str,
                    proposal.confidence,
                    proposal.status.value,
                    proposal.created_at,
                ),
            )
        logger.info(
            "Created memory proposal: %s ('%s' for domain '%s')",
            proposal.id,
            proposal.suggested_title,
            proposal.domain.value,
        )
        return proposal.id

    def get_pending_proposals(self) -> list[MemoryProposal]:
        """Fetch all proposals currently awaiting user review."""
        proposals = []
        with self.db.session("memory_proposals") as conn:
            rows = conn.execute(
                "SELECT * FROM memory_proposals WHERE status = 'PENDING' ORDER BY created_at DESC;"
            ).fetchall()

            for r in rows:
                tags = [t.strip() for t in (r["suggested_tags"] or "").split(",") if t.strip()]
                text = str(r["proposal_text"])
                title = str(r["suggested_title"]) if ("suggested_title" in r.keys() and r["suggested_title"]) else ((text[:40] + "...") if len(text) > 40 else text)
                proposals.append(
                    MemoryProposal(
                        id=str(r["id"]),
                        domain=MemoryDomain(str(r["domain"])),
                        source_conversation_id=r["source_conversation_id"],
                        proposal_text=text,
                        suggested_title=title,
                        suggested_category=str(r["suggested_category"] or "general"),
                        suggested_tags=tags,
                        confidence=float(r["confidence"] or 0.8),
                        status=MemoryProposalStatus.PENDING,
                        created_at=str(r["created_at"]),
                        reviewed_at=r["reviewed_at"],
                    )
                )
        return proposals

    def get_proposal(self, proposal_id: str) -> Optional[MemoryProposal]:
        """Fetch a specific proposal by ID."""
        with self.db.session("memory_proposals") as conn:
            r = conn.execute(
                "SELECT * FROM memory_proposals WHERE id = ?;", (proposal_id,)
            ).fetchone()
            if not r:
                return None

            tags = [t.strip() for t in (r["suggested_tags"] or "").split(",") if t.strip()]
            text = str(r["proposal_text"])
            title = str(r["suggested_title"]) if ("suggested_title" in r.keys() and r["suggested_title"]) else ((text[:40] + "...") if len(text) > 40 else text)
            return MemoryProposal(
                id=str(r["id"]),
                domain=MemoryDomain(str(r["domain"])),
                source_conversation_id=r["source_conversation_id"],
                proposal_text=text,
                suggested_title=title,
                suggested_category=str(r["suggested_category"] or "general"),
                suggested_tags=tags,
                confidence=float(r["confidence"] or 0.8),
                status=MemoryProposalStatus(str(r["status"])),
                created_at=str(r["created_at"]),
                reviewed_at=r["reviewed_at"],
            )

    def approve_proposal(
        self,
        proposal_id: str,
        title: Optional[str] = None,
        content: Optional[str] = None,
        category: Optional[str] = None,
        project_name: Optional[str] = None,
        tags: Optional[list[str]] = None,
        importance: int = 2,
    ) -> tuple[bool, Optional[str]]:
        """Approve a proposal, writing it to the appropriate persistent memory domain.

        Returns:
            tuple[bool, Optional[str]]: (success, created_memory_id)
        """
        proposal = self.get_proposal(proposal_id)
        if not proposal:
            logger.warning("Attempted to approve nonexistent proposal %s", proposal_id)
            return False, None

        final_title = title.strip() if title else proposal.suggested_title
        final_content = content.strip() if content else proposal.proposal_text
        final_tags = tags if tags is not None else proposal.suggested_tags
        final_category = category if category else proposal.suggested_category
        final_project = project_name if project_name else (proposal.suggested_project or "default")

        created_mem_id = None
        if proposal.domain == MemoryDomain.PERSONAL:
            created_mem_id, is_new = self.personal_memory.add_memory(
                title=final_title,
                content=final_content,
                category=final_category,
                tags=final_tags,
                importance=importance,
                source_conversation_id=proposal.source_conversation_id,
            )
        else:
            created_mem_id, is_new = self.work_memory.add_memory(
                title=final_title,
                content=final_content,
                project_name=final_project,
                tags=final_tags,
                importance=importance,
                source_conversation_id=proposal.source_conversation_id,
            )

        # Update proposal record in DB
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with self.db.session("memory_proposals") as conn:
            conn.execute(
                "UPDATE memory_proposals SET status = 'APPROVED', reviewed_at = ? WHERE id = ?;",
                (now, proposal_id),
            )

        logger.info(
            "User APPROVED proposal %s -> saved to %s memory with ID %s",
            proposal_id,
            proposal.domain.value,
            created_mem_id,
        )
        return True, created_mem_id

    def reject_proposal(self, proposal_id: str) -> bool:
        """Reject a proposal, ensuring it is discarded and never saved to long-term memory."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with self.db.session("memory_proposals") as conn:
            cursor = conn.execute(
                "UPDATE memory_proposals SET status = 'REJECTED', reviewed_at = ? WHERE id = ?;",
                (now, proposal_id),
            )
            success = cursor.rowcount > 0

        logger.info("User REJECTED memory proposal %s (never saved)", proposal_id)
        return success

    def get_stats(self) -> dict[str, int]:
        """Return counts of proposals grouped by status."""
        stats = {"PENDING": 0, "APPROVED": 0, "REJECTED": 0}
        with self.db.session("memory_proposals") as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM memory_proposals GROUP BY status;"
            ).fetchall()
            for r in rows:
                stats[str(r["status"])] = int(r["cnt"])
        return stats
