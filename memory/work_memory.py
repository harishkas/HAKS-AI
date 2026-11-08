"""Work Memory Domain Manager for HAKS.

Manages the independent work_memory.db and work_memory_index.json.
Stores engineering decisions, code context, algorithms, and project architecture.
Guarantees complete isolation from Personal Memory. Never uploads memories to cloud AI.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Optional
import uuid

from config.settings import AppConfig, get_settings
from core.logger import get_logger
from memory.encryption import MemoryEncryptionService, get_encryption_service
from memory.models import MemoryBackupBundle, MemoryDomain, MemoryItem, MemorySearchResult
from memory.vector_index import LocalVectorIndex
from storage.database import DatabaseManager, get_db_manager

logger = get_logger("memory.work")


class WorkMemoryManager:
    """Manages Work Memory storage, project organization, semantic indexing, and lifecycle."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        db_manager: Optional[DatabaseManager] = None,
        encryption_service: Optional[MemoryEncryptionService] = None,
    ) -> None:
        self.config = config or get_settings()
        self.db = db_manager or get_db_manager(self.config)
        self.encryption = encryption_service or get_encryption_service(self.config)

        # Independent Vector Index for Work Memory Domain
        index_dir = self.config.paths.data_dir / "work_memory" / "vector_index"
        index_dir.mkdir(parents=True, exist_ok=True)
        index_path = index_dir / "index.json"

        # Auto-migrate legacy root index if present
        legacy_path = self.config.paths.data_dir / "work_memory_index.json"
        if legacy_path.exists() and not index_path.exists():
            try:
                legacy_path.rename(index_path)
                logger.info("Migrated legacy work_memory_index.json to %s", index_path)
            except Exception as exc:
                logger.warning("Could not auto-migrate work_memory_index.json: %s", exc)

        self.vector_index = LocalVectorIndex(index_path, self.encryption)

        logger.info("WorkMemoryManager initialized with %d indexed vectors.", len(self.vector_index))

    def add_memory(
        self,
        title: str,
        content: str,
        project_name: str = "default",
        tags: Optional[list[str]] = None,
        importance: int = 1,
        source_conversation_id: Optional[str] = None,
        check_duplicates: bool = True,
    ) -> tuple[str, bool]:
        """Add a memory item to Work Memory with transparent encryption at rest.

        Returns:
            tuple[str, bool]: (memory_id, is_new)
        """
        tags_list = tags or []
        tags_str = ", ".join(tags_list)
        project_clean = project_name.strip() or "default"

        # Duplicate detection via semantic similarity within project
        if check_duplicates and len(self.vector_index) > 0:
            similar = self.vector_index.search(f"{project_clean} {title} {content}", top_k=1, min_score=0.88)
            if similar:
                existing_id, score, meta = similar[0]
                logger.info(
                    "Duplicate detected (similarity: %.2f) with existing work memory '%s'. Updating record.",
                    score,
                    existing_id,
                )
                self.update_memory(
                    existing_id,
                    title=title,
                    content=content,
                    project_name=project_clean,
                    tags=tags_list,
                    importance=importance,
                )
                return existing_id, False

        mem_id = f"wm_{uuid.uuid4().hex[:10]}"
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        # Encrypt sensitive content at rest
        stored_content = self.encryption.encrypt_field(content)

        with self.db.session("work_memory") as conn:
            conn.execute(
                """
                INSERT INTO work_memories (
                    id, project_name, title, content, tags, importance, created_at, updated_at, source_conversation_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (mem_id, project_clean, title, stored_content, tags_str, importance, now, now, source_conversation_id),
            )

        # Index in independent work vector index (text is encrypted when persisted to disk)
        full_text = f"[{project_clean}] {title}\n{content}\nTags: {tags_str}"
        self.vector_index.add_document(
            doc_id=mem_id,
            text=full_text,
            metadata={
                "project_name": project_clean,
                "title": title,
                "importance": importance,
                "tags": tags_list,
            },
        )

        logger.info("Saved new Work Memory: %s ('%s' under '%s')", mem_id, title, project_clean)
        return mem_id, True

    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve a single work memory item by ID, decrypting content transparently."""
        with self.db.session("work_memory") as conn:
            row = conn.execute(
                "SELECT * FROM work_memories WHERE id = ? AND is_archived = 0;",
                (memory_id,),
            ).fetchone()
            if not row:
                return None

            tags = [t.strip() for t in (row["tags"] or "").split(",") if t.strip()]
            dec_content = self.encryption.decrypt_field(str(row["content"]))
            return MemoryItem(
                id=str(row["id"]),
                domain=MemoryDomain.WORK,
                title=str(row["title"]),
                content=dec_content,
                project_name=str(row["project_name"] or "default"),
                tags=tags,
                importance=int(row["importance"] or 1),
                created_at=str(row["created_at"]),
                updated_at=str(row["updated_at"]),
                source_conversation_id=row["source_conversation_id"],
                is_archived=bool(row["is_archived"]),
            )

    def get_all_memories(
        self, project_name: Optional[str] = None, search: str = ""
    ) -> list[MemoryItem]:
        """Fetch all active work memories with decrypted content and keyword filtering."""
        memories = []
        with self.db.session("work_memory") as conn:
            query = "SELECT * FROM work_memories WHERE is_archived = 0"
            params = []

            if project_name and project_name != "All Projects":
                query += " AND project_name = ?"
                params.append(project_name)

            query += " ORDER BY importance DESC, created_at DESC;"

            search_lower = search.strip().lower()
            for row in conn.execute(query, params).fetchall():
                dec_content = self.encryption.decrypt_field(str(row["content"]))
                title_str = str(row["title"])
                tags_str = str(row["tags"] or "")
                tags = [t.strip() for t in tags_str.split(",") if t.strip()]

                if search_lower:
                    if (search_lower not in title_str.lower() and
                        search_lower not in dec_content.lower() and
                        search_lower not in tags_str.lower()):
                        continue

                memories.append(
                    MemoryItem(
                        id=str(row["id"]),
                        domain=MemoryDomain.WORK,
                        title=title_str,
                        content=dec_content,
                        project_name=str(row["project_name"] or "default"),
                        tags=tags,
                        importance=int(row["importance"] or 1),
                        created_at=str(row["created_at"]),
                        updated_at=str(row["updated_at"]),
                        source_conversation_id=row["source_conversation_id"],
                        is_archived=bool(row["is_archived"]),
                    )
                )
        return memories

    def get_projects(self) -> list[str]:
        """Get distinct project names stored in Work Memory."""
        with self.db.session("work_memory") as conn:
            rows = conn.execute(
                "SELECT DISTINCT project_name FROM work_memories WHERE is_archived = 0 ORDER BY project_name ASC;"
            ).fetchall()
            return [str(r["project_name"]) for r in rows if r["project_name"]]

    def semantic_search(
        self, query: str, top_k: int = 5, min_score: float = 0.20
    ) -> list[MemorySearchResult]:
        """Perform semantic search using the local vector index."""
        raw_results = self.vector_index.search(query, top_k=top_k, min_score=min_score)
        results = []

        for doc_id, score, meta in raw_results:
            item = self.get_memory(doc_id)
            if item:
                results.append(
                    MemorySearchResult(
                        item=item,
                        score=score,
                        match_type="semantic",
                    )
                )
        return results

    def update_memory(
        self,
        memory_id: str,
        title: Optional[str] = None,
        content: Optional[str] = None,
        project_name: Optional[str] = None,
        tags: Optional[list[str]] = None,
        importance: Optional[int] = None,
    ) -> bool:
        """Update an existing work memory item."""
        existing = self.get_memory(memory_id)
        if not existing:
            return False

        new_title = title if title is not None else existing.title
        new_content = content if content is not None else existing.content
        new_project = project_name if project_name is not None else existing.project_name
        new_tags = tags if tags is not None else existing.tags
        new_importance = importance if importance is not None else existing.importance
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        stored_content = self.encryption.encrypt_field(new_content)

        with self.db.session("work_memory") as conn:
            conn.execute(
                """
                UPDATE work_memories
                SET title = ?, content = ?, project_name = ?, tags = ?, importance = ?, updated_at = ?
                WHERE id = ?;
                """,
                (new_title, stored_content, new_project, ", ".join(new_tags), new_importance, now, memory_id),
            )

        # Update vector index
        full_text = f"[{new_project}] {new_title}\n{new_content}\nTags: {', '.join(new_tags)}"
        self.vector_index.add_document(
            doc_id=memory_id,
            text=full_text,
            metadata={
                "project_name": new_project,
                "title": new_title,
                "importance": new_importance,
                "tags": new_tags,
            },
        )
        logger.info("Updated Work Memory item %s ('%s')", memory_id, new_title)
        return True

    def delete_memory(self, memory_id: str) -> bool:
        """Delete an individual work memory record from database and vector index."""
        with self.db.session("work_memory") as conn:
            cursor = conn.execute("DELETE FROM work_memories WHERE id = ?;", (memory_id,))
            deleted = cursor.rowcount > 0

        self.vector_index.remove_document(memory_id)
        logger.info("Deleted Work Memory item %s (success=%s)", memory_id, deleted)
        return deleted

    def clear_domain(self) -> None:
        """Clear all Work Memories and reset vector index. Never touches Personal Memory."""
        self.db.clear_domain("work_memory")
        self.vector_index.clear()
        logger.warning("Work Memory domain cleared.")

    def export_bundle(self, export_path: Path, encrypt: bool = False) -> Path:
        """Export all work memories to an encrypted or plain JSON backup bundle."""
        items = [m.model_dump() for m in self.get_all_memories()]
        bundle = MemoryBackupBundle(
            domain=MemoryDomain.WORK,
            exported_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            item_count=len(items),
            items=items,
            is_encrypted=encrypt,
        )

        export_path.parent.mkdir(parents=True, exist_ok=True)
        raw_json = bundle.model_dump_json(indent=2)

        if encrypt:
            encrypted_data = self.encryption.encrypt_string(raw_json)
            export_path.write_text(encrypted_data, encoding="utf-8")
        else:
            export_path.write_text(raw_json, encoding="utf-8")

        logger.info("Exported %d work memories to %s (encrypted=%s)", len(items), export_path, encrypt)
        return export_path

    def restore_bundle(self, bundle_path: Path, is_encrypted: bool = False) -> int:
        """Restore work memories from a backup bundle."""
        content = bundle_path.read_text(encoding="utf-8")
        if is_encrypted or content.startswith("gAAAA"):
            content = self.encryption.decrypt_string(content)

        data = json.loads(content)
        if data.get("domain") != MemoryDomain.WORK.value:
            raise ValueError(f"Invalid domain in backup: {data.get('domain')} (expected work)")

        restored_count = 0
        for item_data in data.get("items", []):
            self.add_memory(
                title=item_data["title"],
                content=item_data["content"],
                project_name=item_data.get("project_name", "default"),
                tags=item_data.get("tags", []),
                importance=item_data.get("importance", 1),
                source_conversation_id=item_data.get("source_conversation_id"),
                check_duplicates=True,
            )
            restored_count += 1

        logger.info("Restored %d work memories from %s", restored_count, bundle_path.name)
        return restored_count

    def rebuild_index(self) -> int:
        """Rebuild the vector index from SQLite database records."""
        self.vector_index.clear()
        all_items = self.get_all_memories()
        for item in all_items:
            full_text = f"[{item.project_name}] {item.title}\n{item.content}\nTags: {', '.join(item.tags)}"
            self.vector_index.add_document(
                doc_id=item.id,
                text=full_text,
                metadata={
                    "project_name": item.project_name,
                    "title": item.title,
                    "importance": item.importance,
                    "tags": item.tags,
                },
            )
        logger.info("Rebuilt work memory vector index with %d items.", len(all_items))
        return len(all_items)

    def sync_vector_index(self) -> dict[str, Any]:
        """Ensure vector records remain 100% consistent with corresponding SQLite records.

        Performs self-healing synchronization:
        - Identifies and removes orphaned vector records
        - Identifies and indexes missing SQLite records
        """
        all_memories = self.get_all_memories()
        active_ids = {m.id for m in all_memories}
        indexed_ids = set(self.vector_index._entries.keys())

        added_count = 0
        removed_count = 0

        # 1. Remove orphaned vector entries
        orphaned = indexed_ids - active_ids
        for oid in orphaned:
            self.vector_index.remove_document(oid)
            removed_count += 1

        # 2. Add missing entries
        for item in all_memories:
            if item.id not in indexed_ids:
                full_text = f"[{item.project_name}] {item.title}\n{item.content}\nTags: {', '.join(item.tags)}"
                self.vector_index.add_document(
                    doc_id=item.id,
                    text=full_text,
                    metadata={
                        "project_name": item.project_name,
                        "title": item.title,
                        "importance": item.importance,
                        "tags": item.tags,
                    },
                )
                added_count += 1

        self.vector_index.save()
        logger.info(
            "Work Memory vector sync complete: added=%d, removed=%d, total=%d",
            added_count,
            removed_count,
            len(self.vector_index),
        )
        return {
            "domain": "work_memory",
            "synced": True,
            "added": added_count,
            "removed": removed_count,
            "total": len(self.vector_index),
        }

    def verify_consistency(self) -> dict[str, Any]:
        """Verify consistency between SQLite records and vector index entries."""
        all_memories = self.get_all_memories()
        active_ids = {m.id for m in all_memories}
        indexed_ids = set(self.vector_index._entries.keys())
        missing = list(active_ids - indexed_ids)
        orphaned = list(indexed_ids - active_ids)
        is_consistent = (len(missing) == 0 and len(orphaned) == 0)
        return {
            "domain": "work_memory",
            "is_consistent": is_consistent,
            "sqlite_count": len(active_ids),
            "vector_count": len(indexed_ids),
            "missing_in_vector": missing,
            "orphaned_in_vector": orphaned,
        }
