"""Independent Backup, Restore, Export, and Wiping Manager for HAKS Storage Domains.

Guarantees complete domain isolation: operations on Personal Memory never touch
Work Memory or Conversations. Supports authenticated Fernet encrypted archive bundles
(AES-128-CBC with HMAC-SHA256, 256-bit master key), cross-platform JSON, CSV,
and Markdown exports, and guarded deletion.
"""

import csv
from datetime import datetime, timezone
import io
import json
import logging
from pathlib import Path
import sqlite3
from typing import Any, Optional
import zipfile

from config.settings import AppConfig, get_settings
from memory.encryption import MemoryEncryptionService, get_encryption_service
from storage.database import DatabaseManager, get_db_manager
from storage.schema import DATABASE_SCHEMAS

logger = logging.getLogger("haks.storage.backup")


class BackupManager:
    """Manages independent domain backups, restoration, exports, and secure domain wipes."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        db_manager: Optional[DatabaseManager] = None,
        encryption_service: Optional[MemoryEncryptionService] = None,
    ) -> None:
        self.config = config or get_settings()
        self.db = db_manager or get_db_manager(self.config)
        self.encryption = encryption_service or get_encryption_service(self.config)
        self.backups_dir = self.config.paths.backups_dir
        self.backups_dir.mkdir(parents=True, exist_ok=True)

    def _get_domain_files(self, domain: str) -> list[tuple[Path, str]]:
        """Get source filesystem paths and archive-relative paths for a given domain."""
        db_path = self.db.get_db_path(domain)
        files = []
        if db_path.exists():
            files.append((db_path, f"{domain}.db"))

        if domain in ("personal_memory", "work_memory"):
            vector_index_path = self.config.paths.data_dir / domain / "vector_index" / "index.json"
            if vector_index_path.exists():
                files.append((vector_index_path, "vector_index/index.json"))

        return files

    def backup_domain(
        self,
        domain: str,
        target_dir: Optional[Path] = None,
        encrypt: bool = True,
    ) -> Path:
        """Create an independent backup bundle for a single domain.

        Args:
            domain: Domain to back up ('personal_memory', 'work_memory', 'conversations', 'settings')
            target_dir: Optional destination directory (defaults to configured backups_dir)
            encrypt: Whether to encrypt the resulting backup bundle using Fernet (AES-128-CBC + HMAC-SHA256)

        Returns:
            Path to created backup file (.zip or .haksbackup)
        """
        if domain not in ("personal_memory", "work_memory", "conversations", "settings", "memory_proposals"):
            raise ValueError(f"Invalid domain for backup: '{domain}'")

        dest_dir = target_dir or (self.backups_dir / domain)
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint WAL so database file contains all active committed transactions
        try:
            with self.db.session(domain) as conn:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        except Exception as exc:
            logger.warning("Could not flush WAL for %s before backup: %s", domain, exc)

        files = self._get_domain_files(domain)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        manifest = {
            "domain": domain,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "haks_version": self.config.ui.version,
            "is_encrypted": encrypt,
            "files": [rel_path for _, rel_path in files],
        }

        # Build in-memory zip archive
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            for src_path, rel_path in files:
                if src_path.exists():
                    zf.write(src_path, arcname=rel_path)

        zip_bytes = zip_buffer.getvalue()

        if encrypt:
            enc_bytes = self.encryption.encrypt_bytes(zip_bytes)
            output_path = dest_dir / f"{domain}_backup_{timestamp}.haksbackup"
            output_path.write_bytes(enc_bytes)
        else:
            output_path = dest_dir / f"{domain}_backup_{timestamp}.zip"
            output_path.write_bytes(zip_bytes)

        logger.info("Successfully created independent backup for '%s' at %s (encrypted=%s)", domain, output_path, encrypt)
        return output_path

    def restore_domain(self, backup_path: Path, expected_domain: str) -> bool:
        """Restore an independent domain from a backup archive.

        Args:
            backup_path: Path to .haksbackup or .zip file
            expected_domain: Expected domain identifier to prevent accidental cross-domain overwrite

        Returns:
            True if restoration was verified and successful
        """
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")

        raw_bytes = backup_path.read_bytes()

        # Decrypt if encrypted (.haksbackup or starts with Fernet header)
        if backup_path.suffix == ".haksbackup" or raw_bytes.startswith(b"gAAAA"):
            try:
                raw_bytes = self.encryption.decrypt_bytes(raw_bytes)
            except Exception as exc:
                raise ValueError(f"Failed to decrypt backup bundle: {exc}")

        zip_buffer = io.BytesIO(raw_bytes)
        if not zipfile.is_zipfile(zip_buffer):
            raise ValueError("Restored data is not a valid zip archive.")

        with zipfile.ZipFile(zip_buffer, mode="r") as zf:
            namelist = zf.namelist()
            if "manifest.json" not in namelist:
                raise ValueError("Corrupted backup: manifest.json missing from archive.")

            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
            archive_domain = manifest.get("domain")
            if archive_domain != expected_domain:
                raise ValueError(
                    f"Domain mismatch: backup is for domain '{archive_domain}', but restoration was requested for '{expected_domain}'."
                )

            # Flush WAL and checkpoint target before overwriting
            try:
                with self.db.session(expected_domain) as conn:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            except Exception:
                pass

            target_db_path = self.db.get_db_path(expected_domain)
            target_db_path.parent.mkdir(parents=True, exist_ok=True)

            # Extract database file
            db_rel_name = f"{expected_domain}.db"
            if db_rel_name in namelist:
                target_db_path.write_bytes(zf.read(db_rel_name))
                # Remove stale WAL/SHM companion files if present
                for companion in [Path(f"{target_db_path}-wal"), Path(f"{target_db_path}-shm")]:
                    if companion.exists():
                        try:
                            companion.unlink()
                        except Exception:
                            pass

            # Extract vector index if present in archive
            if "vector_index/index.json" in namelist:
                vector_index_path = self.config.paths.data_dir / expected_domain / "vector_index" / "index.json"
                vector_index_path.parent.mkdir(parents=True, exist_ok=True)
                vector_index_path.write_bytes(zf.read("vector_index/index.json"))

        # Verify restored database integrity
        with self.db.session(expected_domain) as conn:
            check_result = conn.execute("PRAGMA integrity_check;").fetchone()[0]
            if str(check_result).lower() != "ok":
                raise ValueError(f"Restored database integrity check failed: {check_result}")

        logger.info("Successfully restored domain '%s' from %s", expected_domain, backup_path.name)
        return True

    def export_domain_markdown(self, domain: str, output_path: Path) -> Path:
        """Export domain records as a clean, human-readable Markdown file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# HAKS Export: {domain.replace('_', ' ').title()}",
            f"*Exported on: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}*",
            "",
            "---",
            "",
        ]

        if domain == "personal_memory":
            with self.db.session(domain) as conn:
                rows = conn.execute("SELECT * FROM personal_memories WHERE is_archived = 0 ORDER BY importance DESC, created_at DESC;").fetchall()
                for r in rows:
                    dec_content = self.encryption.decrypt_field(str(r["content"]))
                    lines.append(f"## {r['title']}")
                    lines.append(f"- **ID:** `{r['id']}`")
                    lines.append(f"- **Category:** {r['category'] or 'general'}")
                    lines.append(f"- **Tags:** {r['tags'] or 'None'}")
                    lines.append(f"- **Importance:** {r['importance']}")
                    lines.append(f"- **Created:** {r['created_at']}")
                    lines.append("")
                    lines.append(dec_content)
                    lines.append("")
                    lines.append("---")
                    lines.append("")

        elif domain == "work_memory":
            with self.db.session(domain) as conn:
                rows = conn.execute("SELECT * FROM work_memories WHERE is_archived = 0 ORDER BY project_name ASC, importance DESC, created_at DESC;").fetchall()
                for r in rows:
                    dec_content = self.encryption.decrypt_field(str(r["content"]))
                    lines.append(f"## [{r['project_name']}] {r['title']}")
                    lines.append(f"- **ID:** `{r['id']}`")
                    lines.append(f"- **Project:** {r['project_name']}")
                    lines.append(f"- **Tags:** {r['tags'] or 'None'}")
                    lines.append(f"- **Importance:** {r['importance']}")
                    lines.append(f"- **Created:** {r['created_at']}")
                    lines.append("")
                    lines.append(dec_content)
                    lines.append("")
                    lines.append("---")
                    lines.append("")

        elif domain == "conversations":
            with self.db.session(domain) as conn:
                convs = conn.execute("SELECT * FROM conversations ORDER BY created_at DESC;").fetchall()
                for c in convs:
                    lines.append(f"## Conversation: {c['title']} (`{c['id']}`)")
                    lines.append(f"- **Model:** {c['model_used'] or 'unknown'}")
                    lines.append(f"- **Created:** {c['created_at']}")
                    lines.append("")
                    msgs = conn.execute("SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC;", (c["id"],)).fetchall()
                    for m in msgs:
                        role_tag = "**User**" if m["role"] == "user" else "**Assistant**"
                        lines.append(f"{role_tag} *({m['timestamp']})*:\n")
                        lines.append(f"{m['content']}\n")
                    lines.append("---")
                    lines.append("")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Exported domain '%s' to Markdown at %s", domain, output_path)
        return output_path

    def export_domain_json(self, domain: str, output_path: Path, decrypt: bool = True) -> Path:
        """Export domain records as structured JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        items = []

        if domain == "personal_memory":
            with self.db.session(domain) as conn:
                for r in conn.execute("SELECT * FROM personal_memories;").fetchall():
                    content = self.encryption.decrypt_field(str(r["content"])) if decrypt else str(r["content"])
                    items.append({
                        "id": str(r["id"]),
                        "title": str(r["title"]),
                        "content": content,
                        "category": str(r["category"] or "general"),
                        "tags": str(r["tags"] or ""),
                        "importance": int(r["importance"] or 1),
                        "created_at": str(r["created_at"]),
                        "updated_at": str(r["updated_at"]),
                        "is_archived": bool(r["is_archived"]),
                    })

        elif domain == "work_memory":
            with self.db.session(domain) as conn:
                for r in conn.execute("SELECT * FROM work_memories;").fetchall():
                    content = self.encryption.decrypt_field(str(r["content"])) if decrypt else str(r["content"])
                    items.append({
                        "id": str(r["id"]),
                        "project_name": str(r["project_name"] or "default"),
                        "title": str(r["title"]),
                        "content": content,
                        "tags": str(r["tags"] or ""),
                        "importance": int(r["importance"] or 1),
                        "created_at": str(r["created_at"]),
                        "updated_at": str(r["updated_at"]),
                        "is_archived": bool(r["is_archived"]),
                    })

        elif domain == "conversations":
            with self.db.session(domain) as conn:
                for c in conn.execute("SELECT * FROM conversations;").fetchall():
                    conv_id = str(c["id"])
                    msgs = [
                        {
                            "id": str(m["id"]),
                            "role": str(m["role"]),
                            "content": str(m["content"]),
                            "timestamp": str(m["timestamp"]),
                            "tokens": int(m["tokens"] or 0),
                        }
                        for m in conn.execute("SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC;", (conv_id,)).fetchall()
                    ]
                    items.append({
                        "id": conv_id,
                        "title": str(c["title"]),
                        "model_used": str(c["model_used"] or ""),
                        "created_at": str(c["created_at"]),
                        "messages": msgs,
                    })

        export_data = {
            "domain": domain,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "count": len(items),
            "decrypted": decrypt,
            "items": items,
        }
        output_path.write_text(json.dumps(export_data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Exported domain '%s' to JSON at %s (%d records)", domain, output_path, len(items))
        return output_path

    def export_domain_csv(self, domain: str, output_path: Path) -> Path:
        """Export domain records as a standard CSV file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if domain == "personal_memory":
            fieldnames = ["id", "title", "content", "category", "tags", "importance", "created_at", "updated_at"]
            with self.db.session(domain) as conn:
                rows = conn.execute("SELECT * FROM personal_memories WHERE is_archived = 0;").fetchall()
                with open(output_path, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for r in rows:
                        writer.writerow({
                            "id": r["id"],
                            "title": r["title"],
                            "content": self.encryption.decrypt_field(str(r["content"])),
                            "category": r["category"],
                            "tags": r["tags"],
                            "importance": r["importance"],
                            "created_at": r["created_at"],
                            "updated_at": r["updated_at"],
                        })

        elif domain == "work_memory":
            fieldnames = ["id", "project_name", "title", "content", "tags", "importance", "created_at", "updated_at"]
            with self.db.session(domain) as conn:
                rows = conn.execute("SELECT * FROM work_memories WHERE is_archived = 0;").fetchall()
                with open(output_path, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for r in rows:
                        writer.writerow({
                            "id": r["id"],
                            "project_name": r["project_name"],
                            "title": r["title"],
                            "content": self.encryption.decrypt_field(str(r["content"])),
                            "tags": r["tags"],
                            "importance": r["importance"],
                            "created_at": r["created_at"],
                            "updated_at": r["updated_at"],
                        })

        logger.info("Exported domain '%s' to CSV at %s", domain, output_path)
        return output_path

    def wipe_domain(self, domain: str, confirm_phrase: str) -> bool:
        """Securely wipe an individual domain.

        Requires exact confirm_phrase ('DELETE <DOMAIN>') as safety guard.
        Never touches any other domain.
        """
        expected_phrase = f"DELETE {domain.upper()}"
        if confirm_phrase.strip() != expected_phrase:
            raise ValueError(f"Confirmation phrase mismatch: expected '{expected_phrase}', got '{confirm_phrase}'")

        logger.warning("Initiating secure wipe for domain '%s'", domain)

        # Clear database records
        self.db.clear_domain(domain)
        self.db.vacuum_domain(domain)

        # Wipe vector index if applicable
        if domain in ("personal_memory", "work_memory"):
            vector_index_path = self.config.paths.data_dir / domain / "vector_index" / "index.json"
            if vector_index_path.exists():
                vector_index_path.write_text("{}", encoding="utf-8")

        logger.info("Secure wipe completed for domain '%s'. All other domains remain intact.", domain)
        return True
