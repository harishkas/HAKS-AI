"""Safe Storage Directory Migration Service for HAKS.

Coordinates seamless, verified migration of the HAKS data directory:
- Pre-flight checks (permissions, disk space, .venv / source code guards)
- Non-destructive conflict detection
- WAL log flushes (checkpoints)
- Cryptographic SHA-256 byte-level verification of all copied files
- SQLite PRAGMA integrity_check on destination databases
- Safe preservation of source data
"""

import hashlib
import logging
import os
from pathlib import Path
import shutil
import sqlite3
from typing import Optional

from config.settings import AppConfig, PROJECT_ROOT, get_settings
from storage.database import DatabaseManager, get_db_manager

logger = logging.getLogger("haks.storage.migration")


class StorageMigrationService:
    """Safely migrates HAKS storage domain databases and vector indexes to a new location."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        db_manager: Optional[DatabaseManager] = None,
    ) -> None:
        self.config = config or get_settings()
        self.db = db_manager or get_db_manager(self.config)

    @staticmethod
    def calculate_file_hash(file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def validate_target_directory(self, new_data_dir: Path) -> tuple[bool, str]:
        """Validate target directory location, permissions, and available disk space."""
        try:
            target = Path(new_data_dir).resolve()
        except Exception as exc:
            return False, f"Invalid path syntax: {exc}"

        current = self.config.paths.data_dir.resolve()
        if target == current:
            return False, "Target directory is identical to current data directory."

        # Guard against storing in .venv
        for part in target.parts:
            if part.lower() in (".venv", "venv", "__pycache__"):
                return False, "Cannot store HAKS memory data inside a Python virtual environment directory."

        # Verify write permissions by probing
        try:
            target.mkdir(parents=True, exist_ok=True)
            probe_file = target / ".haks_probe.tmp"
            probe_file.write_text("probe", encoding="utf-8")
            probe_file.unlink()
        except Exception as exc:
            return False, f"Target directory is not writable: {exc}"

        # Check free disk space (require at least 100 MB free)
        try:
            total, used, free = shutil.disk_usage(target)
            if free < 100 * 1024 * 1024:
                return False, f"Insufficient disk space on target volume: {free / (1024*1024):.1f} MB available (100 MB required)."
        except Exception as exc:
            logger.warning("Could not determine disk usage for %s: %s", target, exc)

        return True, "OK"

    def check_target_conflicts(self, new_data_dir: Path) -> list[str]:
        """Check if destination directory already contains conflicting HAKS database files."""
        conflicts = []
        target = Path(new_data_dir).resolve()
        domain_dbs = [
            "personal_memory/personal_memory.db",
            "work_memory/work_memory.db",
            "conversations/conversations.db",
            "memory_proposals/memory_proposals.db",
            "settings/settings.db",
        ]
        for rel_path in domain_dbs:
            candidate = target / rel_path
            if candidate.exists() and candidate.stat().st_size > 0:
                conflicts.append(str(candidate))
        return conflicts

    def migrate_data_directory(
        self,
        new_data_dir: Path,
        overwrite_approved: bool = False,
    ) -> tuple[bool, str]:
        """Safely migrate all HAKS storage domains to a new data directory.

        Args:
            new_data_dir: Destination directory path
            overwrite_approved: Must be True if existing data was detected at target

        Returns:
            tuple[bool, str]: (Success, message or error explanation)
        """
        valid, msg = self.validate_target_directory(new_data_dir)
        if not valid:
            logger.error("Migration pre-flight validation failed: %s", msg)
            return False, msg

        target_dir = Path(new_data_dir).resolve()
        source_dir = self.config.paths.data_dir.resolve()

        # Step 2: Target Conflict Check
        conflicts = self.check_target_conflicts(target_dir)
        if conflicts and not overwrite_approved:
            err_msg = (
                f"Target directory contains existing data ({len(conflicts)} conflicting files). "
                "Migration aborted to prevent accidental data loss. Explicit user approval is required to overwrite."
            )
            logger.warning(err_msg)
            return False, err_msg

        logger.info("Starting safe migration from %s to %s", source_dir, target_dir)

        # Step 3: Checkpoint all active SQLite connections
        self.db.checkpoint_all()

        # Step 4: Ensure target domain directories exist
        subdirectories = [
            target_dir / "personal_memory",
            target_dir / "personal_memory" / "vector_index",
            target_dir / "work_memory",
            target_dir / "work_memory" / "vector_index",
            target_dir / "conversations",
            target_dir / "memory_proposals",
            target_dir / "settings",
            target_dir / "backups",
            target_dir / ".keys",
        ]
        for d in subdirectories:
            d.mkdir(parents=True, exist_ok=True)

        copied_files: list[tuple[Path, Path, str]] = []

        try:
            # Step 5: Copy all files recursively with checksum verification
            for root, _, files in os.walk(source_dir):
                root_path = Path(root)
                rel_root = root_path.relative_to(source_dir)

                for f in files:
                    # Skip temporary SQLite journal files during copy
                    if f.endswith(("-wal", "-shm")):
                        continue

                    src_file = root_path / f
                    dst_file = target_dir / rel_root / f
                    dst_file.parent.mkdir(parents=True, exist_ok=True)

                    shutil.copy2(src_file, dst_file)
                    src_hash = self.calculate_file_hash(src_file)
                    dst_hash = self.calculate_file_hash(dst_file)

                    if src_hash != dst_hash:
                        raise ValueError(f"Checksum mismatch on {f}: {src_hash} != {dst_hash}")

                    copied_files.append((src_file, dst_file, src_hash))

            # Step 6: Verify destination database integrity
            for db_rel in [
                "personal_memory/personal_memory.db",
                "work_memory/work_memory.db",
                "conversations/conversations.db",
                "memory_proposals/memory_proposals.db",
                "settings/settings.db",
            ]:
                target_db = target_dir / db_rel
                if target_db.exists():
                    conn = sqlite3.connect(str(target_db))
                    try:
                        cursor = conn.cursor()
                        cursor.execute("PRAGMA integrity_check;")
                        res = cursor.fetchone()[0]
                        if str(res).lower() != "ok":
                            raise ValueError(f"Integrity check failed for {db_rel}: {res}")
                    finally:
                        conn.close()

        except Exception as exc:
            logger.error("Migration failed during file copy/verification: %s. Cleaning up copied files...", exc)
            for _, dst_file, _ in copied_files:
                if dst_file.exists():
                    try:
                        dst_file.unlink()
                    except Exception:
                        pass
            return False, f"Migration failed: {exc}"

        # Step 7: Update active configuration and database manager
        self.config.paths.data_dir = target_dir
        self.config.paths.backups_dir = target_dir / "backups"
        self.db.data_dir = target_dir

        logger.info(
            "Migration completed successfully. Copied %d verified files. Source directory preserved at %s.",
            len(copied_files),
            source_dir,
        )
        return True, f"Successfully migrated {len(copied_files)} files to {target_dir}."
