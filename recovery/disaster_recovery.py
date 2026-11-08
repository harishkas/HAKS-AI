"""Disaster Recovery and Self-Healing Engine for HAKS Phase 8.

Performs deep SQLite integrity checks, vector index synchronization and rebuilding,
corrupted file preservation and quarantine, crash recovery detection, and safe restoration.
"""

from datetime import datetime, timezone
import io
import json
import logging
import os
from pathlib import Path
import shutil
import sqlite3
import time
from typing import Any, Optional
import uuid
import zipfile

from config.settings import AppConfig, get_settings
from core.events import ApplicationEvents, get_app_events
from memory.encryption import MemoryEncryptionService, get_encryption_service
from memory.vector_index import LocalVectorIndex
from recovery.backup_service import EnhancedBackupService
from recovery.models import (
    DatabaseIntegrityStatus,
    RecoveryReport,
    VectorIndexStatus,
)
from storage.database import DatabaseManager, get_db_manager
from storage.schema import DATABASE_SCHEMAS

logger = logging.getLogger("haks.recovery.disaster")


class DisasterRecoveryService:
    """Orchestrates system integrity checks, self-healing, crash recovery, and disaster restoration."""

    DOMAINS = [
        "personal_memory",
        "work_memory",
        "conversations",
        "memory_proposals",
        "settings",
    ]

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        db_manager: Optional[DatabaseManager] = None,
        encryption_service: Optional[MemoryEncryptionService] = None,
        backup_service: Optional[EnhancedBackupService] = None,
        events: Optional[ApplicationEvents] = None,
    ) -> None:
        self.config = config or get_settings()
        self.db = db_manager or get_db_manager(self.config)
        self.encryption = encryption_service or get_encryption_service(self.config)
        self.events = events or get_app_events()
        self.backup_service = backup_service or EnhancedBackupService(
            self.config, self.db, self.encryption, self.events
        )

        self.quarantine_dir = self.config.paths.data_dir / "quarantine"
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)

        self._session_lock_file = self.config.paths.data_dir / ".session_lock.json"
        self._unclean_shutdown = False

    # --------------------------------------------------------------------------
    # 1. Crash Recovery & Session Lifecycle
    # --------------------------------------------------------------------------

    def check_and_record_startup(self) -> bool:
        """Inspect previous session lock file. Return True if an unclean shutdown / crash was detected."""
        unclean = False
        if self._session_lock_file.exists():
            try:
                data = json.loads(self._session_lock_file.read_text(encoding="utf-8"))
                if not data.get("clean_exit", False):
                    unclean = True
                    prev_pid = data.get("pid")
                    prev_time = data.get("start_time")
                    logger.warning(
                        "UNCLEAN SHUTDOWN DETECTED! Previous session (PID %s, started %s) crashed or terminated abruptly.",
                        prev_pid,
                        prev_time,
                    )
            except Exception as exc:
                logger.warning("Could not parse previous session lock file: %s", exc)

        self._unclean_shutdown = unclean

        # Write active session lock
        new_state = {
            "pid": os.getpid(),
            "start_time": datetime.now(timezone.utc).isoformat(),
            "clean_exit": False,
            "version": self.config.ui.version,
        }
        try:
            self._session_lock_file.write_text(json.dumps(new_state, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Could not write session lock file: %s", exc)

        if unclean:
            # Run automatic self-healing integrity verification on all databases
            logger.info("Executing automatic post-crash integrity verification across all storage domains...")
            self.check_all_databases()

        return unclean

    def record_clean_shutdown(self) -> None:
        """Mark active session as cleanly terminated."""
        if self._session_lock_file.exists():
            try:
                data = json.loads(self._session_lock_file.read_text(encoding="utf-8"))
                data["clean_exit"] = True
                data["shutdown_time"] = datetime.now(timezone.utc).isoformat()
                self._session_lock_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
                logger.info("Clean application shutdown recorded in session lock.")
            except Exception as exc:
                logger.debug("Failed to record clean shutdown: %s", exc)

    @property
    def was_unclean_shutdown(self) -> bool:
        return self._unclean_shutdown

    # --------------------------------------------------------------------------
    # 2. Comprehensive SQLite Integrity Verification
    # --------------------------------------------------------------------------

    def check_database_integrity(self, domain: str) -> DatabaseIntegrityStatus:
        """Run PRAGMA quick_check and PRAGMA integrity_check on a specific database."""
        db_path = self.db.get_db_path(domain)
        status = DatabaseIntegrityStatus(domain=domain, db_path=str(db_path))

        if not db_path.exists():
            status.exists = False
            status.is_healthy = False
            status.errors.append("Database file does not exist on disk.")
            return status

        status.exists = True
        status.size_bytes = db_path.stat().st_size
        status.wal_active = Path(f"{db_path}-wal").exists()

        if status.size_bytes == 0:
            status.is_healthy = False
            status.errors.append("Database file is 0 bytes (empty/truncated).")
            return status

        # Direct SQLite connection for raw PRAGMA check
        conn = None
        try:
            conn = sqlite3.connect(str(db_path), timeout=5.0)
            conn.row_factory = sqlite3.Row

            # Quick check
            qc_row = conn.execute("PRAGMA quick_check;").fetchone()
            status.quick_check = str(qc_row[0]) if qc_row else "unknown"

            # Full integrity check
            ic_row = conn.execute("PRAGMA integrity_check;").fetchone()
            status.integrity_check = str(ic_row[0]) if ic_row else "unknown"

            # Table count
            tbl_rows = conn.execute("SELECT count(*) FROM sqlite_master WHERE type='table';").fetchone()
            status.table_count = int(tbl_rows[0]) if tbl_rows else 0

            # Count primary table records if applicable
            primary_table_map = {
                "personal_memory": "personal_memories",
                "work_memory": "work_memories",
                "conversations": "conversations",
                "memory_proposals": "memory_proposals",
                "settings": "app_settings",
            }
            target_table = primary_table_map.get(domain)
            if target_table:
                try:
                    c_row = conn.execute(f"SELECT count(*) FROM {target_table};").fetchone()
                    status.record_count = int(c_row[0]) if c_row else 0
                except Exception:
                    pass

            is_ok = (status.quick_check.lower() == "ok" and status.integrity_check.lower() == "ok")
            status.is_healthy = is_ok
            if not is_ok:
                status.errors.append(f"Integrity check failed: {status.integrity_check}")

        except Exception as exc:
            status.is_healthy = False
            status.errors.append(f"SQLite connection or query error: {exc}")
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

        return status

    def check_all_databases(self) -> dict[str, DatabaseIntegrityStatus]:
        """Perform integrity verification across all 5 system databases."""
        results = {}
        for domain in self.DOMAINS:
            res = self.check_database_integrity(domain)
            results[domain] = res
            if not res.is_healthy:
                logger.error("Database health check failed for domain '%s': %s", domain, res.errors)

        self.events.database_health_changed.emit({
            d: st.model_dump() for d, st in results.items()
        })
        return results

    # --------------------------------------------------------------------------
    # 3. Vector Index Synchronization & Self-Healing
    # --------------------------------------------------------------------------

    def check_vector_index(self, domain: str) -> VectorIndexStatus:
        """Inspect vector index synchronization against SQLite records."""
        idx_path = self.config.paths.data_dir / domain / "vector_index" / "index.json"
        status = VectorIndexStatus(domain=domain, index_path=str(idx_path))

        # 1. Count SQLite records
        db_count = 0
        target_table = "personal_memories" if domain == "personal_memory" else "work_memories"
        try:
            with self.db.session(domain) as conn:
                row = conn.execute(f"SELECT count(*) FROM {target_table};").fetchone()
                db_count = int(row[0]) if row else 0
        except Exception as exc:
            status.errors.append(f"Could not count SQLite records: {exc}")

        status.db_record_count = db_count

        # 2. Inspect Vector Index file
        if not idx_path.exists():
            status.exists = False
            status.is_synchronized = (db_count == 0)
            return status

        status.exists = True
        try:
            data = json.loads(idx_path.read_text(encoding="utf-8"))
            vectors_count = len(data) if isinstance(data, (dict, list)) else 0
            status.vector_count = vectors_count
            status.dimension = 384
            status.is_synchronized = (vectors_count == db_count)
            if not status.is_synchronized:
                status.errors.append(
                    f"Vector count mismatch: SQLite has {db_count} items, index has {vectors_count} vectors."
                )
        except Exception as exc:
            status.is_synchronized = False
            status.errors.append(f"Vector index JSON corrupted: {exc}")

        return status

    def rebuild_vector_index(self, domain: str) -> bool:
        """Self-heal and rebuild vector index from scratch using approved SQLite records."""
        if domain not in ("personal_memory", "work_memory"):
            raise ValueError(f"Domain '{domain}' does not support vector indexes.")

        idx_path = self.config.paths.data_dir / domain / "vector_index" / "index.json"

        # Preserve corrupted index if it exists
        if idx_path.exists():
            try:
                corrupted_backup = idx_path.parent / f"index_corrupted_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.bak"
                shutil.copy2(idx_path, corrupted_backup)
                logger.info("Preserved damaged vector index as %s before rebuilding.", corrupted_backup.name)
            except Exception as exc:
                logger.warning("Could not preserve damaged index: %s", exc)

        idx_path.parent.mkdir(parents=True, exist_ok=True)
        new_index = LocalVectorIndex(index_file=idx_path, encryption_service=self.encryption)

        try:
            target_table = "personal_memories" if domain == "personal_memory" else "work_memories"
            with self.db.session(domain) as conn:
                rows = conn.execute(f"SELECT id, content FROM {target_table};").fetchall()
                for r in rows:
                    item_id = str(r["id"])
                    text = str(r["content"])
                    # If field was encrypted, decrypt before embedding
                    if text.startswith("enc:") or text.startswith("gAAAA"):
                        try:
                            text = self.encryption.decrypt_field(text)
                        except Exception:
                            pass
                    new_index.add_document(item_id, text)

            new_index.save()
            logger.info("Successfully rebuilt vector index for '%s' with %d records.", domain, len(new_index))
            return True
        except Exception as exc:
            logger.error("Failed to rebuild vector index for '%s': %s", domain, exc)
            return False

    def verify_and_rebuild_indexes(self) -> dict[str, Any]:
        """Verify all memory vector indexes and automatically rebuild any out-of-sync indexes."""
        report = {}
        for domain in ("personal_memory", "work_memory"):
            st = self.check_vector_index(domain)
            rebuilt = False
            if not st.is_synchronized or not st.exists:
                rebuilt = self.rebuild_vector_index(domain)
            report[domain] = {
                "status": st.model_dump(),
                "rebuilt": rebuilt,
            }
        return report

    # --------------------------------------------------------------------------
    # 4. Corruption Quarantine & Pre-Recovery Snapshots
    # --------------------------------------------------------------------------

    def quarantine_corrupted_database(self, domain: str) -> Optional[Path]:
        """Preserve original damaged data before attempting repair or replacement.

        CRITICAL REQUIREMENT: Never overwrite or delete damaged user data without
        first creating an immutable quarantine copy.
        """
        db_path = self.db.get_db_path(domain)
        if not db_path.exists():
            return None

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        target_quarantine_dir = self.quarantine_dir / f"{domain}_corrupted_{ts}"
        target_quarantine_dir.mkdir(parents=True, exist_ok=True)

        copied: list[Path] = []
        for file in [db_path, Path(f"{db_path}-wal"), Path(f"{db_path}-shm")]:
            if file.exists():
                dest = target_quarantine_dir / file.name
                shutil.copy2(file, dest)
                copied.append(dest)

        logger.warning(
            "QUARANTINED corrupted database for domain '%s' into %s (%d files preserved).",
            domain,
            target_quarantine_dir,
            len(copied),
        )
        return target_quarantine_dir

    def create_pre_recovery_snapshot(self) -> Path:
        """Create a complete raw snapshot of all active databases before any restoration or migration."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        snapshot_dir = self.config.paths.backups_dir / "pre_recovery_snapshots" / f"snapshot_{ts}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        for domain in self.DOMAINS:
            # Checkpoint WAL
            try:
                with self.db.session(domain) as conn:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            except Exception:
                pass

            db_path = self.db.get_db_path(domain)
            if db_path.exists():
                shutil.copy2(db_path, snapshot_dir / f"{domain}.db")

        logger.info("Created pre-recovery safety snapshot at %s", snapshot_dir)
        return snapshot_dir

    # --------------------------------------------------------------------------
    # 5. Missing-File Detection & Safe Reconstruction
    # --------------------------------------------------------------------------

    def detect_and_repair_missing_files(self) -> RecoveryReport:
        """Inspect and self-heal missing database and index files."""
        report = RecoveryReport(
            report_id=f"missing_repair_{uuid.uuid4().hex[:8]}",
            operation="missing_file_repair",
        )

        for domain in self.DOMAINS:
            db_path = self.db.get_db_path(domain)
            if not db_path.exists():
                logger.warning("Missing database detected: %s", db_path)
                # Attempt to find latest backup containing this domain
                restored = False
                backups = self.backup_service.list_backups()
                for b_meta in backups:
                    if domain in b_meta.domains or b_meta.backup_type == "full_system":
                        try:
                            self.restore_system(Path(b_meta.path), specific_domains=[domain], preserve_current=False)
                            report.restored_domains.append(domain)
                            restored = True
                            logger.info("Restored missing domain '%s' from backup %s", domain, b_meta.filename)
                            break
                        except Exception as exc:
                            logger.warning("Could not restore '%s' from backup %s: %s", domain, b_meta.filename, exc)

                if not restored:
                    # Recreate fresh schema so app does not crash
                    try:
                        self.db.initialize()
                        logger.info("Initialized fresh schema for missing domain: %s", domain)
                        report.restored_domains.append(f"{domain} (fresh schema)")
                    except Exception as exc:
                        report.errors.append(f"Failed to recreate schema for {domain}: {exc}")

        # Check and rebuild vector indexes if missing
        idx_res = self.verify_and_rebuild_indexes()
        for dom, info in idx_res.items():
            if info.get("rebuilt"):
                report.rebuilt_indexes.append(dom)

        report.success = len(report.errors) == 0
        return report

    # --------------------------------------------------------------------------
    # 6. Disaster Restoration Engine
    # --------------------------------------------------------------------------

    def restore_system(
        self,
        backup_path: Path,
        target_dir: Optional[Path] = None,
        specific_domains: Optional[list[str]] = None,
        preserve_current: bool = True,
    ) -> RecoveryReport:
        """Perform full or selective restoration from a verified backup archive.

        Args:
            backup_path: Path to .haksbackup or .zip file.
            target_dir: Optional staging / test directory. If None, restores to live system data_dir.
            specific_domains: Optional list of domains to selectively restore.
            preserve_current: Whether to capture a pre-recovery safety snapshot before overwriting.
        """
        report = RecoveryReport(
            report_id=f"restore_{uuid.uuid4().hex[:8]}",
            operation="full_restore" if not specific_domains else "domain_restore",
        )

        if not backup_path.exists():
            report.errors.append(f"Backup file not found: {backup_path}")
            return report

        # 1. Verify backup validity first
        v_res = self.backup_service.verify_backup(backup_path)
        if not v_res.is_valid:
            report.errors.extend(v_res.errors + v_res.checksum_mismatches)
            return report

        # 2. Preserve current data if restoring to active system
        if preserve_current and target_dir is None:
            snapshot_path = self.create_pre_recovery_snapshot()
            report.pre_recovery_snapshot_path = str(snapshot_path)

        # 3. Read and decrypt archive
        raw_bytes = backup_path.read_bytes()
        if backup_path.suffix == ".haksbackup" or raw_bytes.startswith(b"gAAAA"):
            try:
                raw_bytes = self.encryption.decrypt_bytes(raw_bytes)
            except Exception as exc:
                report.errors.append(f"Decryption failed during restoration: {exc}")
                return report

        zip_buf = io.BytesIO(raw_bytes)
        dest_base_dir = target_dir or self.config.paths.data_dir

        try:
            with zipfile.ZipFile(zip_buf, mode="r") as zf:
                namelist = zf.namelist()
                target_domains = specific_domains or self.DOMAINS

                for domain in target_domains:
                    # Check for full-system format ("databases/<domain>.db") or legacy domain format ("<domain>.db")
                    arc_db_path = f"databases/{domain}.db"
                    if arc_db_path not in namelist and f"{domain}.db" in namelist:
                        arc_db_path = f"{domain}.db"

                    if arc_db_path in namelist:
                        # Quarantine current file if corrupt before overwrite
                        if target_dir is None:
                            curr_status = self.check_database_integrity(domain)
                            if not curr_status.is_healthy and curr_status.exists:
                                q_path = self.quarantine_corrupted_database(domain)
                                if q_path:
                                    report.quarantined_files.append(str(q_path))

                        target_db_path = dest_base_dir / domain / f"{domain}.db"
                        target_db_path.parent.mkdir(parents=True, exist_ok=True)

                        # Write database file
                        target_db_path.write_bytes(zf.read(arc_db_path))

                        # Clean up stale WAL/SHM companion files
                        for comp in [Path(f"{target_db_path}-wal"), Path(f"{target_db_path}-shm")]:
                            if comp.exists():
                                try:
                                    comp.unlink()
                                except Exception:
                                    pass

                        # Verify restored database
                        conn = sqlite3.connect(str(target_db_path))
                        check = conn.execute("PRAGMA integrity_check;").fetchone()[0]
                        conn.close()
                        if str(check).lower() != "ok":
                            report.errors.append(f"Restored DB for '{domain}' failed integrity check: {check}")
                        else:
                            report.restored_domains.append(domain)

                    # Restore vector indexes if present
                    if domain in ("personal_memory", "work_memory"):
                        arc_idx = f"indexes/{domain}_index.json"
                        if arc_idx not in namelist and "vector_index/index.json" in namelist:
                            arc_idx = "vector_index/index.json"

                        if arc_idx in namelist:
                            t_idx = dest_base_dir / domain / "vector_index" / "index.json"
                            t_idx.parent.mkdir(parents=True, exist_ok=True)
                            t_idx.write_bytes(zf.read(arc_idx))
                            report.rebuilt_indexes.append(f"{domain} (restored)")
                        else:
                            # Rebuild vector index from the freshly restored SQLite records
                            if target_dir is None:
                                self.rebuild_vector_index(domain)
                                report.rebuilt_indexes.append(f"{domain} (rebuilt)")

            report.success = len(report.errors) == 0

        except Exception as exc:
            report.errors.append(f"Restoration exception: {exc}")
            report.success = False

        self.events.recovery_operation_completed.emit(report.model_dump())
        return report
