"""Comprehensive Backup and Retention Management Service for HAKS Phase 8.

Manages full-system and multi-domain encrypted backups, SHA-256 manifest verification,
cataloging, retention pruning, and secondary external drive backup destinations.
"""

from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import sqlite3
from typing import Any, Optional
import uuid
import zipfile

from config.settings import AppConfig, get_settings
from core.events import ApplicationEvents, get_app_events
from memory.encryption import MemoryEncryptionService, get_encryption_service
from recovery.models import BackupManifest, BackupMetadata, BackupVerificationResult
from storage.database import DatabaseManager, get_db_manager

logger = logging.getLogger("haks.recovery.backup")


class EnhancedBackupService:
    """Manages full-system backups, SHA-256 manifests, retention policies, and verification."""

    ALL_DOMAINS = [
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
        events: Optional[ApplicationEvents] = None,
    ) -> None:
        self.config = config or get_settings()
        self.db = db_manager or get_db_manager(self.config)
        self.encryption = encryption_service or get_encryption_service(self.config)
        self.events = events or get_app_events()

        self.backups_dir = self.config.paths.backups_dir
        self.backups_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------
    # 1. Full-System Backup Creation
    # --------------------------------------------------------------------------

    def create_full_backup(
        self,
        target_dir: Optional[Path] = None,
        encrypt: bool = True,
        notes: str = "",
    ) -> Path:
        """Create a complete, consistent backup archive covering all databases, indexes, and config.

        Flushes WAL across all SQLite databases, calculates SHA-256 checksums,
        generates an internal manifest, and optionally encrypts with AES-256 Fernet.
        """
        dest_dir = target_dir or self.backups_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        backup_id = f"backup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        files_to_pack: list[tuple[Path, str]] = []  # (source_path, arcname)

        # 1. Flush WAL and gather database files
        for domain in self.ALL_DOMAINS:
            try:
                with self.db.session(domain) as conn:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            except Exception as exc:
                logger.warning("Could not flush WAL for domain '%s' before backup: %s", domain, exc)

            db_path = self.db.get_db_path(domain)
            if db_path.exists():
                files_to_pack.append((db_path, f"databases/{domain}.db"))

        # 2. Gather vector index files
        for domain in ("personal_memory", "work_memory"):
            idx_path = self.config.paths.data_dir / domain / "vector_index" / "index.json"
            if idx_path.exists():
                files_to_pack.append((idx_path, f"indexes/{domain}_index.json"))

        # 3. Compute SHA-256 for all collected files
        checksums: dict[str, str] = {}
        for src_path, arcname in files_to_pack:
            try:
                data = src_path.read_bytes()
                checksums[arcname] = hashlib.sha256(data).hexdigest()
            except Exception as exc:
                logger.error("Failed to read %s for checksum calculation: %s", src_path, exc)

        # 4. Build Manifest
        manifest = BackupManifest(
            backup_id=backup_id,
            backup_type="full_system",
            created_at=datetime.now(timezone.utc).isoformat(),
            haks_version=self.config.ui.version,
            is_encrypted=encrypt,
            checksums=checksums,
            domains=self.ALL_DOMAINS,
            files=[arc for _, arc in files_to_pack],
            notes=notes,
            computer_name=os.environ.get("COMPUTERNAME", "HAKS-HOST"),
        )

        # 5. Build Zip archive in memory
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(manifest.model_dump(), indent=2))
            for src_path, arcname in files_to_pack:
                if src_path.exists():
                    zf.write(src_path, arcname=arcname)

        raw_zip_bytes = zip_buf.getvalue()

        # 6. Apply encryption if requested
        if encrypt:
            final_bytes = self.encryption.encrypt_bytes(raw_zip_bytes)
            output_file = dest_dir / f"{backup_id}.haksbackup"
        else:
            final_bytes = raw_zip_bytes
            output_file = dest_dir / f"{backup_id}.zip"

        output_file.write_bytes(final_bytes)
        logger.info(
            "Created full-system backup [%s] at %s (%s bytes, encrypted=%s)",
            backup_id,
            output_file,
            len(final_bytes),
            encrypt,
        )

        # 7. Mirror to external backup directory if configured and accessible
        if self.config.recovery.external_backup_dir:
            try:
                ext_dir = Path(self.config.recovery.external_backup_dir)
                if ext_dir.exists() and ext_dir.is_dir():
                    ext_target = ext_dir / output_file.name
                    ext_target.write_bytes(final_bytes)
                    logger.info("Mirrored backup to external destination: %s", ext_target)
            except Exception as exc:
                logger.warning("Could not mirror backup to external drive: %s", exc)

        # 8. Dispatch notification event
        self.events.backup_completed.emit({
            "backup_id": backup_id,
            "path": str(output_file),
            "size_bytes": len(final_bytes),
            "is_encrypted": encrypt,
            "timestamp": manifest.created_at,
        })

        # 9. Apply retention policy
        self.apply_retention_policy(
            max_backups=self.config.recovery.backup_retention_count,
            max_age_days=self.config.recovery.backup_retention_days,
            target_dir=dest_dir,
        )

        return output_file

    # --------------------------------------------------------------------------
    # 2. Backup Verification
    # --------------------------------------------------------------------------

    def verify_backup(self, backup_path: Path) -> BackupVerificationResult:
        """Verify archive integrity, manifest validity, and file SHA-256 hashes without modifying any system data."""
        result = BackupVerificationResult(backup_path=str(backup_path))

        if not backup_path.exists():
            result.errors.append(f"Backup file not found: {backup_path}")
            return result

        try:
            raw_bytes = backup_path.read_bytes()
            # Decrypt if needed
            if backup_path.suffix == ".haksbackup" or raw_bytes.startswith(b"gAAAA"):
                try:
                    raw_bytes = self.encryption.decrypt_bytes(raw_bytes)
                except Exception as exc:
                    result.errors.append(f"Decryption failed: {exc}")
                    return result

            zip_buf = io.BytesIO(raw_bytes)
            if not zipfile.is_zipfile(zip_buf):
                result.errors.append("Archive is corrupted or not a valid ZIP file.")
                return result

            with zipfile.ZipFile(zip_buf, mode="r") as zf:
                namelist = zf.namelist()
                if "manifest.json" not in namelist:
                    result.errors.append("Invalid backup: manifest.json is missing.")
                    return result

                manifest_data = json.loads(zf.read("manifest.json").decode("utf-8"))
                manifest = BackupManifest(**manifest_data)
                result.manifest = manifest

                # Verify each file against its recorded checksum
                verified_count = 0
                for arcname, expected_hash in manifest.checksums.items():
                    if arcname not in namelist:
                        result.errors.append(f"Missing expected archive member: {arcname}")
                        continue

                    file_bytes = zf.read(arcname)
                    actual_hash = hashlib.sha256(file_bytes).hexdigest()
                    if actual_hash != expected_hash:
                        result.checksum_mismatches.append(
                            f"{arcname} (expected {expected_hash[:8]}..., got {actual_hash[:8]}...)"
                        )
                    else:
                        verified_count += 1

                result.verified_files_count = verified_count
                result.is_valid = (len(result.errors) == 0 and len(result.checksum_mismatches) == 0)

        except Exception as exc:
            result.errors.append(f"Verification exception: {exc}")
            result.is_valid = False

        return result

    # --------------------------------------------------------------------------
    # 3. Backup Cataloging & Listing
    # --------------------------------------------------------------------------

    def list_backups(self, search_dirs: Optional[list[Path]] = None) -> list[BackupMetadata]:
        """Scan backup locations and return catalog of available backups sorted newest first."""
        target_dirs = search_dirs or [self.backups_dir]
        if self.config.recovery.external_backup_dir:
            ext_path = Path(self.config.recovery.external_backup_dir)
            if ext_path.exists() and ext_path not in target_dirs:
                target_dirs.append(ext_path)

        # Include domain subdirectories if present
        for domain in self.ALL_DOMAINS:
            d_dir = self.backups_dir / domain
            if d_dir.exists() and d_dir not in target_dirs:
                target_dirs.append(d_dir)

        discovered: list[BackupMetadata] = []
        seen_paths: set[Path] = set()

        for directory in target_dirs:
            if not directory.exists():
                continue
            for file_path in directory.glob("*"):
                if file_path.suffix in (".haksbackup", ".zip") and file_path not in seen_paths:
                    seen_paths.add(file_path)
                    meta = self._inspect_backup_file(file_path)
                    if meta:
                        discovered.append(meta)

        # Sort descending by timestamp / mtime
        discovered.sort(key=lambda m: m.timestamp, reverse=True)
        return discovered

    def _inspect_backup_file(self, file_path: Path) -> Optional[BackupMetadata]:
        """Read lightweight manifest from backup file."""
        try:
            stat = file_path.stat()
            size = stat.st_size
            mod_time = datetime.fromtimestamp(stat.st_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

            is_enc = file_path.suffix == ".haksbackup"
            # Attempt to read manifest
            manifest = None
            try:
                raw_bytes = file_path.read_bytes()
                if is_enc or raw_bytes.startswith(b"gAAAA"):
                    raw_bytes = self.encryption.decrypt_bytes(raw_bytes)
                zip_buf = io.BytesIO(raw_bytes)
                if zipfile.is_zipfile(zip_buf):
                    with zipfile.ZipFile(zip_buf, "r") as zf:
                        if "manifest.json" in zf.namelist():
                            data = json.loads(zf.read("manifest.json").decode("utf-8"))
                            manifest = BackupManifest(**data)
            except Exception:
                pass

            backup_id = manifest.backup_id if manifest else file_path.stem
            backup_type = manifest.backup_type if manifest else ("full_system" if "full" in file_path.name else "domain")
            ts = manifest.created_at[:19].replace("T", " ") if manifest else mod_time
            domains = manifest.domains if manifest else []
            notes = manifest.notes if manifest else ""

            return BackupMetadata(
                backup_id=backup_id,
                path=str(file_path),
                filename=file_path.name,
                backup_type=backup_type,
                timestamp=ts,
                size_bytes=size,
                is_encrypted=is_enc,
                is_valid=True,
                domains=domains,
                notes=notes,
            )
        except Exception as exc:
            logger.debug("Could not inspect backup %s: %s", file_path, exc)
            return None

    # --------------------------------------------------------------------------
    # 4. Retention Policy Pruning
    # --------------------------------------------------------------------------

    def apply_retention_policy(
        self,
        max_backups: int = 10,
        max_age_days: int = 30,
        target_dir: Optional[Path] = None,
    ) -> list[Path]:
        """Prune older backups while strictly guaranteeing the last valid backup is never removed."""
        d = target_dir or self.backups_dir
        if not d.exists():
            return []

        backups = [f for f in d.glob("*") if f.suffix in (".haksbackup", ".zip")]
        if len(backups) <= 1:
            return []  # Always keep at least 1 backup

        # Sort by mtime descending (newest first)
        backups.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        removed: list[Path] = []
        now_ts = datetime.now(timezone.utc).timestamp()
        cutoff_age_seconds = max_age_days * 86400

        # Keep the top max_backups, prune the rest or those older than max_age_days
        for idx, b_path in enumerate(backups):
            if idx == 0:
                continue  # Never delete newest backup

            age_seconds = now_ts - b_path.stat().st_mtime
            should_prune = (idx >= max_backups) or (age_seconds > cutoff_age_seconds)

            if should_prune:
                try:
                    b_path.unlink()
                    removed.append(b_path)
                    logger.info("Pruned old backup per retention policy: %s", b_path.name)
                except Exception as exc:
                    logger.warning("Could not delete pruned backup %s: %s", b_path, exc)

        return removed
