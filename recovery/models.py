"""Data models and schemas for Phase 8 Offline Operation and Disaster Recovery."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from pydantic import BaseModel, Field


class NetworkStatus(BaseModel):
    """Real-time internet connectivity status."""
    is_online: bool = False
    last_checked: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    )
    latency_ms: Optional[float] = None
    status_message: str = "Checking connectivity..."
    probe_host: str = "1.1.1.1"


class DatabaseIntegrityStatus(BaseModel):
    """Health and SQLite structural integrity evaluation for a database domain."""
    domain: str
    db_path: str
    exists: bool = False
    size_bytes: int = 0
    is_healthy: bool = False
    quick_check: str = "unchecked"
    integrity_check: str = "unchecked"
    table_count: int = 0
    wal_active: bool = False
    record_count: int = 0
    errors: list[str] = Field(default_factory=list)


class VectorIndexStatus(BaseModel):
    """Consistency evaluation between SQLite records and in-memory/file vector index."""
    domain: str
    index_path: str
    exists: bool = False
    vector_count: int = 0
    db_record_count: int = 0
    is_synchronized: bool = False
    dimension: int = 384
    errors: list[str] = Field(default_factory=list)


class BackupManifest(BaseModel):
    """Manifest header embedded inside every backup archive bundle."""
    backup_id: str
    backup_type: str = "full_system"  # full_system, personal_memory, work_memory, etc.
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    haks_version: str = "0.1.0"
    is_encrypted: bool = True
    checksums: dict[str, str] = Field(default_factory=dict)  # relative file path -> SHA-256
    database_versions: dict[str, int] = Field(default_factory=dict)
    domains: list[str] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)
    notes: str = ""
    computer_name: str = ""


class BackupMetadata(BaseModel):
    """Catalog entry for an existing backup archive."""
    backup_id: str
    path: str
    filename: str
    backup_type: str = "full_system"
    timestamp: str
    size_bytes: int = 0
    is_encrypted: bool = True
    is_valid: bool = True
    domains: list[str] = Field(default_factory=list)
    notes: str = ""


class BackupVerificationResult(BaseModel):
    """Diagnostic outcome of backup verification without full extraction."""
    is_valid: bool = False
    backup_path: str
    manifest: Optional[BackupManifest] = None
    verified_files_count: int = 0
    checksum_mismatches: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class RecoveryReport(BaseModel):
    """Structured report detailing actions taken during disaster recovery or restoration."""
    report_id: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    )
    operation: str  # full_restore, domain_restore, crash_recovery, self_heal, index_rebuild
    success: bool = False
    pre_recovery_snapshot_path: Optional[str] = None
    quarantined_files: list[str] = Field(default_factory=list)
    restored_domains: list[str] = Field(default_factory=list)
    rebuilt_indexes: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class SystemHealthOverview(BaseModel):
    """System-wide summary of network, AI engine, databases, and backup status."""
    network: NetworkStatus
    active_engine: str = "local"
    local_model_available: bool = False
    local_model_name: str = "llama3.2:1b"
    databases: dict[str, DatabaseIntegrityStatus] = Field(default_factory=dict)
    vector_indexes: dict[str, VectorIndexStatus] = Field(default_factory=dict)
    last_backup_time: Optional[str] = None
    total_backups_count: int = 0
    storage_destination: str = ""
    unclean_shutdown_detected: bool = False
    quarantined_backups_count: int = 0
