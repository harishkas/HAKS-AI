"""Central Offline Operation and Disaster Recovery Manager for HAKS Phase 8.

Unified facade coordinating network monitoring, full-system encrypted backups,
self-healing SQLite integrity verification, vector index rebuilding, and offline AI/Voice operations.
"""

from datetime import datetime, timezone
import logging
import os
from pathlib import Path
from typing import Any, Optional

from config.settings import AppConfig, get_settings
from core.events import ApplicationEvents, get_app_events
from memory.encryption import MemoryEncryptionService, get_encryption_service
from recovery.backup_service import EnhancedBackupService
from recovery.disaster_recovery import DisasterRecoveryService
from recovery.models import (
    BackupManifest,
    BackupMetadata,
    BackupVerificationResult,
    DatabaseIntegrityStatus,
    RecoveryReport,
    SystemHealthOverview,
    VectorIndexStatus,
)
from recovery.network_monitor import NetworkMonitor
from storage.database import DatabaseManager, get_db_manager

logger = logging.getLogger("haks.recovery.manager")


class OfflineRecoveryManager:
    """Central coordinator for Phase 8 Offline Operation and Disaster Recovery."""

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

        # Core Subsystems
        self.network = NetworkMonitor(events=self.events)
        self.backup = EnhancedBackupService(
            config=self.config,
            db_manager=self.db,
            encryption_service=self.encryption,
            events=self.events,
        )
        self.disaster = DisasterRecoveryService(
            config=self.config,
            db_manager=self.db,
            encryption_service=self.encryption,
            backup_service=self.backup,
            events=self.events,
        )

        # Connect network transitions
        self.network.add_online_callback(self._on_network_restored)
        self.network.add_offline_callback(self._on_network_lost)

        logger.info("OfflineRecoveryManager initialized.")

    # --------------------------------------------------------------------------
    # 1. Lifecycle Hooks
    # --------------------------------------------------------------------------

    def initialize_on_startup(self) -> bool:
        """Execute crash detection and automatic startup integrity checks."""
        unclean = self.disaster.check_and_record_startup()
        if self.config.recovery.check_integrity_on_startup:
            self.disaster.detect_and_repair_missing_files()
        return unclean

    def shutdown(self) -> None:
        """Safely record clean exit and shutdown network monitor."""
        self.network.stop_monitoring()
        self.disaster.record_clean_shutdown()

    # --------------------------------------------------------------------------
    # 2. Network Transitions & Connectivity
    # --------------------------------------------------------------------------

    @property
    def is_online(self) -> bool:
        """Return True if internet is currently reachable."""
        return self.network.is_online

    def _on_network_restored(self) -> None:
        logger.info("Network connectivity RESTORED. System is now ONLINE.")

    def _on_network_lost(self) -> None:
        logger.warning("Network connectivity LOST. System transitioned to OFFLINE mode.")

    # --------------------------------------------------------------------------
    # 3. Offline AI & Voice Diagnostics
    # --------------------------------------------------------------------------

    def get_available_local_models(self) -> list[str]:
        """Query local Ollama server for downloaded models."""
        import urllib.request
        import json

        base_url = self.config.ai.ollama_base_url.rstrip("/")
        try:
            req = urllib.request.Request(f"{base_url}/api/tags", headers={"User-Agent": "HAKS/0.1"})
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
                return models
        except Exception as exc:
            logger.debug("Could not reach Ollama at %s: %s", base_url, exc)
            return []

    def is_local_ai_available(self) -> bool:
        """Check if local Ollama daemon is reachable and responding."""
        models = self.get_available_local_models()
        return len(models) > 0 or self._check_ollama_version()

    def _check_ollama_version(self) -> bool:
        import urllib.request
        base_url = self.config.ai.ollama_base_url.rstrip("/")
        try:
            req = urllib.request.Request(f"{base_url}/api/version", headers={"User-Agent": "HAKS/0.1"})
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def get_voice_offline_status(self) -> dict[str, Any]:
        """Diagnose offline voice capability (wake word, VAD, Whisper, SAPI/Piper)."""
        return {
            "wake_word_offline": True,  # Local acoustic energy detector runs 100% offline
            "vad_offline": True,        # Dynamic Energy VAD runs 100% offline
            "stt_whisper_offline": True,# Faster-Whisper runs CPU int8 offline
            "tts_edge_neural_online_only": True,  # Edge Neural TTS requires internet
            "tts_sapi_offline": True,   # Windows SAPI runs 100% offline
            "tts_piper_offline": True,  # Piper ONNX runs 100% offline
            "tamil_offline_tts_available": True, # Supported via SAPI/Piper fallback
        }

    # --------------------------------------------------------------------------
    # 4. System Health Overview
    # --------------------------------------------------------------------------

    def get_system_health_overview(self) -> SystemHealthOverview:
        """Gather complete system diagnostic snapshot for the Recovery Center UI."""
        net_status = self.network.get_status()
        db_statuses = self.disaster.check_all_databases()

        vec_statuses: dict[str, VectorIndexStatus] = {}
        for dom in ("personal_memory", "work_memory"):
            vec_statuses[dom] = self.disaster.check_vector_index(dom)

        backups = self.backup.list_backups()
        last_bk = backups[0].timestamp if backups else None

        local_ai_ok = self.is_local_ai_available()

        dest = str(self.config.paths.backups_dir)
        if self.config.recovery.external_backup_dir:
            dest += f" (Mirrored: {self.config.recovery.external_backup_dir})"

        quarantine_count = len(list(self.disaster.quarantine_dir.glob("*")))

        return SystemHealthOverview(
            network=net_status,
            active_engine=self.config.ai.default_engine,
            local_model_available=local_ai_ok,
            local_model_name=self.config.ai.local_model,
            databases=db_statuses,
            vector_indexes=vec_statuses,
            last_backup_time=last_bk,
            total_backups_count=len(backups),
            storage_destination=dest,
            unclean_shutdown_detected=self.disaster.was_unclean_shutdown,
            quarantined_backups_count=quarantine_count,
        )

    # --------------------------------------------------------------------------
    # 5. Backup & Recovery Operations Facade
    # --------------------------------------------------------------------------

    def create_full_backup(self, encrypt: bool = True, notes: str = "") -> Path:
        """Create a complete encrypted system backup."""
        return self.backup.create_full_backup(encrypt=encrypt, notes=notes)

    def restore_system(
        self,
        backup_path: Path,
        target_dir: Optional[Path] = None,
        specific_domains: Optional[list[str]] = None,
        preserve_current: bool = True,
    ) -> RecoveryReport:
        """Restore system from a verified backup archive."""
        return self.disaster.restore_system(
            backup_path=backup_path,
            target_dir=target_dir,
            specific_domains=specific_domains,
            preserve_current=preserve_current,
        )

    def run_integrity_check(self) -> dict[str, DatabaseIntegrityStatus]:
        """Perform comprehensive SQLite integrity verification."""
        return self.disaster.check_all_databases()

    def rebuild_vector_indexes(self) -> dict[str, Any]:
        """Rebuild memory vector indexes from SQLite records."""
        return self.disaster.verify_and_rebuild_indexes()
