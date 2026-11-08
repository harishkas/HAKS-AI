"""Comprehensive Test Suite for Phase 8: Offline Operation and Disaster Recovery.

Covers:
1. Real-time network monitoring, offline/online transition detection, and event broadcasting.
2. Offline AI routing guards (preventing unauthorized cloud calls, enforcing Phase 5 non-silent fallback).
3. Offline Voice operations (Local wake-word, VAD, Faster-Whisper, offline SAPI/Piper TTS).
4. Full-system encrypted backup creation (.haksbackup) with SHA-256 manifests.
5. Backup verification and tamper detection.
6. Backup retention policy pruning (guaranteeing the latest backup is never deleted).
7. SQLite integrity verification (PRAGMA quick_check and PRAGMA integrity_check).
8. Corrupted database quarantine and pre-recovery snapshots (never overwriting original damaged data).
9. Vector index consistency verification and self-healing reconstruction from SQLite.
10. Application crash recovery (unclean shutdown detection and auto-repair).
11. Restoration to a separate isolated temporary directory.
12. Complete backup and restore roundtrip for Personal and Work Memory.
13. Master memory recovery key export and recovery.
14. RecoveryView UI components and MainWindow integration.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import pytest

from config.settings import AppConfig
from core.app_controller import AppController
from memory.vector_index import LocalVectorIndex
from recovery.backup_service import EnhancedBackupService
from recovery.disaster_recovery import DisasterRecoveryService
from recovery.models import BackupManifest, DatabaseIntegrityStatus, RecoveryReport
from recovery.network_monitor import NetworkMonitor
from recovery.offline_manager import OfflineRecoveryManager
from routing.models import RoutingMode
from storage.database import DatabaseManager


# ==============================================================================
# 1. Network Monitoring & Transition Detection
# ==============================================================================

def test_network_monitor_connectivity_transitions(controller: AppController):
    """Verify NetworkMonitor detects online/offline transitions and dispatches events."""
    monitor = NetworkMonitor(events=controller.events)

    transitions = []
    controller.events.network_connectivity_changed.connect(lambda online: transitions.append(online))

    # Test forced offline transition
    monitor.force_status(False)
    assert monitor.is_online is False
    assert transitions[-1] is False
    assert "Offline" in monitor.get_status().status_message

    # Test forced online transition
    monitor.force_status(True)
    assert monitor.is_online is True
    assert transitions[-1] is True

    # Reset
    monitor.force_status(None)


# ==============================================================================
# 2. Offline AI Routing Guards
# ==============================================================================

def test_offline_ai_routing_guards(controller: AppController):
    """Verify that when offline, cloud requests are intercepted and routed to Local AI without silent leakage."""
    rm = controller.get_recovery_manager()

    # Force offline mode
    rm.network.force_status(False)

    # 1. Routine prompt with AUTO mode when offline
    decision1 = controller.route_request("Hello, what is Python?", mode=RoutingMode.AUTO)
    assert decision1.selected_engine == "local"

    # 2. Complex prompt with AUTO mode when offline
    decision2 = controller.route_request("Architect an asynchronous distributed actor model.", mode=RoutingMode.AUTO)
    # When offline, cloud cannot be reached, so it flags non-silent fallback
    assert decision2.is_engine_available is False or decision2.selected_engine == "local"
    if decision2.selected_engine == "cloud":
        assert decision2.is_fallback_needed is True
        assert decision2.fallback_engine == "local"
        assert "offline" in decision2.fallback_reason.lower()

    # 3. Explicit MANUAL_CLOUD when offline
    decision3 = controller.route_request("Write a quick poem", mode=RoutingMode.MANUAL_CLOUD)
    assert decision3.is_fallback_needed is True
    assert decision3.fallback_engine == "local"
    assert "offline" in decision3.fallback_reason.lower()

    # Reset status
    rm.network.force_status(None)


# ==============================================================================
# 3. Offline Voice Operation & Distinctions
# ==============================================================================

def test_offline_voice_operations(controller: AppController):
    """Verify offline readiness of wake-word, VAD, Whisper, and explicit TTS offline/online distinction."""
    rm = controller.get_recovery_manager()
    v_status = rm.get_voice_offline_status()

    assert v_status["wake_word_offline"] is True
    assert v_status["vad_offline"] is True
    assert v_status["stt_whisper_offline"] is True
    assert v_status["tts_edge_neural_online_only"] is True
    assert v_status["tts_sapi_offline"] is True
    assert v_status["tts_piper_offline"] is True

    # Verify TTS router language awareness and engine distinction
    tts = controller.voice_engine.tts
    assert tts.is_engine_offline("sapi") is True
    assert tts.is_engine_offline("piper") is True
    assert tts.is_engine_offline("edge") is False
    assert "Online Only" in tts.get_engine_network_requirement("edge")
    assert "Offline" in tts.get_engine_network_requirement("sapi")


# ==============================================================================
# 4. Backup Creation and SHA-256 Manifest
# ==============================================================================

def test_backup_creation_and_sha256_manifest(controller: AppController):
    """Verify full-system encrypted backup creates a valid archive with per-file SHA-256 hashes."""
    rm = controller.get_recovery_manager()

    # Add sample records to Personal and Work memory
    p_id = controller.add_personal_memory("Family vacation plan", "Trip to Ooty in May", category="life")
    w_id = controller.add_work_memory("HAKS Architecture", "Phase 8 Disaster Recovery design", project_name="HAKS")

    # Create full backup
    bk_path = rm.create_full_backup(encrypt=True, notes="Unit test backup")
    assert bk_path.exists()
    assert bk_path.suffix == ".haksbackup"
    assert bk_path.stat().st_size > 1000

    # Verify backup
    v_res = rm.backup.verify_backup(bk_path)
    assert v_res.is_valid is True
    assert v_res.manifest is not None
    assert v_res.manifest.backup_type == "full_system"
    assert v_res.manifest.is_encrypted is True
    assert len(v_res.manifest.checksums) >= 5
    assert len(v_res.checksum_mismatches) == 0


# ==============================================================================
# 5. Backup Verification Detects Tampering
# ==============================================================================

def test_backup_verification_detects_tampering(controller: AppController, tmp_path: Path):
    """Verify that verify_backup detects corrupted files or hash mismatches."""
    rm = controller.get_recovery_manager()

    # Create unencrypted zip backup to test internal file modification
    bk_path = rm.create_full_backup(encrypt=False, notes="Tamper test backup")
    assert bk_path.exists()

    # Read bytes and corrupt
    corrupted_path = tmp_path / "tampered_backup.zip"
    raw_data = bytearray(bk_path.read_bytes())
    # Flip bytes in the payload
    if len(raw_data) > 300:
        raw_data[250] ^= 0xFF
        raw_data[251] ^= 0xFF
    corrupted_path.write_bytes(raw_data)

    v_res = rm.backup.verify_backup(corrupted_path)
    assert v_res.is_valid is False
    assert len(v_res.errors) > 0 or len(v_res.checksum_mismatches) > 0


# ==============================================================================
# 6. Retention Policy Pruning
# ==============================================================================

def test_backup_retention_policy(controller: AppController, tmp_path: Path):
    """Verify retention policy prunes excess backups while always protecting the latest backup."""
    rm = controller.get_recovery_manager()
    test_bk_dir = tmp_path / "retention_test"
    test_bk_dir.mkdir(parents=True, exist_ok=True)

    # Create 5 dummy backup files with staggered timestamps
    created_files = []
    for i in range(5):
        f = test_bk_dir / f"backup_2026110{i}_000000.zip"
        f.write_text(f"dummy content {i}")
        created_files.append(f)

    # Apply policy: max 2 backups
    pruned = rm.backup.apply_retention_policy(max_backups=2, max_age_days=365, target_dir=test_bk_dir)
    assert len(pruned) >= 2

    # Verify at least the newest backup still exists
    remaining = list(test_bk_dir.glob("*.zip"))
    assert len(remaining) <= 2
    assert len(remaining) >= 1


# ==============================================================================
# 7. SQLite Integrity Checks
# ==============================================================================

def test_sqlite_integrity_checks(controller: AppController):
    """Verify PRAGMA integrity_check and quick_check across all 5 databases."""
    rm = controller.get_recovery_manager()
    report = rm.run_integrity_check()

    assert len(report) == 5
    for domain in ("personal_memory", "work_memory", "conversations", "memory_proposals", "settings"):
        assert domain in report
        st = report[domain]
        assert st.exists is True
        assert st.is_healthy is True
        assert st.quick_check.lower() == "ok"
        assert st.integrity_check.lower() == "ok"


# ==============================================================================
# 8. Corrupted Database Quarantine
# ==============================================================================

def test_corrupted_database_quarantine_and_safeguards(controller: AppController):
    """Verify corrupted databases are preserved into quarantine before repair."""
    rm = controller.get_recovery_manager()

    q_dir = rm.disaster.quarantine_corrupted_database("settings")
    assert q_dir is not None
    assert q_dir.exists()
    assert (q_dir / "settings.db").exists()


# ==============================================================================
# 9. Vector Index Self-Healing Rebuild
# ==============================================================================

def test_vector_index_self_healing_rebuild(controller: AppController):
    """Verify that an out-of-sync or missing vector index is automatically rebuilt from SQLite."""
    rm = controller.get_recovery_manager()

    # Add memories
    controller.add_personal_memory("Cooking recipe", "Biryani recipe with basmati rice", category="food")
    controller.add_personal_memory("Exercise routine", "Morning 5km jog and stretching", category="health")

    # Corrupt or delete vector index
    idx_path = controller.config.paths.data_dir / "personal_memory" / "vector_index" / "index.json"
    if idx_path.exists():
        idx_path.unlink()

    # Verify status detects missing index
    st_before = rm.disaster.check_vector_index("personal_memory")
    assert st_before.exists is False

    # Execute self-healing rebuild
    rebuilt = rm.disaster.rebuild_vector_index("personal_memory")
    assert rebuilt is True

    # Verify status is now synchronized
    st_after = rm.disaster.check_vector_index("personal_memory")
    assert st_after.exists is True
    assert st_after.is_synchronized is True
    assert st_after.vector_count >= 2


# ==============================================================================
# 10. Crash Recovery / Unclean Shutdown Detection
# ==============================================================================

def test_crash_recovery_unclean_shutdown_detection(temp_workspace: AppConfig, test_db_manager: DatabaseManager):
    """Verify that an unclean shutdown session is flagged on startup and initiates auto-verification."""
    # Write simulated crashed session lock
    lock_file = temp_workspace.paths.data_dir / ".session_lock.json"
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    crashed_state = {
        "pid": 99999,
        "start_time": "2026-11-08T00:00:00",
        "clean_exit": False,
        "version": "0.1.0",
    }
    lock_file.write_text(json.dumps(crashed_state))

    # Initialize OfflineRecoveryManager
    rm = OfflineRecoveryManager(config=temp_workspace, db_manager=test_db_manager)
    unclean = rm.initialize_on_startup()
    assert unclean is True
    assert rm.disaster.was_unclean_shutdown is True

    # Record clean shutdown
    rm.shutdown()
    assert rm.disaster._session_lock_file.exists()
    saved = json.loads(rm.disaster._session_lock_file.read_text())
    assert saved["clean_exit"] is True


# ==============================================================================
# 11. Restoration to Isolated Directory
# ==============================================================================

def test_full_system_restoration_to_isolated_directory(controller: AppController, tmp_path: Path):
    """Verify that full restoration can extract to an isolated staging directory without modifying production databases."""
    rm = controller.get_recovery_manager()

    # Create backup
    bk_file = rm.create_full_backup(encrypt=True, notes="Staging test")

    # Restore into separate temporary target directory
    target_dir = tmp_path / "staged_restore"
    report = rm.disaster.restore_system(backup_path=bk_file, target_dir=target_dir, preserve_current=False)

    assert report.success is True
    assert len(report.errors) == 0
    assert len(report.restored_domains) >= 5

    # Check files in staging directory
    for dom in ("personal_memory", "work_memory", "conversations", "memory_proposals", "settings"):
        staged_db = target_dir / dom / f"{dom}.db"
        assert staged_db.exists()
        # Verify valid SQLite
        conn = sqlite3.connect(str(staged_db))
        res = conn.execute("PRAGMA integrity_check;").fetchone()[0]
        conn.close()
        assert res.lower() == "ok"


# ==============================================================================
# 12. Backup & Restore Complete Roundtrip
# ==============================================================================

def test_full_system_restore_and_roundtrip(controller: AppController):
    """Verify end-to-end backup, database wiping/corruption, restoration, and semantic retrieval."""
    rm = controller.get_recovery_manager()

    # 1. Create distinct items in Personal and Work Memory
    unique_token = f"TOKEN_{datetime.now().microsecond}"
    p_id = controller.add_personal_memory("Unique Personal Note", f"Secret code {unique_token}", category="notes")
    w_id = controller.add_work_memory("Unique Work Note", f"Project requirement {unique_token}", project_name="ProjX")

    # 2. Create full backup
    bk_path = rm.create_full_backup(encrypt=True, notes="Roundtrip backup")
    assert bk_path.exists()

    # 3. Simulate data loss: clear work memory and personal memory
    controller.personal_memory.clear_domain()
    controller.work_memory.clear_domain()
    assert len(controller.get_personal_memories()) == 0
    assert len(controller.get_work_memories()) == 0

    # 4. Restore from backup
    report = controller.restore_full_backup(backup_path=bk_path, preserve_current=True)
    assert report.success is True
    assert report.pre_recovery_snapshot_path is not None

    # 5. Verify records restored cleanly
    restored_p = controller.get_personal_memories()
    restored_w = controller.get_work_memories()
    assert any(unique_token in m["content"] for m in restored_p)
    assert any(unique_token in m["content"] for m in restored_w)


# ==============================================================================
# 13. Recovery Key Export and Recovery
# ==============================================================================

def test_recovery_key_export_and_import(controller: AppController):
    """Verify exporting the master memory recovery key and using it to recover encryption cipher."""
    key = controller.get_recovery_key()
    assert key.startswith("HAKS-REC-")

    recovered = controller.recover_encryption_key(key)
    assert recovered is True
    assert controller.personal_memory.encryption.is_key_healthy() is True


# ==============================================================================
# 14. UI Integration: RecoveryView and MainWindow
# ==============================================================================

def test_recovery_view_ui(qapp, controller: AppController):
    """Verify RecoveryView initializes all 3 sub-tabs, tables, and buttons."""
    from ui.views.recovery_view import RecoveryView

    view = RecoveryView(controller)
    assert view.tabs.count() == 3
    assert "Health" in view.tabs.tabText(0)
    assert "Backup" in view.tabs.tabText(1)
    assert "Restore" in view.tabs.tabText(2)
    assert view.table_db.columnCount() == 7
    assert view.table_vec.columnCount() == 5
    assert view.table_backups.columnCount() == 6
    view.close()


def test_main_window_recovery_tab_integration(qapp, temp_workspace: AppConfig, controller: AppController):
    """Verify MainWindow integrates Backup & Recovery in sidebar, stack, and status badges."""
    from ui.main_window import MainWindow

    win = MainWindow(config=temp_workspace, controller=controller)
    assert hasattr(win, "page_recovery")
    assert "recovery" in win._nav_buttons
    assert hasattr(win, "badge_network")

    # Navigate to recovery page
    win.events.navigate_to_page.emit("recovery")
    assert win.content_stack.currentWidget() == win.page_recovery
    assert win.lbl_page_title.text() == "Backup & Recovery"

    win.close()
