"""Comprehensive tests for HAKS Memory Storage Architecture.

Covers:
- Persistent %LOCALAPPDATA%\\HAKS\\data path resolution
- 9 independent storage domains
- Encryption at rest for SQLite and vector index
- Key management, export, and disaster recovery
- SQLite-Vector consistency and self-healing synchronization
- Independent domain backup, restore, export (Markdown, JSON, CSV), and wipe
- Safe storage migration with conflict detection, checksum verification, and overwrite guards
"""

import os
from pathlib import Path
import sqlite3
import pytest

from config.settings import AppConfig, SystemPaths, get_default_data_dir, get_default_persistent_root
from core.app_controller import AppController
from memory.encryption import MemoryEncryptionService
from memory.models import MemoryDomain
from storage.backup_manager import BackupManager
from storage.database import DatabaseManager
from storage.migration import StorageMigrationService


def test_persistent_windows_location_defaults():
    """Verify default persistent paths resolve to %LOCALAPPDATA%\\HAKS\\data."""
    default_root = get_default_persistent_root()
    default_data = get_default_data_dir()

    expected_localappdata = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
    assert str(default_root).startswith(expected_localappdata)
    assert default_root.name == "HAKS"
    assert default_data == default_root / "data"


def test_nine_storage_domains_created(temp_workspace: AppConfig):
    """Verify that all 9 independent storage domains are created."""
    data_dir = temp_workspace.paths.data_dir
    logs_dir = temp_workspace.paths.logs_dir

    expected_domains = [
        data_dir / "personal_memory",
        data_dir / "personal_memory" / "vector_index",
        data_dir / "work_memory",
        data_dir / "work_memory" / "vector_index",
        data_dir / "conversations",
        data_dir / "memory_proposals",
        data_dir / "settings",
        data_dir / "backups",
        logs_dir,
    ]

    for domain_path in expected_domains:
        assert domain_path.exists(), f"Domain directory {domain_path} was not created."


def test_encryption_at_rest_in_sqlite(temp_workspace: AppConfig, test_db_manager: DatabaseManager):
    """Verify sensitive memory content is encrypted at rest in SQLite and decrypted in memory."""
    controller = AppController(temp_workspace, test_db_manager)

    secret_note = "User confidential health data: Blood pressure 120/80 mmHg"
    mem_id, is_new = controller.personal_memory.add_memory(
        title="Health Record",
        content=secret_note,
        category="health",
    )
    assert is_new is True

    # 1. Inspect raw SQLite table directly
    with test_db_manager.session("personal_memory") as conn:
        row = conn.execute("SELECT content FROM personal_memories WHERE id = ?;", (mem_id,)).fetchone()
        raw_stored_content = row["content"]

        # Content must NOT be plaintext
        assert raw_stored_content != secret_note
        assert raw_stored_content.startswith("enc:")

    # 2. Inspect through personal_memory manager
    item = controller.personal_memory.get_memory(mem_id)
    assert item is not None
    assert item.content == secret_note

    # 3. Keyword search on decrypted content works
    results = controller.personal_memory.get_all_memories(search="Blood pressure")
    assert len(results) == 1
    assert results[0].id == mem_id


def test_vector_index_encryption_at_rest(temp_workspace: AppConfig, test_db_manager: DatabaseManager):
    """Verify text stored inside vector index file on disk is encrypted at rest."""
    controller = AppController(temp_workspace, test_db_manager)

    confidential_project_info = "Super Secret Algorithm Architecture V2"
    controller.work_memory.add_memory(
        title="Algorithm Design",
        content=confidential_project_info,
        project_name="AI-Core",
    )

    index_file = temp_workspace.paths.data_dir / "work_memory" / "vector_index" / "index.json"
    assert index_file.exists()
    disk_content = index_file.read_text(encoding="utf-8")

    # Plaintext must not be present directly in raw index JSON
    assert confidential_project_info not in disk_content

    # In-memory vector search still retrieves the document cleanly
    results = controller.work_memory.semantic_search("Algorithm Architecture", top_k=1)
    assert len(results) == 1
    assert results[0].item.title == "Algorithm Design"


def test_key_management_and_recovery(temp_workspace: AppConfig):
    """Verify recovery key export and cipher recovery."""
    enc = MemoryEncryptionService(temp_workspace)
    assert enc.is_key_healthy() is True

    rec_key = enc.export_recovery_key()
    assert rec_key.startswith("HAKS-REC-")

    # Encrypt a sample message
    original_text = "Highly classified operating notes"
    ciphertext = enc.encrypt_field(original_text)

    # Corrupt or simulate key loss by initializing a new cipher with fresh random key
    new_enc = MemoryEncryptionService(temp_workspace)
    # Recover with exported recovery key
    success = new_enc.recover_with_recovery_key(rec_key)
    assert success is True

    # Decrypt ciphertext using recovered cipher
    decrypted = new_enc.decrypt_field(ciphertext)
    assert decrypted == original_text

    # Invalid recovery key is rejected
    assert new_enc.recover_with_recovery_key("invalid_bad_key") is False


def test_sqlite_vector_self_healing_sync(temp_workspace: AppConfig, test_db_manager: DatabaseManager):
    """Verify self-healing synchronization between SQLite records and vector index."""
    controller = AppController(temp_workspace, test_db_manager)

    # 1. Add valid memories
    id1, _ = controller.personal_memory.add_memory(title="Item 1", content="First content")
    id2, _ = controller.personal_memory.add_memory(title="Item 2", content="Second content")

    # Check initially consistent
    consistency = controller.personal_memory.verify_consistency()
    assert consistency["is_consistent"] is True

    # 2. Simulate orphaned vector entry by deleting item from SQLite directly
    with test_db_manager.session("personal_memory") as conn:
        conn.execute("DELETE FROM personal_memories WHERE id = ?;", (id1,))

    consistency_after_delete = controller.personal_memory.verify_consistency()
    assert consistency_after_delete["is_consistent"] is False
    assert id1 in consistency_after_delete["orphaned_in_vector"]

    # 3. Simulate missing vector entry by inserting directly into SQLite
    id3 = "pm_direct_insert"
    with test_db_manager.session("personal_memory") as conn:
        conn.execute(
            "INSERT INTO personal_memories (id, title, content) VALUES (?, ?, ?);",
            (id3, "Item 3 Direct", "Directly inserted into db"),
        )

    consistency_after_insert = controller.personal_memory.verify_consistency()
    assert consistency_after_insert["is_consistent"] is False
    assert id3 in consistency_after_insert["missing_in_vector"]

    # 4. Trigger self-healing synchronization
    sync_result = controller.personal_memory.sync_vector_index()
    assert sync_result["synced"] is True
    assert sync_result["removed"] == 1  # id1 removed
    assert sync_result["added"] == 1    # id3 added

    # 5. Confirm 100% consistent again
    final_consistency = controller.personal_memory.verify_consistency()
    assert final_consistency["is_consistent"] is True
    assert final_consistency["sqlite_count"] == 2
    assert final_consistency["vector_count"] == 2


def test_independent_domain_backup_and_restore(temp_workspace: AppConfig, test_db_manager: DatabaseManager):
    """Verify independent backup, encrypted bundle creation, and domain restoration."""
    controller = AppController(temp_workspace, test_db_manager)

    # Seed Personal Memory and Work Memory
    pm_id, _ = controller.personal_memory.add_memory("Doctor Appointment", "Cardiologist visit at 10 AM")
    wm_id, _ = controller.work_memory.add_memory("Sprint Goal", "Ship Phase 3 and 4", project_name="HAKS")

    # Backup Personal Memory (encrypted)
    backup_file = controller.backup_domain("personal_memory", encrypt=True)
    assert backup_file.exists()
    assert backup_file.suffix == ".haksbackup"

    # Mutate Personal Memory (add another, change existing)
    controller.personal_memory.add_memory("Grocery List", "Apples, Milk, Eggs")
    assert len(controller.personal_memory.get_all_memories()) == 2

    # Restore from backup
    restore_success = controller.restore_domain(backup_file, "personal_memory")
    assert restore_success is True

    # Verify Personal Memory restored to original 1 item
    restored_items = controller.personal_memory.get_all_memories()
    assert len(restored_items) == 1
    assert restored_items[0].title == "Doctor Appointment"

    # Verify Work Memory was completely unaffected
    work_items = controller.work_memory.get_all_memories()
    assert len(work_items) == 1
    assert work_items[0].title == "Sprint Goal"

    # Verify cross-domain restoration is rejected (e.g. restoring personal_memory backup into work_memory)
    with pytest.raises(ValueError, match="Domain mismatch"):
        controller.restore_domain(backup_file, "work_memory")


def test_domain_exports_markdown_json_csv(temp_workspace: AppConfig, test_db_manager: DatabaseManager, tmp_path: Path):
    """Verify exporting domain records to Markdown, JSON, and CSV."""
    controller = AppController(temp_workspace, test_db_manager)
    controller.personal_memory.add_memory("Python Tips", "Use list comprehensions wisely", category="dev", tags=["python", "tips"])

    export_dir = tmp_path / "exports"

    # 1. Markdown
    md_path = controller.export_domain("personal_memory", "markdown", export_dir / "notes.md")
    assert md_path.exists()
    md_text = md_path.read_text(encoding="utf-8")
    assert "# HAKS Export: Personal Memory" in md_text
    assert "## Python Tips" in md_text
    assert "Use list comprehensions wisely" in md_text

    # 2. JSON
    json_path = controller.export_domain("personal_memory", "json", export_dir / "notes.json")
    assert json_path.exists()
    json_text = json_path.read_text(encoding="utf-8")
    assert '"title": "Python Tips"' in json_text
    assert '"content": "Use list comprehensions wisely"' in json_text

    # 3. CSV
    csv_path = controller.export_domain("personal_memory", "csv", export_dir / "notes.csv")
    assert csv_path.exists()
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "Python Tips" in csv_text
    assert "dev" in csv_text


def test_domain_secure_wipe_guard(temp_workspace: AppConfig, test_db_manager: DatabaseManager):
    """Verify guarded domain wipe requires exact phrase and never touches other domains."""
    controller = AppController(temp_workspace, test_db_manager)

    controller.personal_memory.add_memory("Private Thought", "Secret diary entry")
    controller.work_memory.add_memory("Production Code", "Backend service logic", project_name="Core")

    # 1. Invalid confirmation phrase is rejected
    with pytest.raises(ValueError, match="Confirmation phrase mismatch"):
        controller.wipe_domain("personal_memory", "delete please")

    # 2. Exact phrase wipes only personal_memory
    wipe_success = controller.wipe_domain("personal_memory", "DELETE PERSONAL_MEMORY")
    assert wipe_success is True

    # Personal Memory is empty
    assert len(controller.personal_memory.get_all_memories()) == 0

    # Work Memory remains untouched
    assert len(controller.work_memory.get_all_memories()) == 1
    assert controller.work_memory.get_all_memories()[0].title == "Production Code"


def test_safe_storage_migration(temp_workspace: AppConfig, test_db_manager: DatabaseManager, tmp_path: Path):
    """Verify safe data migration with conflict checking, checksum verification, and overwrite protection."""
    controller = AppController(temp_workspace, test_db_manager)

    controller.personal_memory.add_memory("Crucial Note", "Cannot afford to lose this", importance=5)
    controller.work_memory.add_memory("Architecture Spec", "Microservices on Kubernetes", project_name="Infra")

    new_data_target = tmp_path / "migrated_haks_data"

    # 1. Attempt migration into illegal location (.venv)
    invalid_target = tmp_path / ".venv" / "data"
    success, err = controller.migrate_storage(invalid_target)
    assert success is False
    assert "virtual environment" in err.lower()

    # 2. Simulate pre-existing data conflict at destination
    conflict_db = new_data_target / "personal_memory" / "personal_memory.db"
    conflict_db.parent.mkdir(parents=True, exist_ok=True)
    conflict_db.write_text("existing_content", encoding="utf-8")

    # Migration without overwrite approval must ABORT
    abort_success, abort_msg = controller.migrate_storage(new_data_target, overwrite_approved=False)
    assert abort_success is False
    assert "Explicit user approval is required" in abort_msg

    # 3. Migration with explicit overwrite approval succeeds
    mig_success, mig_msg = controller.migrate_storage(new_data_target, overwrite_approved=True)
    assert mig_success is True
    assert "Successfully migrated" in mig_msg

    # 4. Verify data in migrated location
    assert (new_data_target / "personal_memory" / "personal_memory.db").exists()
    assert (new_data_target / "work_memory" / "work_memory.db").exists()

    # Query through updated controller
    migrated_pm = controller.personal_memory.get_all_memories()
    assert len(migrated_pm) == 1
    assert migrated_pm[0].title == "Crucial Note"
    assert migrated_pm[0].content == "Cannot afford to lose this"

    migrated_wm = controller.work_memory.get_all_memories()
    assert len(migrated_wm) == 1
    assert migrated_wm[0].title == "Architecture Spec"
