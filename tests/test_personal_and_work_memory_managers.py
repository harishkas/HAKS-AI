"""Unit and integration tests for Personal and Work Memory Managers."""

from pathlib import Path
from config.settings import AppConfig
from memory.models import MemoryDomain
from memory.personal_memory import PersonalMemoryManager
from memory.work_memory import WorkMemoryManager
from storage.database import DatabaseManager


def test_personal_memory_crud_and_semantic_search(temp_workspace: AppConfig, test_db_manager: DatabaseManager):
    """Verify Personal Memory CRUD, categorization, and semantic retrieval."""
    mgr = PersonalMemoryManager(config=temp_workspace, db_manager=test_db_manager)

    mem_id, is_new = mgr.add_memory(
        title="Voice Language Preference",
        content="Prefers Tamil speech synthesis whenever possible",
        category="preference",
        tags=["tamil", "voice"],
        importance=4,
        source_conversation_id="conv_test_1",
    )
    assert is_new is True

    # Retrieve
    item = mgr.get_memory(mem_id)
    assert item is not None
    assert item.title == "Voice Language Preference"
    assert item.category == "preference"
    assert item.importance == 4
    assert item.source_conversation_id == "conv_test_1"

    # Semantic search
    hits = mgr.semantic_search("tamil speech voice", top_k=2)
    assert len(hits) > 0
    assert hits[0].item.id == mem_id
    assert hits[0].score > 0.3

    # Update
    mgr.update_memory(mem_id, importance=5)
    updated = mgr.get_memory(mem_id)
    assert updated.importance == 5

    # Delete
    deleted = mgr.delete_memory(mem_id)
    assert deleted is True
    assert mgr.get_memory(mem_id) is None


def test_work_memory_project_organization(temp_workspace: AppConfig, test_db_manager: DatabaseManager):
    """Verify Work Memory project filtering and distinct project listing."""
    mgr = WorkMemoryManager(config=temp_workspace, db_manager=test_db_manager)

    mgr.add_memory(
        title="SQLite Concurrency Rule",
        content="Must enable PRAGMA journal_mode = WAL for multi-threaded readers",
        project_name="HAKS_CORE",
        tags=["sqlite", "wal"],
        importance=5,
    )
    mgr.add_memory(
        title="Web Scraping Rule",
        content="Use headless browser for dynamic single page applications",
        project_name="WEB_TOOLS",
        tags=["scraping"],
        importance=3,
    )

    projects = mgr.get_projects()
    assert "HAKS_CORE" in projects
    assert "WEB_TOOLS" in projects

    haks_memories = mgr.get_all_memories(project_name="HAKS_CORE")
    assert len(haks_memories) == 1
    assert haks_memories[0].title == "SQLite Concurrency Rule"


def test_duplicate_detection_and_updates(temp_workspace: AppConfig, test_db_manager: DatabaseManager):
    """Verify that adding an item near-identical to an existing one updates it rather than duplicating."""
    mgr = PersonalMemoryManager(config=temp_workspace, db_manager=test_db_manager)

    id1, is_new1 = mgr.add_memory(
        title="Coffee Preference",
        content="I prefer strong South Indian filter coffee in the early morning",
        category="preference",
        importance=3,
    )
    assert is_new1 is True

    # Attempt to add substantially identical memory
    id2, is_new2 = mgr.add_memory(
        title="Coffee Preference",
        content="I prefer strong South Indian filter coffee in the early morning",
        category="preference",
        importance=4,  # updated importance
    )
    # Should detect duplicate and update id1
    assert id2 == id1
    assert is_new2 is False

    # Check total items in DB is still 1
    items = mgr.get_all_memories()
    assert len(items) == 1
    assert items[0].importance == 4


def test_strict_domain_isolation(temp_workspace: AppConfig, test_db_manager: DatabaseManager):
    """Verify that clearing Work Memory never touches Personal Memory and vice versa."""
    pm = PersonalMemoryManager(config=temp_workspace, db_manager=test_db_manager)
    wm = WorkMemoryManager(config=temp_workspace, db_manager=test_db_manager)

    pm_id, _ = pm.add_memory(title="Personal Secret", content="Must persist forever")
    wm_id, _ = wm.add_memory(title="Work Sprint Task", content="Can be cleared")

    # Clear work memory
    wm.clear_domain()

    assert len(wm.get_all_memories()) == 0
    assert len(wm.vector_index) == 0

    # Personal memory must be completely intact
    assert len(pm.get_all_memories()) == 1
    assert len(pm.vector_index) == 1
    assert pm.get_memory(pm_id) is not None


def test_encrypted_export_and_restore(temp_workspace: AppConfig, test_db_manager: DatabaseManager, tmp_path: Path):
    """Verify exporting memory to an encrypted bundle and restoring into a fresh database."""
    pm = PersonalMemoryManager(config=temp_workspace, db_manager=test_db_manager)

    pm.add_memory(title="Memory 1", content="Personal fact 1", category="general", importance=2)
    pm.add_memory(title="Memory 2", content="Personal fact 2", category="preference", importance=5)

    export_file = tmp_path / "personal_backup.enc"
    pm.export_bundle(export_file, encrypt=True)

    assert export_file.exists()
    assert not export_file.read_text(encoding="utf-8").startswith("{")  # Encrypted string!

    # Clear domain
    pm.clear_domain()
    assert len(pm.get_all_memories()) == 0

    # Restore from encrypted bundle
    restored = pm.restore_bundle(export_file, is_encrypted=True)
    assert restored == 2
    assert len(pm.get_all_memories()) == 2
