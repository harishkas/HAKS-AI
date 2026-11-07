"""Unit tests for SQLite storage framework."""

import sqlite3
import pytest
from storage.database import DatabaseManager


def test_database_initialization_creates_5_databases(test_db_manager: DatabaseManager):
    """Verify that all 5 required SQLite databases are created and have valid schemas."""
    expected_dbs = [
        "personal_memory",
        "work_memory",
        "conversations",
        "memory_proposals",
        "settings",
    ]
    for db_name in expected_dbs:
        db_path = test_db_manager.get_db_path(db_name)
        assert db_path.exists(), f"Database file {db_path} does not exist."


def test_wal_mode_and_foreign_keys(test_db_manager: DatabaseManager):
    """Verify WAL journal mode and foreign key enforcement."""
    with test_db_manager.session("conversations") as conn:
        journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert journal_mode.lower() == "wal"

        foreign_keys = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
        assert foreign_keys == 1


def test_domain_isolation(test_db_manager: DatabaseManager):
    """Verify that clearing Work Memory does NOT delete Personal Memory."""
    # 1. Insert into Personal Memory
    with test_db_manager.session("personal_memory") as conn:
        conn.execute(
            "INSERT INTO personal_memories (id, title, content, category) VALUES (?, ?, ?, ?);",
            ("pm-1", "User Preference", "Prefers Tamil responses", "preference"),
        )

    # 2. Insert into Work Memory
    with test_db_manager.session("work_memory") as conn:
        conn.execute(
            "INSERT INTO work_memories (id, project_name, title, content) VALUES (?, ?, ?, ?);",
            ("wm-1", "HAKS Assistant", "Architecture", "Modular Monolithic Architecture",),
        )

    # 3. Clear Work Memory
    test_db_manager.clear_domain("work_memory")

    # 4. Verify Work Memory is empty
    with test_db_manager.session("work_memory") as conn:
        wm_count = conn.execute("SELECT COUNT(*) FROM work_memories;").fetchone()[0]
        assert wm_count == 0

    # 5. Verify Personal Memory still has its record
    with test_db_manager.session("personal_memory") as conn:
        pm_row = conn.execute("SELECT title, content FROM personal_memories WHERE id = ?;", ("pm-1",)).fetchone()
        assert pm_row is not None
        assert pm_row["title"] == "User Preference"
        assert pm_row["content"] == "Prefers Tamil responses"


def test_conversation_cascade_delete(test_db_manager: DatabaseManager):
    """Verify foreign key constraint: deleting a conversation cascades to messages."""
    with test_db_manager.session("conversations") as conn:
        conn.execute(
            "INSERT INTO conversations (id, title, model_used) VALUES (?, ?, ?);",
            ("conv-1", "Test Conversation", "ollama/llama3.2:1b"),
        )
        conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content) VALUES (?, ?, ?, ?);",
            ("msg-1", "conv-1", "user", "Hello HAKS"),
        )
        conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content) VALUES (?, ?, ?, ?);",
            ("msg-2", "conv-1", "assistant", "வணக்கம்! நான் HAKS."),
        )

        # Confirm messages inserted
        msg_count = conn.execute("SELECT COUNT(*) FROM messages WHERE conversation_id = ?;", ("conv-1",)).fetchone()[0]
        assert msg_count == 2

        # Delete conversation
        conn.execute("DELETE FROM conversations WHERE id = ?;", ("conv-1",))

        # Check cascading deletion
        remaining_msgs = conn.execute("SELECT COUNT(*) FROM messages WHERE conversation_id = ?;", ("conv-1",)).fetchone()[0]
        assert remaining_msgs == 0


def test_memory_proposals_status_constraint(test_db_manager: DatabaseManager):
    """Verify proposal status constraint (PENDING, APPROVED, REJECTED)."""
    with test_db_manager.session("memory_proposals") as conn:
        conn.execute(
            "INSERT INTO memory_proposals (id, domain, proposal_text, status) VALUES (?, ?, ?, ?);",
            ("prop-1", "personal", "Remember user's birthday", "PENDING"),
        )

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO memory_proposals (id, domain, proposal_text, status) VALUES (?, ?, ?, ?);",
                ("prop-2", "personal", "Invalid status proposal", "INVALID_STATUS"),
            )
