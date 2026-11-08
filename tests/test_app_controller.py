"""Unit tests for the central AppController."""

import pytest
from config.settings import AppConfig
from core.app_controller import AppController
from storage.database import DatabaseManager


@pytest.fixture
def test_controller(temp_workspace: AppConfig, test_db_manager: DatabaseManager) -> AppController:
    """Fixture providing an isolated AppController."""
    return AppController(config=temp_workspace, db_manager=test_db_manager)


def test_conversation_lifecycle(test_controller: AppController):
    """Test creating, messaging, loading, and deleting conversations."""
    conv_id = test_controller.get_active_conversation_id()
    assert conv_id.startswith("conv_")

    # Send a message
    msg = test_controller.send_message("Hello from test suite!", role="user")
    assert msg["conversation_id"] == conv_id
    assert msg["role"] == "user"
    assert msg["content"] == "Hello from test suite!"

    # Load messages
    messages = test_controller.load_conversation(conv_id)
    assert len(messages) == 1
    assert messages[0]["content"] == "Hello from test suite!"

    # Get all conversations
    convs = test_controller.get_all_conversations()
    assert len(convs) >= 1
    found = [c for c in convs if c["id"] == conv_id]
    assert len(found) == 1
    assert found[0]["message_count"] == 1

    # Delete conversation
    test_controller.delete_conversation(conv_id)
    remaining = test_controller.get_all_conversations()
    assert not any(c["id"] == conv_id for c in remaining)


def test_personal_memory_lifecycle(test_controller: AppController):
    """Test adding, querying, searching, and deleting personal memory."""
    mem_id = test_controller.add_personal_memory(
        title="Favorite Beverage",
        content="Prefers South Indian filter coffee in the morning",
        category="preference",
        tags="coffee, morning, routine",
    )
    assert mem_id.startswith("pm_")

    # Query all
    memories = test_controller.get_personal_memories()
    assert len(memories) == 1
    assert memories[0]["title"] == "Favorite Beverage"

    # Search
    search_res = test_controller.get_personal_memories(search="coffee")
    assert len(search_res) == 1

    empty_search = test_controller.get_personal_memories(search="nonexistent")
    assert len(empty_search) == 0

    # Delete
    test_controller.delete_personal_memory(mem_id)
    assert len(test_controller.get_personal_memories()) == 0


def test_work_memory_lifecycle(test_controller: AppController):
    """Test adding, querying, and domain isolation of work memory."""
    mem_id = test_controller.add_work_memory(
        project_name="HAKS AI",
        title="Architecture Decision: Modular Monolith",
        content="Avoid microservices for local desktop assistant; use internal modular layers.",
        tags="architecture, python",
    )
    assert mem_id.startswith("wm_")

    memories = test_controller.get_work_memories()
    assert len(memories) == 1
    assert memories[0]["project_name"] == "HAKS AI"

    # Test clearing work memory does not affect personal memory
    test_controller.add_personal_memory(title="Personal Note", content="Should persist")
    test_controller.clear_work_memory()

    assert len(test_controller.get_work_memories()) == 0
    assert len(test_controller.get_personal_memories()) == 1


def test_settings_persistence(test_controller: AppController):
    """Test saving and retrieving settings and coordinates."""
    test_controller.save_floating_position(250, 400)
    x, y = test_controller.get_floating_position()
    assert x == 250
    assert y == 400

    test_controller.set_always_on_top(True)
    assert test_controller.get_always_on_top() is True

    test_controller.set_always_on_top(False)
    assert test_controller.get_always_on_top() is False

    test_controller.set_theme("light")
    assert test_controller.get_theme() == "light"
    test_controller.set_theme("dark")
    assert test_controller.get_theme() == "dark"
