"""Unit tests for PySide6 UI foundation."""

import sys
import pytest
from PySide6.QtWidgets import QApplication

from config.settings import AppConfig
from core.events import get_app_events
from storage.database import DatabaseManager
from ui.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    """Ensure a single headless QApplication instance exists for tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def test_main_window_structure_and_navigation(qapp, temp_workspace: AppConfig):
    """Verify that MainWindow initializes all required sections and navigation works."""
    # Initialize DB for workspace
    db_manager = DatabaseManager(temp_workspace)
    db_manager.initialize()

    window = MainWindow(temp_workspace)
    assert window.windowTitle() == temp_workspace.ui.app_title
    assert window.minimumWidth() >= temp_workspace.ui.min_window_width
    assert window.minimumHeight() >= temp_workspace.ui.min_window_height

    # Verify content stack has 10 pages
    assert window.content_stack.count() == 10

    # Verify navigation buttons
    expected_nav_ids = [
        "home",
        "chat",
        "history",
        "personal_memory",
        "work_memory",
        "voice",
        "ai_engine",
        "control",
        "recovery",
        "settings",
    ]
    for nav_id in expected_nav_ids:
        assert nav_id in window._nav_buttons
        btn = window._nav_buttons[nav_id]
        btn.click()
        # Verify active button is checked
        assert btn.isChecked()

    # Test event bus integration
    events = get_app_events()
    events.status_message_updated.emit("Testing status update")
    assert window.lbl_status.text() == "Testing status update"

    events.voice_state_changed.emit("LISTENING")
    assert "LISTENING" in window.badge_voice.text()

    window.close()
