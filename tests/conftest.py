"""Pytest fixtures for HAKS testing suite."""

import os
from pathlib import Path
import sys
import tempfile
import pytest
from PySide6.QtWidgets import QApplication

# Ensure headless Qt testing
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from config.settings import AppConfig, SystemPaths
from core.app_controller import AppController
from storage.database import DatabaseManager


@pytest.fixture(scope="session")
def qapp():
    """Single QApplication instance for entire test session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


@pytest.fixture
def temp_workspace(tmp_path: Path) -> AppConfig:
    """Provide an isolated AppConfig pointing to a temporary test directory."""
    paths = SystemPaths(
        root_dir=tmp_path,
        data_dir=tmp_path / "data",
        logs_dir=tmp_path / "logs",
        assets_dir=tmp_path / "assets",
        backups_dir=tmp_path / "data" / "backups",
    )
    paths.ensure_directories()
    config = AppConfig(paths=paths)
    return config


@pytest.fixture
def test_db_manager(temp_workspace: AppConfig) -> DatabaseManager:
    """Provide an initialized DatabaseManager using temporary storage."""
    db_manager = DatabaseManager(temp_workspace)
    db_manager.initialize()
    return db_manager


@pytest.fixture
def controller(temp_workspace: AppConfig, test_db_manager: DatabaseManager) -> AppController:
    """Fixture providing an isolated AppController."""
    return AppController(config=temp_workspace, db_manager=test_db_manager)
