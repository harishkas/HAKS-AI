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
from storage.database import DatabaseManager


@pytest.fixture(scope="session")
def qapp():
    """Single QApplication instance for entire test session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def pytest_sessionfinish(session, exitstatus):
    """Ensure clean process exit on Windows Python 3.14 without C++ static destructor abort."""
    os._exit(exitstatus)


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
