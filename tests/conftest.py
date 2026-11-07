"""Pytest fixtures for HAKS testing suite."""

import os
from pathlib import Path
import tempfile
import pytest

# Ensure headless Qt testing
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from config.settings import AppConfig, SystemPaths
from storage.database import DatabaseManager


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
