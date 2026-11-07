"""Unit tests for configuration management."""

from pathlib import Path
from config.settings import AppConfig, get_settings


def test_default_config_properties():
    """Verify that default configuration possesses expected attributes and values."""
    config = get_settings()

    assert config.ui.app_name == "HAKS"
    assert config.ui.version == "0.1.0"
    assert config.voice.wake_phrase == "Hey HAKS"
    assert "ta" in config.voice.supported_languages
    assert "en" in config.voice.supported_languages
    assert config.database.enable_wal is True
    assert config.security.permission_enforcement is True


def test_custom_paths_creation(tmp_path: Path):
    """Verify that SystemPaths correctly creates directories on demand."""
    custom_root = tmp_path / "custom_haks"
    from config.settings import SystemPaths
    paths = SystemPaths(
        root_dir=custom_root,
        data_dir=custom_root / "test_data",
        logs_dir=custom_root / "test_logs",
        assets_dir=custom_root / "test_assets",
        backups_dir=custom_root / "test_data" / "backups",
    )
    paths.ensure_directories()

    assert paths.data_dir.exists()
    assert paths.logs_dir.exists()
    assert paths.assets_dir.exists()
    assert paths.backups_dir.exists()
