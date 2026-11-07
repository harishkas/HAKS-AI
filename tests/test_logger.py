"""Unit tests for logging infrastructure."""

from pathlib import Path
from core.logger import setup_logging, get_logger
from config.settings import AppConfig, SystemPaths


def test_logger_initialization_and_file_creation(tmp_path: Path):
    """Verify that logging creates the log file and properly logs messages."""
    paths = SystemPaths(
        root_dir=tmp_path,
        data_dir=tmp_path / "data",
        logs_dir=tmp_path / "logs",
        assets_dir=tmp_path / "assets",
        backups_dir=tmp_path / "data" / "backups",
    )
    config = AppConfig(paths=paths)

    root_logger = setup_logging(config)
    test_logger = get_logger("unit_test")
    test_logger.info("Test message for HAKS logging verification.")

    log_file = paths.logs_dir / config.logging.log_file_name
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "Test message for HAKS logging verification." in content
    assert "[haks.unit_test]" in content
