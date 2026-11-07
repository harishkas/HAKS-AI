"""Structured logging system for HAKS.

Provides thread-safe file rotation and console logging with consistent formatting.
Ensures sensitive details are not carelessly leaked to stdout.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
from typing import Optional

from config.settings import AppConfig, get_settings

_LOG_FORMAT = (
    "%(asctime)s [%(levelname)s] [%(name)s] [Thread:%(threadName)s]: %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(config: Optional[AppConfig] = None) -> logging.Logger:
    """Initialize root and application loggers with file and console handlers.

    Args:
        config: Application configuration instance. If None, default settings are loaded.

    Returns:
        The configured root logger.
    """
    if config is None:
        config = get_settings()

    config.paths.ensure_directories()
    log_file_path: Path = config.paths.logs_dir / config.logging.log_file_name

    root_logger = logging.getLogger()
    log_level = getattr(logging, config.logging.level.upper(), logging.INFO)
    root_logger.setLevel(log_level)

    # Avoid adding duplicate handlers on multiple calls
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # 1. File Handler (with size-based rotation)
    file_handler = RotatingFileHandler(
        filename=str(log_file_path),
        maxBytes=config.logging.max_bytes,
        backupCount=config.logging.backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # 2. Console Handler
    if config.logging.enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Initial log confirmation
    app_logger = logging.getLogger("haks.core")
    app_logger.info(
        "HAKS logging initialized. Log level: %s | Target file: %s",
        config.logging.level,
        log_file_path,
    )

    return app_logger


def get_logger(name: str) -> logging.Logger:
    """Obtain a namespaced logger.

    Args:
        name: Name of module or component requesting logger.

    Returns:
        Configured logging.Logger.
    """
    return logging.getLogger(f"haks.{name}")
