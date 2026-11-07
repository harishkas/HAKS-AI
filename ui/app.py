"""Application bootstrap and Qt lifecycle management for HAKS.

Configures High-DPI scaling, stylesheets, database connection validation,
and application shutdown handlers.
"""

import sys
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from config.settings import AppConfig, get_settings
from core.logger import get_logger, setup_logging
from storage.database import get_db_manager
from ui.main_window import MainWindow
from ui.styles import DARK_THEME_QSS

logger = get_logger("ui.app")


class HaksApplication:
    """Manages the lifecycle of the HAKS PySide6 desktop application."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or get_settings()
        self.qapp: Optional[QApplication] = None
        self.main_window: Optional[MainWindow] = None

    def initialize(self) -> None:
        """Initialize core services and Qt application."""
        # Setup logging
        setup_logging(self.config)
        logger.info("Starting HAKS Desktop Application v%s...", self.config.ui.version)

        # Initialize SQLite database framework and verify schemas
        db_manager = get_db_manager(self.config)
        db_manager.initialize()

        # Qt High DPI configuration
        if hasattr(Qt, "HighDpiScaleFactorRoundingPolicy"):
            QApplication.setHighDpiScaleFactorRoundingPolicy(
                Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
            )

        # Create QApplication if not existing
        self.qapp = QApplication.instance()
        if self.qapp is None:
            self.qapp = QApplication(sys.argv)

        self.qapp.setApplicationName(self.config.ui.app_name)
        self.qapp.setApplicationDisplayName(self.config.ui.app_title)

        # Apply dark theme stylesheet
        self.qapp.setStyleSheet(DARK_THEME_QSS)

        # Connect cleanup on shutdown
        self.qapp.aboutToQuit.connect(self._on_shutdown)

        # Instantiate MainWindow
        self.main_window = MainWindow(self.config)
        logger.info("HAKS Desktop Application initialized successfully.")

    def run(self) -> int:
        """Display main window and start Qt event loop."""
        if not self.main_window or not self.qapp:
            self.initialize()

        assert self.main_window is not None
        assert self.qapp is not None

        self.main_window.show()
        logger.info("Entering Qt event loop.")
        return self.qapp.exec()

    def _on_shutdown(self) -> None:
        """Graceful application shutdown hook."""
        logger.info("HAKS Desktop Application shutting down gracefully.")


def run_app() -> int:
    """Application runner entrypoint."""
    app = HaksApplication()
    return app.run()
