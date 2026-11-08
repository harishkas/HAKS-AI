"""Application bootstrap and Qt lifecycle management for HAKS.

Configures High-DPI scaling, stylesheets, database connection validation,
shared AppController, Main Window, Floating Assistant, and System Tray.
"""

import os
import sys
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from config.settings import AppConfig, get_settings
from core.app_controller import AppController, get_app_controller
from core.logger import get_logger, setup_logging
from storage.database import get_db_manager
from ui.floating_assistant import FloatingAssistant
from ui.main_window import MainWindow
from ui.styles import get_stylesheet
from ui.system_tray import HaksSystemTray

logger = get_logger("ui.app")


class HaksApplication:
    """Manages the lifecycle of the HAKS PySide6 desktop application."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or get_settings()
        self.qapp: Optional[QApplication] = None
        self.controller: Optional[AppController] = None
        self.main_window: Optional[MainWindow] = None
        self.floating_assistant: Optional[FloatingAssistant] = None
        self.tray_icon: Optional[HaksSystemTray] = None

    def initialize(self) -> None:
        """Initialize core services, shared controller, and Qt application."""
        # 1. Setup logging
        setup_logging(self.config)
        logger.info("Starting HAKS Desktop Application v%s...", self.config.ui.version)

        # 2. Initialize SQLite databases
        db_manager = get_db_manager(self.config)
        db_manager.initialize()

        # 3. Qt High DPI configuration
        os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
        os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.window.warning=false")
        if hasattr(Qt, "HighDpiScaleFactorRoundingPolicy"):
            QApplication.setHighDpiScaleFactorRoundingPolicy(
                Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
            )

        # 4. Create QApplication instance
        self.qapp = QApplication.instance()
        if self.qapp is None:
            self.qapp = QApplication(sys.argv)

        self.qapp.setApplicationName(self.config.ui.app_name)
        self.qapp.setApplicationDisplayName(self.config.ui.app_title)

        # 5. Initialize shared AppController
        self.controller = get_app_controller(self.config)

        # 6. Apply initial theme
        self._apply_theme(self.controller.get_theme())
        self.controller.events.theme_changed.connect(self._apply_theme)

        # 7. Connect cleanup on shutdown
        self.qapp.aboutToQuit.connect(self._on_shutdown)

        # 8. Instantiate MainWindow & FloatingAssistant (sharing the same controller!)
        self.main_window = MainWindow(self.config, self.controller)
        self.floating_assistant = FloatingAssistant(self.config, self.controller)

        # 9. Instantiate System Tray
        if HaksSystemTray.isSystemTrayAvailable():
            self.tray_icon = HaksSystemTray(self.controller)
            self.tray_icon.show()

        logger.info("HAKS Desktop Application initialized successfully.")

    def run(self) -> int:
        """Display main window and start Qt event loop."""
        if not self.main_window or not self.qapp:
            self.initialize()

        assert self.main_window is not None
        assert self.qapp is not None

        if self.config.ui.start_minimized_to_tray and self.tray_icon:
            logger.info("Application starting minimized to system tray.")
        else:
            self.main_window.show()

        logger.info("Entering Qt event loop.")
        return self.qapp.exec()

    def _apply_theme(self, theme_name: str) -> None:
        """Apply active theme stylesheet to the entire application."""
        if self.qapp:
            stylesheet = get_stylesheet(theme_name)
            self.qapp.setStyleSheet(stylesheet)
            logger.info("Applied stylesheet for theme: %s", theme_name)

    def _on_shutdown(self) -> None:
        """Graceful application shutdown hook."""
        logger.info("HAKS Desktop Application shutting down gracefully.")
        if self.floating_assistant:
            try:
                self.controller.save_floating_position(
                    self.floating_assistant.x(), self.floating_assistant.y()
                )
                self.floating_assistant.close()
            except Exception as exc:
                logger.warning("Error closing floating assistant on shutdown: %s", exc)
        if self.main_window:
            try:
                self.main_window.close()
            except Exception as exc:
                logger.warning("Error closing main window on shutdown: %s", exc)
        if self.controller:
            try:
                self.controller.shutdown()
            except Exception as exc:
                logger.warning("Error during controller shutdown: %s", exc)


def run_app() -> int:
    """Application runner entrypoint."""
    app = HaksApplication()
    return app.run()
