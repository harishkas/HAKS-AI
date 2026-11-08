"""UI module for HAKS Desktop Application and Floating AI Assistant."""

from ui.app import HaksApplication, run_app
from ui.floating_assistant import FloatingAssistant
from ui.main_window import MainWindow
from ui.styles import get_stylesheet
from ui.system_tray import HaksSystemTray

__all__ = [
    "FloatingAssistant",
    "HaksApplication",
    "HaksSystemTray",
    "MainWindow",
    "get_stylesheet",
    "run_app",
]
