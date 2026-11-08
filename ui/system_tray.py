"""System Tray icon integration for HAKS.

Provides system tray presence, context menu actions, notification alerts,
and toggling between the Main Desktop Window and Floating AI Assistant.
"""

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor, QFont
from PySide6.QtWidgets import QSystemTrayIcon, QMenu

from core.app_controller import AppController, get_app_controller
from core.logger import get_logger

logger = get_logger("ui.system_tray")


def create_tray_icon() -> QIcon:
    """Dynamically generate a high-DPI tray icon pixmap."""
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    # Blue circle background
    painter.setBrush(QColor("#1f6feb"))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(4, 4, 56, 56)

    # White 'H' letter
    painter.setPen(QColor("#ffffff"))
    font = QFont("Segoe UI", 24, QFont.Weight.Bold)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "H")
    painter.end()

    return QIcon(pixmap)


class HaksSystemTray(QSystemTrayIcon):
    """System tray manager for HAKS."""

    def __init__(self, controller: Optional[AppController] = None, parent=None):
        super().__init__(parent)
        self.controller = controller or get_app_controller()
        self.setIcon(create_tray_icon())
        self.setToolTip("HAKS - Personal AI Assistant")

        self._build_menu()
        self.activated.connect(self._on_tray_activated)
        logger.info("HaksSystemTray initialized.")

    def _build_menu(self) -> None:
        """Create contextual tray menu."""
        menu = QMenu()

        act_main = menu.addAction("Open Desktop Window")
        act_main.triggered.connect(lambda: self.controller.events.show_main_window_requested.emit())

        act_floating = menu.addAction("Open Floating Assistant")
        act_floating.triggered.connect(lambda: self.controller.events.show_floating_requested.emit())

        menu.addSeparator()

        self.act_mute = menu.addAction("Mute Microphone")
        self.act_mute.triggered.connect(self._on_toggle_mic)

        self.act_sleep = menu.addAction("Voice: Go to Sleep")
        self.act_sleep.triggered.connect(self._on_toggle_sleep)

        menu.addSeparator()

        act_theme = menu.addAction("Toggle Dark/Light Theme")
        act_theme.triggered.connect(self._on_toggle_theme)

        menu.addSeparator()

        act_exit = menu.addAction("Exit HAKS")
        act_exit.triggered.connect(self._on_exit)

        self.setContextMenu(menu)

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        """Handle clicks on tray icon."""
        if reason in (QSystemTrayIcon.ActivationReason.Trigger, QSystemTrayIcon.ActivationReason.DoubleClick):
            self.controller.events.show_main_window_requested.emit()

    def _on_toggle_mic(self) -> None:
        is_muted = self.controller.toggle_mic_mute()
        self.act_mute.setText("Unmute Microphone" if is_muted else "Mute Microphone")

    def _on_toggle_sleep(self) -> None:
        current = self.controller.get_voice_state()
        if current == "SLEEP":
            self.controller.set_voice_state("STANDBY")
            self.act_sleep.setText("Voice: Go to Sleep")
        else:
            self.controller.set_voice_state("SLEEP")
            self.act_sleep.setText("Voice: Wake Up")

    def _on_toggle_theme(self) -> None:
        current = self.controller.get_theme()
        new_theme = "light" if current == "dark" else "dark"
        self.controller.set_theme(new_theme)

    def _on_exit(self) -> None:
        from PySide6.QtWidgets import QApplication
        qapp = QApplication.instance()
        if qapp:
            qapp.quit()
