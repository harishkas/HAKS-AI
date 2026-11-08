"""Settings and Privacy View for HAKS.

Provides theme configuration, floating assistant behaviors,
storage inspection, and Windows security permission levels overview.
"""

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QGridLayout,
    QComboBox,
    QCheckBox,
    QScrollArea,
)

from core.app_controller import AppController, get_app_controller
from core.logger import get_logger

logger = get_logger("ui.views.settings_view")


class SettingsView(QWidget):
    """Application settings, appearance, and privacy view."""

    def __init__(self, controller: Optional[AppController] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.controller = controller or get_app_controller()
        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none; background: transparent;")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        # Header
        header = QHBoxLayout()
        title = QLabel("Settings & Privacy")
        title.setStyleSheet("font-size: 18px; font-weight: 700; color: #f0f6fc;")
        header.addWidget(title)
        header.addStretch()
        layout.addLayout(header)

        # 1. Appearance Card
        app_card = QFrame()
        app_card.setProperty("class", "stat_card")
        ac_layout = QVBoxLayout(app_card)
        ac_layout.setSpacing(8)

        ac_title = QLabel("APPEARANCE & THEME")
        ac_title.setProperty("class", "card_title")
        ac_layout.addWidget(ac_title)

        theme_row = QHBoxLayout()
        theme_lbl = QLabel("Active UI Theme:")
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark Theme", "Light Theme"])
        current_theme = self.controller.get_theme()
        if current_theme == "light":
            self.theme_combo.setCurrentIndex(1)
        self.theme_combo.currentIndexChanged.connect(self._on_theme_changed)

        theme_row.addWidget(theme_lbl)
        theme_row.addWidget(self.theme_combo)
        theme_row.addStretch()
        ac_layout.addLayout(theme_row)

        layout.addWidget(app_card)

        # 2. Floating Assistant Options
        float_card = QFrame()
        float_card.setProperty("class", "stat_card")
        fc_layout = QVBoxLayout(float_card)
        fc_layout.setSpacing(8)

        fc_title = QLabel("FLOATING AI ASSISTANT OPTIONS")
        fc_title.setProperty("class", "card_title")
        fc_layout.addWidget(fc_title)

        self.chk_always_on_top = QCheckBox("Keep Floating Assistant always-on-top")
        self.chk_always_on_top.setChecked(self.controller.get_always_on_top())
        self.chk_always_on_top.toggled.connect(self._on_always_on_top_toggled)
        fc_layout.addWidget(self.chk_always_on_top)

        btn_float_row = QHBoxLayout()
        self.btn_open_floating = QPushButton("Show Floating Assistant")
        self.btn_open_floating.setProperty("class", "btn_primary")
        self.btn_open_floating.clicked.connect(lambda: self.controller.events.show_floating_requested.emit())
        btn_float_row.addWidget(self.btn_open_floating)

        self.btn_reset_pos = QPushButton("Reset Floating Window Position")
        self.btn_reset_pos.setProperty("class", "btn_secondary")
        self.btn_reset_pos.clicked.connect(self._on_reset_position)
        btn_float_row.addWidget(self.btn_reset_pos)
        btn_float_row.addStretch()
        fc_layout.addLayout(btn_float_row)

        layout.addWidget(float_card)

        # 3. Privacy & Security Framework
        sec_card = QFrame()
        sec_card.setProperty("class", "stat_card")
        sc_layout = QVBoxLayout(sec_card)
        sc_layout.setSpacing(8)

        sc_title = QLabel("WINDOWS COMPUTER CONTROL - RISK CATEGORIES")
        sc_title.setProperty("class", "card_title")
        sc_layout.addWidget(sc_title)

        levels = [
            ("Level 0: Read-only", "Allowed automatically (inspecting files, checking system metrics)."),
            ("Level 1: Reversible", "Allowed with notification (creating temp folders, opening apps)."),
            ("Level 2: Sensitive", "Explicit confirmation required (modifying settings, reading personal docs)."),
            ("Level 3: Destructive/Privileged", "Mandatory explicit approval + preview (file deletion, registry edits)."),
        ]

        for lvl_name, lvl_desc in levels:
            row = QLabel(f"• <b>{lvl_name}</b>: {lvl_desc}")
            row.setStyleSheet("color: #e6edf3; font-size: 12px; margin: 2px 0;")
            sc_layout.addWidget(row)

        layout.addWidget(sec_card)

        # 4. Storage & Diagnostics Card
        diag_card = QFrame()
        diag_card.setProperty("class", "stat_card")
        dc_layout = QVBoxLayout(diag_card)
        dc_layout.setSpacing(6)

        dc_title = QLabel("STORAGE DATABASES (5/5 ACTIVE)")
        dc_title.setProperty("class", "card_title")
        dc_layout.addWidget(dc_title)

        stats = self.controller.db.get_stats()
        for db_name, info in stats.items():
            size_kb = info["size_bytes"] / 1024.0
            lbl = QLabel(f"• <b>{db_name}.db</b> — Size: {size_kb:.1f} KB | Tables: {list(info['tables'].keys())}")
            lbl.setStyleSheet("color: #8b949e; font-size: 11px;")
            dc_layout.addWidget(lbl)

        layout.addWidget(diag_card)
        layout.addStretch()

        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    def _on_theme_changed(self, index: int) -> None:
        theme = "light" if index == 1 else "dark"
        self.controller.set_theme(theme)

    def _on_always_on_top_toggled(self, checked: bool) -> None:
        self.controller.set_always_on_top(checked)

    def _on_reset_position(self) -> None:
        self.controller.save_floating_position(120, 120)
        self.controller.events.show_floating_requested.emit()
