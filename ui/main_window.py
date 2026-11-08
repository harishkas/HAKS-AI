"""Main Window implementation for HAKS Desktop Application.

PySide6 implementation providing responsive modular layout, high-DPI scaling,
navigation sidebar, top status bar, stacked content views, and live system monitoring.
"""

from typing import Optional

from PySide6.QtCore import Qt, QSize
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QFrame,
    QStatusBar,
    QGridLayout,
    QScrollArea,
)

from config.settings import AppConfig, get_settings
from core.app_controller import AppController, get_app_controller
from core.logger import get_logger
from ui.views.chat_view import ChatView
from ui.views.history_view import HistoryView
from ui.views.personal_memory_view import PersonalMemoryView
from ui.views.work_memory_view import WorkMemoryView
from ui.views.voice_view import VoiceView
from ui.views.ai_engine_view import AIEngineView
from ui.views.windows_control_view import WindowsControlView
from ui.views.settings_view import SettingsView

logger = get_logger("ui.main_window")


class MainWindow(QMainWindow):
    """Main desktop interface for HAKS Personal AI Assistant."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        controller: Optional[AppController] = None,
    ) -> None:
        super().__init__()
        self.config = config or get_settings()
        self.controller = controller or get_app_controller(self.config)
        self.events = self.controller.events

        self._nav_buttons: dict[str, QPushButton] = {}
        self._init_ui()
        self._connect_signals()
        logger.info("MainWindow initialized successfully.")

    def _init_ui(self) -> None:
        """Construct the UI hierarchy."""
        self.setWindowTitle(self.config.ui.app_title)
        self.resize(self.config.ui.window_width, self.config.ui.window_height)
        self.setMinimumSize(self.config.ui.min_window_width, self.config.ui.min_window_height)

        # Central Root Container
        root_container = QWidget(self)
        root_container.setObjectName("root_container")
        self.setCentralWidget(root_container)

        root_layout = QHBoxLayout(root_container)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # 1. Left Sidebar
        sidebar = self._build_sidebar()
        root_layout.addWidget(sidebar)

        # 2. Right Main Area (Top Bar + Content Area)
        main_area = QWidget()
        main_layout = QVBoxLayout(main_area)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Top Bar
        top_bar = self._build_top_bar()
        main_layout.addWidget(top_bar)

        # Content Stack with real functional views
        self.content_stack = QStackedWidget()
        self.content_stack.setObjectName("content_area")

        self.page_home = self._build_home_page()
        self.page_chat = ChatView(self.controller)
        self.page_history = HistoryView(self.controller)
        self.page_personal_memory = PersonalMemoryView(self.controller)
        self.page_work_memory = WorkMemoryView(self.controller)
        self.page_voice = VoiceView(self.controller)
        self.page_ai_engine = AIEngineView(self.controller)
        self.page_windows_control = WindowsControlView(self.controller)
        self.page_settings = SettingsView(self.controller)

        self.content_stack.addWidget(self.page_home)            # Index 0
        self.content_stack.addWidget(self.page_chat)            # Index 1
        self.content_stack.addWidget(self.page_history)         # Index 2
        self.content_stack.addWidget(self.page_personal_memory) # Index 3
        self.content_stack.addWidget(self.page_work_memory)     # Index 4
        self.content_stack.addWidget(self.page_voice)           # Index 5
        self.content_stack.addWidget(self.page_ai_engine)       # Index 6
        self.content_stack.addWidget(self.page_windows_control) # Index 7
        self.content_stack.addWidget(self.page_settings)        # Index 8

        main_layout.addWidget(self.content_stack)
        root_layout.addWidget(main_area)

        # 3. Status Bar
        self._build_status_bar()

    def _build_sidebar(self) -> QWidget:
        """Create the left navigation sidebar."""
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(8, 12, 8, 12)
        layout.setSpacing(4)

        # Brand header
        brand_title = QLabel("HAKS")
        brand_title.setObjectName("brand_title")
        brand_subtitle = QLabel("Personal AI Assistant")
        brand_subtitle.setObjectName("brand_subtitle")
        layout.addWidget(brand_title)
        layout.addWidget(brand_subtitle)

        # Nav items
        nav_items = [
            ("home", "Dashboard", 0),
            ("chat", "New Chat", 1),
            ("history", "Conversation History", 2),
            ("personal_memory", "Personal Memory", 3),
            ("work_memory", "Work Memory", 4),
            ("voice", "Voice Assistant", 5),
            ("ai_engine", "AI Engine", 6),
            ("control", "Computer Control", 7),
            ("settings", "Settings", 8),
        ]

        for nav_id, label, index in nav_items:
            btn = QPushButton(label)
            btn.setProperty("class", "nav_button")
            btn.setCheckable(True)
            btn.clicked.connect(lambda _, idx=index, nid=nav_id: self._on_navigate(nid, idx))
            self._nav_buttons[nav_id] = btn
            layout.addWidget(btn)

        layout.addStretch()

        # Offline & Privacy notice in sidebar footer
        privacy_label = QLabel("Offline-First • Privacy-Protected")
        privacy_label.setStyleSheet("color: #484f58; font-size: 10px; padding: 6px;")
        layout.addWidget(privacy_label)

        # Set Dashboard active by default
        if "home" in self._nav_buttons:
            self._nav_buttons["home"].setChecked(True)

        return sidebar

    def _build_top_bar(self) -> QWidget:
        """Create the top status header bar."""
        top_bar = QWidget()
        top_bar.setObjectName("top_bar")
        layout = QHBoxLayout(top_bar)
        layout.setContentsMargins(16, 6, 16, 6)

        self.lbl_page_title = QLabel("System Dashboard")
        self.lbl_page_title.setObjectName("top_bar_title")
        layout.addWidget(self.lbl_page_title)

        layout.addStretch()

        # Engine Badge
        self.badge_engine = QLabel("Engine: Local Ollama (Standby)")
        self.badge_engine.setProperty("class", "status_badge badge_local")
        layout.addWidget(self.badge_engine)

        # Voice Badge
        self.badge_voice = QLabel("Voice: STANDBY")
        self.badge_voice.setProperty("class", "status_badge badge_voice")
        layout.addWidget(self.badge_voice)

        # Storage Badge
        self.badge_storage = QLabel("Storage: 5/5 DBs Active")
        self.badge_storage.setProperty("class", "status_badge badge_offline")
        layout.addWidget(self.badge_storage)

        # Switch to Floating Assistant button
        self.btn_floating_mode = QPushButton("Floating Mode 🗗")
        self.btn_floating_mode.setProperty("class", "btn_secondary")
        self.btn_floating_mode.setToolTip("Switch to compact Floating AI Assistant")
        self.btn_floating_mode.clicked.connect(self._on_switch_to_floating)
        layout.addWidget(self.btn_floating_mode)

        return top_bar

    def _build_home_page(self) -> QWidget:
        """Create the dashboard home page with hardware info and database stats."""
        page = QWidget()
        scroll = QScrollArea(page)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none; background: transparent;")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(20)

        # Welcome Card
        welcome_frame = QFrame()
        welcome_frame.setProperty("class", "stat_card")
        wf_layout = QVBoxLayout(welcome_frame)
        title = QLabel("HAKS Personal AI Assistant - Foundation & Desktop UI Active")
        title.setStyleSheet("font-size: 18px; font-weight: 700; color: #58a6ff;")
        desc = QLabel(
            "Phase 2 Desktop UI is operational. Both Main Window and Floating Assistant share a single controller.\n"
            "Hardware-calibrated for Dell Inspiron 3576 (Intel i5-7200U, 8 GB RAM, Radeon 520)."
        )
        desc.setStyleSheet("color: #8b949e; font-size: 13px; line-height: 1.4;")
        desc.setWordWrap(True)
        wf_layout.addWidget(title)
        wf_layout.addWidget(desc)
        layout.addWidget(welcome_frame)

        # Grid of System Cards
        grid = QGridLayout()
        grid.setSpacing(14)

        # Card 1: Hardware Specs
        c1 = self._create_card(
            "TARGET HARDWARE",
            "Intel i5-7200U | 8 GB RAM",
            "Radeon 520 (2GB) • SSD ~238 GB • Win 10/11"
        )
        grid.addWidget(c1, 0, 0)

        # Card 2: Storage Architecture
        c2 = self._create_card(
            "MEMORY ARCHITECTURE",
            "5 Independent DBs",
            "Personal, Work, Conversations, Proposals, Settings"
        )
        grid.addWidget(c2, 0, 1)

        # Card 3: AI Intelligence
        c3 = self._create_card(
            "AI INTELLIGENCE ROUTER",
            "Hybrid Local + Cloud",
            "Ollama (Routine/Offline) + Cloud Adapter (Complex)"
        )
        grid.addWidget(c3, 1, 0)

        # Card 4: Voice Pipeline
        c4 = self._create_card(
            "VOICE PIPELINE",
            'Hey HAKS',
            "Tamil, English & Mixed • Continuous Voice Support"
        )
        grid.addWidget(c4, 1, 1)

        # Card 5: Computer Control & Automation
        c5 = self._create_card(
            "WINDOWS COMPUTER CONTROL",
            "Full Control Active",
            "CMD & PowerShell • UAC Approval • File/App/Sys Automation"
        )
        grid.addWidget(c5, 2, 0, 1, 2)

        layout.addLayout(grid)
        layout.addStretch()

        scroll_layout = QVBoxLayout(page)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.addWidget(scroll)
        scroll.setWidget(container)

        return page

    def _create_card(self, title: str, value: str, detail: str) -> QFrame:
        """Helper to create a structured metric card."""
        frame = QFrame()
        frame.setProperty("class", "stat_card")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(4)

        t_lbl = QLabel(title)
        t_lbl.setProperty("class", "card_title")
        v_lbl = QLabel(value)
        v_lbl.setProperty("class", "card_value")
        d_lbl = QLabel(detail)
        d_lbl.setProperty("class", "card_detail")
        d_lbl.setWordWrap(True)

        layout.addWidget(t_lbl)
        layout.addWidget(v_lbl)
        layout.addWidget(d_lbl)
        return frame

    def _build_status_bar(self) -> None:
        """Create the bottom status bar."""
        status_bar = QStatusBar(self)
        self.setStatusBar(status_bar)

        self.lbl_status = QLabel("HAKS Ready • All systems normal")
        status_bar.addWidget(self.lbl_status)

        self.lbl_db_info = QLabel("SQLite WAL: Active | Version: 0.1.0")
        status_bar.addPermanentWidget(self.lbl_db_info)

    def _on_navigate(self, nav_id: str, index: int) -> None:
        """Handle sidebar navigation clicks."""
        for nid, btn in self._nav_buttons.items():
            btn.setChecked(nid == nav_id)

        self.content_stack.setCurrentIndex(index)
        btn = self._nav_buttons.get(nav_id)
        if btn:
            self.lbl_page_title.setText(btn.text())
        self.events.navigate_to_page.emit(nav_id)

    def _connect_signals(self) -> None:
        """Connect application event bus to UI slots."""
        self.events.status_message_updated.connect(self._update_status_message)
        self.events.voice_state_changed.connect(self._update_voice_badge)
        self.events.navigate_to_page.connect(self._on_external_navigate)
        self.events.show_main_window_requested.connect(self._on_show_requested)

    def _update_status_message(self, message: str) -> None:
        self.lbl_status.setText(message)

    def _update_voice_badge(self, state: str) -> None:
        self.badge_voice.setText(f"Voice: {state}")
        if state in ("LISTENING", "SPEAKING"):
            self.badge_voice.setProperty("class", "status_badge badge_voice_active")
        else:
            self.badge_voice.setProperty("class", "status_badge badge_voice")
        self.badge_voice.style().unpolish(self.badge_voice)
        self.badge_voice.style().polish(self.badge_voice)

    def _on_external_navigate(self, page_id: str) -> None:
        nav_map = {
            "home": 0,
            "chat": 1,
            "history": 2,
            "personal_memory": 3,
            "work_memory": 4,
            "voice": 5,
            "ai_engine": 6,
            "control": 7,
            "settings": 8,
        }
        if page_id in nav_map:
            self._on_navigate(page_id, nav_map[page_id])

    def _on_switch_to_floating(self) -> None:
        """Switch to compact floating assistant view."""
        self.events.show_floating_requested.emit()
        self.hide()

    def _on_show_requested(self) -> None:
        """Show and raise main window."""
        self.show()
        self.activateWindow()
        self.raise_()
