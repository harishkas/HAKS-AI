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
from core.events import get_app_events
from core.logger import get_logger
from storage.database import get_db_manager

logger = get_logger("ui.main_window")


class MainWindow(QMainWindow):
    """Main desktop interface for HAKS Personal AI Assistant."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        super().__init__()
        self.config = config or get_settings()
        self.events = get_app_events()
        self.db_manager = get_db_manager(self.config)

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

        # Content Stack
        self.content_stack = QStackedWidget()
        self.content_stack.setObjectName("content_area")

        self.page_home = self._build_home_page()
        self.page_chat = self._build_placeholder_page("New Chat", "Chat interface will be fully developed in Phase 2.")
        self.page_history = self._build_placeholder_page("Conversation History", "Session history and archives will be loaded here.")
        self.page_personal_memory = self._build_placeholder_page("Personal Memory", "Independent Personal Memory domain storage.")
        self.page_work_memory = self._build_placeholder_page("Work Memory", "Independent Work Memory domain storage.")
        self.page_voice = self._build_placeholder_page("Voice Assistant", "Voice pipeline, wake word 'Hey HAKS', and speech controls.")
        self.page_ai_engine = self._build_placeholder_page("AI Engine", "Ollama local runtime & Cloud AI hybrid router.")
        self.page_settings = self._build_placeholder_page("Settings", "Application configuration and privacy controls.")

        self.content_stack.addWidget(self.page_home)            # Index 0
        self.content_stack.addWidget(self.page_chat)            # Index 1
        self.content_stack.addWidget(self.page_history)         # Index 2
        self.content_stack.addWidget(self.page_personal_memory) # Index 3
        self.content_stack.addWidget(self.page_work_memory)     # Index 4
        self.content_stack.addWidget(self.page_voice)           # Index 5
        self.content_stack.addWidget(self.page_ai_engine)       # Index 6
        self.content_stack.addWidget(self.page_settings)        # Index 7

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
            ("settings", "Settings", 7),
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
        title = QLabel("HAKS Personal AI Assistant - Foundation Active")
        title.setStyleSheet("font-size: 18px; font-weight: 700; color: #58a6ff;")
        desc = QLabel(
            "Phase 1 Foundation is operational. All 5 isolated SQLite databases are loaded with WAL mode.\n"
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
        db_stats = self.db_manager.get_stats()
        c2 = self._create_card(
            "MEMORY ARCHITECTURE",
            "5 Independent DBs",
            f"Personal, Work, Conversations, Proposals, Settings"
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
            "Hey HAKS",
            "Tamil, English & Mixed • Continuous Voice Support"
        )
        grid.addWidget(c4, 1, 1)

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

    def _build_placeholder_page(self, title: str, description: str) -> QWidget:
        """Create a placeholder page for upcoming phase sections."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(10)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("font-size: 20px; font-weight: 700; color: #f0f6fc;")
        desc_lbl = QLabel(description)
        desc_lbl.setStyleSheet("font-size: 13px; color: #8b949e;")

        layout.addWidget(title_lbl, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc_lbl, alignment=Qt.AlignmentFlag.AlignCenter)
        return page

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

    def _update_status_message(self, message: str) -> None:
        """Update bottom status bar message."""
        self.lbl_status.setText(message)

    def _update_voice_badge(self, state: str) -> None:
        """Update voice status badge."""
        self.badge_voice.setText(f"Voice: {state}")
