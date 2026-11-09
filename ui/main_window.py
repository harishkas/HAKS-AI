"""Main Window implementation for HAKS Desktop Application.

PySide6 implementation providing a clean, modern Windows Light (White mode) UI,
intuitive navigation sidebar, top status bar with quick theme toggle,
friendly home dashboard, stacked content views, and live system monitoring.
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
from ui.views.recovery_view import RecoveryView
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
        self.page_recovery = RecoveryView(self.controller)
        self.page_settings = SettingsView(self.controller)

        self.content_stack.addWidget(self.page_home)            # Index 0
        self.content_stack.addWidget(self.page_chat)            # Index 1
        self.content_stack.addWidget(self.page_history)         # Index 2
        self.content_stack.addWidget(self.page_personal_memory) # Index 3
        self.content_stack.addWidget(self.page_work_memory)     # Index 4
        self.content_stack.addWidget(self.page_voice)           # Index 5
        self.content_stack.addWidget(self.page_ai_engine)       # Index 6
        self.content_stack.addWidget(self.page_windows_control) # Index 7
        self.content_stack.addWidget(self.page_recovery)        # Index 8
        self.content_stack.addWidget(self.page_settings)        # Index 9

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
        brand_title = QLabel("HAKS AI")
        brand_title.setObjectName("brand_title")
        brand_subtitle = QLabel("Personal Windows Assistant")
        brand_subtitle.setObjectName("brand_subtitle")
        layout.addWidget(brand_title)
        layout.addWidget(brand_subtitle)

        # Nav items with intuitive icons and plain language
        nav_items = [
            ("home", "🏠  Home", 0),
            ("chat", "💬  New Chat", 1),
            ("history", "🕒  History", 2),
            ("personal_memory", "👤  Personal Memory", 3),
            ("work_memory", "💼  Work Memory", 4),
            ("voice", "🎙️  Voice Assistant", 5),
            ("ai_engine", "⚡  AI Models", 6),
            ("control", "🖥️  Computer Control", 7),
            ("recovery", "🛡️  Backup & Recovery", 8),
            ("settings", "⚙️  Settings", 9),
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
        privacy_label = QLabel("🔒 100% Private • Offline Ready")
        privacy_label.setStyleSheet("color: #6b7280; font-size: 11px; padding: 6px 10px; font-weight: 500;")
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
        layout.setSpacing(10)

        self.lbl_page_title = QLabel("Home")
        self.lbl_page_title.setObjectName("top_bar_title")
        layout.addWidget(self.lbl_page_title)

        layout.addStretch()

        # Network Connectivity Badge
        is_online = self.controller.is_online()
        self.badge_network = QLabel("Online" if is_online else "Offline")
        self.badge_network.setProperty("class", "status_badge badge_online" if is_online else "status_badge badge_offline")
        layout.addWidget(self.badge_network)

        # Engine Badge
        self.badge_engine = QLabel("Engine: Ready")
        self.badge_engine.setProperty("class", "status_badge badge_local")
        layout.addWidget(self.badge_engine)

        # Voice Badge
        self.badge_voice = QLabel("Voice: STANDBY")
        self.badge_voice.setProperty("class", "status_badge badge_voice")
        layout.addWidget(self.badge_voice)

        # Storage Badge
        self.badge_storage = QLabel("Storage: 5 DBs Ready")
        self.badge_storage.setProperty("class", "status_badge badge_local")
        layout.addWidget(self.badge_storage)

        # Quick Theme Toggle Button (One-click Light/Dark switch)
        current_theme = self.controller.get_theme()
        theme_icon = "🌙 Dark Mode" if current_theme == "light" else "☀️ White Mode"
        self.btn_theme_toggle = QPushButton(theme_icon)
        self.btn_theme_toggle.setProperty("class", "btn_secondary")
        self.btn_theme_toggle.setToolTip("Toggle between Clean White Mode and Dark Mode")
        self.btn_theme_toggle.clicked.connect(self._on_toggle_theme)
        layout.addWidget(self.btn_theme_toggle)

        # Switch to Floating Assistant button
        self.btn_floating_mode = QPushButton("🗗 Floating Mini")
        self.btn_floating_mode.setProperty("class", "btn_secondary")
        self.btn_floating_mode.setToolTip("Switch to compact Floating AI Assistant")
        self.btn_floating_mode.clicked.connect(self._on_switch_to_floating)
        layout.addWidget(self.btn_floating_mode)

        return top_bar

    def _on_toggle_theme(self) -> None:
        """Toggle active UI theme with a single click."""
        cur = self.controller.get_theme()
        new_theme = "dark" if cur == "light" else "light"
        self.controller.set_theme(new_theme)
        self.btn_theme_toggle.setText("🌙 Dark Mode" if new_theme == "light" else "☀️ White Mode")

    def _build_home_page(self) -> QWidget:
        """Create a clean, welcoming, easy-to-understand Windows home screen."""
        page = QWidget()
        scroll = QScrollArea(page)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none; background: transparent;")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(22)

        # 1. Welcome Card
        welcome_frame = QFrame()
        welcome_frame.setProperty("class", "stat_card")
        wf_layout = QVBoxLayout(welcome_frame)
        wf_layout.setContentsMargins(20, 18, 20, 18)
        wf_layout.setSpacing(6)

        title = QLabel("Welcome to HAKS AI")
        title.setStyleSheet("font-size: 22px; font-weight: 700; color: #0067c0;")
        desc = QLabel(
            "Your personal, private AI assistant built for Windows. "
            "Chat with intelligent local or cloud models, speak naturally hands-free, "
            "store your personal preferences, and automate Windows tasks — with complete privacy."
        )
        desc.setStyleSheet("color: #4b5563; font-size: 13px; line-height: 1.5;")
        desc.setWordWrap(True)
        wf_layout.addWidget(title)
        wf_layout.addWidget(desc)
        layout.addWidget(welcome_frame)

        # Section Heading
        lbl_actions = QLabel("QUICK ACTIONS")
        lbl_actions.setProperty("class", "card_title")
        layout.addWidget(lbl_actions)

        # 2. Grid of Intuitive Action Cards (2x2)
        grid = QGridLayout()
        grid.setSpacing(16)

        # Action 1: Chat
        c1 = self._create_action_card(
            icon="💬",
            title="Start a New Chat",
            desc="Ask questions, write documents, brainstorm ideas, and get instant answers with intelligent AI models.",
            button_text="Open Chat →",
            on_click=lambda: self._on_navigate("chat", 1),
        )
        grid.addWidget(c1, 0, 0)

        # Action 2: Voice
        c2 = self._create_action_card(
            icon="🎙️",
            title="Voice Assistant",
            desc="Hands-free speech with wake phrase 'Hey HAKS'. Supports natural conversations in Tamil, English, and Mixed speech.",
            button_text="Open Voice →",
            on_click=lambda: self._on_navigate("voice", 5),
        )
        grid.addWidget(c2, 0, 1)

        # Action 3: PC Control
        c3 = self._create_action_card(
            icon="🖥️",
            title="Computer Control",
            desc="Control Windows with plain language: launch programs, search files, manage processes, and run PowerShell commands safely.",
            button_text="Control PC →",
            on_click=lambda: self._on_navigate("control", 7),
        )
        grid.addWidget(c3, 1, 0)

        # Action 4: Memory
        c4 = self._create_action_card(
            icon="🧠",
            title="Memory & Preferences",
            desc="HAKS securely remembers your personal facts, work projects, and settings on your local PC — never shared with third parties.",
            button_text="View Memory →",
            on_click=lambda: self._on_navigate("personal_memory", 3),
        )
        grid.addWidget(c4, 1, 1)

        layout.addLayout(grid)

        # 3. Simple System Status Overview
        lbl_status_sec = QLabel("SYSTEM OVERVIEW")
        lbl_status_sec.setProperty("class", "card_title")
        layout.addWidget(lbl_status_sec)

        status_grid = QGridLayout()
        status_grid.setSpacing(14)

        s1 = self._create_card("🔒 PRIVACY & STORAGE", "100% On-Device", "Personal memory encrypted locally with SQLite WAL")
        s2 = self._create_card("⚡ AI ENGINE", "Hybrid Ready", "Local Ollama + Cloud provider fallback with intelligent routing")
        s3 = self._create_card("🗗 MINI FLOATING WIDGET", "Always Accessible", "Click 'Floating Mini' at the top right to keep HAKS on screen")

        status_grid.addWidget(s1, 0, 0)
        status_grid.addWidget(s2, 0, 1)
        status_grid.addWidget(s3, 0, 2)
        layout.addLayout(status_grid)

        # 4. Quick Tips Card
        tips_frame = QFrame()
        tips_frame.setProperty("class", "stat_card")
        tf_layout = QVBoxLayout(tips_frame)
        tf_layout.setContentsMargins(16, 12, 16, 12)
        tf_layout.setSpacing(4)
        tips_title = QLabel("💡 Quick Tips")
        tips_title.setStyleSheet("font-weight: 700; color: #111827; font-size: 13px;")
        tips_text = QLabel(
            "• In Chat, press Enter to send your message, or Shift+Enter to start a new line.\n"
            "• Use the 🌙 / ☀️ button in the top bar to easily switch between Clean White Mode and Dark Mode.\n"
            "• You can type commands like /cmd or /powershell directly into Chat to control your computer."
        )
        tips_text.setStyleSheet("color: #4b5563; font-size: 12px; line-height: 1.5;")
        tf_layout.addWidget(tips_title)
        tf_layout.addWidget(tips_text)
        layout.addWidget(tips_frame)

        layout.addStretch()

        scroll_layout = QVBoxLayout(page)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.addWidget(scroll)
        scroll.setWidget(container)

        return page

    def _create_action_card(self, icon: str, title: str, desc: str, button_text: str, on_click) -> QFrame:
        """Create a user-friendly clickable action card for the home screen."""
        frame = QFrame()
        frame.setProperty("class", "stat_card")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(8)

        # Header with icon and title
        hdr = QHBoxLayout()
        hdr.setSpacing(8)
        lbl_icon = QLabel(icon)
        lbl_icon.setStyleSheet("font-size: 22px;")
        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("font-size: 16px; font-weight: 700; color: #111827;")
        hdr.addWidget(lbl_icon)
        hdr.addWidget(lbl_title)
        hdr.addStretch()
        layout.addLayout(hdr)

        # Description
        lbl_desc = QLabel(desc)
        lbl_desc.setStyleSheet("color: #4b5563; font-size: 12px; line-height: 1.4;")
        lbl_desc.setWordWrap(True)
        layout.addWidget(lbl_desc)

        layout.addSpacing(4)

        # Button
        btn = QPushButton(button_text)
        btn.setProperty("class", "btn_primary")
        btn.clicked.connect(on_click)
        layout.addWidget(btn)

        return frame

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

    def _on_navigate(self, nav_id: str, index: int, notify_event_bus: bool = True) -> None:
        """Handle sidebar navigation clicks."""
        for nid, btn in self._nav_buttons.items():
            btn.setChecked(nid == nav_id)

        self.content_stack.setCurrentIndex(index)
        btn = self._nav_buttons.get(nav_id)
        if btn:
            # Clean label (strip leading emoji if present for title bar)
            raw_text = btn.text()
            title_text = raw_text.split("  ")[-1] if "  " in raw_text else raw_text
            self.lbl_page_title.setText(title_text)
        if notify_event_bus:
            self.events.navigate_to_page.emit(nav_id)

    def _connect_signals(self) -> None:
        """Connect application event bus to UI slots."""
        self.events.status_message_updated.connect(self._update_status_message)
        self.events.voice_state_changed.connect(self._update_voice_badge)
        self.events.network_connectivity_changed.connect(self._update_network_badge)
        self.events.navigate_to_page.connect(self._on_external_navigate)
        self.events.show_main_window_requested.connect(self._on_show_requested)
        self.events.theme_changed.connect(self._on_theme_changed)

    def _on_theme_changed(self, theme_name: str) -> None:
        """Update toggle button label when theme changes externally."""
        if hasattr(self, "btn_theme_toggle"):
            self.btn_theme_toggle.setText("🌙 Dark Mode" if theme_name == "light" else "☀️ White Mode")

    def _update_status_message(self, message: str) -> None:
        self.lbl_status.setText(message)

    def _update_network_badge(self, is_online: bool) -> None:
        if is_online:
            self.badge_network.setText("Online")
            self.badge_network.setProperty("class", "status_badge badge_online")
        else:
            self.badge_network.setText("Offline")
            self.badge_network.setProperty("class", "status_badge badge_offline")
        self.badge_network.style().unpolish(self.badge_network)
        self.badge_network.style().polish(self.badge_network)

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
            "recovery": 8,
            "settings": 9,
        }
        if page_id in nav_map:
            self._on_navigate(page_id, nav_map[page_id], notify_event_bus=False)

    def _on_switch_to_floating(self) -> None:
        """Switch to compact floating assistant view."""
        self.events.show_floating_requested.emit()
        self.hide()

    def _on_show_requested(self) -> None:
        """Show and raise main window."""
        self.show()
        self.activateWindow()
        self.raise_()

    def closeEvent(self, event) -> None:
        """Handle main window close event, ensuring child timers and workers are stopped."""
        try:
            if hasattr(self, "page_windows_control") and self.page_windows_control:
                self.page_windows_control.close()
            if hasattr(self, "page_chat") and self.page_chat:
                self.page_chat.close()
        except Exception as exc:
            logger.warning("Error closing main window subviews: %s", exc)
        super().closeEvent(event)
