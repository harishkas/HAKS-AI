"""Conversation History View for HAKS.

Lists previous conversation sessions from conversations.db,
supports keyword searching, reloading conversations, and deletion.
"""

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QFrame,
)

from core.app_controller import AppController, get_app_controller
from core.logger import get_logger

logger = get_logger("ui.views.history_view")


class HistoryView(QWidget):
    """Conversation history browser and manager."""

    def __init__(self, controller: Optional[AppController] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.controller = controller or get_app_controller()
        self._conversations_cache: list[dict] = []
        self._init_ui()
        self._connect_signals()
        self.refresh_list()

    def _init_ui(self) -> None:
        """Construct the view layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        # Header
        header = QHBoxLayout()
        title = QLabel("Conversation History")
        title.setStyleSheet("font-size: 18px; font-weight: 700; color: #111827;")
        header.addWidget(title)
        header.addStretch()

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.setProperty("class", "btn_secondary")
        self.btn_refresh.clicked.connect(self.refresh_list)
        header.addWidget(self.btn_refresh)

        layout.addLayout(header)

        # Search Bar
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search conversations by title...")
        self.search_input.textChanged.connect(self._on_search_changed)
        layout.addWidget(self.search_input)

        # Conversations List
        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.list_widget, stretch=1)

        # Bottom Actions Bar
        actions_bar = QHBoxLayout()
        self.btn_open = QPushButton("Open in Chat")
        self.btn_open.setProperty("class", "btn_primary")
        self.btn_open.clicked.connect(self._on_open_selected)
        actions_bar.addWidget(self.btn_open)

        self.btn_delete = QPushButton("Delete Conversation")
        self.btn_delete.setProperty("class", "btn_danger")
        self.btn_delete.clicked.connect(self._on_delete_selected)
        actions_bar.addWidget(self.btn_delete)

        actions_bar.addStretch()
        layout.addLayout(actions_bar)

    def _connect_signals(self) -> None:
        """Connect to conversation updates."""
        self.controller.events.conversation_list_updated.connect(self.refresh_list)

    def refresh_list(self) -> None:
        """Reload conversation list from database."""
        self._conversations_cache = self.controller.get_all_conversations()
        self._filter_list(self.search_input.text())

    def _on_search_changed(self, text: str) -> None:
        """Filter conversations based on search text."""
        self._filter_list(text)

    def _filter_list(self, query: str) -> None:
        """Populate list widget according to filter query."""
        self.list_widget.clear()
        query = query.strip().lower()

        for conv in self._conversations_cache:
            title = conv["title"]
            if query and query not in title.lower():
                continue

            item_text = f"{title}\nMessages: {conv['message_count']}  |  Last Active: {conv['updated_at']}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, conv["id"])
            self.list_widget.addItem(item)

    def _get_selected_conv_id(self) -> Optional[str]:
        """Return the conversation ID of the selected item."""
        item = self.list_widget.currentItem()
        if item:
            return str(item.data(Qt.ItemDataRole.UserRole))
        return None

    def _on_open_selected(self) -> None:
        """Open the selected conversation in ChatView."""
        conv_id = self._get_selected_conv_id()
        if not conv_id:
            return
        self.controller.select_conversation(conv_id)
        # Navigate to chat page (nav id 'chat')
        self.controller.events.navigate_to_page.emit("chat")

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        """Double click opens conversation in chat."""
        self._on_open_selected()

    def _on_delete_selected(self) -> None:
        """Delete the selected conversation."""
        conv_id = self._get_selected_conv_id()
        if not conv_id:
            return

        confirm = QMessageBox.question(
            self,
            "Confirm Deletion",
            "Are you sure you want to delete this conversation and all its messages?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.controller.delete_conversation(conv_id)
            self.refresh_list()
