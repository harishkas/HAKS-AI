"""Work Memory View for HAKS.

Manages the independent work_memory.db storage domain:
viewing, searching, project organization, adding, deleting, and clearing work memory records.
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
    QDialog,
    QFormLayout,
    QTextEdit,
)

from core.app_controller import AppController, get_app_controller
from core.logger import get_logger

logger = get_logger("ui.views.work_memory_view")


class AddWorkMemoryDialog(QDialog):
    """Dialog to add a new work memory entry."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Work Memory")
        self.setFixedSize(480, 380)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.project_input = QLineEdit()
        self.project_input.setText("HAKS AI")
        form.addRow("Project:", self.project_input)

        self.title_input = QLineEdit()
        self.title_input.setPlaceholderText("e.g., Architecture Design / API Spec")
        form.addRow("Title:", self.title_input)

        self.tags_input = QLineEdit()
        self.tags_input.setPlaceholderText("e.g., python, pyside6, sqlite, architecture")
        form.addRow("Tags:", self.tags_input)

        self.content_input = QTextEdit()
        self.content_input.setPlaceholderText("Enter the technical decision, algorithm or context...")
        form.addRow("Content:", self.content_input)

        layout.addLayout(form)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_cancel)

        self.btn_save = QPushButton("Save Work Memory")
        self.btn_save.setProperty("class", "btn_primary")
        self.btn_save.clicked.connect(self._validate_and_accept)
        btn_row.addWidget(self.btn_save)

        layout.addLayout(btn_row)

    def _validate_and_accept(self) -> None:
        if not self.title_input.text().strip():
            QMessageBox.warning(self, "Validation", "Title cannot be empty.")
            return
        if not self.content_input.toPlainText().strip():
            QMessageBox.warning(self, "Validation", "Content cannot be empty.")
            return
        self.accept()

    def get_data(self) -> dict:
        return {
            "project_name": self.project_input.text().strip() or "default",
            "title": self.title_input.text().strip(),
            "tags": self.tags_input.text().strip(),
            "content": self.content_input.toPlainText().strip(),
        }


class WorkMemoryView(QWidget):
    """Work Memory domain management interface."""

    def __init__(self, controller: Optional[AppController] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.controller = controller or get_app_controller()
        self._init_ui()
        self._connect_signals()
        self.refresh_memories()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        # Header
        header = QHBoxLayout()
        title = QLabel("Work Memory")
        title.setStyleSheet("font-size: 18px; font-weight: 700; color: #f0f6fc;")
        subtitle = QLabel("Independent domain: projects, algorithms & engineering decisions")
        subtitle.setStyleSheet("color: #8b949e; font-size: 12px; margin-left: 8px;")
        header.addWidget(title)
        header.addWidget(subtitle)
        header.addStretch()

        self.btn_add = QPushButton("+ Add Work Memory")
        self.btn_add.setProperty("class", "btn_primary")
        self.btn_add.clicked.connect(self._on_add_memory)
        header.addWidget(self.btn_add)

        self.btn_clear = QPushButton("Clear Work Memory")
        self.btn_clear.setProperty("class", "btn_danger")
        self.btn_clear.clicked.connect(self._on_clear_domain)
        header.addWidget(self.btn_clear)

        layout.addLayout(header)

        # Search Bar
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search work memories by project, title, content, or tags...")
        self.search_input.textChanged.connect(self._on_search_changed)
        layout.addWidget(self.search_input)

        # Memory List
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet(
            "QListWidget { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 6px; }"
            "QListWidget::item { padding: 12px; border-bottom: 1px solid #21262d; border-radius: 4px; color: #e6edf3; }"
            "QListWidget::item:hover { background-color: #21262d; }"
            "QListWidget::item:selected { background-color: #1f3a5f; color: #58a6ff; }"
        )
        layout.addWidget(self.list_widget, stretch=1)

        # Bottom Actions
        actions = QHBoxLayout()
        self.btn_delete = QPushButton("Delete Selected Record")
        self.btn_delete.setProperty("class", "btn_secondary")
        self.btn_delete.clicked.connect(self._on_delete_selected)
        actions.addWidget(self.btn_delete)

        self.lbl_count = QLabel("0 records stored")
        self.lbl_count.setStyleSheet("color: #7d8590; font-size: 12px;")
        actions.addStretch()
        actions.addWidget(self.lbl_count)

        layout.addLayout(actions)

    def _connect_signals(self) -> None:
        self.controller.events.memory_domain_updated.connect(self._on_domain_updated)

    def _on_domain_updated(self, domain: str) -> None:
        if domain == "work":
            self.refresh_memories()

    def refresh_memories(self) -> None:
        query = self.search_input.text().strip()
        memories = self.controller.get_work_memories(search=query)
        self.list_widget.clear()

        for mem in memories:
            tags_txt = f"  [Tags: {mem['tags']}]" if mem['tags'] else ""
            item_text = (
                f"⚙ [{mem['project_name'].upper()}] {mem['title']}{tags_txt}\n"
                f"{mem['content']}\n"
                f"Saved: {mem['created_at']}"
            )
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, mem["id"])
            self.list_widget.addItem(item)

        self.lbl_count.setText(f"{len(memories)} work records stored")

    def _on_search_changed(self, text: str) -> None:
        self.refresh_memories()

    def _on_add_memory(self) -> None:
        dlg = AddWorkMemoryDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            data = dlg.get_data()
            self.controller.add_work_memory(
                project_name=data["project_name"],
                title=data["title"],
                content=data["content"],
                tags=data["tags"],
            )
            self.refresh_memories()

    def _on_delete_selected(self) -> None:
        item = self.list_widget.currentItem()
        if not item:
            return
        mem_id = str(item.data(Qt.ItemDataRole.UserRole))
        confirm = QMessageBox.question(
            self,
            "Delete Work Memory",
            "Are you sure you want to delete this work memory record?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.controller.delete_work_memory(mem_id)
            self.refresh_memories()

    def _on_clear_domain(self) -> None:
        confirm = QMessageBox.warning(
            self,
            "Clear Work Memory",
            "Are you sure you want to clear ALL Work Memories?\n\n"
            "Notice: Personal Memory will remain completely untouched.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.controller.clear_work_memory()
            self.refresh_memories()
