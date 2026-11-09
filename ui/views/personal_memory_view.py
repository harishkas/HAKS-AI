"""Personal Memory View for HAKS.

Manages the independent personal_memory.db storage domain:
viewing, semantic searching, adding, editing, deleting, clearing,
and reviewing AI-detected memory proposals before permanent approval.
"""

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.app_controller import AppController, get_app_controller
from core.logger import get_logger

logger = get_logger("ui.views.personal_memory_view")


class EditPersonalMemoryDialog(QDialog):
    """Dialog to view, edit, or create a personal memory record."""

    def __init__(self, memory_data: Optional[dict] = None, parent=None):
        super().__init__(parent)
        self.is_edit = memory_data is not None
        self.setWindowTitle("Edit Personal Memory" if self.is_edit else "Add Personal Memory")
        self.setFixedSize(500, 440)
        self.memory_data = memory_data or {}
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.title_input = QLineEdit(self.memory_data.get("title", ""))
        self.title_input.setPlaceholderText("e.g., Preferred Language / Personal Goal")
        form.addRow("Title:", self.title_input)

        self.category_combo = QComboBox()
        self.category_combo.addItems(["preference", "experience", "goal", "history", "general"])
        cat = self.memory_data.get("category", "general")
        idx = self.category_combo.findText(cat)
        if idx >= 0:
            self.category_combo.setCurrentIndex(idx)
        form.addRow("Category:", self.category_combo)

        self.tags_input = QLineEdit(self.memory_data.get("tags", ""))
        self.tags_input.setPlaceholderText("e.g., tamil, coffee, daily")
        form.addRow("Tags:", self.tags_input)

        self.importance_spin = QSpinBox()
        self.importance_spin.setRange(1, 5)
        self.importance_spin.setValue(int(self.memory_data.get("importance", 2)))
        form.addRow("Importance (1-5):", self.importance_spin)

        self.content_input = QTextEdit(self.memory_data.get("content", ""))
        self.content_input.setPlaceholderText("Enter the personal memory details...")
        form.addRow("Content:", self.content_input)

        # Show source conversation if editing
        if self.memory_data.get("source_conversation_id"):
            src_lbl = QLabel(f"Origin Session: {self.memory_data['source_conversation_id']}")
            src_lbl.setStyleSheet("color: #6b7280; font-size: 11px;")
            form.addRow("Source:", src_lbl)

        layout.addLayout(form)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_cancel)

        self.btn_save = QPushButton("Save Memory")
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
            "title": self.title_input.text().strip(),
            "category": self.category_combo.currentText(),
            "tags": self.tags_input.text().strip(),
            "importance": self.importance_spin.value(),
            "content": self.content_input.toPlainText().strip(),
        }


class PersonalMemoryView(QWidget):
    """Personal Memory domain management interface with proposal review and backup tools."""

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

        # 1. Header
        header = QHBoxLayout()
        title = QLabel("Personal Memory")
        title.setStyleSheet("font-size: 18px; font-weight: 700; color: #111827;")
        subtitle = QLabel("Independent domain: personal preferences, goals & experiences")
        subtitle.setStyleSheet("color: #4b5563; font-size: 12px; margin-left: 8px;")
        header.addWidget(title)
        header.addWidget(subtitle)
        header.addStretch()

        self.btn_add = QPushButton("+ Add Memory")
        self.btn_add.setProperty("class", "btn_primary")
        self.btn_add.clicked.connect(self._on_add_memory)
        header.addWidget(self.btn_add)

        self.btn_export = QPushButton("Export Backup")
        self.btn_export.setProperty("class", "btn_secondary")
        self.btn_export.clicked.connect(self._on_export_backup)
        header.addWidget(self.btn_export)

        self.btn_restore = QPushButton("Restore Backup")
        self.btn_restore.setProperty("class", "btn_secondary")
        self.btn_restore.clicked.connect(self._on_restore_backup)
        header.addWidget(self.btn_restore)

        self.btn_clear = QPushButton("Clear Domain")
        self.btn_clear.setProperty("class", "btn_danger")
        self.btn_clear.clicked.connect(self._on_clear_domain)
        header.addWidget(self.btn_clear)

        layout.addLayout(header)

        # 2. Memory Proposal Review Banner (AI suggests; user approves)
        self.proposal_frame = QFrame()
        self.proposal_frame.setProperty("class", "stat_card")
        self.proposal_frame.setStyleSheet(
            "background-color: #eff6ff; border: 1px solid #93c5fd; border-radius: 8px; padding: 10px;"
        )
        self.proposal_layout = QVBoxLayout(self.proposal_frame)
        self.proposal_layout.setSpacing(6)

        prop_header = QLabel("💡 PENDING AI MEMORY PROPOSALS (Approval Required)")
        prop_header.setStyleSheet("font-weight: 700; color: #1e40af; font-size: 12px;")
        self.proposal_layout.addWidget(prop_header)

        self.proposals_list = QListWidget()
        self.proposals_list.setMaximumHeight(90)
        self.proposal_layout.addWidget(self.proposals_list)

        prop_btn_row = QHBoxLayout()
        self.btn_approve_prop = QPushButton("✔ Approve Proposal")
        self.btn_approve_prop.setProperty("class", "btn_primary")
        self.btn_approve_prop.clicked.connect(self._on_approve_proposal)
        prop_btn_row.addWidget(self.btn_approve_prop)

        self.btn_reject_prop = QPushButton("✖ Reject Proposal")
        self.btn_reject_prop.setProperty("class", "btn_danger")
        self.btn_reject_prop.clicked.connect(self._on_reject_proposal)
        prop_btn_row.addWidget(self.btn_reject_prop)
        prop_btn_row.addStretch()

        self.proposal_layout.addLayout(prop_btn_row)
        layout.addWidget(self.proposal_frame)
        self.proposal_frame.setVisible(False)

        # 3. Filter Bar (Search + Category Filter)
        filter_row = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Semantic or keyword search (e.g. coffee, goals, tamil)...")
        self.search_input.textChanged.connect(self.refresh_memories)
        filter_row.addWidget(self.search_input, stretch=1)

        self.category_filter = QComboBox()
        self.category_filter.addItems(["All Categories", "preference", "experience", "goal", "history", "general"])
        self.category_filter.currentIndexChanged.connect(self.refresh_memories)
        filter_row.addWidget(self.category_filter)

        layout.addLayout(filter_row)

        # 4. Memory Items List
        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.list_widget, stretch=1)

        # 5. Bottom Status / Actions
        actions = QHBoxLayout()
        self.btn_edit = QPushButton("Edit Selected")
        self.btn_edit.setProperty("class", "btn_secondary")
        self.btn_edit.clicked.connect(self._on_edit_selected)
        actions.addWidget(self.btn_edit)

        self.btn_delete = QPushButton("Delete Selected")
        self.btn_delete.setProperty("class", "btn_danger")
        self.btn_delete.clicked.connect(self._on_delete_selected)
        actions.addWidget(self.btn_delete)

        self.lbl_count = QLabel("0 memories stored")
        self.lbl_count.setStyleSheet("color: #6b7280; font-size: 12px;")
        actions.addStretch()
        actions.addWidget(self.lbl_count)

        layout.addLayout(actions)

    def _connect_signals(self) -> None:
        self.controller.events.memory_domain_updated.connect(self._on_domain_updated)
        self.controller.events.memory_proposal_created.connect(self._on_proposal_created)

    def _on_domain_updated(self, domain: str) -> None:
        if domain in ("personal", "proposals"):
            self.refresh_memories()

    def _on_proposal_created(self, proposal_dict: dict) -> None:
        self.refresh_memories()

    def refresh_memories(self) -> None:
        """Reload personal memories and update proposal banner."""
        query = self.search_input.text().strip()
        cat = self.category_filter.currentText()
        category_param = None if cat == "All Categories" else cat

        memories = self.controller.get_personal_memories(search=query, category=category_param)
        self.list_widget.clear()

        for mem in memories:
            tags_txt = f"  [Tags: {mem['tags']}]" if mem['tags'] else ""
            stars = "★" * mem["importance"] + "☆" * (5 - mem["importance"])
            item_text = (
                f"{stars} {mem['title']}  ({mem['category'].upper()}){tags_txt}\n"
                f"{mem['content']}\n"
                f"Saved: {mem['created_at']}"
            )
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, mem["id"])
            self.list_widget.addItem(item)

        self.lbl_count.setText(f"{len(memories)} personal memories stored (Vector Indexed)")

        # Refresh Pending Proposals
        self._refresh_proposals()

    def _refresh_proposals(self) -> None:
        """Check for pending personal memory proposals."""
        proposals = self.controller.get_pending_proposals()
        personal_props = [p for p in proposals if p.get("domain") == "personal"]

        self.proposals_list.clear()
        if personal_props:
            self.proposal_frame.setVisible(True)
            for p in personal_props:
                item_text = f"[{p.get('suggested_category', 'general').upper()}] {p.get('proposal_text')}"
                item = QListWidgetItem(item_text)
                item.setData(Qt.ItemDataRole.UserRole, p["id"])
                self.proposals_list.addItem(item)
            self.proposals_list.setCurrentRow(0)
        else:
            self.proposal_frame.setVisible(False)

    def _on_approve_proposal(self) -> None:
        """Approve selected AI memory proposal."""
        item = self.proposals_list.currentItem()
        if not item:
            return
        prop_id = str(item.data(Qt.ItemDataRole.UserRole))
        success, mem_id = self.controller.approve_proposal(prop_id)
        if success:
            QMessageBox.information(self, "Proposal Approved", f"Saved to Personal Memory! (ID: {mem_id})")
            self.refresh_memories()

    def _on_reject_proposal(self) -> None:
        """Reject and discard selected memory proposal."""
        item = self.proposals_list.currentItem()
        if not item:
            return
        prop_id = str(item.data(Qt.ItemDataRole.UserRole))
        self.controller.reject_proposal(prop_id)
        self.refresh_memories()

    def _on_add_memory(self) -> None:
        dlg = EditPersonalMemoryDialog(parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            data = dlg.get_data()
            self.controller.add_personal_memory(
                title=data["title"],
                content=data["content"],
                category=data["category"],
                tags=data["tags"],
                importance=data["importance"],
            )
            self.refresh_memories()

    def _on_edit_selected(self) -> None:
        item = self.list_widget.currentItem()
        if not item:
            return
        mem_id = str(item.data(Qt.ItemDataRole.UserRole))
        raw_item = self.controller.personal_memory.get_memory(mem_id)
        if not raw_item:
            return

        dlg = EditPersonalMemoryDialog(
            memory_data={
                "title": raw_item.title,
                "content": raw_item.content,
                "category": raw_item.category,
                "tags": ", ".join(raw_item.tags),
                "importance": raw_item.importance,
                "source_conversation_id": raw_item.source_conversation_id,
            },
            parent=self,
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            data = dlg.get_data()
            self.controller.update_personal_memory(
                memory_id=mem_id,
                title=data["title"],
                content=data["content"],
                category=data["category"],
                tags=data["tags"],
                importance=data["importance"],
            )
            self.refresh_memories()

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        self._on_edit_selected()

    def _on_delete_selected(self) -> None:
        item = self.list_widget.currentItem()
        if not item:
            return
        mem_id = str(item.data(Qt.ItemDataRole.UserRole))
        confirm = QMessageBox.question(
            self,
            "Delete Personal Memory",
            "Are you sure you want to delete this memory record and its vector index?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.controller.delete_personal_memory(mem_id)
            self.refresh_memories()

    def _on_clear_domain(self) -> None:
        confirm = QMessageBox.warning(
            self,
            "Clear Personal Memory",
            "Are you sure you want to clear ALL Personal Memories?\n\n"
            "Notice: Work Memory will remain completely untouched.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.controller.clear_personal_memory()
            self.refresh_memories()

    def _on_export_backup(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Personal Memory Backup",
            str(self.controller.config.paths.backups_dir / "personal_memory_backup.json"),
            "Encrypted Backup (*.enc);;JSON Backup (*.json)",
        )
        if file_path:
            is_encrypted = file_path.endswith(".enc")
            self.controller.export_memory_domain("personal", Path(file_path), encrypt=is_encrypted)
            QMessageBox.information(
                self, "Backup Exported", f"Successfully exported Personal Memory to:\n{file_path}"
            )

    def _on_restore_backup(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Restore Personal Memory Backup",
            str(self.controller.config.paths.backups_dir),
            "Backup Files (*.json *.enc)",
        )
        if file_path:
            try:
                is_encrypted = file_path.endswith(".enc")
                count = self.controller.restore_memory_domain("personal", Path(file_path), is_encrypted=is_encrypted)
                QMessageBox.information(
                    self, "Backup Restored", f"Restored {count} memory records into Personal Memory."
                )
                self.refresh_memories()
            except Exception as exc:
                QMessageBox.critical(self, "Restore Error", f"Failed to restore backup:\n{exc}")
