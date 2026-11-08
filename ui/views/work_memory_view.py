"""Work Memory View for HAKS.

Manages the independent work_memory.db storage domain:
viewing, semantic searching, project organization, adding, editing, deleting,
clearing, and reviewing AI-detected technical memory proposals before permanent approval.
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

logger = get_logger("ui.views.work_memory_view")


class EditWorkMemoryDialog(QDialog):
    """Dialog to view, edit, or create a work memory record."""

    def __init__(self, memory_data: Optional[dict] = None, parent=None):
        super().__init__(parent)
        self.is_edit = memory_data is not None
        self.setWindowTitle("Edit Work Memory" if self.is_edit else "Add Work Memory")
        self.setFixedSize(520, 460)
        self.memory_data = memory_data or {}
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.project_input = QLineEdit(self.memory_data.get("project_name", "HAKS"))
        self.project_input.setPlaceholderText("e.g., HAKS, FIRMWARE, BACKEND")
        form.addRow("Project Name:", self.project_input)

        self.title_input = QLineEdit(self.memory_data.get("title", ""))
        self.title_input.setPlaceholderText("e.g., Modular Monolith Architecture / SQLite Pragmas")
        form.addRow("Title:", self.title_input)

        self.tags_input = QLineEdit(self.memory_data.get("tags", ""))
        self.tags_input.setPlaceholderText("e.g., architecture, sqlite, python, offline")
        form.addRow("Tags:", self.tags_input)

        self.importance_spin = QSpinBox()
        self.importance_spin.setRange(1, 5)
        self.importance_spin.setValue(int(self.memory_data.get("importance", 3)))
        form.addRow("Importance (1-5):", self.importance_spin)

        self.content_input = QTextEdit(self.memory_data.get("content", ""))
        self.content_input.setPlaceholderText("Enter the technical decision, algorithm, or codebase context...")
        form.addRow("Content:", self.content_input)

        # Show source conversation if editing
        if self.memory_data.get("source_conversation_id"):
            src_lbl = QLabel(f"Origin Session: {self.memory_data['source_conversation_id']}")
            src_lbl.setStyleSheet("color: #8b949e; font-size: 11px;")
            form.addRow("Source:", src_lbl)

        layout.addLayout(form)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_cancel)

        self.btn_save = QPushButton("Save Work Record")
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
            "importance": self.importance_spin.value(),
            "content": self.content_input.toPlainText().strip(),
        }


class WorkMemoryView(QWidget):
    """Work Memory domain management interface with proposal review and backup tools."""

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
        title = QLabel("Work Memory")
        title.setStyleSheet("font-size: 18px; font-weight: 700; color: #f0f6fc;")
        subtitle = QLabel("Independent domain: technical decisions, algorithms & projects")
        subtitle.setStyleSheet("color: #8b949e; font-size: 12px; margin-left: 8px;")
        header.addWidget(title)
        header.addWidget(subtitle)
        header.addStretch()

        self.btn_add = QPushButton("+ Add Work Record")
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
            "background-color: #1a2332; border: 1px solid #388bfd; border-radius: 8px; padding: 10px;"
        )
        self.proposal_layout = QVBoxLayout(self.proposal_frame)
        self.proposal_layout.setSpacing(6)

        prop_header = QLabel("💡 PENDING AI WORK MEMORY PROPOSALS (Approval Required)")
        prop_header.setStyleSheet("font-weight: 700; color: #58a6ff; font-size: 12px;")
        self.proposal_layout.addWidget(prop_header)

        self.proposals_list = QListWidget()
        self.proposals_list.setMaximumHeight(90)
        self.proposals_list.setStyleSheet(
            "background-color: #0d1117; border: 1px solid #30363d; border-radius: 6px; color: #e6edf3; font-size: 12px;"
        )
        self.proposal_layout.addWidget(self.proposals_list)

        prop_btn_row = QHBoxLayout()
        self.btn_approve_prop = QPushButton("✔ Approve Work Proposal")
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

        # 3. Filter Bar (Search + Project Selector)
        filter_row = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Semantic or keyword search (e.g. modular monolith, sqlite, vad)...")
        self.search_input.textChanged.connect(self.refresh_memories)
        filter_row.addWidget(self.search_input, stretch=1)

        self.project_filter = QComboBox()
        self.project_filter.addItem("All Projects")
        self.project_filter.currentIndexChanged.connect(self.refresh_memories)
        filter_row.addWidget(self.project_filter)

        layout.addLayout(filter_row)

        # 4. Work Records List
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet(
            "QListWidget { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 6px; }"
            "QListWidget::item { padding: 12px; border-bottom: 1px solid #21262d; border-radius: 4px; color: #e6edf3; }"
            "QListWidget::item:hover { background-color: #21262d; }"
            "QListWidget::item:selected { background-color: #1f3a5f; color: #58a6ff; }"
        )
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.list_widget, stretch=1)

        # 5. Bottom Actions
        actions = QHBoxLayout()
        self.btn_edit = QPushButton("Edit Selected")
        self.btn_edit.setProperty("class", "btn_secondary")
        self.btn_edit.clicked.connect(self._on_edit_selected)
        actions.addWidget(self.btn_edit)

        self.btn_delete = QPushButton("Delete Selected")
        self.btn_delete.setProperty("class", "btn_danger")
        self.btn_delete.clicked.connect(self._on_delete_selected)
        actions.addWidget(self.btn_delete)

        self.lbl_count = QLabel("0 work records stored")
        self.lbl_count.setStyleSheet("color: #7d8590; font-size: 12px;")
        actions.addStretch()
        actions.addWidget(self.lbl_count)

        layout.addLayout(actions)

    def _connect_signals(self) -> None:
        self.controller.events.memory_domain_updated.connect(self._on_domain_updated)
        self.controller.events.memory_proposal_created.connect(self._on_proposal_created)

    def _on_domain_updated(self, domain: str) -> None:
        if domain in ("work", "proposals"):
            self.refresh_memories()

    def _on_proposal_created(self, proposal_dict: dict) -> None:
        self.refresh_memories()

    def refresh_memories(self) -> None:
        """Reload work memories and update proposal banner and project filter."""
        # Update project list in filter dropdown
        current_sel = self.project_filter.currentText()
        projects = self.controller.get_work_projects()
        self.project_filter.blockSignals(True)
        self.project_filter.clear()
        self.project_filter.addItem("All Projects")
        for p in projects:
            self.project_filter.addItem(p)
        idx = self.project_filter.findText(current_sel)
        if idx >= 0:
            self.project_filter.setCurrentIndex(idx)
        self.project_filter.blockSignals(False)

        query = self.search_input.text().strip()
        proj = self.project_filter.currentText()
        proj_param = None if proj == "All Projects" else proj

        memories = self.controller.get_work_memories(project=proj_param or "", search=query)
        self.list_widget.clear()

        for mem in memories:
            tags_txt = f"  [Tags: {mem['tags']}]" if mem['tags'] else ""
            stars = "★" * mem["importance"] + "☆" * (5 - mem["importance"])
            item_text = (
                f"{stars} [{mem['project_name'].upper()}] {mem['title']}{tags_txt}\n"
                f"{mem['content']}\n"
                f"Saved: {mem['created_at']}"
            )
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, mem["id"])
            self.list_widget.addItem(item)

        self.lbl_count.setText(f"{len(memories)} work records stored (Vector Indexed)")

        # Refresh Pending Work Proposals
        self._refresh_proposals()

    def _refresh_proposals(self) -> None:
        """Check for pending work memory proposals."""
        proposals = self.controller.get_pending_proposals()
        work_props = [p for p in proposals if p.get("domain") == "work"]

        self.proposals_list.clear()
        if work_props:
            self.proposal_frame.setVisible(True)
            for p in work_props:
                item_text = f"[{p.get('suggested_project', 'HAKS')}] {p.get('proposal_text')}"
                item = QListWidgetItem(item_text)
                item.setData(Qt.ItemDataRole.UserRole, p["id"])
                self.proposals_list.addItem(item)
            self.proposals_list.setCurrentRow(0)
        else:
            self.proposal_frame.setVisible(False)

    def _on_approve_proposal(self) -> None:
        """Approve selected AI work memory proposal."""
        item = self.proposals_list.currentItem()
        if not item:
            return
        prop_id = str(item.data(Qt.ItemDataRole.UserRole))
        success, mem_id = self.controller.approve_proposal(prop_id)
        if success:
            QMessageBox.information(self, "Proposal Approved", f"Saved to Work Memory! (ID: {mem_id})")
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
        dlg = EditWorkMemoryDialog(parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            data = dlg.get_data()
            self.controller.add_work_memory(
                project_name=data["project_name"],
                title=data["title"],
                content=data["content"],
                tags=data["tags"],
                importance=data["importance"],
            )
            self.refresh_memories()

    def _on_edit_selected(self) -> None:
        item = self.list_widget.currentItem()
        if not item:
            return
        mem_id = str(item.data(Qt.ItemDataRole.UserRole))
        raw_item = self.controller.work_memory.get_memory(mem_id)
        if not raw_item:
            return

        dlg = EditWorkMemoryDialog(
            memory_data={
                "project_name": raw_item.project_name,
                "title": raw_item.title,
                "content": raw_item.content,
                "tags": ", ".join(raw_item.tags),
                "importance": raw_item.importance,
                "source_conversation_id": raw_item.source_conversation_id,
            },
            parent=self,
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            data = dlg.get_data()
            self.controller.update_work_memory(
                memory_id=mem_id,
                title=data["title"],
                content=data["content"],
                project_name=data["project_name"],
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
            "Delete Work Memory",
            "Are you sure you want to delete this work memory record and its vector index?",
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

    def _on_export_backup(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Work Memory Backup",
            str(self.controller.config.paths.backups_dir / "work_memory_backup.json"),
            "Encrypted Backup (*.enc);;JSON Backup (*.json)",
        )
        if file_path:
            is_encrypted = file_path.endswith(".enc")
            self.controller.export_memory_domain("work", Path(file_path), encrypt=is_encrypted)
            QMessageBox.information(
                self, "Backup Exported", f"Successfully exported Work Memory to:\n{file_path}"
            )

    def _on_restore_backup(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Restore Work Memory Backup",
            str(self.controller.config.paths.backups_dir),
            "Backup Files (*.json *.enc)",
        )
        if file_path:
            try:
                is_encrypted = file_path.endswith(".enc")
                count = self.controller.restore_memory_domain("work", Path(file_path), is_encrypted=is_encrypted)
                QMessageBox.information(
                    self, "Backup Restored", f"Restored {count} records into Work Memory."
                )
                self.refresh_memories()
            except Exception as exc:
                QMessageBox.critical(self, "Restore Error", f"Failed to restore backup:\n{exc}")
