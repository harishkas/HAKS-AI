"""Settings and Privacy View for HAKS.

Provides theme configuration, floating assistant behaviors,
storage inspection, and Windows security permission levels overview.
"""

from pathlib import Path
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
    QFileDialog,
    QMessageBox,
    QInputDialog,
    QLineEdit,
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

        # 4. Storage & Persistence Architecture Card
        diag_card = QFrame()
        diag_card.setProperty("class", "stat_card")
        dc_layout = QVBoxLayout(diag_card)
        dc_layout.setSpacing(10)

        dc_title = QLabel("HAKS STORAGE ARCHITECTURE (9 DOMAINS ACTIVE)")
        dc_title.setProperty("class", "card_title")
        dc_layout.addWidget(dc_title)

        cur_data_dir = self.controller.get_storage_path()
        loc_row = QHBoxLayout()
        self.lbl_storage_path = QLabel(f"<b>Persistent Location:</b> <code>{cur_data_dir}</code>")
        self.lbl_storage_path.setStyleSheet("color: #e6edf3; font-size: 12px;")
        loc_row.addWidget(self.lbl_storage_path)
        loc_row.addStretch()

        self.btn_change_location = QPushButton("📁 Change Location & Migrate")
        self.btn_change_location.setProperty("class", "btn_secondary")
        self.btn_change_location.clicked.connect(self._on_change_storage)
        loc_row.addWidget(self.btn_change_location)
        dc_layout.addLayout(loc_row)

        # Domain breakdown
        domains_title = QLabel("<b>Independent Storage Domains & Status:</b>")
        domains_title.setStyleSheet("color: #8b949e; font-size: 11px; margin-top: 4px;")
        dc_layout.addWidget(domains_title)

        stats = self.controller.db.get_stats()
        domains_info = [
            ("1. personal_memory", "personal_memory/personal_memory.db", stats.get("personal_memory", {})),
            ("2. work_memory", "work_memory/work_memory.db", stats.get("work_memory", {})),
            ("3. personal_vector", "personal_memory/vector_index/index.json", {"size_bytes": (cur_data_dir / "personal_memory" / "vector_index" / "index.json").stat().st_size if (cur_data_dir / "personal_memory" / "vector_index" / "index.json").exists() else 0, "tables": {"vectors": len(self.controller.personal_memory.vector_index)}}),
            ("4. work_vector", "work_memory/vector_index/index.json", {"size_bytes": (cur_data_dir / "work_memory" / "vector_index" / "index.json").stat().st_size if (cur_data_dir / "work_memory" / "vector_index" / "index.json").exists() else 0, "tables": {"vectors": len(self.controller.work_memory.vector_index)}}),
            ("5. conversations", "conversations/conversations.db", stats.get("conversations", {})),
            ("6. memory_proposals", "memory_proposals/memory_proposals.db", stats.get("memory_proposals", {})),
            ("7. settings", "settings/settings.db", stats.get("settings", {})),
            ("8. backups", "backups/", {"size_bytes": 0, "tables": {"directory": "Active"}}),
            ("9. logs", "logs/", {"size_bytes": 0, "tables": {"directory": "Active"}}),
        ]

        for d_num, d_path, d_data in domains_info:
            kb = d_data.get("size_bytes", 0) / 1024.0
            lbl = QLabel(f"• <b>{d_num}</b> — <code>{d_path}</code> ({kb:.1f} KB) | {d_data.get('tables', {})}")
            lbl.setStyleSheet("color: #8b949e; font-size: 11px;")
            dc_layout.addWidget(lbl)

        # Action Buttons Row
        actions_box = QHBoxLayout()
        self.btn_backup_pm = QPushButton("💾 Backup Personal")
        self.btn_backup_pm.setProperty("class", "btn_secondary")
        self.btn_backup_pm.clicked.connect(self._on_backup_personal)
        actions_box.addWidget(self.btn_backup_pm)

        self.btn_backup_wm = QPushButton("💾 Backup Work")
        self.btn_backup_wm.setProperty("class", "btn_secondary")
        self.btn_backup_wm.clicked.connect(self._on_backup_work)
        actions_box.addWidget(self.btn_backup_wm)

        self.btn_restore = QPushButton("📥 Restore Domain...")
        self.btn_restore.setProperty("class", "btn_secondary")
        self.btn_restore.clicked.connect(self._on_restore_domain)
        actions_box.addWidget(self.btn_restore)

        self.btn_sync_vectors = QPushButton("🔄 Sync Vector Indexes")
        self.btn_sync_vectors.setProperty("class", "btn_secondary")
        self.btn_sync_vectors.clicked.connect(self._on_sync_vectors)
        actions_box.addWidget(self.btn_sync_vectors)

        self.btn_recovery_key = QPushButton("🔑 Recovery Key")
        self.btn_recovery_key.setProperty("class", "btn_secondary")
        self.btn_recovery_key.clicked.connect(self._on_view_recovery_key)
        actions_box.addWidget(self.btn_recovery_key)

        actions_box.addStretch()
        dc_layout.addLayout(actions_box)

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

    def _on_change_storage(self) -> None:
        """Prompt user for new storage directory and safely migrate data."""
        chosen = QFileDialog.getExistingDirectory(self, "Select New HAKS Data Directory")
        if not chosen:
            return

        target_path = Path(chosen)
        conflicts = self.controller.migration_service.check_target_conflicts(target_path)
        overwrite_approved = False
        if conflicts:
            reply = QMessageBox.question(
                self,
                "Existing Data Detected",
                f"The destination directory already contains {len(conflicts)} existing files.\n\n"
                "Do you want to overwrite them with your current HAKS data?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            overwrite_approved = True

        success, msg = self.controller.migrate_storage(target_path, overwrite_approved=overwrite_approved)
        if success:
            QMessageBox.information(self, "Migration Succeeded", f"Storage successfully migrated to:\n{target_path}")
            self.lbl_storage_path.setText(f"<b>Persistent Location:</b> <code>{target_path}</code>")
        else:
            QMessageBox.warning(self, "Migration Failed", f"Could not migrate data:\n{msg}")

    def _on_backup_personal(self) -> None:
        """Backup Personal Memory domain."""
        try:
            path = self.controller.backup_domain("personal_memory", encrypt=True)
            QMessageBox.information(self, "Backup Complete", f"Personal Memory backed up to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Backup Error", f"Failed to back up Personal Memory:\n{exc}")

    def _on_backup_work(self) -> None:
        """Backup Work Memory domain."""
        try:
            path = self.controller.backup_domain("work_memory", encrypt=True)
            QMessageBox.information(self, "Backup Complete", f"Work Memory backed up to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Backup Error", f"Failed to back up Work Memory:\n{exc}")

    def _on_restore_domain(self) -> None:
        """Restore a domain from a backup archive."""
        domain_choice, ok = QInputDialog.getItem(
            self,
            "Select Domain to Restore",
            "Target domain:",
            ["personal_memory", "work_memory", "conversations"],
            0,
            False,
        )
        if not ok or not domain_choice:
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select Backup Archive for {domain_choice}",
            str(self.controller.config.paths.backups_dir),
            "HAKS Backups (*.haksbackup *.zip)",
        )
        if not file_path:
            return

        try:
            self.controller.restore_domain(Path(file_path), domain_choice)
            QMessageBox.information(self, "Restore Succeeded", f"Domain '{domain_choice}' restored successfully from:\n{file_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Restore Failed", f"Failed to restore domain '{domain_choice}':\n{exc}")

    def _on_sync_vectors(self) -> None:
        """Trigger self-healing vector index synchronization."""
        try:
            res = self.controller.sync_vector_indexes()
            msg = (
                f"Personal Memory: {res['personal']['total']} vectors (added {res['personal']['added']}, removed {res['personal']['removed']})\n"
                f"Work Memory: {res['work']['total']} vectors (added {res['work']['added']}, removed {res['work']['removed']})"
            )
            QMessageBox.information(self, "Vector Indexes Synced", msg)
        except Exception as exc:
            QMessageBox.critical(self, "Sync Error", f"Failed to sync vector indexes:\n{exc}")

    def _on_view_recovery_key(self) -> None:
        """Display recovery key dialog with option to copy."""
        try:
            rec_key = self.controller.get_recovery_key()
            dialog = QMessageBox(self)
            dialog.setWindowTitle("Memory Encryption Recovery Key")
            dialog.setText(
                "Keep this recovery key safe. You can use it to restore access to your encrypted "
                "memory databases and backups if your Windows credentials are reset.\n\n"
                f"<b>Recovery Key:</b><br><code>{rec_key}</code>"
            )
            dialog.exec()
        except Exception as exc:
            QMessageBox.warning(self, "Recovery Key Error", f"Could not retrieve recovery key:\n{exc}")
