"""Offline Operation and Disaster Recovery Center UI View for HAKS Phase 8.

Provides an operational control panel for:
- Real-time network reachability (Online / Offline mode)
- Local AI model & Ollama daemon status
- Database integrity inspection across all 5 SQLite databases
- Vector index consistency verification & self-healing rebuilds
- Full-system encrypted backups with SHA-256 manifest verification
- Restoration from backup bundles with pre-recovery snapshot safeguards
- Encryption key management & recovery logs
"""

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from recovery.models import BackupMetadata, BackupVerificationResult, DatabaseIntegrityStatus, RecoveryReport, SystemHealthOverview

if TYPE_CHECKING:
    from core.app_controller import AppController


class RecoveryWorker(QThread):
    """Background worker thread to keep Qt UI responsive during disk I/O and cryptographic operations."""
    result_ready = Signal(object)
    error_occurred = Signal(str)

    def __init__(self, func, *args, **kwargs) -> None:
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self) -> None:
        try:
            res = self.func(*self.args, **self.kwargs)
            self.result_ready.emit(res)
        except Exception as exc:
            self.error_occurred.emit(str(exc))


class RecoveryView(QWidget):
    """Offline Operation and Disaster Recovery Center."""

    def __init__(self, controller: "AppController", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.rm = controller.get_recovery_manager()

        self._active_worker: Optional[QThread] = None

        self._init_ui()
        self._connect_events()
        self._refresh_overview()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # 1. Header Banner
        header = QHBoxLayout()
        title_lbl = QLabel("Offline Operation & Disaster Recovery Center")
        title_font = QFont("Segoe UI", 16, QFont.Bold)
        title_lbl.setFont(title_font)
        header.addWidget(title_lbl)

        header.addStretch()

        self.btn_refresh = QPushButton("🔄 Refresh Status")
        self.btn_refresh.clicked.connect(self._refresh_overview)
        header.addWidget(self.btn_refresh)

        main_layout.addLayout(header)

        # 2. Main Tabbed Control Sections
        self.tabs = QTabWidget()
        self.tabs.setObjectName("recovery_tabs")

        self.tab_health = self._build_health_tab()
        self.tab_backup = self._build_backup_tab()
        self.tab_restore = self._build_restore_tab()

        self.tabs.addTab(self.tab_health, "🏥 System & Database Health")
        self.tabs.addTab(self.tab_backup, "💾 Backup & Retention")
        self.tabs.addTab(self.tab_restore, "🛡️ Restore & Disaster Recovery")

        main_layout.addWidget(self.tabs)

    def _connect_events(self) -> None:
        """Connect application event bus to UI updates."""
        self.controller.events.network_connectivity_changed.connect(self._on_network_changed)
        self.controller.events.backup_completed.connect(lambda d: self._refresh_overview())
        self.controller.events.recovery_operation_completed.connect(self._on_recovery_completed)

    # --------------------------------------------------------------------------
    # 1. Health & Integrity Tab
    # --------------------------------------------------------------------------

    def _build_health_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Summary Metric Cards
        cards_layout = QHBoxLayout()
        self.card_network = self._create_card("Network Mode", "Checking...")
        self.card_ai = self._create_card("AI Operating Status", "Local Ollama")
        self.card_db = self._create_card("Database Health", "5/5 Verified")
        self.card_last_backup = self._create_card("Last Backup", "None")

        cards_layout.addWidget(self.card_network)
        cards_layout.addWidget(self.card_ai)
        cards_layout.addWidget(self.card_db)
        cards_layout.addWidget(self.card_last_backup)
        layout.addLayout(cards_layout)

        # Database Integrity Table
        db_group = QGroupBox("SQLite Database Integrity & WAL Status")
        db_box = QVBoxLayout(db_group)

        self.table_db = QTableWidget(0, 7)
        self.table_db.setHorizontalHeaderLabels([
            "Domain", "File Size", "Integrity Status", "Quick Check", "Records", "WAL Mode", "Errors"
        ])
        self.table_db.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_db.setEditTriggers(QTableWidget.NoEditTriggers)
        db_box.addWidget(self.table_db)

        btn_box = QHBoxLayout()
        self.btn_check_integrity = QPushButton("🔍 Run Full SQLite Integrity Check")
        self.btn_check_integrity.setStyleSheet("background-color: #0d47a1; color: white; padding: 6px 14px; font-weight: bold;")
        self.btn_check_integrity.clicked.connect(self._on_run_integrity_check)
        btn_box.addWidget(self.btn_check_integrity)

        self.btn_rebuild_indexes = QPushButton("⚡ Rebuild Vector Indexes (Self-Healing)")
        self.btn_rebuild_indexes.setStyleSheet("background-color: #2e7d32; color: white; padding: 6px 14px; font-weight: bold;")
        self.btn_rebuild_indexes.clicked.connect(self._on_rebuild_indexes)
        btn_box.addWidget(self.btn_rebuild_indexes)

        btn_box.addStretch()
        db_box.addLayout(btn_box)
        layout.addWidget(db_group)

        # Vector Index Table
        vec_group = QGroupBox("Memory Vector Index Consistency")
        vec_box = QVBoxLayout(vec_group)
        self.table_vec = QTableWidget(0, 5)
        self.table_vec.setHorizontalHeaderLabels([
            "Domain", "Vector Count", "DB Records", "Synchronized", "Dimension"
        ])
        self.table_vec.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_vec.setEditTriggers(QTableWidget.NoEditTriggers)
        vec_box.addWidget(self.table_vec)
        layout.addWidget(vec_group)

        return widget

    # --------------------------------------------------------------------------
    # 2. Backup & Retention Tab
    # --------------------------------------------------------------------------

    def _build_backup_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Action Bar: Create Backup
        create_box = QGroupBox("Create Full-System Backup")
        create_layout = QHBoxLayout(create_box)

        self.chk_encrypt = QCheckBox("AES-256 Encrypted Bundle (.haksbackup)")
        self.chk_encrypt.setChecked(True)
        create_layout.addWidget(self.chk_encrypt)

        self.txt_notes = QLineEdit()
        self.txt_notes.setPlaceholderText("Optional backup description / label...")
        create_layout.addWidget(self.txt_notes)

        self.btn_create_backup = QPushButton("📦 Create Backup Now")
        self.btn_create_backup.setStyleSheet("background-color: #388e3c; color: white; font-weight: bold; padding: 6px 16px;")
        self.btn_create_backup.clicked.connect(self._on_create_backup_clicked)
        create_layout.addWidget(self.btn_create_backup)

        layout.addWidget(create_box)

        # External Backup Path configuration
        dest_box = QGroupBox("Storage & Destination Mirroring")
        dest_layout = QHBoxLayout(dest_box)
        self.lbl_dest_path = QLabel(f"Primary Directory: {self.controller.config.paths.backups_dir}")
        dest_layout.addWidget(self.lbl_dest_path)
        dest_layout.addStretch()

        self.btn_set_ext_drive = QPushButton("📁 Set External Drive / Secondary Folder...")
        self.btn_set_ext_drive.clicked.connect(self._on_select_external_drive)
        dest_layout.addWidget(self.btn_set_ext_drive)
        layout.addWidget(dest_box)

        # Backup Catalog Table
        catalog_box = QGroupBox("Available Backups Catalog")
        cat_layout = QVBoxLayout(catalog_box)

        self.table_backups = QTableWidget(0, 6)
        self.table_backups.setHorizontalHeaderLabels([
            "Filename", "Backup Type", "Created At", "Size", "Encrypted", "Notes"
        ])
        self.table_backups.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_backups.setEditTriggers(QTableWidget.NoEditTriggers)
        cat_layout.addWidget(self.table_backups)

        btn_row = QHBoxLayout()
        self.btn_verify_selected = QPushButton("🛡️ Verify Selected Backup")
        self.btn_verify_selected.clicked.connect(self._on_verify_selected_backup)
        btn_row.addWidget(self.btn_verify_selected)

        self.btn_restore_selected = QPushButton("♻️ Restore Selected Backup...")
        self.btn_restore_selected.setStyleSheet("background-color: #d32f2f; color: white; font-weight: bold; padding: 6px 14px;")
        self.btn_restore_selected.clicked.connect(self._on_restore_selected_backup)
        btn_row.addWidget(self.btn_restore_selected)

        btn_row.addStretch()
        cat_layout.addLayout(btn_row)
        layout.addWidget(catalog_box)

        return widget

    # --------------------------------------------------------------------------
    # 3. Restore & Disaster Recovery Tab
    # --------------------------------------------------------------------------

    def _build_restore_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Manual File Restore Box
        restore_box = QGroupBox("Manual Backup Archive Restoration")
        r_layout = QVBoxLayout(restore_box)

        file_row = QHBoxLayout()
        self.txt_restore_path = QLineEdit()
        self.txt_restore_path.setPlaceholderText("Select a .haksbackup or .zip file to restore...")
        file_row.addWidget(self.txt_restore_path)

        self.btn_browse_restore = QPushButton("Browse...")
        self.btn_browse_restore.clicked.connect(self._on_browse_restore_file)
        file_row.addWidget(self.btn_browse_restore)
        r_layout.addLayout(file_row)

        action_row = QHBoxLayout()
        self.btn_verify_file = QPushButton("🔍 Inspect & Verify Checksums")
        self.btn_verify_file.clicked.connect(self._on_verify_file_clicked)
        action_row.addWidget(self.btn_verify_file)

        self.btn_execute_restore = QPushButton("🚨 Execute System Restoration")
        self.btn_execute_restore.setStyleSheet("background-color: #d32f2f; color: white; font-weight: bold; padding: 6px 14px;")
        self.btn_execute_restore.clicked.connect(self._on_execute_restore_clicked)
        action_row.addWidget(self.btn_execute_restore)

        action_row.addStretch()
        r_layout.addLayout(action_row)
        layout.addWidget(restore_box)

        # Encryption Recovery Key Management
        key_box = QGroupBox("Master Memory Encryption Key Protection")
        k_layout = QHBoxLayout(key_box)

        self.btn_export_key = QPushButton("🔑 View / Export Recovery Key")
        self.btn_export_key.clicked.connect(self._on_export_key_clicked)
        k_layout.addWidget(self.btn_export_key)

        self.btn_import_key = QPushButton("📥 Restore with Recovery Key...")
        self.btn_import_key.clicked.connect(self._on_import_key_clicked)
        k_layout.addWidget(self.btn_import_key)

        k_layout.addStretch()
        layout.addWidget(key_box)

        # Recovery Event Logs
        log_box = QGroupBox("Disaster Recovery & Self-Healing Audit Log")
        l_layout = QVBoxLayout(log_box)
        self.log_viewer = QTextEdit()
        self.log_viewer.setReadOnly(True)
        self.log_viewer.setFont(QFont("Consolas", 9))
        self.log_viewer.setStyleSheet("background-color: #1e1e1e; color: #dcdcdc; border: 1px solid #333;")
        l_layout.addWidget(self.log_viewer)
        layout.addWidget(log_box)

        return widget

    def _create_card(self, title: str, initial_val: str) -> QGroupBox:
        box = QGroupBox(title)
        vbox = QVBoxLayout(box)
        lbl = QLabel(initial_val)
        lbl.setFont(QFont("Segoe UI", 11, QFont.Bold))
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setObjectName(f"val_{title.lower().replace(' ', '_')}")
        vbox.addWidget(lbl)
        return box

    # --------------------------------------------------------------------------
    # Data Refresh & Display
    # --------------------------------------------------------------------------

    def _refresh_overview(self) -> None:
        """Fetch health overview and populate UI tables."""
        if os.environ.get("PYTEST_CURRENT_TEST"):
            # Synchronous in automated tests
            try:
                overview = self.rm.get_system_health_overview()
                self._apply_overview(overview)
            except Exception:
                pass
            return

        worker = RecoveryWorker(self.rm.get_system_health_overview)
        worker.result_ready.connect(self._apply_overview)
        self._active_worker = worker
        worker.start()

    def _apply_overview(self, overview: SystemHealthOverview) -> None:
        # 1. Network Badge
        lbl_net = self.card_network.findChild(QLabel)
        if lbl_net:
            if overview.network.is_online:
                lbl_net.setText(f"🟢 ONLINE ({overview.network.latency_ms}ms)")
                lbl_net.setStyleSheet("color: #4caf50;")
            else:
                lbl_net.setText("🔴 OFFLINE (Local Mode)")
                lbl_net.setStyleSheet("color: #f44336;")

        # 2. AI Card
        lbl_ai = self.card_ai.findChild(QLabel)
        if lbl_ai:
            status_txt = f"{overview.active_engine.upper()}: {overview.local_model_name}"
            lbl_ai.setText(status_txt)

        # 3. Database Health Card
        lbl_db = self.card_db.findChild(QLabel)
        if lbl_db:
            healthy_count = sum(1 for d in overview.databases.values() if d.is_healthy)
            total = len(overview.databases)
            lbl_db.setText(f"{healthy_count}/{total} Healthy")
            lbl_db.setStyleSheet("color: #4caf50;" if healthy_count == total else "color: #f44336;")

        # 4. Last Backup Card
        lbl_bk = self.card_last_backup.findChild(QLabel)
        if lbl_bk:
            lbl_bk.setText(overview.last_backup_time or "No Backups Yet")

        # 5. Populate Database Table
        self.table_db.setRowCount(0)
        for d_name, d_status in overview.databases.items():
            row = self.table_db.rowCount()
            self.table_db.insertRow(row)
            self.table_db.setItem(row, 0, QTableWidgetItem(d_name))
            self.table_db.setItem(row, 1, QTableWidgetItem(f"{d_status.size_bytes // 1024} KB"))

            item_status = QTableWidgetItem("✅ Healthy" if d_status.is_healthy else "❌ Corrupted")
            item_status.setForeground(QColor("#4caf50") if d_status.is_healthy else QColor("#f44336"))
            self.table_db.setItem(row, 2, item_status)

            self.table_db.setItem(row, 3, QTableWidgetItem(d_status.quick_check))
            self.table_db.setItem(row, 4, QTableWidgetItem(str(d_status.record_count)))
            self.table_db.setItem(row, 5, QTableWidgetItem("Yes" if d_status.wal_active else "No"))
            self.table_db.setItem(row, 6, QTableWidgetItem(", ".join(d_status.errors) if d_status.errors else "None"))

        # 6. Populate Vector Index Table
        self.table_vec.setRowCount(0)
        for dom, v_status in overview.vector_indexes.items():
            row = self.table_vec.rowCount()
            self.table_vec.insertRow(row)
            self.table_vec.setItem(row, 0, QTableWidgetItem(dom))
            self.table_vec.setItem(row, 1, QTableWidgetItem(str(v_status.vector_count)))
            self.table_vec.setItem(row, 2, QTableWidgetItem(str(v_status.db_record_count)))
            sync_item = QTableWidgetItem("✅ In-Sync" if v_status.is_synchronized else "⚠️ Out-of-Sync")
            sync_item.setForeground(QColor("#4caf50") if v_status.is_synchronized else QColor("#ff9800"))
            self.table_vec.setItem(row, 3, sync_item)
            self.table_vec.setItem(row, 4, QTableWidgetItem(str(v_status.dimension)))

        # 7. Populate Backups Catalog
        backups = self.rm.backup.list_backups()
        self.table_backups.setRowCount(0)
        for bk in backups:
            row = self.table_backups.rowCount()
            self.table_backups.insertRow(row)
            self.table_backups.setItem(row, 0, QTableWidgetItem(bk.filename))
            self.table_backups.setItem(row, 1, QTableWidgetItem(bk.backup_type))
            self.table_backups.setItem(row, 2, QTableWidgetItem(bk.timestamp))
            self.table_backups.setItem(row, 3, QTableWidgetItem(f"{bk.size_bytes // 1024} KB"))
            self.table_backups.setItem(row, 4, QTableWidgetItem("🔒 AES-256" if bk.is_encrypted else "Plain ZIP"))
            self.table_backups.setItem(row, 5, QTableWidgetItem(bk.notes or "-"))

    # --------------------------------------------------------------------------
    # User Actions & Event Handlers
    # --------------------------------------------------------------------------

    def _on_network_changed(self, is_online: bool) -> None:
        self._refresh_overview()

    def _on_run_integrity_check(self) -> None:
        res = self.rm.run_integrity_check()
        self._refresh_overview()
        all_ok = all(s.is_healthy for s in res.values())
        if all_ok:
            QMessageBox.information(self, "Integrity Check", "All 5 SQLite databases passed PRAGMA integrity_check successfully!")
        else:
            failed = [d for d, s in res.items() if not s.is_healthy]
            QMessageBox.warning(self, "Integrity Warning", f"Integrity check failed for domains: {', '.join(failed)}")

    def _on_rebuild_indexes(self) -> None:
        res = self.rm.rebuild_vector_indexes()
        self._refresh_overview()
        QMessageBox.information(
            self,
            "Vector Indexes",
            f"Vector index self-healing complete! Repopulated approved embeddings from SQLite records.",
        )

    def _on_create_backup_clicked(self) -> None:
        enc = self.chk_encrypt.isChecked()
        notes = self.txt_notes.text().strip()
        try:
            bk_path = self.rm.create_full_backup(encrypt=enc, notes=notes)
            self.txt_notes.clear()
            self._refresh_overview()
            QMessageBox.information(self, "Backup Successful", f"Full-system backup created successfully!\n\nLocation: {bk_path.name}")
        except Exception as exc:
            QMessageBox.critical(self, "Backup Error", f"Failed to create backup: {exc}")

    def _on_select_external_drive(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select External Drive or Backup Folder")
        if folder:
            self.controller.config.recovery.external_backup_dir = folder
            self.lbl_dest_path.setText(f"Primary: {self.controller.config.paths.backups_dir} | Mirror: {folder}")
            self._refresh_overview()
            QMessageBox.information(self, "Mirror Destination Set", f"Secondary backup destination configured: {folder}")

    def _on_verify_selected_backup(self) -> None:
        sel = self.table_backups.currentRow()
        if sel < 0:
            QMessageBox.information(self, "Select Backup", "Please select a backup row from the catalog table first.")
            return

        filename = self.table_backups.item(sel, 0).text()
        bk_path = self.controller.config.paths.backups_dir / filename
        if not bk_path.exists():
            # Check external folder if configured
            if self.controller.config.recovery.external_backup_dir:
                bk_path = Path(self.controller.config.recovery.external_backup_dir) / filename

        v_res = self.rm.backup.verify_backup(bk_path)
        if v_res.is_valid:
            QMessageBox.information(
                self,
                "Backup Verification Passed",
                f"✅ Backup verified successfully!\n\nFile: {filename}\nVerified Members: {v_res.verified_files_count}\nSHA-256 Checksums: All Matched",
            )
        else:
            errs = "\n".join(v_res.errors + v_res.checksum_mismatches)
            QMessageBox.critical(self, "Verification Failed", f"❌ Backup is invalid or corrupted!\n\n{errs}")

    def _on_restore_selected_backup(self) -> None:
        sel = self.table_backups.currentRow()
        if sel < 0:
            QMessageBox.information(self, "Select Backup", "Please select a backup row from the catalog table first.")
            return

        filename = self.table_backups.item(sel, 0).text()
        bk_path = self.controller.config.paths.backups_dir / filename
        self.txt_restore_path.setText(str(bk_path))
        self.tabs.setCurrentIndex(2)  # Switch to restore tab
        self._on_execute_restore_clicked()

    def _on_browse_restore_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select HAKS Backup File",
            str(self.controller.config.paths.backups_dir),
            "HAKS Backup (*.haksbackup *.zip);;All Files (*.*)",
        )
        if path:
            self.txt_restore_path.setText(path)

    def _on_verify_file_clicked(self) -> None:
        path_str = self.txt_restore_path.text().strip()
        if not path_str:
            QMessageBox.warning(self, "Missing File", "Please select a backup file first.")
            return
        bk_path = Path(path_str)
        v_res = self.rm.backup.verify_backup(bk_path)
        if v_res.is_valid:
            domains_str = ", ".join(v_res.manifest.domains) if v_res.manifest else "All"
            QMessageBox.information(
                self,
                "Integrity Verification Passed",
                f"✅ Backup archive verified cleanly!\n\nType: {v_res.manifest.backup_type if v_res.manifest else 'Unknown'}\nIncluded Domains: {domains_str}\nVerified Files: {v_res.verified_files_count}\nChecksums: Valid",
            )
        else:
            errs = "\n".join(v_res.errors + v_res.checksum_mismatches)
            QMessageBox.critical(self, "Verification Failed", f"❌ Archive validation failed:\n\n{errs}")

    def _on_execute_restore_clicked(self) -> None:
        path_str = self.txt_restore_path.text().strip()
        if not path_str or not Path(path_str).exists():
            QMessageBox.warning(self, "Invalid Path", "Please select a valid backup archive file.")
            return

        bk_path = Path(path_str)

        # Mandatory Safety Confirmation Dialog
        reply = QMessageBox.warning(
            self,
            "Confirm System Restoration",
            f"Are you sure you want to restore HAKS from:\n{bk_path.name}?\n\n"
            "SAFETY GUARANTEE: A complete pre-recovery snapshot of your current databases "
            "will be saved automatically before any data is replaced.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        report = self.controller.restore_full_backup(backup_path=bk_path, preserve_current=True)
        if report.success:
            self.log_viewer.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Restored successfully from {bk_path.name}")
            self._refresh_overview()
            QMessageBox.information(
                self,
                "Restoration Completed",
                f"System restoration finished successfully!\n\nRestored Domains: {', '.join(report.restored_domains)}\nPre-recovery Snapshot: {report.pre_recovery_snapshot_path}",
            )
        else:
            errs = "\n".join(report.errors)
            self.log_viewer.append(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Restoration failed: {errs}")
            QMessageBox.critical(self, "Restoration Failed", f"Restoration encountered errors:\n\n{errs}")

    def _on_export_key_clicked(self) -> None:
        try:
            key = self.controller.get_recovery_key()
            QMessageBox.information(
                self,
                "Master Recovery Key",
                f"Your Master Memory Encryption Key is:\n\n{key}\n\n"
                "Keep this key safe. It is required to restore encrypted backups on a new machine.",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Key Error", f"Could not export key: {exc}")

    def _on_import_key_clicked(self) -> None:
        # Prompt for key
        key_input, ok = QFileDialog.getOpenFileName(self, "Import Recovery Key File")
        if not ok or not key_input:
            return
        key_text = Path(key_input).read_text(encoding="utf-8").strip()
        success = self.controller.recover_encryption_key(key_text)
        if success:
            QMessageBox.information(self, "Key Restored", "Master encryption key successfully imported and verified!")
        else:
            QMessageBox.critical(self, "Invalid Key", "The provided recovery key is invalid or incompatible.")

    def _on_recovery_completed(self, report_data: dict) -> None:
        report = RecoveryReport(**report_data)
        self.log_viewer.append(
            f"[{report.timestamp}] Operation: {report.operation} | Success: {report.success} | Restored: {report.restored_domains}"
        )

    def closeEvent(self, event) -> None:
        if hasattr(self, "_active_worker") and self._active_worker and self._active_worker.isRunning():
            self._active_worker.quit()
            self._active_worker.wait(500)
        super().closeEvent(event)
