"""Phase 7 Windows Control Center View for HAKS.

Provides a unified desktop operational interface:
- System Dashboard (Hardware, CPU/RAM, Admin status, Mode)
- Interactive Command Center (Live streaming CMD & PowerShell terminals)
- Software & Download Manager (HTTP/HTTPS downloads, winget package browser, pip)
- Filesystem & Storage Analyzer (Drives, large files, file operations, ZIP)
- System & Process Manager (Process killer, Windows Services, Network diagnostics, Power)
- Security, Confirmation & Immutable Audit Trail (UAC test, Emergency Stop)
"""

import asyncio
from datetime import datetime
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QColor, QFont, QIcon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from windows_control.models import (
    ActionPlan,
    CommandResult,
    DownloadItem,
    DriveInfo,
    ExecutionMode,
    ExecutionStatus,
    InstalledAppInfo,
    PrivilegeLevel,
    ProcessInfo,
    RiskLevel,
    ServiceInfo,
    StorageReport,
)

if TYPE_CHECKING:
    from core.app_controller import AppController


class BackgroundWorker(QThread):
    """Generic background worker thread to prevent Qt UI freezes during I/O."""
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


class AsyncWorker(QThread):
    """Executes asyncio coroutines in a background Qt thread."""
    result_ready = Signal(object)
    error_occurred = Signal(str)

    def __init__(self, coro) -> None:
        super().__init__()
        self.coro = coro

    def run(self) -> None:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            res = loop.run_until_complete(self.coro)
            loop.close()
            self.result_ready.emit(res)
        except Exception as exc:
            self.error_occurred.emit(str(exc))


class WindowsControlView(QWidget):
    """Windows Computer Control Center UI."""

    def __init__(self, controller: "AppController", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.wc = controller.windows_control

        self._active_worker: Optional[QThread] = None

        self._init_ui()
        self._init_timers()
        self._refresh_dashboard()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # Header with Emergency Stop button
        header_layout = QHBoxLayout()
        title_lbl = QLabel("Windows Computer Control Center")
        title_font = QFont("Segoe UI", 16)
        title_font.setBold(True)
        title_lbl.setFont(title_font)
        header_layout.addWidget(title_lbl)

        header_layout.addStretch()

        self.btn_emergency_stop = QPushButton("🛑 EMERGENCY STOP")
        self.btn_emergency_stop.setStyleSheet(
            "background-color: #d32f2f; color: white; font-weight: bold; padding: 8px 16px; border-radius: 4px;"
        )
        self.btn_emergency_stop.clicked.connect(self._on_emergency_stop_clicked)
        header_layout.addWidget(self.btn_emergency_stop)

        main_layout.addLayout(header_layout)

        # Main Modular Sub-Tabs
        self.tabs = QTabWidget()
        self.tabs.setObjectName("control_tabs")

        self.tab_dashboard = self._build_dashboard_tab()
        self.tab_terminal = self._build_terminal_tab()
        self.tab_software = self._build_software_tab()
        self.tab_filesystem = self._build_filesystem_tab()
        self.tab_system = self._build_system_tab()
        self.tab_audit = self._build_audit_tab()

        self.tabs.addTab(self.tab_dashboard, "📊 Dashboard")
        self.tabs.addTab(self.tab_terminal, "💻 Command Center")
        self.tabs.addTab(self.tab_software, "📦 Software & Downloads")
        self.tabs.addTab(self.tab_filesystem, "📁 File Manager")
        self.tabs.addTab(self.tab_system, "⚙️ System & Network")
        self.tabs.addTab(self.tab_audit, "🛡️ Security & Audit")

        main_layout.addWidget(self.tabs)

    def _init_timers(self) -> None:
        """Periodic timer updating CPU/RAM usage and dashboard metrics."""
        self.metrics_timer = QTimer(self)
        self.metrics_timer.setInterval(3000)
        self.metrics_timer.timeout.connect(self._refresh_quick_metrics)
        if not os.environ.get("PYTEST_CURRENT_TEST"):
            self.metrics_timer.start()

    # --------------------------------------------------------------------------
    # 1. Dashboard Tab
    # --------------------------------------------------------------------------

    def _build_dashboard_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Top Info Cards (User, OS, Admin, Mode)
        cards_layout = QHBoxLayout()

        self.card_user = self._create_metric_card("User & Host", "Loading...")
        self.card_admin = self._create_metric_card("Privileges", "Standard (asInvoker)")
        self.card_mode = self._create_metric_card("Execution Mode", "FULL CONTROL")
        self.card_uptime = self._create_metric_card("System Uptime", "0h 0m")

        cards_layout.addWidget(self.card_user)
        cards_layout.addWidget(self.card_admin)
        cards_layout.addWidget(self.card_mode)
        cards_layout.addWidget(self.card_uptime)
        layout.addLayout(cards_layout)

        # CPU & RAM Resource Gauges
        gauge_group = QGroupBox("System Resource Utilization")
        gauge_layout = QVBoxLayout(gauge_group)

        # CPU Bar
        cpu_box = QHBoxLayout()
        self.lbl_cpu = QLabel("CPU Usage: 0%")
        self.lbl_cpu.setFixedWidth(140)
        self.bar_cpu = QProgressBar()
        self.bar_cpu.setRange(0, 100)
        self.bar_cpu.setValue(0)
        cpu_box.addWidget(self.lbl_cpu)
        cpu_box.addWidget(self.bar_cpu)
        gauge_layout.addLayout(cpu_box)

        # RAM Bar
        ram_box = QHBoxLayout()
        self.lbl_ram = QLabel("RAM Usage: 0 GB / 8 GB (0%)")
        self.lbl_ram.setFixedWidth(240)
        self.bar_ram = QProgressBar()
        self.bar_ram.setRange(0, 100)
        self.bar_ram.setValue(0)
        ram_box.addWidget(self.lbl_ram)
        ram_box.addWidget(self.bar_ram)
        gauge_layout.addLayout(ram_box)

        layout.addWidget(gauge_group)

        # Drives Capacity Overview
        drives_group = QGroupBox("Storage Drives Overview")
        self.drives_layout = QHBoxLayout(drives_group)
        layout.addWidget(drives_group)

        # Quick Control Buttons
        actions_layout = QHBoxLayout()
        btn_refresh = QPushButton("🔄 Refresh Diagnostics")
        btn_refresh.clicked.connect(self._refresh_dashboard)
        actions_layout.addWidget(btn_refresh)

        btn_lock = QPushButton("🔒 Lock Workstation")
        btn_lock.clicked.connect(self._on_lock_clicked)
        actions_layout.addWidget(btn_lock)

        actions_layout.addStretch()
        layout.addLayout(actions_layout)

        layout.addStretch()
        return widget

    def _create_metric_card(self, title: str, initial_val: str) -> QGroupBox:
        box = QGroupBox(title)
        vbox = QVBoxLayout(box)
        lbl = QLabel(initial_val)
        lbl.setFont(QFont("Segoe UI", 11, QFont.Bold))
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setObjectName(f"val_{title.lower().replace(' ', '_')}")
        vbox.addWidget(lbl)
        return box

    def _refresh_dashboard(self) -> None:
        """Fetch real system metrics in background and update dashboard."""
        if os.environ.get("PYTEST_CURRENT_TEST"):
            return

        def _fetch():
            return self.wc.get_dashboard_metrics()

        worker = BackgroundWorker(_fetch)
        worker.result_ready.connect(self._apply_dashboard_metrics)
        self._active_worker = worker
        worker.start()

    def _apply_dashboard_metrics(self, data: dict) -> None:
        lbl_user = self.card_user.findChild(QLabel)
        if lbl_user:
            lbl_user.setText(f"{data['current_user']}@{data['computer_name']}")

        lbl_admin = self.card_admin.findChild(QLabel)
        if lbl_admin:
            is_adm = data["is_admin"]
            lbl_admin.setText("Elevated (Admin)" if is_adm else "Normal (UAC On-Demand)")
            lbl_admin.setStyleSheet("color: #4caf50;" if is_adm else "color: #2196f3;")

        lbl_mode = self.card_mode.findChild(QLabel)
        if lbl_mode:
            lbl_mode.setText(data["mode"].upper().replace("_", " "))

        lbl_uptime = self.card_uptime.findChild(QLabel)
        if lbl_uptime:
            sec = int(data["uptime_seconds"])
            hrs = sec // 3600
            mins = (sec % 3600) // 60
            lbl_uptime.setText(f"{hrs}h {mins}m ({sec}s)")

        # CPU & RAM
        cpu_pct = int(data["cpu_usage_percent"])
        self.bar_cpu.setValue(cpu_pct)
        self.lbl_cpu.setText(f"CPU Usage: {cpu_pct}%")

        ram_pct = int(data.get("ram_percent", 0))
        self.bar_ram.setValue(ram_pct)
        ram_used = data.get("ram_used_gb", round(data.get("ram_total_gb", 8.0) - data.get("ram_available_gb", 0.0), 2))
        self.lbl_ram.setText(f"RAM Usage: {ram_used} GB / {data.get('ram_total_gb', 8.0)} GB ({ram_pct}%)")

        # Drives cards
        while self.drives_layout.count():
            item = self.drives_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for d in data.get("drives", []):
            d_box = QGroupBox(f"Drive {d['mountpoint']}")
            d_vbox = QVBoxLayout(d_box)
            d_lbl = QLabel(f"{d['free_gb']} GB free of {d['total_gb']} GB")
            d_bar = QProgressBar()
            d_bar.setValue(int(d["percent_used"]))
            d_vbox.addWidget(d_lbl)
            d_vbox.addWidget(d_bar)
            self.drives_layout.addWidget(d_box)

    def _refresh_quick_metrics(self) -> None:
        """Lightweight update for CPU/RAM bars every few seconds."""
        try:
            import psutil
            cpu = int(psutil.cpu_percent(interval=None))
            vmem = psutil.virtual_memory()
            self.bar_cpu.setValue(cpu)
            self.lbl_cpu.setText(f"CPU Usage: {cpu}%")
            self.bar_ram.setValue(int(vmem.percent))
            self.lbl_ram.setText(
                f"RAM Usage: {round(vmem.used / (1024**3), 2)} GB / {round(vmem.total / (1024**3), 2)} GB ({int(vmem.percent)}%)"
            )
        except Exception:
            pass

    # --------------------------------------------------------------------------
    # 2. Command Center (CMD & PowerShell)
    # --------------------------------------------------------------------------

    def _build_terminal_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Control Toolbar (Shell type, Working Directory, Clear)
        toolbar = QHBoxLayout()

        toolbar.addWidget(QLabel("Shell:"))
        self.combo_shell = QComboBox()
        self.combo_shell.addItems(["PowerShell", "CMD"])
        toolbar.addWidget(self.combo_shell)

        toolbar.addWidget(QLabel("Working Directory:"))
        self.txt_terminal_cwd = QLineEdit(os.getcwd())
        toolbar.addWidget(self.txt_terminal_cwd)

        btn_browse_cwd = QPushButton("Browse...")
        btn_browse_cwd.clicked.connect(self._on_browse_cwd_clicked)
        toolbar.addWidget(btn_browse_cwd)

        self.btn_clear_terminal = QPushButton("Clear Output")
        self.btn_clear_terminal.clicked.connect(lambda: self.terminal_output.clear())
        toolbar.addWidget(self.btn_clear_terminal)

        self.btn_cancel_cmd = QPushButton("Stop Execution")
        self.btn_cancel_cmd.setEnabled(False)
        self.btn_cancel_cmd.clicked.connect(self._on_cancel_terminal_cmd)
        toolbar.addWidget(self.btn_cancel_cmd)

        layout.addLayout(toolbar)

        # Monospace Terminal Output Box
        self.terminal_output = QTextEdit()
        self.terminal_output.setReadOnly(True)
        self.terminal_output.setFont(QFont("Consolas", 10))
        self.terminal_output.setStyleSheet(
            "background-color: #0c0c0c; color: #cccccc; border: 1px solid #333333; padding: 8px;"
        )
        layout.addWidget(self.terminal_output)

        # Command Input Bar
        input_bar = QHBoxLayout()
        self.txt_command_input = QLineEdit()
        self.txt_command_input.setPlaceholderText("Type a command to execute (e.g., Get-Process, dir, pip list)...")
        self.txt_command_input.returnPressed.connect(self._on_run_terminal_command)
        input_bar.addWidget(self.txt_command_input)

        self.btn_run_command = QPushButton("▶ Run")
        self.btn_run_command.clicked.connect(self._on_run_terminal_command)
        input_bar.addWidget(self.btn_run_command)

        layout.addLayout(input_bar)
        return widget

    def _on_browse_cwd_clicked(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Working Directory", self.txt_terminal_cwd.text())
        if folder:
            self.txt_terminal_cwd.setText(folder)

    def _on_run_terminal_command(self) -> None:
        cmd_text = self.txt_command_input.text().strip()
        if not cmd_text:
            return

        shell = self.combo_shell.currentText()
        cwd = self.txt_terminal_cwd.text().strip()

        # Print command banner
        ts = datetime.now().strftime("%H:%M:%S")
        self.terminal_output.append(f"\n<span style='color: #4fc3f7;'>[{ts}] [{shell}] &gt; {cmd_text}</span>")
        self.txt_command_input.clear()
        self.btn_run_command.setEnabled(False)
        self.btn_cancel_cmd.setEnabled(True)

        async def _run():
            plan = ActionPlan(
                objective=f"Terminal {shell} command",
                command=cmd_text,
                working_dir=cwd,
                source="terminal",
            )
            if shell == "CMD":
                return await self.wc.cmd_service.execute(
                    command=cmd_text,
                    working_dir=cwd,
                    output_callback=self._append_terminal_line,
                    error_callback=self._append_terminal_error,
                )
            else:
                return await self.wc.powershell_service.execute(
                    command=cmd_text,
                    working_dir=cwd,
                    output_callback=self._append_terminal_line,
                    error_callback=self._append_terminal_error,
                )

        worker = AsyncWorker(_run())
        worker.result_ready.connect(self._on_terminal_command_done)
        self._active_worker = worker
        worker.start()

    def _append_terminal_line(self, line: str) -> None:
        self.terminal_output.append(f"<span style='color: #e0e0e0;'>{line.rstrip()}</span>")

    def _append_terminal_error(self, line: str) -> None:
        self.terminal_output.append(f"<span style='color: #ff5252;'>{line.rstrip()}</span>")

    def _on_terminal_command_done(self, result: CommandResult) -> None:
        self.btn_run_command.setEnabled(True)
        self.btn_cancel_cmd.setEnabled(False)
        status_color = "#66bb6a" if result.exit_code == 0 else "#ef5350"
        self.terminal_output.append(
            f"<span style='color: {status_color};'>[Exited with code {result.exit_code} in {round(result.duration_ms, 1)}ms]</span>\n"
        )

    def _on_cancel_terminal_cmd(self) -> None:
        self.wc.executor.emergency_stop()
        self.terminal_output.append("<span style='color: #ffa726;'>[Execution halted by user]</span>")
        self.btn_run_command.setEnabled(True)
        self.btn_cancel_cmd.setEnabled(False)

    # --------------------------------------------------------------------------
    # 3. Software & Downloads Tab
    # --------------------------------------------------------------------------

    def _build_software_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Download Manager Section
        dl_group = QGroupBox("HTTP/HTTPS File Download Manager")
        dl_layout = QVBoxLayout(dl_group)

        dl_input_row = QHBoxLayout()
        self.txt_dl_url = QLineEdit()
        self.txt_dl_url.setPlaceholderText("Enter file download URL (https://...)...")
        dl_input_row.addWidget(self.txt_dl_url)

        self.btn_start_dl = QPushButton("⬇ Download")
        self.btn_start_dl.clicked.connect(self._on_start_download_clicked)
        dl_input_row.addWidget(self.btn_start_dl)
        dl_layout.addLayout(dl_input_row)

        self.lbl_dl_status = QLabel("Ready for downloads.")
        dl_layout.addWidget(self.lbl_dl_status)

        self.dl_progress_bar = QProgressBar()
        self.dl_progress_bar.setValue(0)
        dl_layout.addWidget(self.dl_progress_bar)

        layout.addWidget(dl_group)

        # Package Manager Section (winget / pip)
        pkg_group = QGroupBox("Software Package Manager (winget & pip)")
        pkg_layout = QVBoxLayout(pkg_group)

        pkg_search_row = QHBoxLayout()
        self.txt_pkg_query = QLineEdit()
        self.txt_pkg_query.setPlaceholderText("Search packages (e.g. Git, VisualStudioCode, Python)...")
        self.txt_pkg_query.returnPressed.connect(self._on_search_packages_clicked)
        pkg_search_row.addWidget(self.txt_pkg_query)

        btn_search_pkg = QPushButton("🔍 Search Packages")
        btn_search_pkg.clicked.connect(self._on_search_packages_clicked)
        pkg_search_row.addWidget(btn_search_pkg)

        btn_list_installed = QPushButton("📋 Installed Software")
        btn_list_installed.clicked.connect(self._on_list_installed_software_clicked)
        pkg_search_row.addWidget(btn_list_installed)

        btn_list_pip = QPushButton("🐍 Python Packages")
        btn_list_pip.clicked.connect(self._on_list_pip_clicked)
        pkg_search_row.addWidget(btn_list_pip)

        pkg_layout.addLayout(pkg_search_row)

        # Software results table
        self.table_packages = QTableWidget(0, 4)
        self.table_packages.setHorizontalHeaderLabels(["Name", "ID / Package", "Version", "Action"])
        self.table_packages.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table_packages.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table_packages.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table_packages.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        pkg_layout.addWidget(self.table_packages)

        layout.addWidget(pkg_group)
        return widget

    def _on_start_download_clicked(self) -> None:
        url = self.txt_dl_url.text().strip()
        if not url:
            return

        self.lbl_dl_status.setText(f"Starting download: {url}...")
        self.dl_progress_bar.setValue(0)

        def _progress(item: DownloadItem):
            self.dl_progress_bar.setValue(int(item.progress_percent))
            self.lbl_dl_status.setText(
                f"Downloading: {item.filename} ({round(item.downloaded_bytes / (1024*1024), 2)} MB / "
                f"{round(item.total_bytes / (1024*1024), 2)} MB) - Status: {item.status.upper()}"
            )

        async def _dl():
            return await self.wc.download.start_download(
                url=url,
                progress_callback=_progress,
            )

        worker = AsyncWorker(_dl())
        worker.result_ready.connect(lambda res: self.lbl_dl_status.setText(f"Download complete: {res.destination}"))
        worker.error_occurred.connect(lambda err: self.lbl_dl_status.setText(f"Download error: {err}"))
        self._active_worker = worker
        worker.start()

    def _on_search_packages_clicked(self) -> None:
        query = self.txt_pkg_query.text().strip()
        if not query:
            return

        async def _search():
            return await self.wc.software.search_packages(query)

        worker = AsyncWorker(_search())
        worker.result_ready.connect(self._display_packages)
        self._active_worker = worker
        worker.start()

    def _display_packages(self, packages: list) -> None:
        self.table_packages.setRowCount(0)
        for pkg in packages:
            row = self.table_packages.rowCount()
            self.table_packages.insertRow(row)
            self.table_packages.setItem(row, 0, QTableWidgetItem(pkg.get("name", "")))
            self.table_packages.setItem(row, 1, QTableWidgetItem(pkg.get("id", "")))
            self.table_packages.setItem(row, 2, QTableWidgetItem(pkg.get("version", "")))

            btn_install = QPushButton("Install")
            pkg_id = pkg.get("id", "")
            btn_install.clicked.connect(lambda _, pid=pkg_id: self._on_install_package_clicked(pid))
            self.table_packages.setCellWidget(row, 3, btn_install)

    def _on_install_package_clicked(self, package_id: str) -> None:
        reply = QMessageBox.question(
            self,
            "Confirm Installation",
            f"Are you sure you want to install package '{package_id}' via winget?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            cmd = self.wc.software.build_winget_install_command(package_id)
            self.tabs.setCurrentWidget(self.tab_terminal)
            self.txt_command_input.setText(cmd)
            self._on_run_terminal_command()

    def _on_list_installed_software_clicked(self) -> None:
        def _fetch():
            return self.wc.application.list_installed_applications()

        worker = BackgroundWorker(_fetch)
        worker.result_ready.connect(self._display_installed_apps)
        self._active_worker = worker
        worker.start()

    def _display_installed_apps(self, apps: list[InstalledAppInfo]) -> None:
        self.table_packages.setRowCount(0)
        for app in apps:
            row = self.table_packages.rowCount()
            self.table_packages.insertRow(row)
            self.table_packages.setItem(row, 0, QTableWidgetItem(app.name))
            self.table_packages.setItem(row, 1, QTableWidgetItem(app.publisher))
            self.table_packages.setItem(row, 2, QTableWidgetItem(app.version))
            self.table_packages.setItem(row, 3, QTableWidgetItem("Installed"))

    def _on_list_pip_clicked(self) -> None:
        async def _fetch():
            return await self.wc.software.list_pip_packages()

        worker = AsyncWorker(_fetch())
        worker.result_ready.connect(self._display_pip_packages)
        self._active_worker = worker
        worker.start()

    def _display_pip_packages(self, pkgs: list[dict]) -> None:
        self.table_packages.setRowCount(0)
        for p in pkgs:
            row = self.table_packages.rowCount()
            self.table_packages.insertRow(row)
            self.table_packages.setItem(row, 0, QTableWidgetItem(p.get("name", "")))
            self.table_packages.setItem(row, 1, QTableWidgetItem("Python pip"))
            self.table_packages.setItem(row, 2, QTableWidgetItem(p.get("version", "")))
            self.table_packages.setItem(row, 3, QTableWidgetItem("Installed"))

    # --------------------------------------------------------------------------
    # 4. File Manager Tab
    # --------------------------------------------------------------------------

    def _build_filesystem_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # File Search & Storage Analysis Row
        search_box = QHBoxLayout()
        self.txt_file_search = QLineEdit()
        self.txt_file_search.setPlaceholderText("Search files by name or extension (e.g. *.py, report.pdf)...")
        search_box.addWidget(self.txt_file_search)

        btn_search = QPushButton("🔍 Search Files")
        btn_search.clicked.connect(self._on_search_files_clicked)
        search_box.addWidget(btn_search)

        btn_analyze = QPushButton("📊 Analyze Storage Usage")
        btn_analyze.clicked.connect(self._on_analyze_storage_clicked)
        search_box.addWidget(btn_analyze)

        layout.addLayout(search_box)

        # Files Table
        self.table_files = QTableWidget(0, 4)
        self.table_files.setHorizontalHeaderLabels(["Name", "Path", "Size", "Modified"])
        self.table_files.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table_files.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table_files.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table_files.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        layout.addWidget(self.table_files)

        # Operations row (New folder, Open with app, Delete)
        ops_row = QHBoxLayout()
        btn_new_folder = QPushButton("📁 New Folder...")
        btn_new_folder.clicked.connect(self._on_new_folder_clicked)
        ops_row.addWidget(btn_new_folder)

        btn_open_file = QPushButton("📄 Open Selected File")
        btn_open_file.clicked.connect(self._on_open_selected_file_clicked)
        ops_row.addWidget(btn_open_file)

        btn_delete_file = QPushButton("🗑️ Send to Recycle Bin")
        btn_delete_file.clicked.connect(self._on_delete_selected_file_clicked)
        ops_row.addWidget(btn_delete_file)

        ops_row.addStretch()
        layout.addLayout(ops_row)
        return widget

    def _on_search_files_clicked(self) -> None:
        query = self.txt_file_search.text().strip()
        cwd = self.txt_terminal_cwd.text().strip()

        def _search():
            return self.wc.filesystem.search_files(query=query, start_dir=cwd, max_results=100)

        worker = BackgroundWorker(_search)
        worker.result_ready.connect(self._display_file_results)
        self._active_worker = worker
        worker.start()

    def _display_file_results(self, files: list[dict]) -> None:
        self.table_files.setRowCount(0)
        for f in files:
            row = self.table_files.rowCount()
            self.table_files.insertRow(row)
            self.table_files.setItem(row, 0, QTableWidgetItem(f["name"]))
            self.table_files.setItem(row, 1, QTableWidgetItem(f["path"]))
            sz_kb = round(f["size_bytes"] / 1024, 1)
            self.table_files.setItem(row, 2, QTableWidgetItem(f"{sz_kb} KB"))
            self.table_files.setItem(row, 3, QTableWidgetItem(str(f["modified_time"])))

    def _on_analyze_storage_clicked(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Directory to Analyze", os.getcwd())
        if not folder:
            return

        def _analyze():
            return self.wc.filesystem.analyze_storage(folder)

        worker = BackgroundWorker(_analyze)
        worker.result_ready.connect(self._display_storage_report)
        self._active_worker = worker
        worker.start()

    def _display_storage_report(self, report: StorageReport) -> None:
        msg = (
            f"Storage Report for {report.target_path}\n"
            f"Total Size: {report.total_size_mb} MB ({report.file_count} files, {report.dir_count} directories)\n\n"
            f"Largest Files:\n"
        )
        for lf in report.large_files[:5]:
            msg += f"- {lf['path']} ({lf['size_mb']} MB)\n"

        msg += "\nExtension Breakdown:\n"
        for ext, mb in report.extension_breakdown.items():
            msg += f"- {ext}: {mb} MB\n"

        QMessageBox.information(self, "Storage Analysis", msg)

    def _on_new_folder_clicked(self) -> None:
        folder_path = QFileDialog.getExistingDirectory(self, "Select Parent Directory", os.getcwd())
        if folder_path:
            name, ok = QFileDialog.getSaveFileName(self, "New Folder Name", folder_path)
            if ok and name:
                self.wc.filesystem.create_directory(name)
                QMessageBox.information(self, "Success", f"Folder created: {name}")

    def _on_open_selected_file_clicked(self) -> None:
        curr = self.table_files.currentRow()
        if curr >= 0:
            item = self.table_files.item(curr, 1)
            if item:
                self.wc.application.open_file_with_associated_app(item.text())

    def _on_delete_selected_file_clicked(self) -> None:
        curr = self.table_files.currentRow()
        if curr >= 0:
            item = self.table_files.item(curr, 1)
            if item:
                fpath = item.text()
                reply = QMessageBox.question(
                    self,
                    "Confirm Deletion",
                    f"Send '{fpath}' to the Windows Recycle Bin?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply == QMessageBox.Yes:
                    self.wc.filesystem.delete_item(fpath, permanent=False)
                    self.table_files.removeRow(curr)

    # --------------------------------------------------------------------------
    # 5. System & Network Tab
    # --------------------------------------------------------------------------

    def _build_system_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        sub_tabs = QTabWidget()

        # Processes Tab
        tab_proc = QWidget()
        proc_layout = QVBoxLayout(tab_proc)
        proc_btn_row = QHBoxLayout()
        btn_refresh_proc = QPushButton("🔄 Refresh Processes")
        btn_refresh_proc.clicked.connect(self._refresh_processes)
        proc_btn_row.addWidget(btn_refresh_proc)

        btn_kill_proc = QPushButton("🛑 Terminate Selected Process")
        btn_kill_proc.clicked.connect(self._on_kill_process_clicked)
        proc_btn_row.addWidget(btn_kill_proc)
        proc_btn_row.addStretch()
        proc_layout.addLayout(proc_btn_row)

        self.table_proc = QTableWidget(0, 5)
        self.table_proc.setHorizontalHeaderLabels(["PID", "Name", "Memory (MB)", "Status", "Protected"])
        self.table_proc.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        proc_layout.addWidget(self.table_proc)
        sub_tabs.addTab(tab_proc, "Active Processes")

        # Services Tab
        tab_serv = QWidget()
        serv_layout = QVBoxLayout(tab_serv)
        btn_refresh_serv = QPushButton("🔄 Refresh Services")
        btn_refresh_serv.clicked.connect(self._refresh_services)
        serv_layout.addWidget(btn_refresh_serv)

        self.table_services = QTableWidget(0, 4)
        self.table_services.setHorizontalHeaderLabels(["Name", "Display Name", "Status", "Start Type"])
        self.table_services.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        serv_layout.addWidget(self.table_services)
        sub_tabs.addTab(tab_serv, "Windows Services")

        # Network Diagnostics Tab
        tab_net = QWidget()
        net_layout = QVBoxLayout(tab_net)
        btn_diag_net = QPushButton("🔍 Diagnose Network Connectivity")
        btn_diag_net.clicked.connect(self._diagnose_network)
        net_layout.addWidget(btn_diag_net)

        self.txt_net_diag = QTextEdit()
        self.txt_net_diag.setReadOnly(True)
        self.txt_net_diag.setFont(QFont("Consolas", 10))
        net_layout.addWidget(self.txt_net_diag)
        sub_tabs.addTab(tab_net, "Network Diagnostics")

        # Power Controls Tab
        tab_power = QWidget()
        power_layout = QVBoxLayout(tab_power)
        power_layout.addWidget(QLabel("Windows System Power Operations (Safety Confirmed):"))

        btn_p_lock = QPushButton("🔒 Lock Session")
        btn_p_lock.clicked.connect(self._on_lock_clicked)
        power_layout.addWidget(btn_p_lock)

        btn_p_sleep = QPushButton("💤 Sleep Standby")
        btn_p_sleep.clicked.connect(self._on_sleep_clicked)
        power_layout.addWidget(btn_p_sleep)

        btn_p_restart = QPushButton("🔄 Restart Windows...")
        btn_p_restart.setStyleSheet("color: #d32f2f;")
        btn_p_restart.clicked.connect(self._on_restart_clicked)
        power_layout.addWidget(btn_p_restart)

        btn_p_shutdown = QPushButton("⏻ Shut Down Windows...")
        btn_p_shutdown.setStyleSheet("color: #d32f2f;")
        btn_p_shutdown.clicked.connect(self._on_shutdown_clicked)
        power_layout.addWidget(btn_p_shutdown)

        power_layout.addStretch()
        sub_tabs.addTab(tab_power, "Power Management")

        layout.addWidget(sub_tabs)
        return widget

    def _refresh_processes(self) -> None:
        def _fetch():
            return self.wc.process.list_processes(limit=100)

        worker = BackgroundWorker(_fetch)
        worker.result_ready.connect(self._display_processes)
        self._active_worker = worker
        worker.start()

    def _display_processes(self, procs: list[ProcessInfo]) -> None:
        self.table_proc.setRowCount(0)
        for p in procs:
            row = self.table_proc.rowCount()
            self.table_proc.insertRow(row)
            self.table_proc.setItem(row, 0, QTableWidgetItem(str(p.pid)))
            self.table_proc.setItem(row, 1, QTableWidgetItem(p.name))
            self.table_proc.setItem(row, 2, QTableWidgetItem(str(p.memory_mb)))
            self.table_proc.setItem(row, 3, QTableWidgetItem(p.status))
            prot_item = QTableWidgetItem("Yes" if p.is_critical else "No")
            if p.is_critical:
                prot_item.setForeground(QColor("#f44336"))
            self.table_proc.setItem(row, 4, prot_item)

    def _on_kill_process_clicked(self) -> None:
        curr = self.table_proc.currentRow()
        if curr >= 0:
            pid_item = self.table_proc.item(curr, 0)
            name_item = self.table_proc.item(curr, 1)
            if pid_item and name_item:
                pid = int(pid_item.text())
                name = name_item.text()
                reply = QMessageBox.question(
                    self,
                    "Terminate Process",
                    f"Terminate process '{name}' (PID: {pid})?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply == QMessageBox.Yes:
                    ok, msg = self.wc.process.terminate_process(pid, force=False)
                    QMessageBox.information(self, "Result", msg)
                    self._refresh_processes()

    def _refresh_services(self) -> None:
        def _fetch():
            return self.wc.system.list_services(limit=100)

        worker = BackgroundWorker(_fetch)
        worker.result_ready.connect(self._display_services)
        self._active_worker = worker
        worker.start()

    def _display_services(self, servs: list[ServiceInfo]) -> None:
        self.table_services.setRowCount(0)
        for s in servs:
            row = self.table_services.rowCount()
            self.table_services.insertRow(row)
            self.table_services.setItem(row, 0, QTableWidgetItem(s.name))
            self.table_services.setItem(row, 1, QTableWidgetItem(s.display_name))
            self.table_services.setItem(row, 2, QTableWidgetItem(s.status))
            self.table_services.setItem(row, 3, QTableWidgetItem(s.start_type))

    def _diagnose_network(self) -> None:
        def _fetch():
            return self.wc.system.inspect_network()

        worker = BackgroundWorker(_fetch)
        worker.result_ready.connect(self._display_network_diag)
        self._active_worker = worker
        worker.start()

    def _display_network_diag(self, net: Any) -> None:
        txt = (
            f"Host Name: {net.hostname}\n"
            f"Internet Connected: {'YES' if net.internet_connected else 'NO'}\n"
            f"Ping Latency to 8.8.8.8: {net.ping_latency_ms} ms\n\n"
            f"Network Interfaces:\n"
        )
        for iface in net.interfaces:
            txt += f"- {iface['name']}: IPv4={iface['ipv4']} (Active: {iface['is_up']})\n"
        self.txt_net_diag.setText(txt)

    def _on_lock_clicked(self) -> None:
        ok, msg = self.wc.system.lock_workstation()
        if not ok:
            QMessageBox.warning(self, "Lock Failed", msg)

    def _on_sleep_clicked(self) -> None:
        reply = QMessageBox.question(self, "Confirm Sleep", "Put computer to sleep standby now?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.wc.system.sleep_computer()

    def _on_restart_clicked(self) -> None:
        reply = QMessageBox.warning(
            self,
            "⚠️ Confirm System Restart",
            "Are you sure you want to RESTART Windows? Unsaved work will be closed.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            cmd = self.wc.system.build_restart_command()
            self.tabs.setCurrentWidget(self.tab_terminal)
            self.txt_command_input.setText(cmd)
            self._on_run_terminal_command()

    def _on_shutdown_clicked(self) -> None:
        reply = QMessageBox.warning(
            self,
            "⚠️ Confirm System Shutdown",
            "Are you sure you want to SHUT DOWN Windows? Unsaved work will be closed.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            cmd = self.wc.system.build_shutdown_command()
            self.tabs.setCurrentWidget(self.tab_terminal)
            self.txt_command_input.setText(cmd)
            self._on_run_terminal_command()

    # --------------------------------------------------------------------------
    # 6. Security & Audit Tab
    # --------------------------------------------------------------------------

    def _build_audit_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Policy & Execution Mode Settings
        mode_group = QGroupBox("Execution Policy & Administrator Settings")
        mode_layout = QHBoxLayout(mode_group)

        mode_layout.addWidget(QLabel("Execution Mode:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Full Control (Default)", "Safe Confirmation", "Restricted (Read-Only)"])
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.combo_mode)

        mode_layout.addSpacing(20)

        btn_test_uac = QPushButton("🛡️ Test Windows UAC Elevation")
        btn_test_uac.clicked.connect(self._on_test_uac_clicked)
        mode_layout.addWidget(btn_test_uac)

        mode_layout.addStretch()
        layout.addWidget(mode_group)

        # Audit Trail Table
        audit_group = QGroupBox("Immutable Computer Control Audit Log")
        audit_layout = QVBoxLayout(audit_group)

        audit_bar = QHBoxLayout()
        btn_refresh_audit = QPushButton("🔄 Refresh Audit Logs")
        btn_refresh_audit.clicked.connect(self._refresh_audit_logs)
        audit_bar.addWidget(btn_refresh_audit)
        audit_bar.addStretch()
        audit_layout.addLayout(audit_bar)

        self.table_audit = QTableWidget(0, 7)
        self.table_audit.setHorizontalHeaderLabels([
            "Timestamp", "Source", "Instruction", "Operation", "Privilege", "Risk", "Status"
        ])
        self.table_audit.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        audit_layout.addWidget(self.table_audit)

        layout.addWidget(audit_group)
        return widget

    def _on_mode_changed(self, index: int) -> None:
        modes = [ExecutionMode.FULL_CONTROL, ExecutionMode.SAFE_CONFIRMATION, ExecutionMode.RESTRICTED]
        chosen = modes[index]
        self.wc.policy.set_execution_mode(chosen)
        QMessageBox.information(self, "Policy Updated", f"Computer Control mode updated to: {chosen.value.upper()}")
        self._refresh_dashboard()

    def _on_test_uac_clicked(self) -> None:
        async def _test():
            plan = ActionPlan(
                objective="Test UAC Elevation",
                command="whoami /priv",
                privilege_level=PrivilegeLevel.ELEVATED,
                risk_level=RiskLevel.LOW,
            )
            return await self.wc.executor.execute_plan(plan, bypass_confirmation=True)

        worker = AsyncWorker(_test())
        worker.result_ready.connect(lambda res: QMessageBox.information(
            self, "UAC Result", f"Exit Code: {res.exit_code}\nElevated: {res.elevated}\nOutput:\n{res.stdout[:400]}"
        ))
        self._active_worker = worker
        worker.start()

    def _refresh_audit_logs(self) -> None:
        if os.environ.get("PYTEST_CURRENT_TEST"):
            return

        def _fetch():
            return self.wc.audit.query_logs(limit=100)

        worker = BackgroundWorker(_fetch)
        worker.result_ready.connect(self._display_audit_logs)
        self._active_worker = worker
        worker.start()

    def _display_audit_logs(self, logs: list) -> None:
        self.table_audit.setRowCount(0)
        for log in logs:
            row = self.table_audit.rowCount()
            self.table_audit.insertRow(row)
            self.table_audit.setItem(row, 0, QTableWidgetItem(log.timestamp))
            self.table_audit.setItem(row, 1, QTableWidgetItem(log.source))
            self.table_audit.setItem(row, 2, QTableWidgetItem(log.instruction))
            self.table_audit.setItem(row, 3, QTableWidgetItem(log.operation_type))
            self.table_audit.setItem(row, 4, QTableWidgetItem(log.privilege_level))
            self.table_audit.setItem(row, 5, QTableWidgetItem(log.risk_level))
            self.table_audit.setItem(row, 6, QTableWidgetItem(log.status))

    def _on_emergency_stop_clicked(self) -> None:
        count = self.controller.trigger_emergency_stop()
        QMessageBox.warning(
            self,
            "EMERGENCY STOP",
            f"Emergency stop executed! {count} active task(s) and background child processes terminated.",
        )
        self.terminal_output.append("<span style='color: #d32f2f;'>[EMERGENCY STOP EXECUTED]</span>")

    def closeEvent(self, event) -> None:
        if hasattr(self, "metrics_timer") and self.metrics_timer.isActive():
            self.metrics_timer.stop()
        if hasattr(self, "_active_worker") and self._active_worker and self._active_worker.isRunning():
            self._active_worker.quit()
            self._active_worker.wait(500)
        super().closeEvent(event)
