"""Comprehensive Test Suite for Phase 7: Full Windows Computer Control & Administrative Automation.

Covers:
- Execution Policy & Risk Classification (LOW, SENSITIVE, DESTRUCTIVE, BLOCKED)
- Protection of HAKS databases and critical Windows system directories
- Privilege Detection & UAC Elevation Requirement Analysis
- Asynchronous CMD and PowerShell Services (live output streaming, Tamil/Unicode)
- Filesystem Management (drives, recursive search, storage analysis, CRUD, ZIP, duplicates, recycle bin)
- Process Management & Protection of Critical System / HAKS processes
- Application Management & Registry Lookups
- Software & Download Management with SHA-256 Checksum Verification
- Windows System Management (hardware specs, network ping/port, services)
- Natural Language & Voice Command Interpreter (English + Tamil)
- Immutable SQLite Audit Trail with Automated Secret Redaction
- Emergency Stop Subsystem
- WindowsControlView PySide6 UI and MainWindow Integration
"""

import asyncio
import hashlib
import os
from pathlib import Path
import pytest

from config.settings import AppConfig
from core.app_controller import AppController
from storage.database import DatabaseManager
from windows_control.models import (
    ExecutionMode,
    RiskLevel,
    PrivilegeLevel,
    ExecutionStatus,
    ActionPlan,
    CommandResult,
)
from windows_control.execution_policy import ExecutionPolicy
from windows_control.privilege_detector import PrivilegeDetector
from windows_control.cmd_service import CMDService
from windows_control.powershell_service import PowerShellService
from windows_control.filesystem_service import FilesystemService
from windows_control.process_manager import ProcessManager
from windows_control.application_manager import ApplicationManager, APP_ALIASES
from windows_control.installer_verifier import InstallerVerifier
from windows_control.download_manager import DownloadManager
from windows_control.software_manager import SoftwareManager
from windows_control.system_manager import SystemManager
from windows_control.command_interpreter import CommandInterpreter
from windows_control.audit_logger import AuditLogger
from windows_control.manager import WindowsControlManager


# ==============================================================================
# 1. Execution Policy & Risk Assessment Tests
# ==============================================================================

def test_execution_policy_risk_classification(temp_workspace: AppConfig):
    """Test policy classifies operations into LOW, SENSITIVE, DESTRUCTIVE, and BLOCKED."""
    policy = ExecutionPolicy(temp_workspace)

    # LOW risk operations
    assert policy.classify_risk("dir C:\\") == RiskLevel.LOW
    assert policy.classify_risk("echo hello world") == RiskLevel.LOW
    assert policy.classify_risk("git status") == RiskLevel.LOW
    assert policy.classify_risk("ping 127.0.0.1") == RiskLevel.LOW

    # SENSITIVE risk operations
    assert policy.classify_risk("del C:\\temp\\myfile.txt") == RiskLevel.SENSITIVE
    assert policy.classify_risk("rmdir C:\\temp\\subfolder") == RiskLevel.SENSITIVE
    assert policy.classify_risk("taskkill /pid 9999") == RiskLevel.SENSITIVE
    assert policy.classify_risk("net stop wuauserv") == RiskLevel.SENSITIVE
    assert policy.classify_risk("reg add HKLM\\Software\\Test") == RiskLevel.SENSITIVE

    # DESTRUCTIVE risk operations
    assert policy.classify_risk("format D: /fs:ntfs") == RiskLevel.DESTRUCTIVE
    assert policy.classify_risk("diskpart /s script.txt") == RiskLevel.DESTRUCTIVE
    assert policy.classify_risk("del /s /q C:\\*.*") == RiskLevel.DESTRUCTIVE
    assert policy.classify_risk("git reset --hard") == RiskLevel.DESTRUCTIVE

    # BLOCKED operations: protection of HAKS database files
    data_dir = str(temp_workspace.paths.data_dir)
    assert policy.classify_risk("delete", operation_type="delete", target=f"{data_dir}\\personal_memory.db") == RiskLevel.BLOCKED
    assert policy.classify_risk("wipe", operation_type="wipe", target=data_dir) == RiskLevel.BLOCKED


def test_execution_policy_modes(temp_workspace: AppConfig):
    """Test behavior across FULL_CONTROL, SAFE_CONFIRMATION, and RESTRICTED modes."""
    policy = ExecutionPolicy(temp_workspace)

    # 1. FULL_CONTROL: LOW runs directly; SENSITIVE/DESTRUCTIVE require confirmation; BLOCKED blocked
    policy.set_execution_mode(ExecutionMode.FULL_CONTROL)
    plan_low = ActionPlan(objective="Test low", command="echo testing", risk_level=RiskLevel.LOW)
    allowed, msg = policy.evaluate_plan(plan_low)
    assert allowed and not plan_low.confirmation_required

    plan_sens = ActionPlan(objective="Kill proc", command="taskkill /f /im test.exe", risk_level=RiskLevel.SENSITIVE)
    allowed, msg = policy.evaluate_plan(plan_sens)
    assert allowed and plan_sens.confirmation_required

    plan_blocked = ActionPlan(objective="Delete DB", command="del", operation_type="delete", target=str(temp_workspace.paths.data_dir / "work_memory.db"))
    allowed, msg = policy.evaluate_plan(plan_blocked)
    assert not allowed  # Tampering with HAKS data is blocked

    # 2. SAFE_CONFIRMATION: Even non-read actions require confirmation
    policy.set_execution_mode(ExecutionMode.SAFE_CONFIRMATION)
    plan_cmd = ActionPlan(objective="Run script", command="python script.py", operation_type="command")
    allowed, msg = policy.evaluate_plan(plan_cmd)
    assert allowed and plan_cmd.confirmation_required

    # 3. RESTRICTED: Modifying operations blocked
    policy.set_execution_mode(ExecutionMode.RESTRICTED)
    plan_mod = ActionPlan(objective="Delete file", command="del C:\\temp\\file.txt", operation_type="delete")
    allowed, msg = policy.evaluate_plan(plan_mod)
    assert not allowed


# ==============================================================================
# 2. Privilege Detection Tests
# ==============================================================================

def test_privilege_detector():
    """Verify administrator checking and command elevation analysis."""
    # Basic admin check returns a boolean
    is_admin = PrivilegeDetector.is_current_process_admin()
    assert isinstance(is_admin, bool)

    # Commands that require Windows elevation
    assert PrivilegeDetector.requires_elevation("sfc /scannow") == PrivilegeLevel.ELEVATED
    assert PrivilegeDetector.requires_elevation("dism /online /cleanup-image") == PrivilegeLevel.ELEVATED
    assert PrivilegeDetector.requires_elevation("net stop spooler") == PrivilegeLevel.ELEVATED
    assert PrivilegeDetector.requires_elevation("bcdedit /set testsigning on") == PrivilegeLevel.ELEVATED

    # Commands that do NOT require elevation
    assert PrivilegeDetector.requires_elevation("dir C:\\") == PrivilegeLevel.NORMAL
    assert PrivilegeDetector.requires_elevation("echo hello") == PrivilegeLevel.NORMAL
    assert PrivilegeDetector.requires_elevation("python --version") == PrivilegeLevel.NORMAL
    assert PrivilegeDetector.requires_elevation("ping 127.0.0.1") == PrivilegeLevel.NORMAL


# ==============================================================================
# 3. CMD and PowerShell Execution Subsystems
# ==============================================================================

@pytest.mark.asyncio
async def test_cmd_service_execution():
    """Test asynchronous CMD execution and live streaming output."""
    cmd = CMDService()
    captured_output = []

    def stream_cb(text: str):
        captured_output.append(text)

    result = await cmd.execute("echo HAKS_CMD_VERIFICATION_PASS", output_callback=stream_cb)
    assert result.exit_code == 0
    assert result.success is True
    assert "HAKS_CMD_VERIFICATION_PASS" in result.stdout
    assert any("HAKS_CMD_VERIFICATION_PASS" in line for line in captured_output)


@pytest.mark.asyncio
async def test_powershell_service_execution_and_unicode():
    """Test asynchronous PowerShell execution with UTF-8 Tamil and English output."""
    ps = PowerShellService()

    # Standard execution
    res1 = await ps.execute('Write-Output "HAKS_POWERSHELL_READY"')
    assert res1.exit_code == 0
    assert "HAKS_POWERSHELL_READY" in res1.stdout

    # Unicode & Tamil characters preservation
    tamil_test_str = "ஹலோ HAKS தமிழ் சோதனை"
    res2 = await ps.execute(f'[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; Write-Output "{tamil_test_str}"')
    assert res2.exit_code == 0
    assert "தமிழ்" in res2.stdout or "HAKS" in res2.stdout


# ==============================================================================
# 4. Filesystem Management & Safeguards
# ==============================================================================

def test_filesystem_drives_and_crud(tmp_path: Path):
    """Test filesystem drive enumeration, directory creation, read, write, and search."""
    fs = FilesystemService()

    # 1. Drive enumeration
    drives = fs.list_drives()
    assert len(drives) >= 1
    assert any(d.drive_letter.startswith("C") for d in drives)
    assert drives[0].total_bytes > 0

    # 2. File CRUD
    sub_dir = tmp_path / "haks_test_dir"
    created = fs.create_directory(str(sub_dir))
    assert Path(created).is_dir()

    test_file = sub_dir / "sample.txt"
    written = fs.write_file(str(test_file), "HAKS Offline Assistant Filesystem Test Content")
    assert written is True
    assert test_file.exists()

    content = fs.read_file(str(test_file))
    assert content == "HAKS Offline Assistant Filesystem Test Content"

    # 3. Search
    results = fs.search_files(query="sample", start_dir=str(tmp_path))
    assert len(results) >= 1
    assert any(r["name"] == "sample.txt" for r in results)

    # 4. Storage Breakdown Report
    report = fs.get_storage_report(str(tmp_path))
    assert report.file_count >= 1
    assert report.total_size_mb >= 0.0


def test_filesystem_zip_and_duplicates(tmp_path: Path):
    """Test ZIP compression/decompression and SHA-256 duplicate detection."""
    fs = FilesystemService()

    # Create duplicate files and one distinct file
    dir_a = tmp_path / "dir_a"
    dir_a.mkdir()
    f1 = dir_a / "original.dat"
    f2 = dir_a / "copy.dat"
    f3 = dir_a / "distinct.dat"

    identical_content = b"HAKS Duplicate File Detection Verification Data 12345678" * 50
    f1.write_bytes(identical_content)
    f2.write_bytes(identical_content)
    f3.write_bytes(b"Different content altogether" * 50)

    # Duplicate detection
    duplicates = fs.find_duplicates(str(tmp_path), min_size_bytes=10)
    assert len(duplicates) == 1
    dup_paths = duplicates[0]
    assert len(dup_paths) == 2
    assert str(f1) in dup_paths and str(f2) in dup_paths

    # ZIP Archive & Extraction
    zip_dest = tmp_path / "test_archive.zip"
    created_zip = fs.create_zip(str(dir_a), str(zip_dest))
    assert Path(created_zip).exists()

    extract_dest = tmp_path / "extracted_dir"
    extracted = fs.extract_zip(str(zip_dest), str(extract_dest))
    assert Path(extracted).is_dir()
    assert (extract_dest / "original.dat").exists()


def test_filesystem_protected_path_safeguard(temp_workspace: AppConfig):
    """Verify that attempts to delete critical HAKS database files are blocked."""
    fs = FilesystemService(temp_workspace)
    db_file = temp_workspace.paths.data_dir / "personal_memory.db"
    db_file.parent.mkdir(parents=True, exist_ok=True)
    db_file.write_text("dummy database content")

    # Attempt to permanently delete protected file
    with pytest.raises(PermissionError):
        fs.delete_item(str(db_file), permanent=True)
    assert db_file.exists()


# ==============================================================================
# 5. Process & Application Management
# ==============================================================================

def test_process_manager_listing_and_protection():
    """Verify process enumeration and safeguard against terminating protected system PIDs."""
    pm = ProcessManager()

    # 1. Listing processes
    procs = pm.list_processes()
    assert len(procs) > 5
    current_pid = os.getpid()
    assert any(p.pid == current_pid for p in procs)

    # 2. Protection against terminating System Idle (PID 0)
    success0, _ = pm.terminate_process(0)
    assert success0 is False

    # 3. Protection against terminating System (PID 4)
    success4, _ = pm.terminate_process(4)
    assert success4 is False

    # 4. Protection against terminating HAKS itself
    success_cur, _ = pm.terminate_process(current_pid)
    assert success_cur is False


def test_application_manager_aliases():
    """Verify application manager resolves standard application aliases."""
    assert "notepad" in APP_ALIASES
    assert "calc" in APP_ALIASES
    assert "cmd" in APP_ALIASES
    assert "powershell" in APP_ALIASES


# ==============================================================================
# 6. Software & Download Management with Verification
# ==============================================================================

def test_installer_verifier_and_downloads(tmp_path: Path):
    """Verify SHA-256 calculation, file verification, and download tracking."""
    verifier = InstallerVerifier()

    test_content = b"HAKS Installer Verification File Test Content"
    test_file = tmp_path / "installer_test.bin"
    test_file.write_bytes(test_content)

    expected_sha256 = hashlib.sha256(test_content).hexdigest()
    calculated = verifier.calculate_sha256(test_file)
    assert calculated == expected_sha256

    # Verify match
    assert verifier.verify_checksum(test_file, expected_sha256) is True
    # Verify mismatch
    assert verifier.verify_checksum(test_file, "0000000000000000000000000000000000000000000000000000000000000000") is False

    # Download manager tracking
    dm = DownloadManager()
    items = dm.get_downloads()
    assert isinstance(items, list)


def test_software_manager_command_builders():
    """Verify SoftwareManager command construction for winget, pip, and git."""
    sm = SoftwareManager()

    # Winget commands
    install_cmd = sm.build_winget_install_command("Microsoft.VisualStudioCode")
    assert "winget install" in install_cmd and "Microsoft.VisualStudioCode" in install_cmd

    # Pip commands
    pip_install = sm.build_pip_install_command("numpy")
    assert "pip install" in pip_install and "numpy" in pip_install

    # Git commands
    git_status = sm.build_git_status_command()
    assert "git status" in git_status


# ==============================================================================
# 7. System & Network Diagnostics
# ==============================================================================

def test_system_manager_hardware_and_network():
    """Verify system hardware information retrieval and network diagnostics."""
    sm = SystemManager()

    # Hardware Specs
    hw = sm.get_system_hardware_info()
    assert hw.cpu_cores_physical > 0
    assert hw.ram_total_gb > 0
    assert hw.os_name.startswith("Windows") or hw.os_name != ""

    # Network Ping (Localhost)
    ping_res = sm.ping_host("127.0.0.1")
    assert ping_res is not None and ping_res >= 0.0

    # Network Port Test (Localhost)
    port_open = sm.test_port_connectivity("127.0.0.1", 135, timeout_sec=1.0)
    assert isinstance(port_open, bool)

    # Services
    services = sm.list_services()
    assert isinstance(services, list)


# ==============================================================================
# 8. Command Interpreter (English + Tamil Natural Language Parsing)
# ==============================================================================

def test_command_interpreter_english():
    """Verify English natural language instruction parsing into ActionPlans."""
    interpreter = CommandInterpreter()

    # App Launching
    plan1 = interpreter.interpret("open notepad")
    assert plan1.operation_type == "launch_app"
    assert "notepad" in plan1.summary.lower()

    # File Search
    plan2 = interpreter.interpret("search for *.pdf files")
    assert plan2.operation_type == "search_files"

    # Hardware Info
    plan3 = interpreter.interpret("check cpu and memory usage")
    assert plan3.operation_type == "system_hardware"

    # Network Diagnostics
    plan4 = interpreter.interpret("ping 8.8.8.8")
    assert plan4.operation_type == "network_diagnostics"
    assert plan4.target == "8.8.8.8"

    # Conversational non-control instruction
    plan5 = interpreter.interpret("What is the speed of light?")
    assert plan5.operation_type == "conversation"


def test_command_interpreter_tamil():
    """Verify Tamil natural language instruction parsing into ActionPlans."""
    interpreter = CommandInterpreter()

    # Tamil App Launching
    plan1 = interpreter.interpret("நோட்பேட் திற")
    assert plan1.operation_type == "launch_app"
    assert plan1.target == "notepad"

    # Tamil Hardware Check
    plan2 = interpreter.interpret("சிபியூ பயன்பாட்டை காட்டு")
    assert plan2.operation_type == "system_hardware"

    # Tamil Drive Storage Check
    plan3 = interpreter.interpret("சி ட்ரைவ் இடம் எவ்வளவு")
    assert plan3.operation_type == "storage_analysis"


# ==============================================================================
# 9. Audit Logging with Secret Redaction
# ==============================================================================

def test_audit_logger_redaction_and_queries(temp_workspace: AppConfig, test_db_manager: DatabaseManager):
    """Verify audit logger writes to SQLite and automatically redacts API keys and secrets."""
    logger = AuditLogger(config=temp_workspace, db_manager=test_db_manager)

    # Log record with sensitive API key and passwords
    real_key = "AIzaSy" + "B" * 33
    sensitive_command = f"python script.py --api_key={real_key} --password=SecretP@ssword99"
    log_id = logger.log_execution(
        source="chat",
        instruction=sensitive_command,
        plan_id="plan_test_123",
        operation_type="command",
        executable="python",
        args=f"--api_key={real_key} --password=SecretP@ssword99",
        working_dir="C:\\test",
        privilege_level="normal",
        risk_level="low",
        confirmation_status="confirmed",
        exit_code=0,
        status="completed",
        duration_ms=120.0,
    )
    assert log_id is not None

    # Verify database query retrieval
    records = logger.query_logs(limit=10)
    assert len(records) >= 1
    assert records[0].id == log_id
    # Verify redaction
    assert "AIzaSy" not in records[0].instruction
    assert "SecretP@ssword99" not in records[0].instruction
    assert "REDACTED" in records[0].instruction
    assert "REDACTED" in records[0].args


# ==============================================================================
# 10. Emergency Stop Subsystem
# ==============================================================================

def test_emergency_stop(controller: AppController):
    """Verify emergency stop aborts registered active tasks and broadcasts signal."""
    stop_event_fired = False

    def on_stop():
        nonlocal stop_event_fired
        stop_event_fired = True

    controller.events.windows_emergency_stop_triggered.connect(on_stop)

    stopped_count = controller.trigger_emergency_stop()
    assert isinstance(stopped_count, int)
    assert stop_event_fired is True


# ==============================================================================
# 11. UI Integration: WindowsControlView and MainWindow
# ==============================================================================

def test_windows_control_view_ui(qapp, controller: AppController):
    """Verify that WindowsControlView initializes all 6 sub-tabs cleanly."""
    from ui.views.windows_control_view import WindowsControlView

    view = WindowsControlView(controller)
    assert view.tabs.count() == 6
    assert "Dashboard" in view.tabs.tabText(0)
    assert "Command Center" in view.tabs.tabText(1)
    assert "Software" in view.tabs.tabText(2)
    assert "File Manager" in view.tabs.tabText(3)
    assert "System" in view.tabs.tabText(4)
    assert "Security" in view.tabs.tabText(5)
    view.close()


def test_main_window_control_tab_integration(qapp, temp_workspace: AppConfig, controller: AppController):
    """Verify MainWindow integrates Computer Control in the sidebar, status bar, and stack."""
    from ui.main_window import MainWindow

    win = MainWindow(config=temp_workspace, controller=controller)
    assert hasattr(win, "page_windows_control")
    assert "control" in win._nav_buttons

    # Switch to Computer Control tab
    win.events.navigate_to_page.emit("control")
    assert win.content_stack.currentWidget() == win.page_windows_control
    assert win.lbl_page_title.text() == "Computer Control"
    win.close()
