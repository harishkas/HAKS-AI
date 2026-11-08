"""Tests for HAKS System-Wide Windows Resource Discovery & Hardware-Aware Profiling.

Verifies:
- SystemHardwareProfiler: CPU, RAM, GPU compute capabilities, display, audio, storage
- Hardware capacity pre-flight evaluations (disk headroom, RAM reserve, GPU checks)
- SystemResourceDiscoveryService: multi-drive detection, user dirs, Python runtimes,
  Git ecosystem, dev tools, installed apps, running processes, and Windows services
- FilesystemService: cross-drive traversal, permission-aware SearchResult tracking
  (including inaccessible directories), large file detection, rename, and system safeguards
- WindowsControlManager & AppController integration
"""

import os
from pathlib import Path
import pytest

from config.settings import AppConfig
from core.app_controller import AppController
from storage.database import DatabaseManager
from windows_control.filesystem_service import FilesystemService
from windows_control.hardware_profiler import SystemHardwareProfiler
from windows_control.manager import WindowsControlManager
from windows_control.models import (
    CompleteHardwareProfile,
    DriveInfo,
    RAMInfo,
    ResourceSafetyCheck,
    SearchResult,
    SystemDiscoveryReport,
)
from windows_control.resource_discovery import SystemResourceDiscoveryService


# ==============================================================================
# 1. SystemHardwareProfiler Tests
# ==============================================================================

def test_hardware_profiler_specs():
    """Verify hardware profiler accurately measures CPU, RAM, display, audio, and storage."""
    profiler = SystemHardwareProfiler()
    profile = profiler.get_complete_profile(refresh=True)

    assert isinstance(profile, CompleteHardwareProfile)

    # CPU specifications
    assert profile.cpu.physical_cores >= 1
    assert profile.cpu.logical_processors >= 1
    assert len(profile.cpu.model) > 0

    # RAM specifications
    assert profile.ram.total_gb > 0
    assert profile.ram.available_gb > 0
    assert 0 <= profile.ram.percent_used <= 100

    # Storage overview across drives
    assert profile.storage.total_gb > 0
    assert len(profile.storage.drives) >= 1

    # Display & Audio
    assert profile.display.width > 0
    assert profile.display.height > 0
    assert isinstance(profile.audio.input_devices, list)
    assert isinstance(profile.audio.output_devices, list)


def test_hardware_profiler_gpu_evaluation():
    """Verify GPU inspection accurately evaluates compute acceleration without false assumptions."""
    profiler = SystemHardwareProfiler()
    gpus = profiler.detect_gpus()
    assert isinstance(gpus, list)

    for gpu in gpus:
        # GPUs on this hardware (Intel HD Graphics 620, AMD Radeon 520) must be evaluated honestly
        if "520" in gpu.name or "620" in gpu.name or "Intel" in gpu.name or "Radeon" in gpu.name:
            assert gpu.supports_llm_acceleration is False
            assert "cpu" in gpu.acceleration_notes.lower() or "quantized" in gpu.acceleration_notes.lower() or "not supported" in gpu.acceleration_notes.lower()


def test_hardware_profiler_safety_check_pass(monkeypatch):
    """Verify capacity evaluation passes for modest operations with sufficient resources."""
    profiler = SystemHardwareProfiler()

    # Guarantee healthy RAM availability in test environment regardless of host memory load
    monkeypatch.setattr(profiler, "detect_ram", lambda: RAMInfo(
        total_gb=8.0,
        available_gb=3.0,
        used_gb=5.0,
        percent_used=62.5,
    ))

    # Small operation (10 MB disk, 50 MB RAM, no GPU required)
    safety = profiler.evaluate_operation_capacity(
        operation="Download small utility",
        required_disk_gb=0.01,
        required_ram_gb=0.05,
        requires_gpu=False,
    )

    assert isinstance(safety, ResourceSafetyCheck)
    assert safety.is_safe is True
    assert len(safety.insufficient_resources) == 0


def test_hardware_profiler_safety_check_fail_disk_or_ram():
    """Verify capacity evaluation warns and blocks operations that exceed available capacity."""
    profiler = SystemHardwareProfiler()

    # Absurdly large disk request (5000 GB disk on ~238 GB SSD)
    safety_disk = profiler.evaluate_operation_capacity(
        operation="Download massive 5TB dataset",
        required_disk_gb=5000.0,
        required_ram_gb=0.1,
    )
    assert safety_disk.is_safe is False
    assert safety_disk.requires_user_approval is True
    assert any("disk" in item.lower() for item in safety_disk.insufficient_resources)

    # Absurdly large RAM request (64 GB on an 8 GB machine)
    safety_ram = profiler.evaluate_operation_capacity(
        operation="Load 70B parameter model in FP16",
        required_disk_gb=0.1,
        required_ram_gb=64.0,
    )
    assert safety_ram.is_safe is False
    assert any("ram" in item.lower() for item in safety_ram.insufficient_resources)

    # Unaccelerated GPU request
    safety_gpu = profiler.evaluate_operation_capacity(
        operation="Run CUDA TensorRT engine",
        requires_gpu=True,
    )
    assert safety_gpu.gpu_compatible is False
    assert safety_gpu.is_safe is False


# ==============================================================================
# 2. SystemResourceDiscoveryService Tests
# ==============================================================================

def test_system_resource_discovery_drives():
    """Verify discovery locates all logical drives including C:, D:, E:."""
    discovery = SystemResourceDiscoveryService()
    drives = discovery.discover_drives()

    assert len(drives) >= 1
    c_drive = next((d for d in drives if d.mountpoint.upper().startswith("C")), None)
    assert c_drive is not None
    assert c_drive.is_ready is True
    assert c_drive.total_gb > 0


def test_system_resource_discovery_user_directories():
    """Verify discovery resolves standard Windows user and system directories."""
    discovery = SystemResourceDiscoveryService()
    dirs = discovery.discover_user_directories()

    assert "user_profile" in dirs
    assert "desktop" in dirs
    assert "documents" in dirs
    assert "downloads" in dirs
    assert "local_app_data" in dirs
    assert Path(dirs["user_profile"]).exists()


def test_system_resource_discovery_pythons_and_git():
    """Verify discovery finds active Python runtimes and Git tooling."""
    discovery = SystemResourceDiscoveryService()

    # Python environments
    pythons = discovery.discover_python_installations()
    assert len(pythons) >= 1
    assert any("3.14" in p.get("version", "") or p.get("is_current") for p in pythons)

    # Git ecosystem
    git_eco = discovery.discover_git_ecosystem()
    assert git_eco.get("installed") is True
    assert "version" in git_eco


def test_system_resource_discovery_dev_tools_and_apps():
    """Verify discovery detects installed dev tools and Windows Registry applications."""
    discovery = SystemResourceDiscoveryService()

    # Development tools
    tools = discovery.discover_development_tools()
    tool_names = [t["name"].lower() for t in tools]
    assert "git" in tool_names or "powershell" in tool_names

    # Installed software
    apps = discovery.discover_installed_software(limit=20)
    assert len(apps) >= 1
    assert apps[0].name != ""


def test_system_resource_discovery_report():
    """Verify complete SystemDiscoveryReport generation."""
    discovery = SystemResourceDiscoveryService()
    report = discovery.discover_all(include_apps=True, include_processes=False)

    assert isinstance(report, SystemDiscoveryReport)
    assert len(report.drives) >= 1
    assert len(report.user_directories) >= 5
    assert len(report.python_installations) >= 1
    assert report.git_ecosystem.get("installed") is True


# ==============================================================================
# 3. FilesystemService Multi-Drive & Permission-Aware Traversal Tests
# ==============================================================================

def test_filesystem_get_drive_for_path():
    """Verify drive letter detection for paths on various drives."""
    fs = FilesystemService()
    assert fs.get_drive_for_path("C:\\Users\\Admin") == "C:\\"
    assert fs.get_drive_for_path("E:\\Projects\\Code") == "E:\\"


def test_filesystem_permission_aware_search_result(tmp_path: Path):
    """Verify search_files returns SearchResult tracking matched files and scanned counts."""
    fs = FilesystemService()

    # Create dummy files
    sub = tmp_path / "search_test"
    sub.mkdir()
    (sub / "alpha.txt").write_text("Alpha file", encoding="utf-8")
    (sub / "beta.log").write_text("Beta file", encoding="utf-8")
    (sub / "gamma.txt").write_text("Gamma file", encoding="utf-8")

    result = fs.search_files(query="alpha", start_dir=str(sub))
    assert isinstance(result, SearchResult)
    assert len(result) == 1
    assert result.matched_files[0]["name"] == "alpha.txt"
    assert result.total_files_scanned >= 3
    assert result.total_dirs_scanned >= 1
    assert isinstance(result.inaccessible_directories, list)


def test_filesystem_find_large_files(tmp_path: Path):
    """Verify large file detection filters files by size threshold."""
    fs = FilesystemService()

    # Create a 2 MB file and a small 10 KB file
    large_file = tmp_path / "big_data.bin"
    with open(large_file, "wb") as f:
        f.write(b"0" * (2 * 1024 * 1024))

    small_file = tmp_path / "small.txt"
    small_file.write_text("Hello", encoding="utf-8")

    # Find files >= 1 MB
    found = fs.find_large_files(start_dir=str(tmp_path), min_size_mb=1.0)
    assert len(found) >= 1
    assert found[0]["name"] == "big_data.bin"
    assert found[0]["size_mb"] >= 2.0


def test_filesystem_rename_item(tmp_path: Path):
    """Verify rename_item changes file name in place."""
    fs = FilesystemService()

    orig = tmp_path / "old_name.txt"
    orig.write_text("Rename me", encoding="utf-8")

    renamed = fs.rename_item(str(orig), "new_name.txt")
    assert Path(renamed).exists()
    assert Path(renamed).name == "new_name.txt"
    assert not orig.exists()


def test_filesystem_protected_system_path_detection():
    """Verify critical OS and HAKS directories are protected from modification."""
    fs = FilesystemService()

    # Drive root protection
    is_prot, reason = fs.is_protected_system_path("C:\\")
    assert is_prot is True
    assert "Drive Root" in reason

    # Windows root protection
    is_prot, reason = fs.is_protected_system_path("C:\\Windows")
    assert is_prot is True

    # System32 protection
    is_prot, reason = fs.is_protected_system_path("C:\\Windows\\System32")
    assert is_prot is True
    assert "System32" in reason

    # Safe user path
    is_prot, _ = fs.is_protected_system_path("C:\\Users\\Admin\\Music\\song.mp3")
    assert is_prot is False


# ==============================================================================
# 4. WindowsControlManager & AppController High-Level Integration
# ==============================================================================

def test_manager_system_wide_methods(temp_workspace: AppConfig, test_db_manager: DatabaseManager):
    """Verify WindowsControlManager exposes discovery, hardware profiling, and safety checks."""
    manager = WindowsControlManager(config=temp_workspace, db_manager=test_db_manager)

    # 1. Hardware Profile
    hw = manager.get_complete_hardware_profile()
    assert isinstance(hw, CompleteHardwareProfile)
    assert hw.cpu.physical_cores > 0

    # 2. Resource Safety Check
    check = manager.check_resource_safety("Test Build", required_disk_gb=0.01, required_ram_gb=0.01)
    assert isinstance(check, ResourceSafetyCheck)
    assert check.target_drive is not None

    # 3. Discovery Report
    report = manager.discover_system_resources(include_apps=False, include_processes=False)
    assert isinstance(report, SystemDiscoveryReport)
    assert len(report.drives) >= 1

    # 4. Multi-drive listing
    drives = manager.list_system_drives()
    assert len(drives) >= 1


def test_app_controller_system_wide_helpers(controller: AppController):
    """Verify AppController exposes system-wide resource helpers."""
    # 1. Hardware Profile
    hw = controller.get_hardware_profile()
    assert hw.cpu.logical_processors >= 1

    # 2. Operation Capacity
    safety = controller.evaluate_operation_capacity(
        operation="Test operation",
        required_disk_gb=0.01,
        required_ram_gb=0.01,
    )
    assert isinstance(safety, ResourceSafetyCheck)
    assert safety.target_drive is not None

    # 3. Drives
    drives = controller.list_system_drives()
    assert len(drives) >= 1

    # 4. Discovery
    report = controller.discover_system_resources(include_apps=False, include_processes=False)
    assert "user_profile" in report.user_directories
