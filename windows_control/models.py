"""Data models and enums for HAKS Windows Computer Control System.

Defines execution modes, risk classifications, privilege levels,
action plans, execution results, system hardware diagnostics,
and audit log structures.
"""

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from pydantic import BaseModel, Field


class ExecutionMode(str, Enum):
    """Computer control execution mode."""
    FULL_CONTROL = "full_control"          # Auto-executes low-risk actions; requests UAC/confirmation for sensitive/destructive
    SAFE_CONFIRMATION = "safe_confirmation"# Requires explicit confirmation for all modifying actions
    RESTRICTED = "restricted"              # Read-only inspection; execution blocked


class PrivilegeLevel(str, Enum):
    """Required OS privilege level."""
    NORMAL = "normal"      # Standard user privileges (asInvoker)
    ELEVATED = "elevated"  # Administrator privileges via Windows UAC


class RiskLevel(str, Enum):
    """Four-tier risk classification for Windows operations."""
    LOW = "low"                  # Read-only, inspection, app launch, user script in project
    SENSITIVE = "sensitive"      # Software install/uninstall, service changes, system config, downloaded scripts
    DESTRUCTIVE = "destructive"  # Permanent deletion, mass deletion, formatting, partition change, reboot/shutdown
    BLOCKED = "blocked"          # Strictly prohibited (silent Defender disable, firewall wipe, system file corruption)


class ExecutionStatus(str, Enum):
    """Lifecycle status of a computer control action."""
    PENDING = "pending"
    CONFIRMATION_REQUIRED = "confirmation_required"
    ELEVATION_REQUIRED = "elevation_required"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class ActionPlan(BaseModel):
    """Structured execution plan generated from user instruction."""
    id: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")[:18])
    objective: str
    target: str = ""
    operation_type: str = "command"  # command, file, process, software, system, gui
    command: str = ""
    args: list[str] = Field(default_factory=list)
    working_dir: Optional[str] = None
    privilege_level: PrivilegeLevel = PrivilegeLevel.NORMAL
    risk_level: RiskLevel = RiskLevel.LOW
    confirmation_required: bool = False
    confirmation_message: Optional[str] = None
    expected_outcome: str = ""
    timeout_seconds: int = 120
    status: ExecutionStatus = ExecutionStatus.PENDING
    source: str = "chat"  # chat, voice, ui, script
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))

    @property
    def summary(self) -> str:
        return self.objective

    @property
    def intent(self) -> str:
        return self.operation_type

    @property
    def is_actionable(self) -> bool:
        return self.operation_type not in ("", "conversation")


class CommandResult(BaseModel):
    """Output and execution metrics from a completed command or tool action."""
    action_id: str
    command: str
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    duration_ms: float = 0.0
    status: ExecutionStatus = ExecutionStatus.COMPLETED
    error: Optional[str] = None
    pid: Optional[int] = None
    elevated: bool = False
    verified: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Return True if command finished with exit code 0."""
        return self.exit_code == 0 and self.status == ExecutionStatus.COMPLETED


class ProcessInfo(BaseModel):
    """Running Windows process metadata."""
    pid: int
    name: str
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    status: str = "running"
    username: Optional[str] = None
    create_time: float = 0.0
    is_critical: bool = False
    exe: Optional[str] = None


class ServiceInfo(BaseModel):
    """Windows system service metadata."""
    name: str
    display_name: str
    status: str  # running, stopped, paused, start_pending, stop_pending
    start_type: str = "unknown"  # auto, manual, disabled
    pid: Optional[int] = None
    description: str = ""


class DriveInfo(BaseModel):
    """Storage drive capacity and health details."""
    device: str
    mountpoint: str
    fstype: str
    total_gb: float
    used_gb: float
    free_gb: float
    percent_used: float
    drive_type: str = "fixed"  # fixed, removable, cdrom, network, ramdisk, unknown
    volume_name: str = ""
    is_ready: bool = True

    @property
    def drive_letter(self) -> str:
        """Return the drive letter / mountpoint."""
        return self.mountpoint

    @property
    def total_bytes(self) -> int:
        return int(self.total_gb * (1024**3))

    @property
    def free_bytes(self) -> int:
        return int(self.free_gb * (1024**3))


class SearchResult(BaseModel):
    """Permission-aware filesystem search result tracking accessible and inaccessible directories."""
    matched_files: list[dict[str, Any]] = Field(default_factory=list)
    inaccessible_directories: list[str] = Field(default_factory=list)
    total_files_scanned: int = 0
    total_dirs_scanned: int = 0
    search_duration_seconds: float = 0.0
    truncated: bool = False
    query: str = ""
    root_directory: str = ""

    def __iter__(self):
        return iter(self.matched_files)

    def __getitem__(self, item):
        return self.matched_files[item]

    def __len__(self):
        return len(self.matched_files)


class CPUInfo(BaseModel):
    """Detailed CPU specification and real-time load."""
    model: str
    physical_cores: int
    logical_processors: int
    architecture: str
    current_percent: float = 0.0
    frequency_mhz: float = 0.0


class RAMInfo(BaseModel):
    """Physical and virtual system memory statistics."""
    total_gb: float
    available_gb: float
    used_gb: float
    percent_used: float
    swap_total_gb: float = 0.0
    swap_free_gb: float = 0.0


class GPUInfo(BaseModel):
    """Graphics card hardware details and real compute acceleration capability evaluation."""
    name: str
    driver_version: str = "Unknown"
    adapter_ram_mb: float = 0.0
    status: str = "OK"
    supports_cuda: bool = False
    supports_rocm: bool = False
    supports_directml: bool = False
    supports_llm_acceleration: bool = False
    acceleration_notes: str = ""


class DisplayInfo(BaseModel):
    """Active display monitor metrics, resolution, and scaling."""
    width: int
    height: int
    dpi: int = 96
    scaling_percent: float = 100.0
    refresh_rate_hz: int = 60
    monitors_count: int = 1
    monitors: list[dict[str, Any]] = Field(default_factory=list)


class AudioDeviceInfo(BaseModel):
    """Audio input (microphones) and output (speakers) hardware inventory."""
    input_devices: list[dict[str, Any]] = Field(default_factory=list)
    output_devices: list[dict[str, Any]] = Field(default_factory=list)
    default_input: Optional[str] = None
    default_output: Optional[str] = None


class StorageSummary(BaseModel):
    """Cross-drive storage capacity overview."""
    total_gb: float
    free_gb: float
    used_gb: float
    percent_used: float
    drives: list[DriveInfo] = Field(default_factory=list)


class CompleteHardwareProfile(BaseModel):
    """Comprehensive hardware diagnostic profile collected by SystemHardwareProfiler."""
    cpu: CPUInfo
    ram: RAMInfo
    gpus: list[GPUInfo] = Field(default_factory=list)
    display: DisplayInfo
    audio: AudioDeviceInfo
    storage: StorageSummary
    os_name: str
    os_version: str
    os_build: str
    os_architecture: str
    is_admin: bool = False
    computer_name: str = ""
    current_user: str = ""
    profiled_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))


class ResourceSafetyCheck(BaseModel):
    """Pre-flight resource evaluation before consequential downloads, model pulls, or builds."""
    is_safe: bool
    operation: str
    target_drive: Optional[str] = None
    required_disk_gb: float = 0.0
    available_disk_gb: float = 0.0
    required_ram_gb: float = 0.0
    available_ram_gb: float = 0.0
    gpu_required: bool = False
    gpu_compatible: bool = True
    warnings: list[str] = Field(default_factory=list)
    insufficient_resources: list[str] = Field(default_factory=list)
    requires_user_approval: bool = False
    recommendation: str = ""


class SystemDiscoveryReport(BaseModel):
    """System-wide Windows resource discovery report."""
    discovered_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
    drives: list[DriveInfo] = Field(default_factory=list)
    user_directories: dict[str, str] = Field(default_factory=dict)
    installed_applications_count: int = 0
    installed_applications: list["InstalledAppInfo"] = Field(default_factory=list)
    python_installations: list[dict[str, Any]] = Field(default_factory=list)
    git_ecosystem: dict[str, Any] = Field(default_factory=dict)
    development_tools: list[dict[str, Any]] = Field(default_factory=list)
    running_processes_count: int = 0
    running_processes: list["ProcessInfo"] = Field(default_factory=list)
    windows_services_count: int = 0
    windows_services: list["ServiceInfo"] = Field(default_factory=list)
    network: Optional["NetworkInfo"] = None
    system_configuration: dict[str, Any] = Field(default_factory=dict)


class StorageReport(BaseModel):
    """Detailed filesystem storage analysis report."""
    target_path: str
    total_size_mb: float
    file_count: int
    dir_count: int
    large_files: list[dict[str, Any]] = Field(default_factory=list)
    extension_breakdown: dict[str, float] = Field(default_factory=dict)  # ext -> size_mb


class NetworkInfo(BaseModel):
    """Network configuration and connectivity diagnostics."""
    hostname: str
    interfaces: list[dict[str, Any]] = Field(default_factory=list)
    default_gateway: Optional[str] = None
    dns_servers: list[str] = Field(default_factory=list)
    internet_connected: bool = True
    ping_latency_ms: Optional[float] = None


class SystemHardwareInfo(BaseModel):
    """Host machine hardware and operating system specification."""
    os_name: str
    os_version: str
    os_build: str
    os_architecture: str
    cpu_model: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    cpu_usage_percent: float
    ram_total_gb: float
    ram_available_gb: float
    ram_used_percent: float
    gpu_devices: list[str] = Field(default_factory=list)
    system_uptime_seconds: float
    is_admin: bool = False
    computer_name: str = ""
    current_user: str = ""


class InstalledAppInfo(BaseModel):
    """Installed Windows desktop application details."""
    name: str
    version: str = "Unknown"
    publisher: str = "Unknown"
    install_location: Optional[str] = None
    uninstall_string: Optional[str] = None
    source: str = "Registry"  # Registry, winget, start_menu


class DownloadItem(BaseModel):
    """Tracked software or file download."""
    id: str
    url: str
    destination: str
    filename: str
    total_bytes: int = 0
    downloaded_bytes: int = 0
    progress_percent: float = 0.0
    status: str = "pending"  # pending, downloading, completed, failed, cancelled, paused
    sha256_expected: Optional[str] = None
    sha256_actual: Optional[str] = None
    checksum_verified: bool = False
    error: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))


class AuditLogRecord(BaseModel):
    """Audit log entry for a computer control action."""
    id: Optional[int] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
    source: str = "chat"
    instruction: str
    plan_id: str
    operation_type: str
    executable: str
    args: str
    working_dir: str
    privilege_level: str
    risk_level: str
    confirmation_status: str  # auto_approved, user_approved, rejected, cancelled
    exit_code: int = 0
    status: str = "completed"
    duration_ms: float = 0.0
    error: Optional[str] = None
