"""HAKS Windows Computer Control and Administrative Automation Package.

Provides comprehensive, genuine control over Windows commands, applications,
files, processes, services, network, software installations, GUI automation,
system-wide resource discovery, and hardware-aware capacity evaluation
under full control execution mode with native UAC elevation and immutable audit trails.
"""

from windows_control.application_controller import ApplicationActionResult, ApplicationController
from windows_control.application_intents import AppActionType, ApplicationIntent, ApplicationIntentParser
from windows_control.application_manager import ApplicationManager
from windows_control.application_registry import ApplicationEntry, ApplicationRegistry
from windows_control.audit_logger import AuditLogger
from windows_control.cmd_service import CMDService
from windows_control.command_executor import CommandExecutor
from windows_control.command_interpreter import CommandInterpreter
from windows_control.download_manager import DownloadManager
from windows_control.elevation_manager import ElevationManager
from windows_control.execution_policy import ExecutionPolicy
from windows_control.filesystem_service import FilesystemService
from windows_control.gui_automation import GUIAutomationService
from windows_control.hardware_profiler import SystemHardwareProfiler
from windows_control.manager import WindowsControlManager
from windows_control.window_manager import WindowInfo, WindowManager
from windows_control.models import (
    ActionPlan,
    AudioDeviceInfo,
    AuditLogRecord,
    CommandResult,
    CompleteHardwareProfile,
    CPUInfo,
    DisplayInfo,
    DownloadItem,
    DriveInfo,
    ExecutionMode,
    ExecutionStatus,
    GPUInfo,
    InstalledAppInfo,
    NetworkInfo,
    PrivilegeLevel,
    ProcessInfo,
    RAMInfo,
    ResourceSafetyCheck,
    RiskLevel,
    SearchResult,
    ServiceInfo,
    StorageReport,
    StorageSummary,
    SystemDiscoveryReport,
    SystemHardwareInfo,
)
from windows_control.powershell_service import PowerShellService
from windows_control.privilege_detector import PrivilegeDetector
from windows_control.resource_discovery import SystemResourceDiscoveryService
from windows_control.software_manager import SoftwareManager
from windows_control.system_manager import SystemManager

__all__ = [
    "ActionPlan",
    "AppActionType",
    "ApplicationActionResult",
    "ApplicationController",
    "ApplicationEntry",
    "ApplicationIntent",
    "ApplicationIntentParser",
    "ApplicationManager",
    "ApplicationRegistry",
    "AudioDeviceInfo",
    "AuditLogRecord",
    "AuditLogger",
    "CMDService",
    "CommandExecutor",
    "CommandInterpreter",
    "CommandResult",
    "CompleteHardwareProfile",
    "CPUInfo",
    "DisplayInfo",
    "DownloadItem",
    "DownloadManager",
    "DriveInfo",
    "ElevationManager",
    "ExecutionMode",
    "ExecutionPolicy",
    "ExecutionStatus",
    "FilesystemService",
    "GUIAutomationService",
    "GPUInfo",
    "InstalledAppInfo",
    "NetworkInfo",
    "PowerShellService",
    "PrivilegeDetector",
    "PrivilegeLevel",
    "ProcessInfo",
    "RAMInfo",
    "ResourceSafetyCheck",
    "RiskLevel",
    "SearchResult",
    "ServiceInfo",
    "SoftwareManager",
    "StorageReport",
    "StorageSummary",
    "SystemDiscoveryReport",
    "SystemHardwareInfo",
    "SystemHardwareProfiler",
    "SystemManager",
    "SystemResourceDiscoveryService",
    "WindowInfo",
    "WindowManager",
    "WindowsControlManager",
]
