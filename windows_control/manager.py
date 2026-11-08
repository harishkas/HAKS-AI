"""Windows Computer Control Manager for HAKS.

Unifies all 15 computer control subsystems:
- Command Executor (CMD, PowerShell, Direct Executables)
- Windows UAC Elevation Manager
- Filesystem & Storage Analysis Service (Multi-drive, permission-aware)
- Process & Application Managers
- Software & Download Managers
- System Diagnostics & Network Troubleshooting
- Desktop GUI Automation Service
- System Hardware Profiler (CPU, RAM, GPU, Display, Audio, Storage)
- System-Wide Resource Discovery Service (Drives, User Dirs, Apps, Pythons, Git, Tools)
- Natural-Language Command Interpreter
- Immutable Redacted Audit Logger
"""

import asyncio
import logging
import os
from pathlib import Path
import time
from typing import Any, Callable, Optional

from config.settings import AppConfig, get_settings
from storage.database import DatabaseManager, get_db_manager
from windows_control.application_manager import ApplicationManager
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
from windows_control.models import (
    ActionPlan,
    CommandResult,
    CompleteHardwareProfile,
    DriveInfo,
    ExecutionMode,
    ExecutionStatus,
    PrivilegeLevel,
    ResourceSafetyCheck,
    RiskLevel,
    SearchResult,
    SystemDiscoveryReport,
    SystemHardwareInfo,
)
from windows_control.powershell_service import PowerShellService
from windows_control.privilege_detector import PrivilegeDetector
from windows_control.process_manager import ProcessManager
from windows_control.resource_discovery import SystemResourceDiscoveryService
from windows_control.software_manager import SoftwareManager
from windows_control.system_manager import SystemManager

logger = logging.getLogger("haks.windows_control.manager")


class WindowsControlManager:
    """Central coordinator for Windows Computer Control, Resource Discovery, and Hardware Diagnostics."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        db_manager: Optional[DatabaseManager] = None,
    ) -> None:
        self.config = config or get_settings()
        self.db = db_manager or get_db_manager(self.config)

        # Core subsystems
        self.policy = ExecutionPolicy(self.config)
        self.privilege_detector = PrivilegeDetector()
        self.elevation_manager = ElevationManager()
        self.cmd_service = CMDService()
        self.powershell_service = PowerShellService()
        self.executor = CommandExecutor(self.config)
        self.filesystem = FilesystemService(self.config)
        self.process = ProcessManager()
        self.application = ApplicationManager()
        self.download = DownloadManager(self.config)
        self.software = SoftwareManager()
        self.system = SystemManager()
        self.gui = GUIAutomationService()
        self.interpreter = CommandInterpreter(self.policy)
        self.audit = AuditLogger(self.config, self.db)

        # System-wide discovery and hardware-aware profiler
        self.hardware_profiler = SystemHardwareProfiler()
        self.discovery = SystemResourceDiscoveryService(self.config)

        # Link policy mode
        self.policy.set_execution_mode(ExecutionMode.FULL_CONTROL)

    # --------------------------------------------------------------------------
    # 1. Natural-Language & High-Level Execution
    # --------------------------------------------------------------------------

    async def execute_instruction(
        self,
        instruction: str,
        source: str = "chat",
        working_dir: Optional[str] = None,
        bypass_confirmation: bool = False,
        output_callback: Optional[Callable[[str], None]] = None,
        error_callback: Optional[Callable[[str], None]] = None,
    ) -> CommandResult:
        """Interpret natural-language or voice request into ActionPlan, evaluate, and execute."""
        logger.info("Interpreting instruction [%s]: '%s'", source, instruction)

        # 1. Interpret into structured ActionPlan
        plan = self.interpreter.interpret(
            user_instruction=instruction,
            source=source,
            working_dir=working_dir,
        )

        # 2. Execute plan
        return await self.execute_plan(
            plan=plan,
            bypass_confirmation=bypass_confirmation,
            output_callback=output_callback,
            error_callback=error_callback,
        )

    async def execute_plan(
        self,
        plan: ActionPlan,
        bypass_confirmation: bool = False,
        output_callback: Optional[Callable[[str], None]] = None,
        error_callback: Optional[Callable[[str], None]] = None,
    ) -> CommandResult:
        """Execute a typed ActionPlan, handle confirmations, and log to audit trail."""
        # Check special operation types (e.g. app launch, storage analysis) before shell
        if plan.operation_type == "launch_app":
            success, msg = self.application.launch_application(plan.command, plan.args)
            res = CommandResult(
                action_id=plan.id,
                command=plan.command,
                exit_code=0 if success else 1,
                stdout=msg if success else "",
                stderr=msg if not success else "",
                status=ExecutionStatus.COMPLETED if success else ExecutionStatus.FAILED,
            )
            self._log_plan_result(plan, res, confirmation_status="auto_approved")
            return res

        # Run through executor (policy evaluation, elevation, and execution)
        result = await self.executor.execute_plan(
            plan=plan,
            bypass_confirmation=bypass_confirmation,
            output_callback=output_callback,
            error_callback=error_callback,
        )

        # Determine confirmation status for audit
        conf_status = "auto_approved"
        if plan.confirmation_required:
            conf_status = "user_approved" if bypass_confirmation else "pending_confirmation"
        if result.status == ExecutionStatus.REJECTED:
            conf_status = "rejected"
        elif result.status == ExecutionStatus.CANCELLED:
            conf_status = "cancelled"

        self._log_plan_result(plan, result, confirmation_status=conf_status)
        return result

    def resolve_confirmation(self, plan_id: str, approved: bool) -> Optional[ActionPlan]:
        """User responded to confirmation dialog for an ActionPlan."""
        return self.executor.resolve_confirmation(plan_id, approved)

    # --------------------------------------------------------------------------
    # 2. System Hardware Profiling & Resource Safety Pre-Flight
    # --------------------------------------------------------------------------

    def get_complete_hardware_profile(self, refresh: bool = False) -> CompleteHardwareProfile:
        """Collect comprehensive diagnostic profile of CPU, RAM, GPUs, Display, Audio, Storage, and OS."""
        return self.hardware_profiler.get_complete_profile(refresh=refresh)

    def check_resource_safety(
        self,
        operation: str,
        target_path: Optional[str] = None,
        required_disk_gb: float = 0.0,
        required_ram_gb: float = 0.0,
        requires_gpu: bool = False,
    ) -> ResourceSafetyCheck:
        """Evaluate resource capacity before expensive operations (large downloads, model pulls, builds)."""
        return self.hardware_profiler.evaluate_operation_capacity(
            operation=operation,
            target_path=target_path,
            required_disk_gb=required_disk_gb,
            required_ram_gb=required_ram_gb,
            requires_gpu=requires_gpu,
        )

    # --------------------------------------------------------------------------
    # 3. System-Wide Resource Discovery & Cross-Drive Scope
    # --------------------------------------------------------------------------

    def discover_system_resources(
        self,
        include_apps: bool = True,
        include_processes: bool = True,
    ) -> SystemDiscoveryReport:
        """Scan system-wide Windows resources across all drives, registry, environment, and services."""
        return self.discovery.discover_all(
            include_apps=include_apps,
            include_processes=include_processes,
        )

    def search_system_files(
        self,
        query: str,
        start_dir: Optional[str] = None,
        extension: Optional[str] = None,
        max_results: int = 100,
    ) -> SearchResult:
        """Search files anywhere on authorized drives with permission-aware tracking."""
        return self.filesystem.search_files(
            query=query,
            start_dir=start_dir,
            extension=extension,
            max_results=max_results,
        )

    def find_large_files(
        self,
        start_dir: Optional[str] = None,
        min_size_mb: float = 50.0,
        max_results: int = 50,
    ) -> list[dict[str, Any]]:
        """Identify large files consuming storage on any drive."""
        return self.filesystem.find_large_files(
            start_dir=start_dir,
            min_size_mb=min_size_mb,
            max_results=max_results,
        )

    def list_system_drives(self) -> list[DriveInfo]:
        """List all detected logical storage drives across the Windows host."""
        return self.filesystem.list_drives()

    # --------------------------------------------------------------------------
    # 4. Dashboard & System Metrics
    # --------------------------------------------------------------------------

    def get_dashboard_metrics(self) -> dict[str, Any]:
        """Gather real-time system metrics for the Control Center UI."""
        hw = self.system.get_system_hardware_info()
        drives = self.filesystem.list_drives()
        recent_logs = self.audit.query_logs(limit=5)
        active_dl = [d for d in self.download.get_downloads() if d.status == "downloading"]

        return {
            "mode": self.policy.mode.value,
            "is_admin": hw.is_admin,
            "current_user": hw.current_user,
            "computer_name": hw.computer_name,
            "os_name": hw.os_name,
            "os_build": hw.os_build,
            "cpu_model": hw.cpu_model,
            "cpu_usage_percent": hw.cpu_usage_percent,
            "ram_total_gb": hw.ram_total_gb,
            "ram_available_gb": hw.ram_available_gb,
            "ram_percent": hw.ram_used_percent,
            "uptime_seconds": hw.system_uptime_seconds,
            "drives": [d.model_dump() for d in drives],
            "active_tasks_count": len(self.executor.cmd_service._active_processes) + len(self.executor.powershell_service._active_processes),
            "active_downloads_count": len(active_dl),
            "recent_audit_count": len(recent_logs),
        }

    # --------------------------------------------------------------------------
    # 5. Emergency Stop & Safety
    # --------------------------------------------------------------------------

    def emergency_stop(self) -> int:
        """Instantly kill all child processes, scripts, and active downloads."""
        logger.warning("EMERGENCY STOP CALLED on WindowsControlManager.")
        killed_procs = self.executor.emergency_stop()
        cancelled_dls = self.download.cancel_all()
        return killed_procs + cancelled_dls

    # --------------------------------------------------------------------------
    # Internal Helpers
    # --------------------------------------------------------------------------

    def _log_plan_result(self, plan: ActionPlan, result: CommandResult, confirmation_status: str) -> None:
        """Record execution outcome in the audit database."""
        try:
            self.audit.log_execution(
                source=plan.source,
                instruction=plan.objective,
                plan_id=plan.id,
                operation_type=plan.operation_type,
                executable=plan.command.split()[0] if plan.command else "",
                args=" ".join(plan.command.split()[1:]) if plan.command else "",
                working_dir=plan.working_dir or os.getcwd(),
                privilege_level=plan.privilege_level.value,
                risk_level=plan.risk_level.value,
                confirmation_status=confirmation_status,
                exit_code=result.exit_code,
                status=result.status.value,
                duration_ms=result.duration_ms,
                error=result.stderr if result.exit_code != 0 else None,
            )
        except Exception as exc:
            logger.warning("Could not log action to audit trail: %s", exc)
