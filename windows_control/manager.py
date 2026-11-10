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
from file_intelligence.service import FileIntelligenceService
from file_intelligence.models import SearchMode
from storage.database import DatabaseManager, get_db_manager
from windows_control.application_controller import ApplicationController, ApplicationActionResult
from windows_control.application_intents import ApplicationIntentParser
from windows_control.application_manager import ApplicationManager
from windows_control.application_registry import ApplicationRegistry
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
from windows_control.window_manager import WindowManager
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

        # Phase 9.1 Application Automation Subsystems
        self.app_registry = ApplicationRegistry(self.config)
        self.window_manager = WindowManager()
        self.app_controller = ApplicationController(
            registry=self.app_registry,
            window_manager=self.window_manager,
            policy=self.policy,
            config=self.config,
        )
        self.app_intents = ApplicationIntentParser(registry=self.app_registry)
        self.interpreter = CommandInterpreter(policy=self.policy, registry=self.app_registry)
        self.audit = AuditLogger(self.config, self.db)

        # Phase 9.2 Industrial File Intelligence Subsystem
        self.file_intelligence = FileIntelligenceService(config=self.config)

        # System-wide discovery and hardware-aware profiler
        self.hardware_profiler = SystemHardwareProfiler()
        self.discovery = SystemResourceDiscoveryService(self.config)

        # Link policy mode
        self.policy.set_execution_mode(ExecutionMode.FULL_CONTROL)

    # --------------------------------------------------------------------------
    # 1. Natural-Language & High-Level Execution
    # --------------------------------------------------------------------------

    async def reason_with_gemini(
        self,
        instruction: str,
        ai_manager: Optional[Any] = None,
        source: str = "chat",
        working_dir: Optional[str] = None,
    ) -> ActionPlan:
        """Use Gemini cloud reasoning to interpret a natural-language Windows control instruction into an ActionPlan."""
        # 1. Deterministic heuristic interpretation as baseline
        baseline_plan = self.interpreter.interpret(
            user_instruction=instruction,
            source=source,
            working_dir=working_dir,
        )

        if ai_manager is None:
            return baseline_plan

        # 2. Use Gemini to reason about command structure, intent, and parameters
        prompt = (
            "You are HAKS Windows Computer Control reasoning engine.\n"
            "Analyze this user instruction and determine the optimal Windows action plan.\n"
            f"Instruction: {instruction}\n\n"
            "Respond in strictly valid JSON with the following schema:\n"
            "{\n"
            '  "objective": "short summary of the action",\n'
            '  "operation_type": "command" or "launch_app" or "file" or "process" or "system",\n'
            '  "command": "executable or shell command",\n'
            '  "args": ["optional", "args"],\n'
            '  "privilege_level": "normal" or "elevated",\n'
            '  "risk_level": "low" or "sensitive" or "destructive",\n'
            '  "expected_outcome": "what happens after execution"\n'
            "}\n"
            "Do not include markdown fences or any other text, only valid JSON."
        )

        try:
            from ai.models import AIMessage
            resp = await ai_manager.generate(
                messages=[AIMessage(role="user", content=prompt)],
                engine_type="cloud",
                temperature=0.1,
                num_predict=256,
                timeout=15.0,
            )
            raw = resp.content.strip() if resp and resp.content else ""
            if raw.startswith("```"):
                lines = raw.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw = "\n".join(lines).strip()

            import json
            data = json.loads(raw)
            priv = PrivilegeLevel.ELEVATED if data.get("privilege_level") == "elevated" else PrivilegeLevel.NORMAL
            risk_val = str(data.get("risk_level", "low")).lower()
            risk = RiskLevel.LOW
            if risk_val == "sensitive":
                risk = RiskLevel.SENSITIVE
            elif risk_val == "destructive":
                risk = RiskLevel.DESTRUCTIVE
            elif risk_val == "blocked":
                risk = RiskLevel.BLOCKED

            plan = ActionPlan(
                objective=data.get("objective", baseline_plan.objective or instruction),
                operation_type=data.get("operation_type", baseline_plan.operation_type or "command"),
                command=data.get("command", baseline_plan.command or instruction),
                args=data.get("args", baseline_plan.args),
                working_dir=working_dir,
                privilege_level=priv,
                risk_level=risk,
                expected_outcome=data.get("expected_outcome", baseline_plan.expected_outcome),
                source=source,
            )
            # Evaluate against execution policy
            allowed, conf_msg = self.policy.evaluate_plan(plan)
            if not allowed:
                plan.risk_level = RiskLevel.BLOCKED
                plan.confirmation_required = True
                plan.confirmation_message = conf_msg
            elif conf_msg:
                plan.confirmation_required = True
                plan.confirmation_message = conf_msg
            logger.info("Gemini reasoned Windows ActionPlan: '%s' [%s]", plan.objective, plan.command)
            return plan
        except Exception as exc:
            logger.warning("Gemini Windows Control reasoning failed: %s. Using baseline plan.", exc)
            return baseline_plan

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
        # Check special operation types (e.g. app control, storage analysis) before shell
        if plan.operation_type in ("launch_app", "switch_app", "minimize_app", "maximize_app", "restore_app", "close_app", "app_control"):
            action = plan.operation_type
            app_target = plan.command or plan.target
            if action == "launch_app":
                res_obj = self.app_controller.launch_application(app_target, plan.args)
            elif action == "switch_app":
                res_obj = self.app_controller.switch_to_application(app_target)
            elif action == "minimize_app":
                res_obj = self.app_controller.minimize_application(app_target)
            elif action == "maximize_app":
                res_obj = self.app_controller.maximize_application(app_target)
            elif action == "restore_app":
                res_obj = self.app_controller.restore_application(app_target)
            elif action == "close_app":
                res_obj = self.app_controller.close_application(app_target)
            else:
                res_obj = self.app_controller.launch_application(app_target, plan.args)

            res = CommandResult(
                action_id=plan.id,
                command=f"{action} {app_target}",
                exit_code=0 if res_obj.success else 1,
                stdout=res_obj.message if res_obj.success else "",
                stderr=res_obj.message if not res_obj.success else "",
                status=ExecutionStatus.COMPLETED if res_obj.success else ExecutionStatus.FAILED,
                pid=res_obj.pid,
                metadata={"action": res_obj.action, "app_id": res_obj.app_id, "window_count": res_obj.window_count},
            )
            self._log_plan_result(plan, res, confirmation_status="auto_approved")
            return res

        # File Intelligence Operations (Phase 9.2)
        if plan.operation_type in ("file_search", "search_files", "semantic_search", "content_search", "hybrid_search"):
            search_query = plan.command or plan.target
            mode = SearchMode.HYBRID
            if plan.operation_type == "semantic_search":
                mode = SearchMode.DENSE
            elif plan.args and plan.args[0] in [m.value for m in SearchMode]:
                mode = SearchMode(plan.args[0])

            scope = plan.working_dir if (plan.working_dir and Path(plan.working_dir).exists()) else None
            results = self.file_intelligence.search(
                natural_query=search_query,
                search_mode=mode,
                top_k=10,
                rerank=True,
                directory_scope=scope,
            )
            if results:
                summary_lines = [f"Found {len(results)} file(s) matching '{search_query}':"]
                for i, r in enumerate(results[:5], 1):
                    summary_lines.append(f"{i}. {r.file_name} ({r.file_path}) - Score: {r.score:.3f}")
                stdout_str = "\n".join(summary_lines)
            else:
                stdout_str = f"No files found matching '{search_query}'."

            res = CommandResult(
                action_id=plan.id,
                command=f"search {search_query}",
                exit_code=0,
                stdout=stdout_str,
                status=ExecutionStatus.COMPLETED,
                metadata={"results": [r.model_dump() for r in results]},
            )
            self._log_plan_result(plan, res, confirmation_status="auto_approved")
            return res

        if plan.operation_type == "find_duplicates":
            target_dir = plan.target or plan.working_dir or os.getcwd()
            groups, benchmarks = self.file_intelligence.find_duplicates(
                root_dir=target_dir,
                use_blake3_benchmark=True,
            )
            total_wasted = sum(g.wasted_bytes for g in groups)
            wasted_mb = total_wasted / (1024 * 1024)
            if groups:
                lines = [f"Detected {len(groups)} duplicate group(s) with {wasted_mb:.2f} MB potential space savings:"]
                for idx, g in enumerate(groups[:5], 1):
                    lines.append(f"Group {idx} ({g.size_bytes / 1024:.1f} KB, hash: {g.hash_value[:10]}...):")
                    for f in g.files:
                        lines.append(f"  - {f}")
                lines.append("Note: Duplicate files are never deleted automatically. Please specify files to move to Recycle Bin.")
                stdout_str = "\n".join(lines)
            else:
                stdout_str = f"No duplicate files detected in {target_dir}."

            res = CommandResult(
                action_id=plan.id,
                command=f"find_duplicates {target_dir}",
                exit_code=0,
                stdout=stdout_str,
                status=ExecutionStatus.COMPLETED,
                metadata={
                    "duplicate_groups": [g.model_dump() for g in groups],
                    "benchmark": benchmarks,
                },
            )
            self._log_plan_result(plan, res, confirmation_status="auto_approved")
            return res

        if plan.operation_type in ("storage_analysis", "disk_usage"):
            target_dir = plan.target or plan.working_dir or os.getcwd()
            summary = self.file_intelligence.analyze_disk_usage(root_dir=target_dir)
            lines = [
                f"Storage Analysis for {summary.root_path}:",
                f"- Total Size: {summary.total_size_mb:.2f} MB ({summary.total_files} files, {summary.total_directories} directories)",
            ]
            if summary.extension_counts:
                ext_str = ", ".join(f"{k}: {v}" for k, v in list(summary.extension_counts.items())[:6])
                lines.append(f"- File Types: {ext_str}")
            if summary.largest_files:
                lines.append("- Largest Files:")
                for lf in summary.largest_files[:5]:
                    lines.append(f"  * {lf.file_name} ({lf.size_mb:.2f} MB) - {lf.file_path}")
            stdout_str = "\n".join(lines)

            res = CommandResult(
                action_id=plan.id,
                command=f"storage_analysis {target_dir}",
                exit_code=0,
                stdout=stdout_str,
                status=ExecutionStatus.COMPLETED,
                metadata={"disk_usage": summary.model_dump()},
            )
            self._log_plan_result(plan, res, confirmation_status="auto_approved")
            return res

        if plan.operation_type in ("recycle_bin", "recycle_file", "delete_file"):
            target_file = plan.target or plan.command
            if plan.confirmation_required and not bypass_confirmation:
                res = CommandResult(
                    action_id=plan.id,
                    command=f"recycle {target_file}",
                    exit_code=1,
                    status=ExecutionStatus.CONFIRMATION_REQUIRED,
                    stderr=plan.confirmation_message or f"Confirmation required to recycle {target_file}",
                )
                self._log_plan_result(plan, res, confirmation_status="pending_confirmation")
                return res

            success, msg = self.file_intelligence.send_to_recycle_bin(target_file)
            res = CommandResult(
                action_id=plan.id,
                command=f"recycle {target_file}",
                exit_code=0 if success else 1,
                stdout=msg if success else "",
                stderr="" if success else msg,
                status=ExecutionStatus.COMPLETED if success else ExecutionStatus.FAILED,
            )
            self._log_plan_result(plan, res, confirmation_status="user_approved" if bypass_confirmation else "auto_approved")
            return res

        if plan.operation_type in ("copy_file", "move_file", "rename_file"):
            target_file = plan.target
            dest = plan.args[0] if plan.args else plan.command
            if plan.operation_type == "copy_file":
                success, msg = self.file_intelligence.copy_file(target_file, dest)
            elif plan.operation_type == "move_file":
                success, msg = self.file_intelligence.move_file(target_file, dest)
            else:
                success, msg = self.file_intelligence.rename_file(target_file, dest)

            res = CommandResult(
                action_id=plan.id,
                command=f"{plan.operation_type} {target_file} -> {dest}",
                exit_code=0 if success else 1,
                stdout=msg if success else "",
                stderr="" if success else msg,
                status=ExecutionStatus.COMPLETED if success else ExecutionStatus.FAILED,
            )
            self._log_plan_result(plan, res, confirmation_status="auto_approved")
            return res

        if plan.operation_type == "index_directory":
            target_dir = plan.target or plan.working_dir or os.getcwd()
            stats = self.file_intelligence.index_directory(target_dir)
            stdout_str = (
                f"Directory indexing complete for {target_dir}:\n"
                f"- Total files scanned: {stats.get('total_files', 0)}\n"
                f"- Files indexed: {stats.get('indexed_files', 0)}\n"
                f"- Chunks indexed: {stats.get('indexed_chunks', 0)}\n"
                f"- Files skipped: {stats.get('skipped_files', 0)}"
            )
            res = CommandResult(
                action_id=plan.id,
                command=f"index_directory {target_dir}",
                exit_code=0,
                stdout=stdout_str,
                status=ExecutionStatus.COMPLETED,
                metadata={"index_stats": stats},
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
            "ram_used_gb": round(hw.ram_total_gb - hw.ram_available_gb, 2),
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
