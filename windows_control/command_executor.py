"""Central Command Execution Engine for HAKS Windows Computer Control.

Routes commands to CMD, PowerShell, direct executable arrays, or the Windows
UAC Elevation Manager based on execution policy and privilege requirements.
"""

import asyncio
import logging
import os
from pathlib import Path
import subprocess
import time
from typing import Callable, Optional

from config.settings import AppConfig, get_settings
from windows_control.cmd_service import CMDService
from windows_control.elevation_manager import ElevationManager
from windows_control.execution_policy import ExecutionPolicy
from windows_control.models import (
    ActionPlan,
    CommandResult,
    ExecutionStatus,
    PrivilegeLevel,
    RiskLevel,
)
from windows_control.powershell_service import PowerShellService
from windows_control.privilege_detector import PrivilegeDetector

logger = logging.getLogger("haks.windows_control.executor")


class CommandExecutor:
    """Central coordinator for command execution across shells and elevation boundaries."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or get_settings()
        self.policy = ExecutionPolicy(self.config)
        self.cmd_service = CMDService()
        self.powershell_service = PowerShellService()
        self.elevation_manager = ElevationManager()

        self._pending_confirmations: dict[str, ActionPlan] = {}

    async def execute_plan(
        self,
        plan: ActionPlan,
        bypass_confirmation: bool = False,
        output_callback: Optional[Callable[[str], None]] = None,
        error_callback: Optional[Callable[[str], None]] = None,
    ) -> CommandResult:
        """Evaluate policy, check elevation, and execute the planned action."""
        # 1. Policy Evaluation
        allowed, reason = self.policy.evaluate_plan(plan)
        if not allowed:
            logger.warning("Action [%s] rejected by policy: %s", plan.id, reason)
            return CommandResult(
                action_id=plan.id,
                command=plan.command,
                exit_code=1,
                status=ExecutionStatus.FAILED,
                stderr=reason or "Operation blocked by security policy.",
                verified=False,
            )

        # 2. Check Confirmation requirement
        if plan.confirmation_required and not bypass_confirmation:
            logger.info("Action [%s] requires user confirmation: %s", plan.id, plan.confirmation_message)
            plan.status = ExecutionStatus.CONFIRMATION_REQUIRED
            self._pending_confirmations[plan.id] = plan
            return CommandResult(
                action_id=plan.id,
                command=plan.command,
                exit_code=0,
                status=ExecutionStatus.CONFIRMATION_REQUIRED,
                stdout=f"Confirmation required before executing: {plan.objective}",
                stderr="",
                metadata={"confirmation_message": plan.confirmation_message, "plan": plan.model_dump()},
            )

        # 3. Check Privilege & Elevation Requirement
        elevation_needed = (
            plan.privilege_level == PrivilegeLevel.ELEVATED
            or PrivilegeDetector.requires_elevation(plan.command, plan.operation_type, plan.target) == PrivilegeLevel.ELEVATED
        )

        plan.status = ExecutionStatus.RUNNING

        # 4. Route Execution
        if elevation_needed and not PrivilegeDetector.is_current_process_admin():
            logger.info("Routing action [%s] through Windows UAC Elevation Manager...", plan.id)
            result = await self.elevation_manager.execute_elevated(
                command=plan.command,
                working_dir=plan.working_dir,
                timeout_seconds=plan.timeout_seconds,
                action_id=plan.id,
            )
            return result

        # Non-elevated execution via chosen shell or direct runner
        cmd = plan.command.strip()
        working_dir = plan.working_dir or os.getcwd()

        # Decide shell: use CMD if explicit cmd or batch, otherwise PowerShell or direct
        if plan.operation_type == "direct_exec" and plan.args:
            return await self._execute_direct_binary(
                plan.command, plan.args, working_dir, plan.timeout_seconds, plan.id
            )

        is_cmd = (
            cmd.lower().startswith("cmd")
            or cmd.lower().startswith("dir ")
            or cmd.lower().endswith(".bat")
            or cmd.lower().endswith(".cmd")
            or self.config.control.default_shell == "cmd"
        )

        if is_cmd:
            return await self.cmd_service.execute(
                command=cmd,
                working_dir=working_dir,
                timeout_seconds=plan.timeout_seconds,
                action_id=plan.id,
                output_callback=output_callback,
                error_callback=error_callback,
            )
        else:
            return await self.powershell_service.execute(
                command=cmd,
                working_dir=working_dir,
                timeout_seconds=plan.timeout_seconds,
                action_id=plan.id,
                output_callback=output_callback,
                error_callback=error_callback,
            )

    async def _execute_direct_binary(
        self,
        executable: str,
        args: list[str],
        working_dir: str,
        timeout_seconds: int,
        action_id: str,
    ) -> CommandResult:
        """Launch an executable directly using argument arrays without shell interpolation."""
        start_time = time.perf_counter()
        full_cmd = [executable] + args
        try:
            proc = await asyncio.create_subprocess_exec(
                *full_cmd,
                cwd=working_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_seconds
            )
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            stdout_str = stdout_bytes.decode("utf-8", errors="replace")
            stderr_str = stderr_bytes.decode("utf-8", errors="replace")
            exit_code = proc.returncode if proc.returncode is not None else 0

            return CommandResult(
                action_id=action_id,
                command=" ".join(full_cmd),
                exit_code=exit_code,
                stdout=stdout_str,
                stderr=stderr_str,
                duration_ms=duration_ms,
                status=ExecutionStatus.COMPLETED if exit_code == 0 else ExecutionStatus.FAILED,
                pid=proc.pid,
                verified=True,
            )
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return CommandResult(
                action_id=action_id,
                command=" ".join(full_cmd),
                exit_code=1,
                status=ExecutionStatus.FAILED,
                stderr=str(exc),
                duration_ms=duration_ms,
            )

    def resolve_confirmation(self, plan_id: str, approved: bool) -> Optional[ActionPlan]:
        """User responded to confirmation prompt for action plan."""
        plan = self._pending_confirmations.pop(plan_id, None)
        if plan:
            plan.status = ExecutionStatus.PENDING if approved else ExecutionStatus.REJECTED
        return plan

    def emergency_stop(self) -> int:
        """Instantly kill all child processes across CMD and PowerShell."""
        logger.warning("EMERGENCY STOP TRIGGERED: Terminating all active commands.")
        killed_cmd = self.cmd_service.cancel_all()
        killed_ps = self.powershell_service.cancel_all()
        self._pending_confirmations.clear()
        return killed_cmd + killed_ps
