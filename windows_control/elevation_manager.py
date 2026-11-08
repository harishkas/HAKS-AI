"""Windows UAC Elevation Manager for HAKS.

Coordinates on-demand administrative execution via Windows ShellExecute ('runas')
and the dedicated isolated AdminHelper process.
"""

import asyncio
import ctypes
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Optional

from windows_control.admin_ipc import AdminIPC
from windows_control.models import CommandResult, ExecutionStatus
from windows_control.privilege_detector import PrivilegeDetector

logger = logging.getLogger("haks.windows_control.elevation")


class ElevationManager:
    """Manages legitimate Windows UAC elevation workflows."""

    def __init__(self) -> None:
        self.helper_script = Path(__file__).resolve().parent / "admin_helper.py"

    async def execute_elevated(
        self,
        command: str,
        working_dir: Optional[str] = None,
        timeout_seconds: int = 120,
        action_id: str = "elevated_action",
    ) -> CommandResult:
        """Execute a command with administrator privileges via Windows UAC.

        If already running as administrator, executes directly.
        Otherwise, triggers the legitimate Windows UAC consent dialog via ShellExecute('runas').
        """
        # 1. If current process is already Administrator, execute directly
        if PrivilegeDetector.is_current_process_admin():
            logger.info("Current process is already elevated. Executing directly.")
            return await self._execute_direct_admin(command, working_dir, timeout_seconds, action_id)

        # 2. Package request for elevated helper
        token, req_path, res_path = AdminIPC.create_request_package(
            command=command,
            working_dir=working_dir,
            timeout_seconds=timeout_seconds,
        )

        logger.info("Requesting Windows UAC elevation for action '%s'...", action_id)
        start_time = time.perf_counter()

        try:
            # 3. Call ShellExecuteW with 'runas' verb
            # Parameters: hwnd, lpOperation, lpFile, lpParameters, lpDirectory, nShowCmd
            # nShowCmd: 0 = SW_HIDE, 1 = SW_SHOWNORMAL
            python_exe = sys.executable
            args = f'"{self.helper_script}" "{req_path}"'
            work_dir = working_dir or os.getcwd()

            ret = await asyncio.to_thread(
                ctypes.windll.shell32.ShellExecuteW,
                None,
                "runas",
                python_exe,
                args,
                work_dir,
                0,  # Run helper hidden
            )

            # ShellExecute returns > 32 on success.
            # 5 = SE_ERR_ACCESSDENIED (User cancelled / denied UAC prompt)
            if ret <= 32:
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                AdminIPC.cleanup(req_path, res_path)
                if ret == 5:
                    logger.warning("Windows UAC elevation was denied or cancelled by user.")
                    return CommandResult(
                        action_id=action_id,
                        command=command,
                        exit_code=1223,  # ERROR_CANCELLED
                        status=ExecutionStatus.REJECTED,
                        stderr="Administrator elevation (UAC) was cancelled or denied by the user.",
                        duration_ms=duration_ms,
                        elevated=True,
                    )
                else:
                    err_msg = f"ShellExecute elevation failed with error code: {ret}"
                    logger.error(err_msg)
                    return CommandResult(
                        action_id=action_id,
                        command=command,
                        exit_code=ret,
                        status=ExecutionStatus.FAILED,
                        stderr=err_msg,
                        duration_ms=duration_ms,
                        elevated=True,
                    )

            # 4. Wait for response package
            deadline = time.time() + timeout_seconds + 10.0
            response_data = None
            while time.time() < deadline:
                if res_path.exists():
                    response_data = AdminIPC.read_response_package(res_path, delete_after=True)
                    if response_data:
                        break
                await asyncio.sleep(0.15)

            duration_ms = (time.perf_counter() - start_time) * 1000.0
            AdminIPC.cleanup(req_path, res_path)

            if not response_data:
                logger.error("Elevated helper timed out or did not return response.")
                return CommandResult(
                    action_id=action_id,
                    command=command,
                    exit_code=1460,
                    status=ExecutionStatus.FAILED,
                    stderr=f"Elevated helper timed out after {timeout_seconds} seconds.",
                    duration_ms=duration_ms,
                    elevated=True,
                )

            return CommandResult(
                action_id=action_id,
                command=command,
                exit_code=response_data.get("exit_code", 0),
                stdout=response_data.get("stdout", ""),
                stderr=response_data.get("stderr", ""),
                duration_ms=duration_ms,
                status=ExecutionStatus.COMPLETED if response_data.get("exit_code") == 0 else ExecutionStatus.FAILED,
                elevated=True,
                verified=True,
            )

        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            AdminIPC.cleanup(req_path, res_path)
            logger.error("Exception during elevated execution: %s", exc)
            return CommandResult(
                action_id=action_id,
                command=command,
                exit_code=1,
                status=ExecutionStatus.FAILED,
                stderr=str(exc),
                duration_ms=duration_ms,
                elevated=True,
            )

    async def _execute_direct_admin(
        self,
        command: str,
        working_dir: Optional[str],
        timeout_seconds: int,
        action_id: str,
    ) -> CommandResult:
        """Internal helper when current process is already elevated."""
        start_time = time.perf_counter()
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=working_dir or os.getcwd(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_seconds
            )
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            stdout_text = stdout_bytes.decode("utf-8", errors="replace")
            stderr_text = stderr_bytes.decode("utf-8", errors="replace")

            return CommandResult(
                action_id=action_id,
                command=command,
                exit_code=proc.returncode if proc.returncode is not None else 0,
                stdout=stdout_text,
                stderr=stderr_text,
                duration_ms=duration_ms,
                status=ExecutionStatus.COMPLETED if proc.returncode == 0 else ExecutionStatus.FAILED,
                pid=proc.pid,
                elevated=True,
                verified=True,
            )
        except asyncio.TimeoutError:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return CommandResult(
                action_id=action_id,
                command=command,
                exit_code=1460,
                status=ExecutionStatus.FAILED,
                stderr=f"Command timed out after {timeout_seconds} seconds.",
                duration_ms=duration_ms,
                elevated=True,
            )
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return CommandResult(
                action_id=action_id,
                command=command,
                exit_code=1,
                status=ExecutionStatus.FAILED,
                stderr=str(exc),
                duration_ms=duration_ms,
                elevated=True,
            )
