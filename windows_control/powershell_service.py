"""PowerShell Service for HAKS.

Executes native Windows PowerShell commands and .ps1 scripts asynchronously,
supports UTF-8 encoding for Unicode/Tamil text, real-time output streaming,
structured output inspection, and non-blocking process management.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Callable, Optional

import psutil

from windows_control.models import CommandResult, ExecutionStatus

logger = logging.getLogger("haks.windows_control.powershell")


class PowerShellService:
    """Asynchronous PowerShell execution service."""

    def __init__(self) -> None:
        self._active_processes: dict[str, asyncio.subprocess.Process] = {}

    async def execute(
        self,
        command: str,
        working_dir: Optional[str] = None,
        timeout_seconds: int = 120,
        action_id: str = "ps_action",
        output_callback: Optional[Callable[[str], None]] = None,
        error_callback: Optional[Callable[[str], None]] = None,
    ) -> CommandResult:
        """Execute a PowerShell command with UTF-8 encoding and streaming."""
        cwd = working_dir or os.getcwd()
        if not Path(cwd).exists():
            cwd = os.getcwd()

        # Enforce UTF-8 output encoding inside the PowerShell session to support Tamil & Unicode
        wrapped_command = (
            f"[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
            f"$OutputEncoding = [System.Text.Encoding]::UTF8; "
            f"{command}"
        )

        args = [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            wrapped_command,
        ]

        logger.info("Executing PowerShell [%s]: %s (cwd=%s)", action_id, command, cwd)
        start_time = time.perf_counter()
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self._active_processes[action_id] = proc

            async def _stream_reader(stream: asyncio.StreamReader, is_stderr: bool) -> None:
                while True:
                    line_bytes = await stream.readline()
                    if not line_bytes:
                        break
                    line = line_bytes.decode("utf-8", errors="replace")
                    if is_stderr:
                        stderr_chunks.append(line)
                        if error_callback:
                            error_callback(line)
                    else:
                        stdout_chunks.append(line)
                        if output_callback:
                            output_callback(line)

            await asyncio.wait_for(
                asyncio.gather(
                    _stream_reader(proc.stdout, False),  # type: ignore
                    _stream_reader(proc.stderr, True),   # type: ignore
                    proc.wait(),
                ),
                timeout=timeout_seconds,
            )

            duration_ms = (time.perf_counter() - start_time) * 1000.0
            full_stdout = "".join(stdout_chunks)
            full_stderr = "".join(stderr_chunks)
            exit_code = proc.returncode if proc.returncode is not None else 0

            return CommandResult(
                action_id=action_id,
                command=command,
                exit_code=exit_code,
                stdout=full_stdout,
                stderr=full_stderr,
                duration_ms=duration_ms,
                status=ExecutionStatus.COMPLETED if exit_code == 0 else ExecutionStatus.FAILED,
                pid=proc.pid,
                verified=True,
            )

        except asyncio.TimeoutError:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            self.cancel(action_id)
            err_msg = f"PowerShell command timed out after {timeout_seconds} seconds."
            logger.warning("[%s] %s", action_id, err_msg)
            return CommandResult(
                action_id=action_id,
                command=command,
                exit_code=1460,
                stdout="".join(stdout_chunks),
                stderr=err_msg,
                duration_ms=duration_ms,
                status=ExecutionStatus.FAILED,
                error=err_msg,
            )
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error("[%s] Error executing PowerShell: %s", action_id, exc)
            return CommandResult(
                action_id=action_id,
                command=command,
                exit_code=1,
                stdout="".join(stdout_chunks),
                stderr=str(exc),
                duration_ms=duration_ms,
                status=ExecutionStatus.FAILED,
                error=str(exc),
            )
        finally:
            self._active_processes.pop(action_id, None)

    async def execute_script_file(
        self,
        script_path: Path,
        script_args: Optional[list[str]] = None,
        working_dir: Optional[str] = None,
        timeout_seconds: int = 120,
        action_id: str = "ps_script",
    ) -> CommandResult:
        """Execute a .ps1 script file."""
        if not script_path.exists():
            return CommandResult(
                action_id=action_id,
                command=str(script_path),
                exit_code=2,
                status=ExecutionStatus.FAILED,
                stderr=f"PowerShell script file not found: {script_path}",
            )

        arg_str = " ".join(f'"{a}"' for a in (script_args or []))
        cmd = f'& "{script_path}" {arg_str}'.strip()
        return await self.execute(cmd, working_dir, timeout_seconds, action_id)

    async def query_json(self, command: str, working_dir: Optional[str] = None) -> Optional[Any]:
        """Execute command and parse structured JSON output (via ConvertTo-Json)."""
        cmd_with_json = f"{command} | ConvertTo-Json -Depth 3 -Compress"
        res = await self.execute(cmd_with_json, working_dir=working_dir, timeout_seconds=30)
        if res.exit_code == 0 and res.stdout.strip():
            try:
                return json.loads(res.stdout.strip())
            except Exception as exc:
                logger.warning("Could not parse PowerShell JSON output: %s", exc)
        return None

    def cancel(self, action_id: str) -> bool:
        """Terminate active PowerShell subprocess and any spawned children."""
        proc = self._active_processes.get(action_id)
        if proc and proc.returncode is None:
            try:
                if proc.pid:
                    try:
                        parent = psutil.Process(proc.pid)
                        for child in parent.children(recursive=True):
                            try:
                                child.kill()
                            except Exception:
                                pass
                        parent.kill()
                    except Exception:
                        proc.kill()
                logger.info("Cancelled PowerShell process for action: %s", action_id)
                return True
            except Exception as exc:
                logger.warning("Error cancelling PowerShell process: %s", exc)
        return False

    def cancel_all(self) -> int:
        """Cancel all active PowerShell processes (Emergency Stop)."""
        count = 0
        for action_id in list(self._active_processes.keys()):
            if self.cancel(action_id):
                count += 1
        return count
