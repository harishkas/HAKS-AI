"""CMD Command Service for HAKS.

Executes native Windows Command Prompt commands asynchronously,
supports real-time line-by-line output streaming, working directory selection,
cancellation, and non-blocking process management.
"""

import asyncio
import logging
import os
from pathlib import Path
import time
from typing import Callable, Optional

import psutil

from windows_control.models import CommandResult, ExecutionStatus

logger = logging.getLogger("haks.windows_control.cmd")


class CMDService:
    """Asynchronous CMD execution service with streaming and cancellation."""

    def __init__(self) -> None:
        self._active_processes: dict[str, asyncio.subprocess.Process] = {}

    async def execute(
        self,
        command: str,
        working_dir: Optional[str] = None,
        timeout_seconds: int = 120,
        action_id: str = "cmd_action",
        output_callback: Optional[Callable[[str], None]] = None,
        error_callback: Optional[Callable[[str], None]] = None,
    ) -> CommandResult:
        """Execute a CMD command with live output streaming and timeout management."""
        cwd = working_dir or os.getcwd()
        if not Path(cwd).exists():
            cwd = os.getcwd()

        cmd_line = f'cmd.exe /c "{command}"'
        logger.info("Executing CMD [%s]: %s (cwd=%s)", action_id, command, cwd)

        start_time = time.perf_counter()
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        try:
            # Create asynchronous subprocess
            proc = await asyncio.create_subprocess_shell(
                cmd_line,
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
                    # Decode with fallback for Windows OEM code pages
                    try:
                        line = line_bytes.decode("utf-8")
                    except UnicodeDecodeError:
                        try:
                            line = line_bytes.decode("cp1252")
                        except UnicodeDecodeError:
                            line = line_bytes.decode("cp850", errors="replace")

                    if is_stderr:
                        stderr_chunks.append(line)
                        if error_callback:
                            error_callback(line)
                    else:
                        stdout_chunks.append(line)
                        if output_callback:
                            output_callback(line)

            # Read both streams concurrently with timeout
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
            err_msg = f"CMD command timed out after {timeout_seconds} seconds."
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
            logger.error("[%s] Error executing CMD: %s", action_id, exc)
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

    def cancel(self, action_id: str) -> bool:
        """Terminate active CMD subprocess and any child processes."""
        proc = self._active_processes.get(action_id)
        if proc and proc.returncode is None:
            try:
                # Terminate entire process tree using psutil
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
                logger.info("Cancelled CMD process for action: %s", action_id)
                return True
            except Exception as exc:
                logger.warning("Error cancelling CMD process: %s", exc)
        return False

    def cancel_all(self) -> int:
        """Cancel all active CMD processes (Emergency Stop)."""
        count = 0
        for action_id in list(self._active_processes.keys()):
            if self.cancel(action_id):
                count += 1
        return count
