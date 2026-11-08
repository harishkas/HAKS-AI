"""Process Management Service for HAKS Windows Computer Control.

Monitors active Windows processes, tracks CPU and memory usage,
and handles safe process termination with critical system protections.
"""

import logging
import os
import time
from typing import Any, Optional

import psutil

from windows_control.execution_policy import CRITICAL_WINDOWS_PROCESSES
from windows_control.models import ProcessInfo

logger = logging.getLogger("haks.windows_control.process")


class ProcessManager:
    """Windows process inspection, resource monitoring, and lifecycle control."""

    def __init__(self) -> None:
        self._current_pid = os.getpid()

    def list_processes(self, limit: int = 150) -> list[ProcessInfo]:
        """List active processes sorted by CPU and memory consumption."""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'status', 'create_time', 'exe']):
            try:
                info = proc.info
                pid = info['pid']
                name = info['name'] or f"PID_{pid}"

                # Calculate memory in MB
                mem_info = proc.memory_info()
                mem_mb = round(mem_info.rss / (1024 * 1024), 2)
                mem_percent = round(proc.memory_percent(), 2)

                is_critical = (
                    name.lower() in CRITICAL_WINDOWS_PROCESSES
                    or pid in (0, 4)
                    or pid == self._current_pid
                )

                processes.append(
                    ProcessInfo(
                        pid=pid,
                        name=name,
                        cpu_percent=0.0,  # Snapshot
                        memory_mb=mem_mb,
                        memory_percent=mem_percent,
                        status=str(info.get('status', 'running')),
                        create_time=info.get('create_time', 0.0),
                        is_critical=is_critical,
                        exe=info.get('exe'),
                    )
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        # Sort by memory usage descending
        processes.sort(key=lambda p: p.memory_mb, reverse=True)
        return processes[:limit]

    def get_process_by_pid(self, pid: int) -> Optional[ProcessInfo]:
        """Retrieve details for a specific process ID."""
        try:
            proc = psutil.Process(pid)
            mem_info = proc.memory_info()
            name = proc.name()
            return ProcessInfo(
                pid=pid,
                name=name,
                cpu_percent=round(proc.cpu_percent(interval=0.05), 1),
                memory_mb=round(mem_info.rss / (1024 * 1024), 2),
                memory_percent=round(proc.memory_percent(), 2),
                status=proc.status(),
                create_time=proc.create_time(),
                is_critical=name.lower() in CRITICAL_WINDOWS_PROCESSES or pid in (0, 4, self._current_pid),
                exe=proc.exe() if hasattr(proc, "exe") else None,
            )
        except Exception:
            return None

    def terminate_process(self, pid: int, force: bool = False) -> tuple[bool, str]:
        """Terminate a process by PID after verifying critical protection guards."""
        if pid in (0, 4):
            return False, "Cannot terminate Windows System Idle or Kernel process."
        if pid == self._current_pid:
            return False, "Cannot terminate HAKS assistant process."

        try:
            proc = psutil.Process(pid)
            name = proc.name().lower()
            if name in CRITICAL_WINDOWS_PROCESSES:
                return False, f"Protected Windows process '{name}' cannot be terminated."

            if force:
                proc.kill()
                logger.warning("Forcefully killed process [%d: %s]", pid, name)
            else:
                proc.terminate()
                logger.info("Gracefully requested termination for process [%d: %s]", pid, name)

            return True, f"Process {name} (PID: {pid}) was terminated."
        except psutil.NoSuchProcess:
            return False, f"Process with PID {pid} was not found."
        except psutil.AccessDenied:
            return False, f"Access denied terminating PID {pid}. Administrator elevation may be required."
        except Exception as exc:
            return False, f"Error terminating process {pid}: {exc}"

    def get_system_resource_usage(self) -> dict[str, Any]:
        """Retrieve overall CPU, RAM, and swap memory utilization."""
        vmem = psutil.virtual_memory()
        cpu_pct = psutil.cpu_percent(interval=0.1)
        return {
            "cpu_percent": cpu_pct,
            "ram_total_gb": round(vmem.total / (1024**3), 2),
            "ram_used_gb": round(vmem.used / (1024**3), 2),
            "ram_available_gb": round(vmem.available / (1024**3), 2),
            "ram_percent": vmem.percent,
        }
