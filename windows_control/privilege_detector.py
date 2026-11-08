"""Privilege and Administrator Elevation Detection for Windows.

Provides real-time checks for administrative privileges and identifies
operations requiring elevation before execution.
"""

import ctypes
import os
from pathlib import Path
import re
from typing import Optional
from windows_control.models import PrivilegeLevel

# Commands and tools requiring Windows Administrator (UAC) elevation
ELEVATED_COMMAND_PATTERNS = [
    re.compile(r"\bnet\s+(start|stop|pause|continue)\b", re.IGNORECASE),
    re.compile(r"\b(Start-Service|Stop-Service|Restart-Service|Set-Service)\b", re.IGNORECASE),
    re.compile(r"\bsc\s+(start|stop|create|delete|config)\b", re.IGNORECASE),
    re.compile(r"\bnetsh\s+(advfirewall|interface|wlan)\b", re.IGNORECASE),
    re.compile(r"\bsetx\s+.*\/m\b", re.IGNORECASE),
    re.compile(r"\b(reg\s+(add|delete|import))\s+hklm\b", re.IGNORECASE),
    re.compile(r"\b(Set-ItemProperty|New-ItemProperty|Remove-ItemProperty)\s+.*hklm\b", re.IGNORECASE),
    re.compile(r"\b(takeown|icacls)\s+.*\/grant\b", re.IGNORECASE),
    re.compile(r"\bwinget\s+install\s+.*--scope\s+machine\b", re.IGNORECASE),
    re.compile(r"\b(dism|sfc)\s+\/", re.IGNORECASE),
    re.compile(r"\bshutdown\s+[/-][sr]\b", re.IGNORECASE),
    re.compile(r"\bRestart-Computer\b", re.IGNORECASE),
    re.compile(r"\bStop-Computer\b", re.IGNORECASE),
    re.compile(r"\bchkdisk\b", re.IGNORECASE),
    re.compile(r"\bbcdedit\b", re.IGNORECASE),
]

# Sensitive system directory paths where writes require elevation
ELEVATED_DIRECTORIES = [
    Path(os.environ.get("SystemRoot", r"C:\Windows")).resolve(),
    Path(os.environ.get("ProgramFiles", r"C:\Program Files")).resolve(),
    Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")).resolve(),
]


class PrivilegeDetector:
    """Detects current process privileges and operation elevation requirements."""

    @staticmethod
    def is_current_process_admin() -> bool:
        """Return True if the current Python process has administrative privileges."""
        try:
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        except Exception:
            return False

    @classmethod
    def requires_elevation(
        cls,
        command: str,
        operation_type: str = "command",
        target_path: Optional[str] = None,
    ) -> PrivilegeLevel:
        """Determine whether the specified operation requires administrative elevation."""
        clean_cmd = command.strip()

        # 1. Check command pattern heuristics
        for pattern in ELEVATED_COMMAND_PATTERNS:
            if pattern.search(clean_cmd):
                return PrivilegeLevel.ELEVATED

        # 2. Check operation type heuristics
        if operation_type in ("service_start", "service_stop", "system_reboot", "system_shutdown", "firewall_config"):
            return PrivilegeLevel.ELEVATED

        # 3. Check target path write permissions
        if target_path and operation_type in ("write", "delete", "create_dir", "modify"):
            try:
                from pathlib import Path
                p = Path(target_path).resolve()
                for sys_dir in ELEVATED_DIRECTORIES:
                    if p == sys_dir or sys_dir in p.parents:
                        return PrivilegeLevel.ELEVATED
            except Exception:
                pass

        return PrivilegeLevel.NORMAL
