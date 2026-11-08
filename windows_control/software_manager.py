"""Software and Package Management Service for HAKS Windows Computer Control.

Manages software lifecycle using Windows Package Manager (winget),
Python/pip environment dependencies, and Git repository operations.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Optional

from windows_control.models import CommandResult, ExecutionStatus

logger = logging.getLogger("haks.windows_control.software")


class SoftwareManager:
    """Manages system package installation, Python libraries, and Git environments."""

    def __init__(self) -> None:
        self.winget_path: Optional[str] = shutil.which("winget")

    def is_winget_available(self) -> bool:
        """Return True if Windows Package Manager (winget) is installed and in PATH."""
        return self.winget_path is not None

    # --------------------------------------------------------------------------
    # 1. Windows Package Manager (winget)
    # --------------------------------------------------------------------------

    async def search_packages(self, query: str, limit: int = 20) -> list[dict[str, str]]:
        """Search available applications via winget."""
        if not self.is_winget_available():
            logger.warning("winget is not available on this system.")
            return []

        cmd = ["winget", "search", query, "--accept-source-agreements"]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=30.0)
            text = stdout_bytes.decode("utf-8", errors="replace")

            # Parse tabular output
            packages = []
            lines = text.strip().splitlines()
            header_idx = -1
            for i, line in enumerate(lines):
                if "---" in line:
                    header_idx = i
                    break

            if header_idx > 0 and header_idx + 1 < len(lines):
                for line in lines[header_idx + 1 :]:
                    parts = line.split()
                    if len(parts) >= 2:
                        packages.append(
                            {
                                "name": parts[0],
                                "id": parts[1],
                                "version": parts[2] if len(parts) > 2 else "Unknown",
                                "source": parts[3] if len(parts) > 3 else "winget",
                            }
                        )
                        if len(packages) >= limit:
                            break

            return packages
        except Exception as exc:
            logger.error("Error searching winget packages for '%s': %s", query, exc)
            return []

    async def get_package_info(self, package_id: str) -> Optional[dict[str, Any]]:
        """Query detailed metadata for a package from winget."""
        if not self.is_winget_available():
            return None

        cmd = ["winget", "show", "--id", package_id, "--accept-source-agreements"]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=20.0)
            text = stdout_bytes.decode("utf-8", errors="replace")

            info: dict[str, Any] = {"id": package_id}
            for line in text.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    info[k.strip().lower().replace(" ", "_")] = v.strip()
            return info
        except Exception as exc:
            logger.warning("Could not fetch winget package info for %s: %s", package_id, exc)
            return None

    def build_winget_install_command(self, package_id: str, scope: str = "user") -> str:
        """Construct standard non-interactive winget installation command line."""
        scope_arg = f"--scope {scope}" if scope else ""
        return (
            f"winget install --id {package_id} -e {scope_arg} "
            f"--accept-package-agreements --accept-source-agreements"
        ).strip()

    def build_winget_uninstall_command(self, package_id: str) -> str:
        """Construct standard non-interactive winget uninstallation command line."""
        return f"winget uninstall --id {package_id} -e"

    def build_winget_upgrade_command(self, package_id: str) -> str:
        """Construct standard winget upgrade command line."""
        return f"winget upgrade --id {package_id} -e --accept-package-agreements --accept-source-agreements"

    # --------------------------------------------------------------------------
    # 2. Python Environment & pip
    # --------------------------------------------------------------------------

    async def list_pip_packages(self) -> list[dict[str, str]]:
        """Query installed Python libraries in the current Python environment."""
        python_exe = sys.executable
        cmd = [python_exe, "-m", "pip", "list", "--format=json"]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=20.0)
            text = stdout_bytes.decode("utf-8", errors="replace")
            return json.loads(text)
        except Exception as exc:
            logger.warning("Error querying pip packages: %s", exc)
            return []

    def build_pip_install_command(self, package_spec: str) -> str:
        """Construct pip install command for the active Python environment."""
        python_exe = sys.executable
        return f'"{python_exe}" -m pip install {package_spec}'

    def build_pip_requirements_command(self, requirements_file: str) -> str:
        """Construct pip install -r requirements.txt command line."""
        python_exe = sys.executable
        return f'"{python_exe}" -m pip install -r "{requirements_file}"'

    # --------------------------------------------------------------------------
    # 3. Git Operations
    # --------------------------------------------------------------------------

    def build_git_clone_command(self, repo_url: str, destination_dir: str) -> str:
        """Construct git clone command."""
        return f'git clone "{repo_url}" "{destination_dir}"'

    def build_git_pull_command(self) -> str:
        """Construct git pull command."""
        return "git pull"

    def build_git_status_command(self) -> str:
        """Construct git status command."""
        return "git status"
