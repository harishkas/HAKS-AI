"""Application Management Service for HAKS Windows Computer Control.

Discovers installed Windows applications, launches programs, opens documents
with associated default software, and manages desktop application lifecycles.
"""

import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Optional
import winreg

import psutil

from windows_control.models import InstalledAppInfo

logger = logging.getLogger("haks.windows_control.application")

# Common shorthand application names mapped to their standard Windows executables
APP_ALIASES = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "calc": "calc.exe",
    "explorer": "explorer.exe",
    "file explorer": "explorer.exe",
    "cmd": "cmd.exe",
    "command prompt": "cmd.exe",
    "powershell": "powershell.exe",
    "paint": "mspaint.exe",
    "task manager": "taskmgr.exe",
    "taskmgr": "taskmgr.exe",
    "vs code": "code.cmd",
    "vscode": "code.cmd",
    "code": "code.cmd",
    "chrome": "chrome.exe",
    "edge": "msedge.exe",
    "firefox": "firefox.exe",
    "control panel": "control.exe",
    "settings": "start ms-settings:",
}


class ApplicationManager:
    """Manages application launching, file association opening, and software discovery."""

    def launch_application(self, app_name_or_path: str, args: Optional[list[str]] = None) -> tuple[bool, str]:
        """Launch an application by shorthand name, registered app path, or full file path."""
        target = app_name_or_path.strip().lower()

        # 1. Check known aliases
        resolved_cmd = APP_ALIASES.get(target, app_name_or_path)

        # 2. Check ms-settings or URI schemes
        if resolved_cmd.startswith("start "):
            try:
                os.system(resolved_cmd)
                return True, f"Launched Windows settings URI: {resolved_cmd}"
            except Exception as exc:
                return False, f"Failed to launch URI: {exc}"

        # 3. Check App Paths in Windows Registry if target is a simple name
        if not Path(resolved_cmd).is_absolute() and not Path(resolved_cmd).exists():
            app_path_from_reg = self._lookup_app_path_in_registry(resolved_cmd)
            if app_path_from_reg:
                resolved_cmd = app_path_from_reg

        # 4. Launch using os.startfile or subprocess.Popen
        try:
            full_args = (args or [])
            if full_args:
                cmd_list = [resolved_cmd] + full_args
                subprocess.Popen(cmd_list, shell=True)
            else:
                try:
                    os.startfile(resolved_cmd)
                except Exception:
                    subprocess.Popen(resolved_cmd, shell=True)

            logger.info("Launched application: %s", resolved_cmd)
            return True, f"Application launched: {resolved_cmd}"
        except Exception as exc:
            logger.error("Failed to launch application '%s': %s", app_name_or_path, exc)
            return False, f"Error launching application: {exc}"

    def open_file_with_associated_app(self, file_path: str) -> tuple[bool, str]:
        """Open a file or URL using the user's default associated Windows program."""
        fp = Path(file_path).resolve()
        if not fp.exists():
            return False, f"Target file not found: {file_path}"

        try:
            os.startfile(str(fp))
            logger.info("Opened file with default program: %s", fp)
            return True, f"Opened file: {fp.name}"
        except Exception as exc:
            logger.error("Failed to open file '%s': %s", file_path, exc)
            return False, f"Error opening file: {exc}"

    def close_application_by_name(self, app_name: str) -> tuple[int, str]:
        """Close running instances of an application by name."""
        clean_name = app_name.lower().replace(".exe", "")
        terminated_count = 0

        for proc in psutil.process_iter(['pid', 'name']):
            try:
                p_name = proc.info['name'].lower().replace(".exe", "")
                if clean_name in p_name or p_name in clean_name:
                    proc.terminate()
                    terminated_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if terminated_count > 0:
            return terminated_count, f"Closed {terminated_count} instance(s) of '{app_name}'."
        return 0, f"No running instances of '{app_name}' found."

    def list_installed_applications(self, search_filter: Optional[str] = None) -> list[InstalledAppInfo]:
        """Scan Windows Registry Uninstall keys for installed software inventory."""
        installed: list[InstalledAppInfo] = []
        reg_keys = [
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
            (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
        ]

        seen_names: set[str] = set()
        clean_filter = search_filter.lower().strip() if search_filter else None

        for hkey, subkey_path in reg_keys:
            try:
                with winreg.OpenKey(hkey, subkey_path, 0, winreg.KEY_READ) as key:
                    num_subkeys = winreg.QueryInfoKey(key)[0]
                    for i in range(num_subkeys):
                        try:
                            sub_name = winreg.EnumKey(key, i)
                            with winreg.OpenKey(key, sub_name, 0, winreg.KEY_READ) as sub_key:
                                try:
                                    display_name, _ = winreg.QueryValueEx(sub_key, "DisplayName")
                                    if not display_name or display_name in seen_names:
                                        continue

                                    # Filter if requested
                                    if clean_filter and clean_filter not in display_name.lower():
                                        continue

                                    version = self._get_reg_val(sub_key, "DisplayVersion", "Unknown")
                                    publisher = self._get_reg_val(sub_key, "Publisher", "Unknown")
                                    install_loc = self._get_reg_val(sub_key, "InstallLocation", None)
                                    uninstall_str = self._get_reg_val(sub_key, "UninstallString", None)

                                    seen_names.add(display_name)
                                    installed.append(
                                        InstalledAppInfo(
                                            name=display_name,
                                            version=str(version),
                                            publisher=str(publisher),
                                            install_location=install_loc,
                                            uninstall_string=uninstall_str,
                                            source="Registry",
                                        )
                                    )
                                except FileNotFoundError:
                                    continue
                        except OSError:
                            continue
            except OSError:
                continue

        installed.sort(key=lambda a: a.name.lower())
        return installed

    def _lookup_app_path_in_registry(self, app_exe: str) -> Optional[str]:
        """Look up executable path in Windows App Paths registry key."""
        if not app_exe.endswith(".exe"):
            app_exe += ".exe"

        for root in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            key_path = rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{app_exe}"
            try:
                with winreg.OpenKey(root, key_path, 0, winreg.KEY_READ) as key:
                    val, _ = winreg.QueryValueEx(key, "")
                    if val and Path(val).exists():
                        return str(val)
            except OSError:
                continue
        return None

    def _get_reg_val(self, key: Any, val_name: str, default: Any = None) -> Any:
        try:
            v, _ = winreg.QueryValueEx(key, val_name)
            return v if v else default
        except FileNotFoundError:
            return default
