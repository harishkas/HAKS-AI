"""System-Wide Windows Resource Discovery Service for HAKS.

Discovers system-wide resources across all accessible storage volumes and OS layers:
- All available drive letters (Fixed, Removable, External USB, CD-ROM, Network)
- Standard Windows user and system directories
- Python installations and virtual environments
- Git installations, global configurations, and local repositories
- Development tools, compilers, and SDKs (Node, VS Code, Rust, Go, Java, .NET, Docker)
- Installed applications across Windows Registry, Start Menu, and Winget
- Running processes and system services
- Network adapters and connectivity
- Complete host system configuration
"""

import ctypes
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Optional
import winreg

import psutil

from windows_control.application_manager import ApplicationManager
from windows_control.models import (
    DriveInfo,
    InstalledAppInfo,
    NetworkInfo,
    ProcessInfo,
    ServiceInfo,
    SystemDiscoveryReport,
)
from windows_control.privilege_detector import PrivilegeDetector
from windows_control.system_manager import SystemManager

logger = logging.getLogger("haks.windows_control.resource_discovery")


class SystemResourceDiscoveryService:
    """Discovers host Windows hardware, filesystem scopes, runtimes, and software inventory."""

    def __init__(self, config: Optional[Any] = None) -> None:
        self.config = config
        self.app_manager = ApplicationManager()
        self.system_manager = SystemManager()

    # --------------------------------------------------------------------------
    # 1. Drive Discovery
    # --------------------------------------------------------------------------

    def discover_drives(self) -> list[DriveInfo]:
        """Discover all logical drive letters and storage volumes across the host."""
        drives: list[DriveInfo] = []
        k32 = ctypes.windll.kernel32

        drive_types_map = {
            0: "unknown",
            1: "no_root_dir",
            2: "removable",  # USB flash drives, SD cards
            3: "fixed",      # HDDs, SSDs
            4: "network",    # Mapped network drives
            5: "cdrom",      # Optical discs
            6: "ramdisk",
        }

        # Query all partitions including unformatted / offline drives
        try:
            for part in psutil.disk_partitions(all=True):
                mount = part.mountpoint
                dev = part.device
                fstype = part.fstype

                dtype_code = k32.GetDriveTypeW(mount)
                dtype_str = drive_types_map.get(dtype_code, "unknown")

                # Query Volume Name
                vol_buf = ctypes.create_unicode_buffer(261)
                fs_buf = ctypes.create_unicode_buffer(261)
                vol_name = ""
                try:
                    if k32.GetVolumeInformationW(
                        mount, vol_buf, ctypes.sizeof(vol_buf), None, None, None, fs_buf, ctypes.sizeof(fs_buf)
                    ):
                        vol_name = vol_buf.value
                        if not fstype and fs_buf.value:
                            fstype = fs_buf.value
                except Exception:
                    pass

                tot_gb = 0.0
                usd_gb = 0.0
                fre_gb = 0.0
                pct = 0.0
                is_ready = False

                try:
                    usage = psutil.disk_usage(mount)
                    tot_gb = round(usage.total / (1024**3), 2)
                    usd_gb = round(usage.used / (1024**3), 2)
                    fre_gb = round(usage.free / (1024**3), 2)
                    pct = usage.percent
                    is_ready = True
                except (PermissionError, OSError):
                    is_ready = False

                drives.append(
                    DriveInfo(
                        device=dev,
                        mountpoint=mount,
                        fstype=fstype or "Unknown",
                        total_gb=tot_gb,
                        used_gb=usd_gb,
                        free_gb=fre_gb,
                        percent_used=pct,
                        drive_type=dtype_str,
                        volume_name=vol_name,
                        is_ready=is_ready,
                    )
                )
        except Exception as exc:
            logger.warning("Error during drive discovery: %s", exc)

        return drives

    # --------------------------------------------------------------------------
    # 2. User & System Directories
    # --------------------------------------------------------------------------

    def discover_user_directories(self) -> dict[str, str]:
        """Discover canonical Windows user profile and system folder locations."""
        user_profile = os.environ.get("USERPROFILE", str(Path.home()))
        app_data = os.environ.get("APPDATA", str(Path(user_profile) / "AppData" / "Roaming"))
        local_app_data = os.environ.get("LOCALAPPDATA", str(Path(user_profile) / "AppData" / "Local"))
        program_data = os.environ.get("ProgramData", r"C:\ProgramData")
        system_root = os.environ.get("SystemRoot", r"C:\Windows")
        temp_dir = os.environ.get("TEMP", str(Path(local_app_data) / "Temp"))

        # Default shell folder locations
        desktop = str(Path(user_profile) / "Desktop")
        documents = str(Path(user_profile) / "Documents")
        downloads = str(Path(user_profile) / "Downloads")
        music = str(Path(user_profile) / "Music")
        pictures = str(Path(user_profile) / "Pictures")
        videos = str(Path(user_profile) / "Videos")

        # Query Windows Registry for redirected / OneDrive user shell folders
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders") as key:
                for reg_name, path_var in [
                    ("Desktop", "desktop"),
                    ("Personal", "documents"),
                    ("{374DE290-123F-4565-9164-39C4925E467B}", "downloads"),
                    ("My Music", "music"),
                    ("My Pictures", "pictures"),
                    ("My Video", "videos"),
                ]:
                    try:
                        val, _ = winreg.QueryValueEx(key, reg_name)
                        expanded = os.path.expandvars(str(val))
                        if Path(expanded).exists():
                            if path_var == "desktop":
                                desktop = expanded
                            elif path_var == "documents":
                                documents = expanded
                            elif path_var == "downloads":
                                downloads = expanded
                            elif path_var == "music":
                                music = expanded
                            elif path_var == "pictures":
                                pictures = expanded
                            elif path_var == "videos":
                                videos = expanded
                    except OSError:
                        pass
        except Exception:
            pass

        # Fallback to OneDrive folder if default desktop/documents don't exist
        one_drive = Path(user_profile) / "OneDrive"
        if not Path(desktop).exists() and (one_drive / "Desktop").exists():
            desktop = str(one_drive / "Desktop")
        if not Path(documents).exists() and (one_drive / "Documents").exists():
            documents = str(one_drive / "Documents")

        dirs = {
            "user_profile": user_profile,
            "desktop": desktop,
            "documents": documents,
            "downloads": downloads,
            "music": music,
            "pictures": pictures,
            "videos": videos,
            "local_app_data": local_app_data,
            "roaming_app_data": app_data,
            "program_data": program_data,
            "system_root": system_root,
            "system32": str(Path(system_root) / "System32"),
            "program_files": os.environ.get("ProgramFiles", r"C:\Program Files"),
            "program_files_x86": os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
            "temp": temp_dir,
        }

        # Filter down to directories that exist on disk
        return {k: v for k, v in dirs.items() if Path(v).exists()}

    # --------------------------------------------------------------------------
    # 3. Python Installations & Virtual Environments
    # --------------------------------------------------------------------------

    def discover_python_environments(self) -> list[dict[str, Any]]:
        """Discover all Python runtimes, launcher interpreters, and active virtual environments."""
        environments: list[dict[str, Any]] = []
        seen_paths: set[str] = set()

        # 1. Current Running Python
        cur_exe = str(Path(sys.executable).resolve())
        is_cur_venv = hasattr(sys, "real_prefix") or (sys.base_prefix != sys.prefix)
        environments.append(
            {
                "name": "Current Active Python",
                "executable": cur_exe,
                "version": platform.python_version(),
                "is_virtualenv": is_cur_venv,
                "base_prefix": sys.base_prefix,
                "prefix": sys.prefix,
                "active": True,
            }
        )
        seen_paths.add(cur_exe.lower())

        # 2. Windows Python Launcher (py --list)
        py_launcher = shutil.which("py")
        if py_launcher:
            try:
                res = subprocess.run(["py", "--list-paths"], capture_output=True, text=True, timeout=5)
                if res.returncode == 0:
                    for line in res.stdout.strip().splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        # Format: -V:3.14 * <path> or -V:3.12 <path>
                        parts = line.split(maxsplit=2)
                        exe_candidate = parts[-1] if len(parts) >= 2 else None
                        if exe_candidate and Path(exe_candidate).exists():
                            p_clean = str(Path(exe_candidate).resolve())
                            if p_clean.lower() not in seen_paths:
                                seen_paths.add(p_clean.lower())
                                ver = parts[0].replace("-V:", "").strip() if len(parts) > 0 else "Unknown"
                                environments.append(
                                    {
                                        "name": f"Python {ver} (via py launcher)",
                                        "executable": p_clean,
                                        "version": ver,
                                        "is_virtualenv": False,
                                        "base_prefix": str(Path(p_clean).parent),
                                        "prefix": str(Path(p_clean).parent),
                                        "active": False,
                                    }
                                )
            except Exception as exc:
                logger.warning("Error querying py launcher: %s", exc)

        # 3. Windows Registry PythonCore Keys
        reg_keys = [
            (winreg.HKEY_CURRENT_USER, r"Software\Python\PythonCore"),
            (winreg.HKEY_LOCAL_MACHINE, r"Software\Python\PythonCore"),
            (winreg.HKEY_LOCAL_MACHINE, r"Software\WOW6432Node\Python\PythonCore"),
        ]
        for root_key, subkey_path in reg_keys:
            try:
                with winreg.OpenKey(root_key, subkey_path, 0, winreg.KEY_READ) as key:
                    num_versions = winreg.QueryInfoKey(key)[0]
                    for i in range(num_versions):
                        ver_name = winreg.EnumKey(key, i)
                        try:
                            install_path_key = rf"{subkey_path}\{ver_name}\InstallPath"
                            with winreg.OpenKey(root_key, install_path_key, 0, winreg.KEY_READ) as ip_key:
                                exe_path, _ = winreg.QueryValueEx(ip_key, "ExecutablePath")
                                if exe_path and Path(exe_path).exists():
                                    clean = str(Path(exe_path).resolve())
                                    if clean.lower() not in seen_paths:
                                        seen_paths.add(clean.lower())
                                        environments.append(
                                            {
                                                "name": f"Python {ver_name} (Registry)",
                                                "executable": clean,
                                                "version": ver_name,
                                                "is_virtualenv": False,
                                                "base_prefix": str(Path(clean).parent),
                                                "prefix": str(Path(clean).parent),
                                                "active": False,
                                            }
                                        )
                        except OSError:
                            continue
            except OSError:
                continue

        # 4. Search for project virtual environments in workspace and user directories
        candidate_roots = [
            os.getcwd(),
            os.environ.get("USERPROFILE", ""),
        ]
        venv_names = [".venv", "venv", "env", ".env"]

        for cand_root in candidate_roots:
            if not cand_root or not Path(cand_root).exists():
                continue
            for vn in venv_names:
                v_path = Path(cand_root) / vn / "Scripts" / "python.exe"
                if v_path.exists():
                    clean = str(v_path.resolve())
                    if clean.lower() not in seen_paths:
                        seen_paths.add(clean.lower())
                        environments.append(
                            {
                                "name": f"Virtualenv ({vn} in {Path(cand_root).name})",
                                "executable": clean,
                                "version": platform.python_version(),
                                "is_virtualenv": True,
                                "base_prefix": str(v_path.parent.parent),
                                "prefix": str(v_path.parent.parent),
                                "active": False,
                            }
                        )

        return environments

    # --------------------------------------------------------------------------
    # 4. Git Ecosystem & Repositories
    # --------------------------------------------------------------------------

    def discover_git_ecosystem(self) -> dict[str, Any]:
        """Discover Git executable, global configuration, and accessible repositories."""
        git_exe = shutil.which("git")
        git_version = "Not Installed"
        config = {}
        repos: list[str] = []

        if git_exe:
            try:
                res = subprocess.run(["git", "--version"], capture_output=True, text=True, timeout=5)
                if res.returncode == 0:
                    git_version = res.stdout.strip()
            except Exception:
                pass

            # Query global config
            for key in ["user.name", "user.email", "core.editor", "init.defaultBranch"]:
                try:
                    res = subprocess.run(["git", "config", "--global", "--get", key], capture_output=True, text=True, timeout=3)
                    if res.returncode == 0 and res.stdout.strip():
                        config[key] = res.stdout.strip()
                except Exception:
                    pass

            # Discover local git repositories in current workspace and immediate user dirs
            search_dirs = [
                os.getcwd(),
                os.environ.get("USERPROFILE", ""),
            ]
            for sdir in search_dirs:
                if not sdir or not Path(sdir).exists():
                    continue
                try:
                    # Check current dir
                    if (Path(sdir) / ".git").exists():
                        repos.append(str(Path(sdir).resolve()))
                    # Check immediate subdirectories (1 level deep)
                    for item in Path(sdir).iterdir():
                        try:
                            if item.is_dir() and (item / ".git").exists():
                                repos.append(str(item.resolve()))
                        except (PermissionError, OSError):
                            continue
                except (PermissionError, OSError):
                    continue

        return {
            "installed": git_exe is not None,
            "executable": git_exe,
            "version": git_version,
            "global_config": config,
            "discovered_repositories": repos[:20],
        }

    # --------------------------------------------------------------------------
    # 5. Development Tools & SDKs
    # --------------------------------------------------------------------------

    def discover_development_tools(self) -> list[dict[str, Any]]:
        """Identify installed programming languages, compilers, package managers, and SDKs."""
        tool_definitions = [
            ("git", ["git", "--version"], "Version Control"),
            ("code", ["code", "--version"], "IDE (VS Code)"),
            ("node", ["node", "--version"], "JavaScript Runtime (Node.js)"),
            ("npm", ["npm", "--version"], "Node Package Manager"),
            ("yarn", ["yarn", "--version"], "Yarn Package Manager"),
            ("pnpm", ["pnpm", "--version"], "PNPM Package Manager"),
            ("rustc", ["rustc", "--version"], "Rust Compiler"),
            ("cargo", ["cargo", "--version"], "Rust Package Manager"),
            ("go", ["go", "version"], "Go Compiler"),
            ("javac", ["javac", "-version"], "Java Compiler (JDK)"),
            ("java", ["java", "-version"], "Java Runtime (JRE)"),
            ("mvn", ["mvn", "-version"], "Apache Maven"),
            ("dotnet", ["dotnet", "--version"], ".NET Core / SDK"),
            ("docker", ["docker", "--version"], "Docker Container Platform"),
            ("winget", ["winget", "--version"], "Windows Package Manager"),
            ("pwsh", ["pwsh", "--version"], "PowerShell Core"),
            ("powershell", ["powershell.exe", "$PSVersionTable.PSVersion.ToString()"], "Windows PowerShell"),
        ]

        detected: list[dict[str, Any]] = []
        for name, cmd, category in tool_definitions:
            exe_path = shutil.which(name)
            if exe_path:
                version = "Installed"
                try:
                    res = subprocess.run(cmd, capture_output=True, text=True, timeout=4)
                    if res.returncode == 0 and res.stdout.strip():
                        # Take first line of output
                        version = res.stdout.strip().splitlines()[0]
                    elif res.stderr.strip():
                        version = res.stderr.strip().splitlines()[0]
                except Exception:
                    pass

                detected.append(
                    {
                        "name": name,
                        "category": category,
                        "executable": exe_path,
                        "version": version,
                    }
                )

        return detected

    # --------------------------------------------------------------------------
    # 6. Installed Applications
    # --------------------------------------------------------------------------

    def discover_installed_software(
        self, query: Optional[str] = None, limit: Optional[int] = None
    ) -> list[InstalledAppInfo]:
        """Aggregate installed software across Windows Registry and Start Menu."""
        apps = self.app_manager.list_installed_applications(search_filter=query)
        if limit and limit > 0:
            return apps[:limit]
        return apps

    def discover_python_installations(self) -> list[dict[str, Any]]:
        """Alias for discover_python_environments."""
        return self.discover_python_environments()

    # --------------------------------------------------------------------------
    # 7. Running Processes & Windows Services
    # --------------------------------------------------------------------------

    def discover_running_processes(self, limit: int = 50) -> list[ProcessInfo]:
        """Enumerate active processes ordered by memory usage."""
        procs: list[ProcessInfo] = []
        for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent", "status", "username", "exe"]):
            try:
                info = p.info
                vmem = p.memory_info()
                mb = round(vmem.rss / (1024 * 1024), 1) if vmem else 0.0

                procs.append(
                    ProcessInfo(
                        pid=info["pid"],
                        name=info.get("name") or f"PID {info['pid']}",
                        cpu_percent=info.get("cpu_percent") or 0.0,
                        memory_mb=mb,
                        memory_percent=round(info.get("memory_percent") or 0.0, 1),
                        status=info.get("status") or "running",
                        username=info.get("username"),
                        exe=info.get("exe"),
                    )
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        procs.sort(key=lambda x: x.memory_mb, reverse=True)
        return procs[:limit]

    def discover_windows_services(self, limit: int = 50) -> list[ServiceInfo]:
        """Enumerate system services using SystemManager."""
        return self.system_manager.list_services(limit=limit)

    # --------------------------------------------------------------------------
    # 8. Network Adapters & System Configuration
    # --------------------------------------------------------------------------

    def discover_network_adapters(self) -> NetworkInfo:
        """Inspect host network configuration and ping connectivity."""
        return self.system_manager.inspect_network()

    def discover_system_configuration(self) -> dict[str, Any]:
        """Query host operating system build, computer identity, and environment."""
        win_ver = sys.getwindowsversion()
        return {
            "computer_name": platform.node(),
            "username": os.environ.get("USERNAME", "User"),
            "user_domain": os.environ.get("USERDOMAIN", ""),
            "os_name": f"Windows {platform.release()}",
            "os_build": f"{win_ver.major}.{win_ver.minor}.{win_ver.build}",
            "os_version": platform.version(),
            "os_architecture": platform.machine(),
            "processor_architecture": os.environ.get("PROCESSOR_ARCHITECTURE", "AMD64"),
            "processor_count": os.environ.get("NUMBER_OF_PROCESSORS", "4"),
            "is_admin": PrivilegeDetector.is_current_process_admin(),
            "path_entries_count": len(os.environ.get("PATH", "").split(os.pathsep)),
        }

    # --------------------------------------------------------------------------
    # 9. Unified Comprehensive Discovery
    # --------------------------------------------------------------------------

    def perform_full_system_discovery(
        self, include_apps: bool = True, include_processes: bool = True
    ) -> SystemDiscoveryReport:
        """Execute complete system discovery across drives, user dirs, tools, apps, and OS."""
        logger.info("Executing comprehensive Windows system resource discovery...")

        drives = self.discover_drives()
        user_dirs = self.discover_user_directories()
        py_envs = self.discover_python_environments()
        git_eco = self.discover_git_ecosystem()
        dev_tools = self.discover_development_tools()
        apps = self.discover_installed_software() if include_apps else []
        procs = self.discover_running_processes(limit=30) if include_processes else []
        services = self.discover_windows_services(limit=30) if include_processes else []
        net = self.discover_network_adapters()
        sys_config = self.discover_system_configuration()

        report = SystemDiscoveryReport(
            drives=drives,
            user_directories=user_dirs,
            installed_applications_count=len(apps),
            installed_applications=apps,
            python_installations=py_envs,
            git_ecosystem=git_eco,
            development_tools=dev_tools,
            running_processes_count=len(procs),
            running_processes=procs,
            windows_services_count=len(services),
            windows_services=services,
            network=net,
            system_configuration=sys_config,
        )
        logger.info(
            "System discovery completed: %d drives, %d apps, %d dev tools, %d Python runtimes.",
            len(drives),
            len(apps),
            len(dev_tools),
            len(py_envs),
        )
        return report

    def discover_all(
        self, include_apps: bool = True, include_processes: bool = True
    ) -> SystemDiscoveryReport:
        """Alias for perform_full_system_discovery."""
        return self.perform_full_system_discovery(
            include_apps=include_apps, include_processes=include_processes
        )
