"""Application Registry for HAKS Windows Computer Control (Phase 9.1).

Maintains validated application definitions, aliases (English, Tamil, Tanglish),
safe executable resolution, and installation discovery.
"""

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import re
import shutil
from typing import Any, Optional
import winreg

from config.settings import AppConfig, get_settings

logger = logging.getLogger("haks.windows_control.application_registry")


@dataclass
class ApplicationEntry:
    """Metadata describing a validated Windows application."""

    app_id: str
    display_name: str
    executable_name: str
    aliases: list[str] = field(default_factory=list)
    process_names: list[str] = field(default_factory=list)
    window_title_keywords: list[str] = field(default_factory=list)
    uri_scheme: Optional[str] = None
    verified_path: Optional[str] = None
    is_installed: bool = False
    is_system_app: bool = False
    description: str = ""

    def matches_alias(self, query: str) -> bool:
        """Check if query matches canonical id, display name, or any registered alias."""
        clean = query.strip().lower()
        if not clean:
            return False
        if clean == self.app_id.lower() or clean == self.display_name.lower():
            return True
        return any(clean == a.strip().lower() for a in self.aliases)


# Initial pre-seeded application definitions for Phase 9.1
INITIAL_APPLICATIONS: list[ApplicationEntry] = [
    ApplicationEntry(
        app_id="notepad",
        display_name="Notepad",
        executable_name="notepad.exe",
        aliases=[
            "notepad",
            "note pad",
            "text editor",
            "editor",
            "நோட்பேட்",
            "குறிப்பேடு",
        ],
        process_names=["notepad.exe"],
        window_title_keywords=["Notepad", "நோட்பேட்"],
        is_system_app=True,
        description="Windows standard text editor",
    ),
    ApplicationEntry(
        app_id="calculator",
        display_name="Calculator",
        executable_name="calc.exe",
        aliases=[
            "calculator",
            "calc",
            "கணக்கி",
            "கால்குலேட்டர்",
        ],
        process_names=["calc.exe", "calculatorapp.exe", "calculator.exe"],
        window_title_keywords=["Calculator", "கணக்கி"],
        uri_scheme="calculator:",
        is_system_app=True,
        description="Windows Calculator app",
    ),
    ApplicationEntry(
        app_id="chrome",
        display_name="Google Chrome",
        executable_name="chrome.exe",
        aliases=[
            "chrome",
            "google chrome",
            "google",
            "குரோம்",
            "கூகிள் குரோம்",
            "browser",
            "உலாவி",
        ],
        process_names=["chrome.exe"],
        window_title_keywords=["Google Chrome", "Chrome"],
        description="Google Chrome web browser",
    ),
    ApplicationEntry(
        app_id="edge",
        display_name="Microsoft Edge",
        executable_name="msedge.exe",
        aliases=[
            "edge",
            "microsoft edge",
            "ms edge",
            "எட்ஜ்",
        ],
        process_names=["msedge.exe"],
        window_title_keywords=["Microsoft Edge", "Edge"],
        description="Microsoft Edge web browser",
    ),
    ApplicationEntry(
        app_id="vscode",
        display_name="Visual Studio Code",
        executable_name="code.cmd",
        aliases=[
            "vs code",
            "vscode",
            "code",
            "visual studio code",
            "விஎஸ் கோட்",
            "கோட்",
        ],
        process_names=["code.exe"],
        window_title_keywords=["Visual Studio Code"],
        description="Visual Studio Code development environment",
    ),
    ApplicationEntry(
        app_id="explorer",
        display_name="File Explorer",
        executable_name="explorer.exe",
        aliases=[
            "explorer",
            "file explorer",
            "windows explorer",
            "files",
            "folders",
            "கோப்பு மேலாளர்",
            "கோப்புகள்",
        ],
        process_names=["explorer.exe"],
        window_title_keywords=["File Explorer", "Explorer"],
        is_system_app=True,
        description="Windows File Explorer",
    ),
    ApplicationEntry(
        app_id="settings",
        display_name="Windows Settings",
        executable_name="SystemSettings.exe",
        aliases=[
            "settings",
            "windows settings",
            "ms-settings",
            "control panel",
            "அமைப்புகள்",
        ],
        process_names=["systemsettings.exe"],
        window_title_keywords=["Settings", "அமைப்புகள்"],
        uri_scheme="ms-settings:",
        is_system_app=True,
        description="Windows Settings configuration portal",
    ),
    ApplicationEntry(
        app_id="taskmgr",
        display_name="Task Manager",
        executable_name="taskmgr.exe",
        aliases=[
            "task manager",
            "taskmgr",
            "task man",
            "பணி மேலாளர்",
        ],
        process_names=["taskmgr.exe"],
        window_title_keywords=["Task Manager", "பணி மேலாளர்"],
        is_system_app=True,
        description="Windows Task Manager performance monitor",
    ),
    ApplicationEntry(
        app_id="paint",
        display_name="Paint",
        executable_name="mspaint.exe",
        aliases=[
            "paint",
            "mspaint",
            "ms paint",
            "பெயின்ட்",
        ],
        process_names=["mspaint.exe", "paint.exe"],
        window_title_keywords=["Paint", "Untitled - Paint", "பெயின்ட்"],
        is_system_app=True,
        description="Microsoft Paint graphics editor",
    ),
]


class ApplicationRegistry:
    """Registry maintaining verified applications, aliases, and executable paths."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or get_settings()
        self._apps: dict[str, ApplicationEntry] = {}
        self._alias_index: dict[str, str] = {}  # alias -> app_id

        # Register default applications
        for app in INITIAL_APPLICATIONS:
            self.register_app(app)

        # Initial discovery
        self.discover_installed_apps()

    def register_app(self, entry: ApplicationEntry) -> None:
        """Register or update an application entry."""
        self._apps[entry.app_id] = entry
        # Index canonical id and display name
        self._alias_index[entry.app_id.lower()] = entry.app_id
        self._alias_index[entry.display_name.lower()] = entry.app_id
        # Index all aliases
        for alias in entry.aliases:
            clean_alias = alias.strip().lower()
            if clean_alias:
                self._alias_index[clean_alias] = entry.app_id

    def get_app(self, app_id: str) -> Optional[ApplicationEntry]:
        """Retrieve application by canonical ID."""
        return self._apps.get(app_id.lower())

    def list_supported_apps(self) -> list[ApplicationEntry]:
        """List all registered applications."""
        return list(self._apps.values())

    def resolve_alias(self, query: str) -> Optional[ApplicationEntry]:
        """Resolve a natural-language query or alias to an ApplicationEntry.

        Handles common Tamil/Tanglish affixes like '-ku', '-ai', ' ah', ' la', etc.
        """
        if not query:
            return None

        clean = query.strip().lower()

        # 1. Exact alias index lookup
        if clean in self._alias_index:
            app_id = self._alias_index[clean]
            return self._apps.get(app_id)

        # 2. Strip common speech/colloquial noise particles
        # Tamil suffixes: -ku, -க்கு, -ai, -ஐ, -oda, -la, -il, -ல, -இல்
        # Tanglish particles: ah, pa, da, va, please, open, close, switch
        normalized = re.sub(
            r"(-?(?:ku|க்கு|ai|ஐ|oda|la|ல|il|இல்)|(?:\s+(?:ah|pa|da|va|nu|app|application|please|the|a|an)))+$",
            "",
            clean,
        ).strip()

        if normalized in self._alias_index:
            app_id = self._alias_index[normalized]
            return self._apps.get(app_id)

        # 3. Leading noise removal (e.g. "open chrome" -> "chrome")
        leading_stripped = re.sub(
            r"^(?:open|launch|start|run|switch\s+to|focus|bring\s+up|close|quit|exit|minimize|maximize|restore|திற|துவக்கு|மூடு|மாற்று)\s+",
            "",
            clean,
        ).strip()

        if leading_stripped in self._alias_index:
            app_id = self._alias_index[leading_stripped]
            return self._apps.get(app_id)

        # Check stripped of both leading and trailing
        both_stripped = re.sub(
            r"(-?(?:ku|க்கு|ai|ஐ|oda|la|ல|il|இல்)|(?:\s+(?:ah|pa|da|va|nu|app|application)))+$",
            "",
            leading_stripped,
        ).strip()

        if both_stripped in self._alias_index:
            app_id = self._alias_index[both_stripped]
            return self._apps.get(app_id)

        # 4. Partial substring match across registered applications
        for app in self._apps.values():
            if clean in app.display_name.lower() or app.display_name.lower() in clean:
                return app
            for alias in app.aliases:
                if clean in alias or alias in clean:
                    return app

        return None

    def validate_executable_path(self, path_or_uri: str) -> tuple[bool, str]:
        """Validate an executable path or URI against security requirements.

        Prevents command injection, disallows unauthorized script execution,
        and ensures target is safe to launch.
        """
        target = path_or_uri.strip()
        if not target:
            return False, "Target path is empty."

        # 1. Allowed URI schemes
        allowed_schemes = ("ms-settings:", "calculator:", "microsoft.windows.camera:")
        for scheme in allowed_schemes:
            if target.lower().startswith(scheme):
                return True, f"Valid Windows URI scheme: {target}"

        # 2. Reject shell injection characters
        forbidden_chars = [";", "&", "|", "`", "$", ">", "<", "\n", "\r"]
        for ch in forbidden_chars:
            if ch in target:
                return False, f"Prohibited shell injection character detected: '{ch}'."

        # 3. Path traversal check
        if ".." in target:
            return False, "Path traversal sequence '..' is prohibited."

        # 4. Resolve absolute path or search PATH
        resolved = None
        target_path = Path(target)
        if target_path.is_absolute() and target_path.exists():
            resolved = str(target_path.resolve())
        else:
            which_path = shutil.which(target)
            if which_path and Path(which_path).exists():
                resolved = str(Path(which_path).resolve())
            else:
                reg_path = self.lookup_registry_app_path(target)
                if reg_path and Path(reg_path).exists():
                    resolved = str(Path(reg_path).resolve())

        if not resolved:
            return False, f"Executable target '{target}' does not exist on disk or in PATH."

        # 5. Extension check
        allowed_extensions = (".exe", ".cmd", ".bat")
        if not any(resolved.lower().endswith(ext) for ext in allowed_extensions):
            return False, f"Target '{resolved}' is not an approved executable format {allowed_extensions}."

        return True, resolved

    def lookup_registry_app_path(self, app_exe: str) -> Optional[str]:
        """Safely query Windows App Paths in registry (HKLM and HKCU)."""
        exe_name = app_exe if app_exe.endswith(".exe") else f"{app_exe}.exe"

        reg_roots = [winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER]
        for root in reg_roots:
            key_path = rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{exe_name}"
            try:
                with winreg.OpenKey(root, key_path, 0, winreg.KEY_READ) as key:
                    val, _ = winreg.QueryValueEx(key, "")
                    if val and Path(val).exists():
                        return str(Path(val).resolve())
            except OSError:
                continue

        return None

    def discover_installed_apps(self, force_refresh: bool = False) -> list[ApplicationEntry]:
        """Safely discover which registered applications are actually installed on the system."""
        for app in self._apps.values():
            if app.is_installed and not force_refresh:
                continue

            # 1. URI scheme based apps (e.g. Settings, modern Calculator) are always available on Windows
            if app.uri_scheme and app.is_system_app:
                app.is_installed = True
                app.verified_path = app.uri_scheme
                continue

            # 2. Check shutil.which in PATH
            which_res = shutil.which(app.executable_name)
            if which_res and Path(which_res).exists():
                app.is_installed = True
                app.verified_path = str(Path(which_res).resolve())
                continue

            # 3. Check Windows App Paths in Registry
            reg_res = self.lookup_registry_app_path(app.executable_name)
            if reg_res and Path(reg_res).exists():
                app.is_installed = True
                app.verified_path = str(Path(reg_res).resolve())
                continue

            # 4. Check known Windows installation directories
            standard_dirs = [
                os.environ.get("ProgramFiles", r"C:\Program Files"),
                os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                os.environ.get("LOCALAPPDATA", ""),
                os.environ.get("APPDATA", ""),
                os.environ.get("SystemRoot", r"C:\Windows"),
                rf"{os.environ.get('SystemRoot', r'C:\Windows')}\System32",
            ]

            found_path = None
            for base in standard_dirs:
                if not base or not os.path.isdir(base):
                    continue
                # Common subpaths
                candidate = Path(base) / app.executable_name
                if candidate.exists():
                    found_path = str(candidate.resolve())
                    break

            if found_path:
                app.is_installed = True
                app.verified_path = found_path
            elif app.is_system_app:
                # Fallback for system apps like calc or explorer
                app.is_installed = True
                app.verified_path = app.executable_name
            else:
                app.is_installed = False
                app.verified_path = None

        installed = [a for a in self._apps.values() if a.is_installed]
        logger.debug("Application registry discovered %d installed applications.", len(installed))
        return installed

    def is_app_installed(self, app_id_or_alias: str) -> bool:
        """Check whether the requested application is installed on this machine."""
        app = self.resolve_alias(app_id_or_alias)
        if not app:
            return False
        return app.is_installed

