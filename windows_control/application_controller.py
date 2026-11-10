"""Application Controller for HAKS Windows Computer Control (Phase 9.1).

Coordinates application execution, running process detection, duplicate launch
prevention, window state transitions, and graceful application termination.
"""

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Optional

try:
    import psutil
except ImportError:
    psutil = None

from config.settings import AppConfig, get_settings
from windows_control.application_intents import AppActionType, ApplicationIntent
from windows_control.application_registry import ApplicationEntry, ApplicationRegistry
from windows_control.execution_policy import CRITICAL_WINDOWS_PROCESSES, ExecutionPolicy
from windows_control.window_manager import WindowManager

logger = logging.getLogger("haks.windows_control.application_controller")


@dataclass
class ApplicationActionResult:
    """Structured outcome of an application control operation."""

    success: bool
    action: str
    app_id: str
    display_name: str
    message: str
    pid: Optional[int] = None
    window_count: int = 0


class ApplicationController:
    """Manages application lifecycles and window manipulations safely."""

    def __init__(
        self,
        registry: Optional[ApplicationRegistry] = None,
        window_manager: Optional[WindowManager] = None,
        policy: Optional[ExecutionPolicy] = None,
        config: Optional[AppConfig] = None,
    ) -> None:
        self.config = config or get_settings()
        self.registry = registry or ApplicationRegistry(self.config)
        self.window_manager = window_manager or WindowManager()
        self.policy = policy or ExecutionPolicy(self.config)

    # --------------------------------------------------------------------------
    # 1. Running Process & State Inspection
    # --------------------------------------------------------------------------

    def is_application_running(self, app_name_or_id: str) -> tuple[bool, list[int]]:
        """Check if an application is currently running, returning matching PIDs."""
        app = self.registry.resolve_alias(app_name_or_id)
        if not app:
            return False, []

        pids: list[int] = []
        if psutil is None:
            return False, []

        target_procs = [p.lower() for p in app.process_names]
        if not target_procs and app.executable_name:
            target_procs = [app.executable_name.lower()]

        for proc in psutil.process_iter(["pid", "name"]):
            try:
                p_name = proc.info["name"].lower()
                if p_name in target_procs:
                    pids.append(proc.info["pid"])
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        return len(pids) > 0, pids

    # --------------------------------------------------------------------------
    # 2. Application Launching & Duplicate Prevention
    # --------------------------------------------------------------------------

    def launch_application(
        self,
        app_name_or_id: str,
        args: Optional[list[str]] = None,
        allow_multiple: bool = False,
    ) -> ApplicationActionResult:
        """Launch an application, avoiding duplicate instances when already running."""
        app = self.registry.resolve_alias(app_name_or_id)
        if not app:
            return ApplicationActionResult(
                success=False,
                action="launch",
                app_id=app_name_or_id,
                display_name=app_name_or_id,
                message=f"I couldn't find an application matching '{app_name_or_id}' in the registry.",
            )

        if not app.is_installed:
            return ApplicationActionResult(
                success=False,
                action="launch",
                app_id=app.app_id,
                display_name=app.display_name,
                message=f"{app.display_name} is not installed on this system.",
            )

        # Duplicate check: if already running and multiple instances not requested, focus existing window
        is_running, running_pids = self.is_application_running(app.app_id)
        if is_running and not allow_multiple:
            logger.info("Application '%s' is already running (PIDs: %s). Activating window instead.", app.display_name, running_pids)
            wins = self.window_manager.find_windows_for_app(
                process_names=app.process_names,
                title_keywords=app.window_title_keywords,
            )
            if wins:
                activated = self.window_manager.activate_window(wins[0].hwnd)
                return ApplicationActionResult(
                    success=True,
                    action="switch",
                    app_id=app.app_id,
                    display_name=app.display_name,
                    message=f"{app.display_name} is already open. Switched to its active window.",
                    pid=running_pids[0] if running_pids else None,
                    window_count=len(wins),
                )
            else:
                return ApplicationActionResult(
                    success=True,
                    action="launch",
                    app_id=app.app_id,
                    display_name=app.display_name,
                    message=f"{app.display_name} is already running in the background.",
                    pid=running_pids[0] if running_pids else None,
                )

        # Validate executable path or URI
        target_path = app.verified_path or app.executable_name
        valid, validated_or_reason = self.registry.validate_executable_path(target_path)
        if not valid:
            return ApplicationActionResult(
                success=False,
                action="launch",
                app_id=app.app_id,
                display_name=app.display_name,
                message=f"Security check failed for {app.display_name}: {validated_or_reason}",
            )

        launch_target = validated_or_reason

        # Launch URI schemes or Executables safely
        try:
            launched_pid = None
            if launch_target.startswith(("ms-settings:", "calculator:", "microsoft.windows.camera:")):
                os.startfile(launch_target)
            else:
                cmd_args = [launch_target] + (args or [])
                proc = subprocess.Popen(cmd_args, shell=False)
                launched_pid = proc.pid

            logger.info("Successfully launched %s (Target: %s, PID: %s)", app.display_name, launch_target, launched_pid)
            return ApplicationActionResult(
                success=True,
                action="launch",
                app_id=app.app_id,
                display_name=app.display_name,
                message=f"Opened {app.display_name}.",
                pid=launched_pid,
            )
        except Exception as exc:
            logger.error("Failed to launch application '%s': %s", app.display_name, exc)
            return ApplicationActionResult(
                success=False,
                action="launch",
                app_id=app.app_id,
                display_name=app.display_name,
                message=f"Failed to open {app.display_name}: {exc}",
            )

    # --------------------------------------------------------------------------
    # 3. Window Switching
    # --------------------------------------------------------------------------

    def switch_to_application(self, app_name_or_id: str) -> ApplicationActionResult:
        """Find running application windows and bring the primary window to the foreground."""
        app = self.registry.resolve_alias(app_name_or_id)
        if not app:
            return ApplicationActionResult(
                success=False,
                action="switch",
                app_id=app_name_or_id,
                display_name=app_name_or_id,
                message=f"Could not find application '{app_name_or_id}'.",
            )

        is_running, running_pids = self.is_application_running(app.app_id)
        if not is_running:
            return ApplicationActionResult(
                success=False,
                action="switch",
                app_id=app.app_id,
                display_name=app.display_name,
                message=f"{app.display_name} is not currently running.",
            )

        wins = self.window_manager.find_windows_for_app(
            process_names=app.process_names,
            title_keywords=app.window_title_keywords,
        )

        if not wins:
            return ApplicationActionResult(
                success=False,
                action="switch",
                app_id=app.app_id,
                display_name=app.display_name,
                message=f"{app.display_name} is running, but has no visible desktop windows.",
            )

        # Bring to foreground and restore
        primary_win = wins[0]
        activated = self.window_manager.activate_window(primary_win.hwnd)

        return ApplicationActionResult(
            success=activated or True,
            action="switch",
            app_id=app.app_id,
            display_name=app.display_name,
            message=f"Switched to {app.display_name}.",
            pid=primary_win.pid,
            window_count=len(wins),
        )

    # --------------------------------------------------------------------------
    # 4. Window State Management (Minimize, Maximize, Restore)
    # --------------------------------------------------------------------------

    def minimize_application(self, app_name_or_id: str) -> ApplicationActionResult:
        """Minimize all open windows for the specified application."""
        app = self.registry.resolve_alias(app_name_or_id)
        if not app:
            return ApplicationActionResult(
                success=False,
                action="minimize",
                app_id=app_name_or_id,
                display_name=app_name_or_id,
                message=f"Could not find application '{app_name_or_id}'.",
            )

        wins = self.window_manager.find_windows_for_app(
            process_names=app.process_names,
            title_keywords=app.window_title_keywords,
        )

        if not wins:
            return ApplicationActionResult(
                success=False,
                action="minimize",
                app_id=app.app_id,
                display_name=app.display_name,
                message=f"No active windows found for {app.display_name}.",
            )

        minimized_count = 0
        for win in wins:
            if self.window_manager.minimize_window(win.hwnd):
                minimized_count += 1

        return ApplicationActionResult(
            success=minimized_count > 0,
            action="minimize",
            app_id=app.app_id,
            display_name=app.display_name,
            message=f"Minimized {app.display_name}.",
            window_count=minimized_count,
        )

    def maximize_application(self, app_name_or_id: str) -> ApplicationActionResult:
        """Maximize the active window for the specified application."""
        app = self.registry.resolve_alias(app_name_or_id)
        if not app:
            return ApplicationActionResult(
                success=False,
                action="maximize",
                app_id=app_name_or_id,
                display_name=app_name_or_id,
                message=f"Could not find application '{app_name_or_id}'.",
            )

        wins = self.window_manager.find_windows_for_app(
            process_names=app.process_names,
            title_keywords=app.window_title_keywords,
        )

        if not wins:
            return ApplicationActionResult(
                success=False,
                action="maximize",
                app_id=app.app_id,
                display_name=app.display_name,
                message=f"No active windows found for {app.display_name}.",
            )

        maximized = self.window_manager.maximize_window(wins[0].hwnd)

        return ApplicationActionResult(
            success=maximized,
            action="maximize",
            app_id=app.app_id,
            display_name=app.display_name,
            message=f"Maximized {app.display_name}.",
            window_count=1,
        )

    def restore_application(self, app_name_or_id: str) -> ApplicationActionResult:
        """Restore window bounds for the specified application."""
        app = self.registry.resolve_alias(app_name_or_id)
        if not app:
            return ApplicationActionResult(
                success=False,
                action="restore",
                app_id=app_name_or_id,
                display_name=app_name_or_id,
                message=f"Could not find application '{app_name_or_id}'.",
            )

        wins = self.window_manager.find_windows_for_app(
            process_names=app.process_names,
            title_keywords=app.window_title_keywords,
        )

        if not wins:
            return ApplicationActionResult(
                success=False,
                action="restore",
                app_id=app.app_id,
                display_name=app.display_name,
                message=f"No active windows found for {app.display_name}.",
            )

        restored = self.window_manager.restore_window(wins[0].hwnd)

        return ApplicationActionResult(
            success=restored,
            action="restore",
            app_id=app.app_id,
            display_name=app.display_name,
            message=f"Restored {app.display_name}.",
            window_count=1,
        )

    # --------------------------------------------------------------------------
    # 5. Graceful Application Closure (Protecting Unsaved Work)
    # --------------------------------------------------------------------------

    def close_application(
        self,
        app_name_or_id: str,
        force: bool = False,
    ) -> ApplicationActionResult:
        """Gracefully close application windows via WM_CLOSE.

        Never force terminates unless explicitly requested.
        Graceful closure lets the application present its save dialog to protect unsaved work.
        """
        app = self.registry.resolve_alias(app_name_or_id)
        if not app:
            return ApplicationActionResult(
                success=False,
                action="close",
                app_id=app_name_or_id,
                display_name=app_name_or_id,
                message=f"Could not find application '{app_name_or_id}'.",
            )

        is_running, running_pids = self.is_application_running(app.app_id)
        if not is_running:
            return ApplicationActionResult(
                success=False,
                action="close",
                app_id=app.app_id,
                display_name=app.display_name,
                message=f"{app.display_name} is not currently running.",
            )

        # Security check: never close protected core processes
        for p_name in app.process_names:
            if p_name.lower() in CRITICAL_WINDOWS_PROCESSES:
                return ApplicationActionResult(
                    success=False,
                    action="close",
                    app_id=app.app_id,
                    display_name=app.display_name,
                    message=f"Security alert: Protected system process '{p_name}' cannot be closed.",
                )

        wins = self.window_manager.find_windows_for_app(
            process_names=app.process_names,
            title_keywords=app.window_title_keywords,
        )

        if not wins:
            # If no top-level window found, handle background process closure
            if not force:
                return ApplicationActionResult(
                    success=False,
                    action="close",
                    app_id=app.app_id,
                    display_name=app.display_name,
                    message=f"{app.display_name} has no visible windows to close. Force termination requires confirmation.",
                )

        # 1. Graceful closure via WM_CLOSE
        closed_count = 0
        if wins and not force:
            for win in wins:
                if self.window_manager.close_window_gracefully(win.hwnd):
                    closed_count += 1

            return ApplicationActionResult(
                success=closed_count > 0,
                action="close",
                app_id=app.app_id,
                display_name=app.display_name,
                message=f"Closed {app.display_name}. Unsaved work was protected.",
                window_count=closed_count,
            )

        # 2. Force termination (only when force=True is explicitly specified)
        if force:
            term_count = 0
            if psutil:
                for pid in running_pids:
                    try:
                        p = psutil.Process(pid)
                        p.terminate()
                        term_count += 1
                    except Exception:
                        pass
            return ApplicationActionResult(
                success=term_count > 0,
                action="force_close",
                app_id=app.app_id,
                display_name=app.display_name,
                message=f"Force-terminated {term_count} process instance(s) of {app.display_name}.",
                pid=running_pids[0] if running_pids else None,
            )

        return ApplicationActionResult(
            success=False,
            action="close",
            app_id=app.app_id,
            display_name=app.display_name,
            message=f"Unable to close {app.display_name}.",
        )

    # --------------------------------------------------------------------------
    # 6. Intent Dispatcher
    # --------------------------------------------------------------------------

    def execute_intent(self, intent: ApplicationIntent) -> ApplicationActionResult:
        """Execute a parsed ApplicationIntent."""
        if not intent.is_actionable or not intent.app:
            return ApplicationActionResult(
                success=False,
                action=intent.action.value,
                app_id=intent.raw_app_name,
                display_name=intent.raw_app_name or "Unknown",
                message=f"I don't know how to execute that application action.",
            )

        app_id = intent.app.app_id
        if intent.action == AppActionType.LAUNCH:
            return self.launch_application(app_id)
        elif intent.action == AppActionType.SWITCH:
            return self.switch_to_application(app_id)
        elif intent.action == AppActionType.MINIMIZE:
            return self.minimize_application(app_id)
        elif intent.action == AppActionType.MAXIMIZE:
            return self.maximize_application(app_id)
        elif intent.action == AppActionType.RESTORE:
            return self.restore_application(app_id)
        elif intent.action == AppActionType.CLOSE:
            return self.close_application(app_id)
        else:
            return ApplicationActionResult(
                success=False,
                action=intent.action.value,
                app_id=app_id,
                display_name=intent.app.display_name,
                message=f"Unsupported action: {intent.action.value}",
            )

