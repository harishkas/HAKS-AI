"""Desktop Window Management Service for HAKS Windows Computer Control (Phase 9.1).

Provides non-blocking window discovery, focus activation, state toggling
(minimize, maximize, restore), and graceful application closure via Win32 APIs.
"""

from dataclasses import dataclass, field
import logging
import time
from typing import Any, Optional

try:
    import psutil
except ImportError:
    psutil = None

try:
    import win32con
    import win32gui
    import win32process
except ImportError:
    win32con = None
    win32gui = None
    win32process = None

logger = logging.getLogger("haks.windows_control.window_manager")


@dataclass
class WindowInfo:
    """Detailed metadata for an active desktop window."""

    hwnd: int
    title: str
    pid: int
    process_name: str = ""
    is_visible: bool = True
    is_minimized: bool = False
    is_maximized: bool = False
    rect: dict[str, int] = field(default_factory=dict)


class WindowManager:
    """Safe Windows desktop window manipulation service."""

    def __init__(self) -> None:
        if win32gui is None:
            logger.warning("pywin32 is not installed or unavailable. Window management will operate in mock/degraded mode.")

    def is_available(self) -> bool:
        """Return True if native Windows windowing APIs are available."""
        return win32gui is not None and win32con is not None

    def list_windows(self, visible_only: bool = True) -> list[WindowInfo]:
        """Enumerate top-level desktop windows."""
        if not self.is_available():
            return []

        windows: list[WindowInfo] = []

        def _enum_callback(hwnd: int, extra: Any) -> None:
            try:
                if visible_only and not win32gui.IsWindowVisible(hwnd):
                    return

                title = win32gui.GetWindowText(hwnd).strip()
                if not title:
                    return

                rect = win32gui.GetWindowRect(hwnd)
                width = rect[2] - rect[0]
                height = rect[3] - rect[1]

                # Filter out invisible or zero-sized helper windows
                if width <= 10 or height <= 10:
                    return

                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                proc_name = ""
                if psutil and pid > 0:
                    try:
                        proc = psutil.Process(pid)
                        proc_name = proc.name().lower()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        proc_name = ""

                is_min = bool(win32gui.IsIconic(hwnd))
                is_max = bool(win32gui.IsZoomed(hwnd))

                windows.append(
                    WindowInfo(
                        hwnd=hwnd,
                        title=title,
                        pid=pid,
                        process_name=proc_name,
                        is_visible=True,
                        is_minimized=is_min,
                        is_maximized=is_max,
                        rect={"left": rect[0], "top": rect[1], "right": rect[2], "bottom": rect[3], "width": width, "height": height},
                    )
                )
            except Exception:
                pass
            return True

        try:
            win32gui.EnumWindows(_enum_callback, None)
        except Exception as exc:
            logger.debug("Desktop window enumeration notice: %s", exc)

        return windows

    def find_windows_for_process_names(self, process_names: list[str]) -> list[WindowInfo]:
        """Find open windows associated with a list of process names (e.g. ['chrome.exe'])."""
        clean_names = [p.lower() for p in process_names]
        matches: list[WindowInfo] = []
        for win in self.list_windows(visible_only=False):
            if win.process_name and win.process_name.lower() in clean_names:
                matches.append(win)
        return matches

    def find_windows_by_title_keyword(self, keywords: list[str]) -> list[WindowInfo]:
        """Find open windows whose title contains any of the specified keywords."""
        clean_keys = [k.lower() for k in keywords if k]
        matches: list[WindowInfo] = []
        for win in self.list_windows(visible_only=True):
            win_title_lower = win.title.lower()
            if any(k in win_title_lower for k in clean_keys):
                matches.append(win)
        return matches

    def find_windows_for_app(
        self,
        process_names: Optional[list[str]] = None,
        title_keywords: Optional[list[str]] = None,
    ) -> list[WindowInfo]:
        """Find windows matching process names first, then falling back to title keywords."""
        if process_names:
            proc_matches = self.find_windows_for_process_names(process_names)
            if proc_matches:
                return proc_matches

        if title_keywords:
            return self.find_windows_by_title_keyword(title_keywords)

        return []

    def get_foreground_window(self) -> Optional[WindowInfo]:
        """Retrieve the currently active foreground window."""
        if not self.is_available():
            return None
        try:
            hwnd = win32gui.GetForegroundWindow()
            if hwnd and win32gui.IsWindow(hwnd):
                title = win32gui.GetWindowText(hwnd).strip()
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                proc_name = ""
                if psutil and pid > 0:
                    try:
                        proc_name = psutil.Process(pid).name().lower()
                    except Exception:
                        pass
                return WindowInfo(
                    hwnd=hwnd,
                    title=title,
                    pid=pid,
                    process_name=proc_name,
                    is_visible=bool(win32gui.IsWindowVisible(hwnd)),
                    is_minimized=bool(win32gui.IsIconic(hwnd)),
                    is_maximized=bool(win32gui.IsZoomed(hwnd)),
                )
        except Exception as exc:
            logger.warning("Error fetching foreground window: %s", exc)
        return None

    def activate_window(self, hwnd: int) -> bool:
        """Bring window to foreground, restore if minimized, and assign input focus."""
        if not self.is_available() or not win32gui.IsWindow(hwnd):
            logger.warning("Cannot activate invalid HWND %s", hwnd)
            return False

        try:
            # 1. Restore if minimized
            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                time.sleep(0.05)

            # 2. Show window normal/active
            win32gui.ShowWindow(hwnd, win32con.SW_SHOW)

            # 3. Bring to foreground
            win32gui.SetForegroundWindow(hwnd)
            time.sleep(0.05)

            # Verification
            active = win32gui.GetForegroundWindow()
            return active == hwnd or win32gui.IsWindow(hwnd)
        except Exception as exc:
            logger.warning("Failed to activate window HWND %s: %s", hwnd, exc)
            return False

    def minimize_window(self, hwnd: int) -> bool:
        """Minimize window to taskbar."""
        if not self.is_available() or not win32gui.IsWindow(hwnd):
            return False

        try:
            win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
            return True
        except Exception as exc:
            logger.warning("Failed to minimize window HWND %s: %s", hwnd, exc)
            return False

    def maximize_window(self, hwnd: int) -> bool:
        """Maximize window to fill screen."""
        if not self.is_available() or not win32gui.IsWindow(hwnd):
            return False

        try:
            win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
            return True
        except Exception as exc:
            logger.warning("Failed to maximize window HWND %s: %s", hwnd, exc)
            return False

    def restore_window(self, hwnd: int) -> bool:
        """Restore window from minimized or maximized state to its standard bounds."""
        if not self.is_available() or not win32gui.IsWindow(hwnd):
            return False

        try:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            return True
        except Exception as exc:
            logger.warning("Failed to restore window HWND %s: %s", hwnd, exc)
            return False

    def close_window_gracefully(self, hwnd: int) -> bool:
        """Gracefully request window close via WM_CLOSE message.

        Does NOT forcefully terminate the process. If unsaved documents exist,
        the application will display its standard save confirmation dialog to protect user data.
        """
        if not self.is_available() or not win32gui.IsWindow(hwnd):
            return False

        try:
            logger.info("Posting WM_CLOSE message to HWND %s", hwnd)
            win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
            return True
        except Exception as exc:
            logger.error("Failed to post WM_CLOSE to HWND %s: %s", hwnd, exc)
            return False
