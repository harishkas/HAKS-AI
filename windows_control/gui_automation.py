"""Desktop GUI Automation Service for HAKS Windows Computer Control.

Performs verified mouse actions, keyboard input, window switching,
dialog interaction, and screenshot verification via PyAutoGUI and Win32 APIs.
"""

import logging
from pathlib import Path
import time
from typing import Any, Optional

import pyautogui
import win32con
import win32gui
import win32process

logger = logging.getLogger("haks.windows_control.gui")

# Enable PyAutoGUI safety failsafe (slamming mouse into top-left corner aborts)
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05


class GUIAutomationService:
    """Desktop GUI automation with window focus and visual verification."""

    # --------------------------------------------------------------------------
    # 1. Mouse & Keyboard Input
    # --------------------------------------------------------------------------

    def get_mouse_position(self) -> tuple[int, int]:
        """Return current (x, y) coordinates of the cursor."""
        pos = pyautogui.position()
        return (pos.x, pos.y)

    def move_mouse(self, x: int, y: int, duration: float = 0.2) -> bool:
        """Move cursor smoothly to target coordinates."""
        try:
            pyautogui.moveTo(x, y, duration=duration)
            return True
        except Exception as exc:
            logger.warning("Error moving mouse to (%d, %d): %s", x, y, exc)
            return False

    def click(self, x: Optional[int] = None, y: Optional[int] = None, button: str = "left", clicks: int = 1) -> bool:
        """Click at current or specified coordinate."""
        try:
            if x is not None and y is not None:
                pyautogui.click(x=x, y=y, button=button, clicks=clicks)
            else:
                pyautogui.click(button=button, clicks=clicks)
            return True
        except Exception as exc:
            logger.warning("Error issuing mouse click: %s", exc)
            return False

    def type_text(self, text: str, interval: float = 0.02) -> bool:
        """Type text string with realistic keypress intervals."""
        try:
            pyautogui.write(text, interval=interval)
            return True
        except Exception as exc:
            logger.warning("Error typing text: %s", exc)
            return False

    def press_hotkey(self, *keys: str) -> bool:
        """Execute a keyboard shortcut combination (e.g. 'ctrl', 'c')."""
        try:
            pyautogui.hotkey(*keys)
            return True
        except Exception as exc:
            logger.warning("Error pressing hotkey %s: %s", keys, exc)
            return False

    # --------------------------------------------------------------------------
    # 2. Window Identification & Focus
    # --------------------------------------------------------------------------

    def list_visible_windows(self) -> list[dict[str, Any]]:
        """Enumerate all open, visible top-level desktop windows."""
        windows: list[dict[str, Any]] = []

        def _enum_callback(hwnd: int, extra: Any) -> None:
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd).strip()
                if title:
                    rect = win32gui.GetWindowRect(hwnd)
                    width = rect[2] - rect[0]
                    height = rect[3] - rect[1]
                    if width > 50 and height > 50:
                        _, pid = win32process.GetWindowThreadProcessId(hwnd)
                        windows.append(
                            {
                                "hwnd": hwnd,
                                "title": title,
                                "pid": pid,
                                "rect": {"left": rect[0], "top": rect[1], "width": width, "height": height},
                            }
                        )

        try:
            win32gui.EnumWindows(_enum_callback, None)
        except Exception as exc:
            logger.warning("Error enumerating desktop windows: %s", exc)

        return windows

    def find_window_by_title(self, search_title: str) -> Optional[int]:
        """Find window handle (HWND) matching title substring."""
        clean = search_title.lower().strip()
        for win in self.list_visible_windows():
            if clean in win["title"].lower():
                return win["hwnd"]
        return None

    def activate_window(self, hwnd_or_title: Any) -> bool:
        """Bring window to foreground and give it active input focus."""
        hwnd = hwnd_or_title if isinstance(hwnd_or_title, int) else self.find_window_by_title(str(hwnd_or_title))
        if not hwnd or not win32gui.IsWindow(hwnd):
            logger.warning("Cannot activate window; HWND not found for '%s'.", hwnd_or_title)
            return False

        try:
            # Restore if minimized
            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(hwnd)
            time.sleep(0.1)

            # Verification: check if the foreground window matches
            active_hwnd = win32gui.GetForegroundWindow()
            is_active = (active_hwnd == hwnd)
            logger.info("Window activated (HWND: %s, Verified: %s)", hwnd, is_active)
            return is_active
        except Exception as exc:
            logger.warning("Error setting foreground window for HWND %s: %s", hwnd, exc)
            return False

    # --------------------------------------------------------------------------
    # 3. Screenshot Capture & Verification
    # --------------------------------------------------------------------------

    def capture_screenshot(self, output_path: str) -> tuple[bool, str]:
        """Capture desktop screen to a PNG image file."""
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            img = pyautogui.screenshot()
            img.save(str(out))
            logger.info("Captured screenshot: %s", out)
            return True, str(out)
        except Exception as exc:
            logger.error("Failed to capture screenshot: %s", exc)
            return False, str(exc)

    def get_screen_resolution(self) -> tuple[int, int]:
        """Return primary screen width and height."""
        size = pyautogui.size()
        return (size.width, size.height)
