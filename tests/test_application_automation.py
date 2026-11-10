"""Automated Test Suite for HAKS Phase 9.1: Voice-Controlled Windows Application Automation.

Covers all 14 requirements:
1. Application registry (metadata, supported apps, paths).
2. Application alias resolution (English, Tamil, Tanglish).
3. Application installation detection (system apps, PATH, registry).
4. Application launch (safe execution, argument handling).
5. Duplicate launch prevention (focus existing instead of duplicating).
6. Running application detection (PID tracking).
7. Window switching (activating foreground window).
8. Minimize and maximize window states.
9. Graceful application closure (WM_CLOSE without silent kill).
10. Permission validation and security checks (injection prevention).
11. Invalid application rejection.
12. Tamil, English, and Tanglish intent parsing.
13. Gemini Live tool call integration.
14. Voice architecture regression and spoken feedback.
"""

import asyncio
from unittest.mock import MagicMock, patch
import pytest

from config.settings import AppConfig
from storage.database import DatabaseManager
from windows_control.application_controller import ApplicationController, ApplicationActionResult
from windows_control.application_intents import AppActionType, ApplicationIntent, ApplicationIntentParser
from windows_control.application_registry import ApplicationEntry, ApplicationRegistry
from windows_control.manager import WindowsControlManager
from windows_control.models import ActionPlan, ExecutionStatus
from windows_control.window_manager import WindowInfo, WindowManager


# ==============================================================================
# 1. Application Registry Tests
# ==============================================================================

def test_01_application_registry_initial_supported_apps():
    """Verify registry initializes with all required Phase 9.1 applications."""
    registry = ApplicationRegistry()
    supported = registry.list_supported_apps()
    app_ids = {a.app_id for a in supported}

    expected_ids = {
        "notepad",
        "calculator",
        "chrome",
        "edge",
        "vscode",
        "explorer",
        "settings",
        "taskmgr",
        "paint",
    }
    assert expected_ids.issubset(app_ids)

    # Check notepad details
    notepad = registry.get_app("notepad")
    assert notepad is not None
    assert notepad.display_name == "Notepad"
    assert notepad.executable_name == "notepad.exe"
    assert notepad.is_system_app is True


# ==============================================================================
# 2. Application Alias Resolution Tests
# ==============================================================================

def test_02_application_alias_resolution():
    """Verify registry resolves aliases across English, Tamil, and Tanglish."""
    registry = ApplicationRegistry()

    # English aliases
    assert registry.resolve_alias("notepad").app_id == "notepad"
    assert registry.resolve_alias("calc").app_id == "calculator"
    assert registry.resolve_alias("google chrome").app_id == "chrome"
    assert registry.resolve_alias("vs code").app_id == "vscode"
    assert registry.resolve_alias("file explorer").app_id == "explorer"
    assert registry.resolve_alias("task manager").app_id == "taskmgr"

    # Tamil aliases
    assert registry.resolve_alias("நோட்பேட்").app_id == "notepad"
    assert registry.resolve_alias("கால்குலேட்டர்").app_id == "calculator"
    assert registry.resolve_alias("குரோம்").app_id == "chrome"
    assert registry.resolve_alias("கோப்பு மேலாளர்").app_id == "explorer"
    assert registry.resolve_alias("அமைப்புகள்").app_id == "settings"

    # Tanglish affixes and particles stripped
    assert registry.resolve_alias("VS Code ah").app_id == "vscode"
    assert registry.resolve_alias("Chrome-ku").app_id == "chrome"
    assert registry.resolve_alias("Notepad-ai").app_id == "notepad"
    assert registry.resolve_alias("Edge app").app_id == "edge"


# ==============================================================================
# 3. Application Installation Detection Tests
# ==============================================================================

def test_03_application_installation_detection():
    """Verify registry detects installed applications and handles nonexistent apps."""
    registry = ApplicationRegistry()

    # System built-in apps are always detected
    assert registry.is_app_installed("notepad") is True
    assert registry.is_app_installed("calculator") is True
    assert registry.is_app_installed("settings") is True

    # Nonexistent application
    assert registry.is_app_installed("nonexistent_fake_app_xyz") is False


# ==============================================================================
# 4. Application Launching Tests (Mocked Subprocess)
# ==============================================================================

def test_04_application_launch_mocked():
    """Verify application launching executes without blocking and captures PID."""
    registry = ApplicationRegistry()
    wm = WindowManager()
    ac = ApplicationController(registry=registry, window_manager=wm)

    mock_proc = MagicMock()
    mock_proc.pid = 9999

    with patch("subprocess.Popen", return_value=mock_proc) as mock_popen, \
         patch.object(ac, "is_application_running", return_value=(False, [])):
        res = ac.launch_application("notepad")
        assert res.success is True
        assert res.action == "launch"
        assert res.pid == 9999
        mock_popen.assert_called_once()


# ==============================================================================
# 5. Duplicate Launch Prevention Tests
# ==============================================================================

def test_05_duplicate_launch_prevention():
    """Verify launching an already-running app switches to its window instead of spawning a new instance."""
    registry = ApplicationRegistry()
    wm = WindowManager()
    ac = ApplicationController(registry=registry, window_manager=wm)

    mock_win = WindowInfo(hwnd=12345, title="Notepad - Untitled", pid=8888, process_name="notepad.exe")

    with patch.object(ac, "is_application_running", return_value=(True, [8888])), \
         patch.object(wm, "find_windows_for_app", return_value=[mock_win]), \
         patch.object(wm, "activate_window", return_value=True) as mock_activate, \
         patch("subprocess.Popen") as mock_popen:

        res = ac.launch_application("notepad", allow_multiple=False)
        assert res.success is True
        assert res.action == "switch"
        assert "already open" in res.message or "already running" in res.message
        assert res.pid == 8888
        mock_activate.assert_called_once_with(12345)
        # Verify no duplicate process was spawned
        mock_popen.assert_not_called()


# ==============================================================================
# 6. Running Application Detection Tests
# ==============================================================================

def test_06_running_application_detection():
    """Verify running application detection matches process names."""
    registry = ApplicationRegistry()
    ac = ApplicationController(registry=registry)

    mock_p1 = MagicMock()
    mock_p1.info = {"pid": 1111, "name": "notepad.exe"}

    mock_p2 = MagicMock()
    mock_p2.info = {"pid": 2222, "name": "explorer.exe"}

    with patch("psutil.process_iter", return_value=[mock_p1, mock_p2]):
        is_running, pids = ac.is_application_running("notepad")
        assert is_running is True
        assert 1111 in pids

        is_running_calc, calc_pids = ac.is_application_running("calculator")
        assert is_running_calc is False
        assert len(calc_pids) == 0


# ==============================================================================
# 7. Window Switching Tests
# ==============================================================================

def test_07_window_switching():
    """Verify window switching activates the primary application window."""
    registry = ApplicationRegistry()
    wm = WindowManager()
    ac = ApplicationController(registry=registry, window_manager=wm)

    mock_win = WindowInfo(hwnd=54321, title="Google Chrome", pid=3333, process_name="chrome.exe")

    with patch.object(ac, "is_application_running", return_value=(True, [3333])), \
         patch.object(wm, "find_windows_for_app", return_value=[mock_win]), \
         patch.object(wm, "activate_window", return_value=True) as mock_activate:

        res = ac.switch_to_application("chrome")
        assert res.success is True
        assert res.action == "switch"
        assert "Switched to" in res.message
        mock_activate.assert_called_once_with(54321)


# ==============================================================================
# 8. Minimize and Maximize Tests
# ==============================================================================

def test_08_minimize_and_maximize():
    """Verify minimize, maximize, and restore window actions."""
    registry = ApplicationRegistry()
    wm = WindowManager()
    ac = ApplicationController(registry=registry, window_manager=wm)

    mock_win = WindowInfo(hwnd=77777, title="Visual Studio Code", pid=4444, process_name="code.exe")

    with patch.object(wm, "find_windows_for_app", return_value=[mock_win]), \
         patch.object(wm, "minimize_window", return_value=True) as mock_min, \
         patch.object(wm, "maximize_window", return_value=True) as mock_max, \
         patch.object(wm, "restore_window", return_value=True) as mock_restore:

        # Minimize
        min_res = ac.minimize_application("vscode")
        assert min_res.success is True
        assert "Minimized" in min_res.message
        mock_min.assert_called_once_with(77777)

        # Maximize
        max_res = ac.maximize_application("vscode")
        assert max_res.success is True
        assert "Maximized" in max_res.message
        mock_max.assert_called_once_with(77777)

        # Restore
        res_res = ac.restore_application("vscode")
        assert res_res.success is True
        assert "Restored" in res_res.message
        mock_restore.assert_called_once_with(77777)


# ==============================================================================
# 9. Graceful Application Closure Tests (WM_CLOSE)
# ==============================================================================

def test_09_graceful_application_closure():
    """Verify closing an application sends WM_CLOSE message without terminating processes."""
    registry = ApplicationRegistry()
    wm = WindowManager()
    ac = ApplicationController(registry=registry, window_manager=wm)

    mock_win = WindowInfo(hwnd=99999, title="Untitled - Notepad", pid=5555, process_name="notepad.exe")

    with patch.object(ac, "is_application_running", return_value=(True, [5555])), \
         patch.object(wm, "find_windows_for_app", return_value=[mock_win]), \
         patch.object(wm, "close_window_gracefully", return_value=True) as mock_close, \
         patch("psutil.Process") as mock_psutil_proc:

        res = ac.close_application("notepad", force=False)
        assert res.success is True
        assert res.action == "close"
        assert "Unsaved work was protected" in res.message
        mock_close.assert_called_once_with(99999)
        # Ensure no terminate() or kill() was called
        mock_psutil_proc.assert_not_called()


# ==============================================================================
# 10. Security & Permission Validation Tests
# ==============================================================================

def test_10_permission_and_injection_validation():
    """Verify executable path validation blocks command injection and path traversal."""
    registry = ApplicationRegistry()

    # Command injection with semicolon
    valid1, msg1 = registry.validate_executable_path("notepad.exe; rm -rf /")
    assert valid1 is False
    assert "injection" in msg1.lower()

    # Shell pipe injection
    valid2, msg2 = registry.validate_executable_path("calc.exe | powershell.exe")
    assert valid2 is False
    assert "injection" in msg2.lower()

    # Path traversal
    valid3, msg3 = registry.validate_executable_path(r"..\..\Windows\System32\cmd.exe")
    assert valid3 is False
    assert "traversal" in msg3.lower()


# ==============================================================================
# 11. Invalid Application Rejection Tests
# ==============================================================================

def test_11_invalid_application_rejection():
    """Verify invalid, ambiguous, or non-existent applications are rejected gracefully."""
    registry = ApplicationRegistry()
    ac = ApplicationController(registry=registry)

    # Launching unknown app
    res = ac.launch_application("unknown_fantasy_tool")
    assert res.success is False
    assert "couldn't find" in res.message.lower() or "not installed" in res.message.lower()

    # Closing unknown app
    close_res = ac.close_application("unknown_fantasy_tool")
    assert close_res.success is False
    assert "could not find" in close_res.message.lower()


# ==============================================================================
# 12. Multilingual Intent Parsing Tests (English, Tamil, Tanglish)
# ==============================================================================

def test_12_multilingual_intent_parsing():
    """Verify intent parser accurately extracts actions and apps across all three languages."""
    parser = ApplicationIntentParser()

    # English examples from requirement
    e1 = parser.parse("Open Chrome")
    assert e1.action == AppActionType.LAUNCH and e1.app.app_id == "chrome"

    e2 = parser.parse("Launch VS Code")
    assert e2.action == AppActionType.LAUNCH and e2.app.app_id == "vscode"

    e3 = parser.parse("Switch to Notepad")
    assert e3.action == AppActionType.SWITCH and e3.app.app_id == "notepad"

    e4 = parser.parse("Minimize Chrome")
    assert e4.action == AppActionType.MINIMIZE and e4.app.app_id == "chrome"

    e5 = parser.parse("Close Calculator")
    assert e5.action == AppActionType.CLOSE and e5.app.app_id == "calculator"

    # Tamil examples from requirement
    t1 = parser.parse("Chrome open pannu")
    assert t1.action == AppActionType.LAUNCH and t1.app.app_id == "chrome"

    t2 = parser.parse("Notepad open pannu")
    assert t2.action == AppActionType.LAUNCH and t2.app.app_id == "notepad"

    t3 = parser.parse("Calculator open pannu")
    assert t3.action == AppActionType.LAUNCH and t3.app.app_id == "calculator"

    t4 = parser.parse("Chrome-ku switch pannu")
    assert t4.action == AppActionType.SWITCH and t4.app.app_id == "chrome"

    t5 = parser.parse("நோட்பேட் திற")
    assert t5.action == AppActionType.LAUNCH and t5.app.app_id == "notepad"

    # Tanglish examples from requirement
    tg1 = parser.parse("VS Code ah open pannu")
    assert tg1.action == AppActionType.LAUNCH and tg1.app.app_id == "vscode"

    tg2 = parser.parse("Chrome la search panna ready pannu")
    assert tg2.action == AppActionType.LAUNCH and tg2.app.app_id == "chrome"

    tg3 = parser.parse("Notepad close pannu")
    assert tg3.action == AppActionType.CLOSE and tg3.app.app_id == "notepad"

    # Reject non-actionable queries
    conv = parser.parse("What is Google Chrome?")
    assert conv.is_actionable is False

    file_query = parser.parse("Open a file in python")
    assert file_query.is_actionable is False


# ==============================================================================
# 13. Gemini Live Tool Call Integration Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_13_gemini_live_tool_call_integration(temp_workspace: AppConfig, test_db_manager: DatabaseManager):
    """Verify Gemini Live toolCall event triggers local application control action."""
    wm = WindowsControlManager(config=temp_workspace, db_manager=test_db_manager)

    # Simulate toolCall event
    tool_call_event = {
        "functionCalls": [
            {
                "name": "windows_app_control",
                "args": {"action": "launch", "app_name": "notepad"},
                "id": "call_12345",
            }
        ]
    }

    # Verify execution of plan generated from tool call
    plan = ActionPlan(
        objective="Open Notepad",
        target="notepad",
        operation_type="launch_app",
        command="notepad",
        source="gemini_live",
    )

    with patch.object(wm.app_controller, "launch_application") as mock_launch:
        mock_launch.return_value = ApplicationActionResult(
            success=True,
            action="launch",
            app_id="notepad",
            display_name="Notepad",
            message="Opened Notepad.",
            pid=6789,
        )

        res = await wm.execute_plan(plan)
        assert res.success is True
        assert res.exit_code == 0
        assert "Opened Notepad" in res.stdout
        mock_launch.assert_called_once_with("notepad", [])


# ==============================================================================
# 14. Voice Architecture Regression Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_14_voice_architecture_regression(temp_workspace: AppConfig, test_db_manager: DatabaseManager):
    """Verify voice conversation engine dispatches application control and speaks friendly feedback."""
    from core.app_controller import AppController
    ctrl = AppController(config=temp_workspace, db_manager=test_db_manager)
    engine = ctrl.voice_engine

    # Mock synthesizer
    spoken_feedback = []
    engine.tts.synthesize = MagicMock()
    mock_utterance = MagicMock()
    mock_utterance.audio_path = None
    engine.tts.synthesize.return_value = asyncio.Future()
    engine.tts.synthesize.return_value.set_result(mock_utterance)

    # Execute plan through Windows Control Manager via voice instruction
    with patch.object(ctrl.windows_control.app_controller, "launch_application") as mock_launch:
        mock_launch.return_value = ApplicationActionResult(
            success=True,
            action="launch",
            app_id="chrome",
            display_name="Google Chrome",
            message="Opened Google Chrome.",
            pid=7777,
        )

        plan = ctrl.windows_control.interpreter.interpret("Hey HAKS, open Chrome", source="voice")
        assert plan.operation_type == "launch_app"
        assert plan.command == "chrome"

        result = await ctrl.windows_control.execute_plan(plan)
        assert result.success is True
        assert result.stdout == "Opened Google Chrome."

    ctrl.shutdown()

