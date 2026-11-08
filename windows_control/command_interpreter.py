"""Natural-Language Command Interpreter for HAKS Windows Computer Control.

Translates conversational user instructions in English and Tamil into
typed, policy-governed ActionPlans.
"""

import logging
import os
from pathlib import Path
import re
from typing import Optional

from windows_control.execution_policy import ExecutionPolicy
from windows_control.models import (
    ActionPlan,
    PrivilegeLevel,
    RiskLevel,
)
from windows_control.privilege_detector import PrivilegeDetector

logger = logging.getLogger("haks.windows_control.interpreter")

TAMIL_APP_MAP = {
    "நோட்பேட்": "notepad",
    "கால்குலேட்டர்": "calc",
    "கணக்கி": "calc",
    "கமாண்ட் ப்ராம்ப்ட்": "cmd",
    "முனையம்": "cmd",
    "கோப்பு மேலாளர்": "explorer",
    "உலாவி": "chrome",
    "குரோம்": "chrome",
    "பெயின்ட்": "mspaint",
}

APP_ALIAS_MAP = {
    "note pad": "notepad",
    "notepad": "notepad",
    "camera": "start microsoft.windows.camera:",
    "webcam": "start microsoft.windows.camera:",
    "calculator": "calc",
    "calc": "calc",
    "terminal": "wt",
    "command prompt": "cmd",
    "cmd": "cmd",
    "browser": "start msedge",
    "chrome": "start chrome",
    "edge": "start msedge",
    "file explorer": "explorer",
    "explorer": "explorer",
    "files": "explorer",
    "task manager": "taskmgr",
    "paint": "mspaint",
    "settings": "start ms-settings:",
}


class CommandInterpreter:
    """Translates user natural-language queries into validated ActionPlans."""

    def __init__(self, policy: Optional[ExecutionPolicy] = None) -> None:
        self.policy = policy or ExecutionPolicy()

    def interpret(self, user_instruction: str, source: str = "chat", working_dir: Optional[str] = None) -> ActionPlan:
        """Parse natural-language request into an executable ActionPlan."""
        text = user_instruction.strip()
        # Strip wake-word / conversational assistant address if present (e.g., "Hey HAKS, open notepad" -> "open notepad")
        stripped = re.sub(r"^(?:hey|hi|hello)?\s*haks[\s,:\-]+", "", text, flags=re.IGNORECASE).strip()
        if stripped:
            text = stripped
        lower = text.lower()
        cwd = working_dir or os.getcwd()

        # 0. Check conversational requests first (e.g. general questions or standalone greetings)
        question_words = ("what is", "what are", "who is", "who are", "tell me", "explain", "how do", "how to", "why is", "why does", "when is", "யார்", "என்ன", "எப்படி")
        standalone_greetings = ("hello", "hi", "hey", "வணக்கம்")
        is_greeting = any(lower == g or lower.startswith(f"{g} ") and not any(cmd in lower for cmd in ("open", "launch", "start", "run", "close", "kill", "restart", "shutdown", "install", "check", "திற")) for g in standalone_greetings)

        if any(lower.startswith(qw) for qw in question_words) or lower.endswith("?") or is_greeting:
            return ActionPlan(
                objective="General Conversation Query",
                target="AI Assistant",
                operation_type="conversation",
                command="",
                working_dir=cwd,
                privilege_level=PrivilegeLevel.NORMAL,
                risk_level=RiskLevel.LOW,
                expected_outcome="Conversational assistant response.",
                source=source,
            )

        # 1. Power Controls (Restart & Shutdown)
        if re.search(r"\b(restart\s+(my\s+)?(computer|laptop|pc|windows)|மீண்டும்\s+தொடங்கு)\b", lower):
            return ActionPlan(
                objective="Restart Windows",
                target="Operating System",
                operation_type="restart",
                command="shutdown.exe /r /t 0",
                working_dir=cwd,
                privilege_level=PrivilegeLevel.ELEVATED,
                risk_level=RiskLevel.DESTRUCTIVE,
                confirmation_required=True,
                confirmation_message="⚠️ Warning: Windows will restart immediately. Any unsaved work in other apps will be closed. Confirm restart?",
                expected_outcome="Reboots the Windows operating system.",
                source=source,
            )

        if re.search(r"\b(shut\s*down\s+(my\s+)?(computer|laptop|pc|windows)|கணினியை\s+அணை)\b", lower):
            return ActionPlan(
                objective="Shut Down Windows",
                target="Operating System",
                operation_type="shutdown",
                command="shutdown.exe /s /t 0",
                working_dir=cwd,
                privilege_level=PrivilegeLevel.ELEVATED,
                risk_level=RiskLevel.DESTRUCTIVE,
                confirmation_required=True,
                confirmation_message="⚠️ Warning: Windows will shut down immediately. Confirm shutdown?",
                expected_outcome="Powers off the computer.",
                source=source,
            )

        if re.search(r"\b(lock\s+(my\s+)?(workstation|computer|screen|pc)|கணினியைப்\s+பூட்டு)\b", lower):
            return ActionPlan(
                objective="Lock Windows Workstation",
                target="Active User Session",
                operation_type="lock",
                command="rundll32.exe user32.dll,LockWorkStation",
                working_dir=cwd,
                privilege_level=PrivilegeLevel.NORMAL,
                risk_level=RiskLevel.LOW,
                expected_outcome="Locks the active desktop session.",
                source=source,
            )

        # 2. Application Launching (English + Tamil)
        tamil_app_match = re.search(r"([a-zA-Z0-9_\-\u0B80-\u0BFF]+)\s*(ஐ|ை)?\s*(திற|துவக்கு|இயக்கு)", text)
        if tamil_app_match:
            raw_target = tamil_app_match.group(1).strip()
            target_app = TAMIL_APP_MAP.get(raw_target, raw_target)
            return ActionPlan(
                objective=f"Open {target_app}",
                target=target_app,
                operation_type="launch_app",
                command=target_app,
                working_dir=cwd,
                privilege_level=PrivilegeLevel.NORMAL,
                risk_level=RiskLevel.LOW,
                expected_outcome=f"Launches {target_app} on the desktop.",
                source=source,
            )

        app_match = re.search(r"\b(open|launch|start|திற)\s+([a-zA-Z0-9_\-\s]+)", text, re.IGNORECASE)
        if app_match and not any(k in lower for k in ("pip", "file", "folder", "dependency", "dependencies", "repo", "git")):
            target_app = app_match.group(2).strip()
            for noise in ("the", "a", "an", "please", "for me", "now", "app", "application"):
                target_app = re.sub(rf"^\b{noise}\b|\b{noise}\b$", "", target_app, flags=re.IGNORECASE).strip()

            canonical_target = APP_ALIAS_MAP.get(target_app.lower(), TAMIL_APP_MAP.get(target_app, target_app))
            command_str = canonical_target
            display_target = target_app
            return ActionPlan(
                objective=f"Open {display_target}",
                target=display_target,
                operation_type="launch_app",
                command=command_str,
                working_dir=cwd,
                privilege_level=PrivilegeLevel.NORMAL,
                risk_level=RiskLevel.LOW,
                expected_outcome=f"Launches {display_target} on the desktop.",
                source=source,
            )

        # 3. Hardware / CPU / Memory Inspection
        if any(w in lower for w in ("cpu", "processor", "ram", "memory usage", "hardware", "சிபியூ", "நினைவகம்")):
            return ActionPlan(
                objective="Inspect System Hardware and CPU Usage",
                target="Hardware",
                operation_type="system_hardware",
                command="Get-CimInstance Win32_Processor | Select-Object Name, LoadPercentage",
                working_dir=cwd,
                privilege_level=PrivilegeLevel.NORMAL,
                risk_level=RiskLevel.LOW,
                expected_outcome="Retrieves CPU, RAM, and GPU hardware metrics.",
                source=source,
            )

        # 4. Pip and Python Dependency Management
        pip_req_match = re.search(r"\b(install\s+.*dependencies|install\s+requirements(\.txt)?)\b", lower)
        if pip_req_match:
            req_file = "requirements.txt"
            return ActionPlan(
                objective="Install Python requirements",
                target=req_file,
                operation_type="pip_install_requirements",
                command=f'python -m pip install -r "{req_file}"',
                working_dir=cwd,
                privilege_level=PrivilegeLevel.NORMAL,
                risk_level=RiskLevel.SENSITIVE,
                confirmation_required=True,
                confirmation_message=f"Install Python dependencies listed in '{req_file}'?",
                expected_outcome="Installs all packages specified in requirements.txt into the active Python environment.",
                source=source,
            )

        pip_pkg_match = re.search(r"\b(pip\s+install|install\s+package)\s+([a-zA-Z0-9_\-]+)\b", lower)
        if pip_pkg_match:
            pkg_name = pip_pkg_match.group(2)
            return ActionPlan(
                objective=f"Install Python package {pkg_name}",
                target=pkg_name,
                operation_type="pip_install",
                command=f"python -m pip install {pkg_name}",
                working_dir=cwd,
                privilege_level=PrivilegeLevel.NORMAL,
                risk_level=RiskLevel.SENSITIVE,
                confirmation_required=True,
                confirmation_message=f"Proceed with installing '{pkg_name}' into Python environment?",
                expected_outcome=f"Installs '{pkg_name}' via pip.",
                source=source,
            )

        # 5. Storage & Storage Analysis
        if any(w in lower for w in ("check storage", "storage consumption", "consuming the most storage", "disk usage", "large files", "சி ட்ரைவ் இடம்", "வட்டு இடம்", "இடம் எவ்வளவு")):
            return ActionPlan(
                objective="Analyze Storage Consumption",
                target=cwd,
                operation_type="storage_analysis",
                command="Get-ChildItem -Recurse | Measure-Object -Property Length -Sum",
                working_dir=cwd,
                privilege_level=PrivilegeLevel.NORMAL,
                risk_level=RiskLevel.LOW,
                expected_outcome="Generates a detailed storage report of large files and extension breakdown.",
                source=source,
            )

        # 6. File Searching
        search_match = re.search(r"\b(find|search|தேடு)\s+(for\s+)?(all\s+)?([a-zA-Z0-9_\-\*\.]+)\s*(files?|கோப்புகள்?)?\b", lower)
        if search_match:
            search_pattern = search_match.group(4)
            ext = search_pattern.replace("*", "").replace(".", "") if "." in search_pattern or search_pattern in ("python", "text", "json", "pdf") else None
            if search_pattern == "python":
                ext = "py"
            return ActionPlan(
                objective=f"Search for {search_pattern} files",
                target=search_pattern,
                operation_type="search_files",
                command=f"dir /s /b *.{ext or '*'}",
                working_dir=cwd,
                privilege_level=PrivilegeLevel.NORMAL,
                risk_level=RiskLevel.LOW,
                expected_outcome=f"Scans directory and lists all matching files.",
                source=source,
            )

        # 7. Windows Services & Network Inspection
        if any(w in lower for w in ("check services", "list services", "windows services", "running services")):
            return ActionPlan(
                objective="Inspect Windows Services",
                target="Services",
                operation_type="service_list",
                command="Get-Service | Where-Object {$_.Status -eq 'Running'}",
                working_dir=cwd,
                privilege_level=PrivilegeLevel.NORMAL,
                risk_level=RiskLevel.LOW,
                expected_outcome="Lists active Windows system services.",
                source=source,
            )

        ping_match = re.search(r"\b(ping|பிங்)\s+([a-zA-Z0-9\.\-]+)\b", lower)
        if ping_match:
            target_host = ping_match.group(2)
            return ActionPlan(
                objective=f"Ping host {target_host}",
                target=target_host,
                operation_type="network_diagnostics",
                command=f"ping -n 4 {target_host}",
                working_dir=cwd,
                privilege_level=PrivilegeLevel.NORMAL,
                risk_level=RiskLevel.LOW,
                expected_outcome=f"Pings {target_host} and returns latency metrics.",
                source=source,
            )

        if any(w in lower for w in ("troubleshoot network", "check network", "network diagnostics", "ip config")):
            return ActionPlan(
                objective="Diagnose Network Connectivity",
                target="Network",
                operation_type="network_diagnostics",
                command="ipconfig /all",
                working_dir=cwd,
                privilege_level=PrivilegeLevel.NORMAL,
                risk_level=RiskLevel.LOW,
                expected_outcome="Tests ping latency and network adapter status.",
                source=source,
            )

        # 8. Git Repositories
        git_clone_match = re.search(r"\b(clone|download\s+repository)\s+(https?://[^\s]+)\b", text, re.IGNORECASE)
        if git_clone_match:
            repo_url = git_clone_match.group(2)
            repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
            target_path = str(Path(cwd) / repo_name)
            return ActionPlan(
                objective=f"Clone Git repository {repo_name}",
                target=repo_url,
                operation_type="git_clone",
                command=f'git clone "{repo_url}" "{target_path}"',
                working_dir=cwd,
                privilege_level=PrivilegeLevel.NORMAL,
                risk_level=RiskLevel.LOW,
                expected_outcome=f"Clones repository into '{target_path}'.",
                source=source,
            )

        # 9. Direct Shell Commands ("run cmd ...", "run powershell ...", or raw command execution)
            return ActionPlan(
                objective=f"Clone Git repository {repo_name}",
                target=repo_url,
                operation_type="git_clone",
                command=f'git clone "{repo_url}" "{target_path}"',
                working_dir=cwd,
                privilege_level=PrivilegeLevel.NORMAL,
                risk_level=RiskLevel.LOW,
                expected_outcome=f"Clones repository into '{target_path}'.",
                source=source,
            )

        # 8. Direct Shell Commands ("run cmd ...", "run powershell ...", or raw command execution)
        clean_cmd = text
        for pfx in ("haks, run cmd ", "haks, run powershell ", "haks, run ", "run cmd ", "run powershell ", "run "):
            if lower.startswith(pfx):
                clean_cmd = text[len(pfx):].strip()
                break

        risk = self.policy.classify_risk(clean_cmd)
        priv = PrivilegeDetector.requires_elevation(clean_cmd)

        plan = ActionPlan(
            objective=f"Execute command: {clean_cmd[:50]}",
            target=clean_cmd,
            operation_type="command",
            command=clean_cmd,
            working_dir=cwd,
            privilege_level=priv,
            risk_level=risk,
            source=source,
            expected_outcome="Executes requested shell command and reports real terminal output.",
        )
        self.policy.evaluate_plan(plan)
        return plan
