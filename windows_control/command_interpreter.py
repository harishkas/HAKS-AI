"""Natural-Language Command Interpreter for HAKS Windows Computer Control.

Translates conversational user instructions in English and Tamil into
typed, policy-governed ActionPlans.
"""

import logging
import os
from pathlib import Path
import re
from typing import Optional

from windows_control.application_intents import ApplicationIntentParser, AppActionType
from windows_control.application_registry import ApplicationRegistry
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

    def __init__(
        self,
        policy: Optional[ExecutionPolicy] = None,
        registry: Optional[ApplicationRegistry] = None,
    ) -> None:
        self.policy = policy or ExecutionPolicy()
        self.registry = registry or ApplicationRegistry()
        self.app_intent_parser = ApplicationIntentParser(self.registry)

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

        # 2. Application Control Actions (Launch, Switch, Minimize, Maximize, Restore, Close) - English, Tamil, Tanglish
        app_intent = self.app_intent_parser.parse(text)
        if app_intent.is_actionable and app_intent.app:
            action_verb_map = {
                AppActionType.LAUNCH: "Open",
                AppActionType.SWITCH: "Switch to",
                AppActionType.MINIMIZE: "Minimize",
                AppActionType.MAXIMIZE: "Maximize",
                AppActionType.RESTORE: "Restore",
                AppActionType.CLOSE: "Close",
            }
            verb = action_verb_map.get(app_intent.action, "Control")
            return ActionPlan(
                objective=f"{verb} {app_intent.app.display_name}",
                target=app_intent.app.app_id,
                operation_type=app_intent.action.value,
                command=app_intent.app.app_id,
                working_dir=cwd,
                privilege_level=PrivilegeLevel.NORMAL,
                risk_level=RiskLevel.LOW,
                expected_outcome=f"{verb}s {app_intent.app.display_name} on the Windows desktop.",
                source=source,
            )

        # Fallback to legacy tamil app match if any edge dialect was missed
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

        # 5. File Intelligence - Duplicate Detection (English, Tamil, Tanglish)
        dup_match = re.search(
            r"\b(find|check|scan|detect|identify)?\s*(duplicate|identical|doubles?)\s*(files?|documents?|docs?)?\b|"
            r"\b(போலி|இரட்டை)\s*கோப்புக(ள்|ளை)\b|"
            r"\bduplicate\s*(files?|docs?|folders?)?\s*(kandupidi|thedu|check|pannu)\b",
            lower,
        )
        if dup_match:
            return ActionPlan(
                objective="Find Duplicate Files",
                target=cwd,
                operation_type="find_duplicates",
                command=f"find_duplicates {cwd}",
                working_dir=cwd,
                privilege_level=PrivilegeLevel.NORMAL,
                risk_level=RiskLevel.LOW,
                expected_outcome="Detects duplicate files using size grouping, SHA-256, and BLAKE3 verification.",
                source=source,
            )

        # 6. File Intelligence - Safe Recycle Bin Deletion (English, Tamil, Tanglish)
        # Note: Never permanently deletes; always uses Windows Recycle Bin via send2trash
        recycle_match = re.search(
            r"\b(?:delete|remove|recycle|trash|move\s+to\s+recycle\s+bin)\s+(?:file\s+)?([^\s]+\.[a-zA-Z0-9]{1,10})\b|"
            r"([^\s]+\.[a-zA-Z0-9]{1,10})\s*(?:கோப்பை|ஐ)?\s*(?:நீக்கு|அழி|குப்பைத்தொட்டிக்கு\s*அனுப்பு)\b|"
            r"([^\s]+\.[a-zA-Z0-9]{1,10})\s*(?:file\s+)?(?:delete\s+pannu|trash\s+la\s+podu|recycle\s+pannu)\b",
            text,
            re.IGNORECASE,
        )
        if recycle_match:
            target_f = recycle_match.group(1) or recycle_match.group(2) or recycle_match.group(3)
            clean_target = target_f.strip().strip("'\"")
            return ActionPlan(
                objective=f"Move '{clean_target}' to Recycle Bin",
                target=clean_target,
                operation_type="recycle_bin",
                command=f"recycle {clean_target}",
                working_dir=cwd,
                privilege_level=PrivilegeLevel.NORMAL,
                risk_level=RiskLevel.SENSITIVE,
                confirmation_required=True,
                confirmation_message=f"Are you sure you want to move '{clean_target}' to the Windows Recycle Bin?",
                expected_outcome=f"Safely sends '{clean_target}' to the Windows Recycle Bin without permanent deletion.",
                source=source,
            )

        # 7. File Intelligence - Directory Indexing
        index_match = re.search(
            r"\b(index\s+(?:my\s+)?(?:files|documents|directory|folder)|reindex)\b|"
            r"\b(கோப்பகத்தை\s+அட்டவணைப்படுத்து)\b|"
            r"\b(?:files?|documents?)\s*(?:index\s+pannu|index\s+seiyyum)\b",
            lower,
        )
        if index_match:
            return ActionPlan(
                objective="Index Directory for File Intelligence",
                target=cwd,
                operation_type="index_directory",
                command=f"index_directory {cwd}",
                working_dir=cwd,
                privilege_level=PrivilegeLevel.NORMAL,
                risk_level=RiskLevel.LOW,
                expected_outcome="Incrementally extracts text, chunks, and builds BGE-M3 dense and sparse indexes.",
                source=source,
            )

        # 8. Storage & Storage Analysis (English, Tamil, Tanglish)
        if any(w in lower for w in (
            "check storage", "storage consumption", "consuming the most storage",
            "disk usage", "large files", "சி ட்ரைவ் இடம்", "வட்டு இடம்", "இடம் எவ்வளவு",
            "storage evvalavu", "disk space check", "space analysis"
        )):
            return ActionPlan(
                objective="Analyze Storage Consumption",
                target=cwd,
                operation_type="storage_analysis",
                command=f"storage_analysis {cwd}",
                working_dir=cwd,
                privilege_level=PrivilegeLevel.NORMAL,
                risk_level=RiskLevel.LOW,
                expected_outcome="Generates a detailed storage report of large files and extension breakdown.",
                source=source,
            )

        # 9. File Searching & Multilingual Semantic Retrieval (English, Tamil, Tanglish)
        search_query_match = re.search(
            r"\b(?:find|search|look\s+for|where\s+is|where\s+are)\s+(?:for\s+)?(?:all\s+)?(.+?)(?:\s+files?|\s+documents?|\s+docs?)?$",
            text,
            re.IGNORECASE,
        )
        tamil_search_match = re.search(
            r"(.+?)\s*(?:கோப்பை|ஆவணத்தை|கோப்புகளை|ஆவணங்களை)?\s*(?:தேடு|கண்டுபிடி|காட்டு)$",
            text,
        )
        tanglish_search_match = re.search(
            r"(.+?)\s*(?:file|document|doc)?\s*(?:enga\s+irukku|engae\s+irukku|thedu|kandupidi|search\s+pannu)$",
            text,
            re.IGNORECASE,
        )

        if search_query_match or tamil_search_match or tanglish_search_match:
            if search_query_match:
                extracted_q = search_query_match.group(1).strip()
            elif tamil_search_match:
                extracted_q = tamil_search_match.group(1).strip()
            else:
                extracted_q = tanglish_search_match.group(1).strip()

            clean_q = re.sub(r"\b(files?|documents?|docs?|all)\b", "", extracted_q, flags=re.IGNORECASE).strip() or extracted_q
            return ActionPlan(
                objective=f"Search for files matching '{clean_q}'",
                target=clean_q,
                operation_type="search_files",
                command=clean_q,
                working_dir=cwd,
                privilege_level=PrivilegeLevel.NORMAL,
                risk_level=RiskLevel.LOW,
                expected_outcome=f"Performs hybrid retrieval (Dense + Sparse + BM25/FTS + Windows Search) for '{clean_q}'.",
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
