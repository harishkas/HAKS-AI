"""Execution Policy Engine for HAKS Windows Computer Control.

Enforces Full Control vs Safe Confirmation modes, classifies operation
risk tiers, protects critical system and HAKS database assets, and
determines whether automatic execution, user confirmation, or blocking applies.
"""

import os
from pathlib import Path
import re
from typing import Optional

from config.settings import AppConfig, get_settings
from windows_control.models import (
    ActionPlan,
    ExecutionMode,
    PrivilegeLevel,
    RiskLevel,
)

# Critical Windows core processes that must never be terminated
CRITICAL_WINDOWS_PROCESSES = {
    "system", "system idle process", "smss.exe", "csrss.exe", "wininit.exe",
    "services.exe", "lsass.exe", "svchost.exe", "winlogon.exe", "explorer.exe",
    "dwm.exe", "spoolsv.exe", "taskhostw.exe", "sihost.exe",
}

# Critical system directories that must never be mass deleted or overwritten
PROTECTED_SYSTEM_PATHS = [
    Path(os.environ.get("SystemRoot", r"C:\Windows")).resolve(),
    Path(os.environ.get("ProgramFiles", r"C:\Program Files")).resolve(),
    Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")).resolve(),
    Path(r"C:\Boot").resolve(),
]

# Destructive command patterns requiring targeted warning confirmation
DESTRUCTIVE_PATTERNS = [
    re.compile(r"\bformat\s+[a-z]:", re.IGNORECASE),
    re.compile(r"\bdiskpart\b", re.IGNORECASE),
    re.compile(r"\b(del|erase|rmdir|rd)\s+.*\/s\b", re.IGNORECASE),
    re.compile(r"\bRemove-Item\s+.*-Recurse\b", re.IGNORECASE),
    re.compile(r"\b(shutdown|restart-computer)\b.*[/-][sr]", re.IGNORECASE),
    re.compile(r"\bgit\s+reset\s+--hard\b", re.IGNORECASE),
    re.compile(r"\bgit\s+clean\s+-[a-z]*f\b", re.IGNORECASE),
    re.compile(r"\b(drop\s+database|truncate\s+table)\b", re.IGNORECASE),
    re.compile(r"\breg\s+delete\b", re.IGNORECASE),
]

# Blocked patterns that violate security boundaries
BLOCKED_PATTERNS = [
    re.compile(r"\bSet-MpPreference\s+-DisableRealtimeMonitoring\s+\$true\b", re.IGNORECASE),
    re.compile(r"\bnetsh\s+advfirewall\s+set\s+allprofiles\s+state\s+off\b", re.IGNORECASE),
    re.compile(r"\bvssadmin\s+delete\s+shadows\b", re.IGNORECASE),
    re.compile(r"\bwevtutil\s+cl\b", re.IGNORECASE),
]

# Sensitive command patterns requiring explicit confirmation
SENSITIVE_PATTERNS = [
    re.compile(r"\bwinget\s+(install|uninstall|upgrade)\b", re.IGNORECASE),
    re.compile(r"\bpip\s+(install|uninstall)\b", re.IGNORECASE),
    re.compile(r"\b(net\s+start|net\s+stop|Stop-Service|Start-Service|Restart-Service)\b", re.IGNORECASE),
    re.compile(r"\b(Set-ItemProperty|New-ItemProperty|Remove-ItemProperty)\s+.*HKLM\b", re.IGNORECASE),
    re.compile(r"\breg\s+add\s+hklm\b", re.IGNORECASE),
    re.compile(r"\b(taskkill|Stop-Process)\b", re.IGNORECASE),
    re.compile(r"\b(takeown|icacls)\b", re.IGNORECASE),
    re.compile(r"\bsetx\s+.*\/m\b", re.IGNORECASE),
    re.compile(r"\b(del|erase|rmdir|rd|Remove-Item)\b", re.IGNORECASE),
    re.compile(r"\b(curl|wget|Invoke-WebRequest)\s+.*\|\s*(iex|powershell|cmd)\b", re.IGNORECASE),
]


class ExecutionPolicy:
    """Evaluates and enforces computer control safety constraints."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or get_settings()
        self.mode: ExecutionMode = ExecutionMode.FULL_CONTROL
        self.protected_haks_paths: list[Path] = [
            self.config.paths.data_dir.resolve(),
            self.config.paths.logs_dir.resolve(),
        ]

    def set_execution_mode(self, mode: ExecutionMode) -> None:
        """Update active execution policy mode."""
        self.mode = mode

    def classify_risk(self, command: str, operation_type: str = "command", target: str = "") -> RiskLevel:
        """Classify operation into LOW, SENSITIVE, DESTRUCTIVE, or BLOCKED."""
        clean_cmd = command.strip()

        # 1. Check strictly prohibited / malware-adjacent patterns
        for pattern in BLOCKED_PATTERNS:
            if pattern.search(clean_cmd):
                return RiskLevel.BLOCKED

        # 2. Check protected path tampering
        if target:
            try:
                target_path = Path(target).resolve()
                # Check HAKS database / system directory protection
                for haks_path in self.protected_haks_paths:
                    if target_path == haks_path or haks_path in target_path.parents:
                        if operation_type in ("delete", "format", "wipe"):
                            return RiskLevel.BLOCKED
                for sys_path in PROTECTED_SYSTEM_PATHS:
                    if target_path == sys_path:
                        if operation_type in ("delete", "format", "wipe"):
                            return RiskLevel.BLOCKED
            except Exception:
                pass

        # 3. Check destructive patterns
        for pattern in DESTRUCTIVE_PATTERNS:
            if pattern.search(clean_cmd):
                return RiskLevel.DESTRUCTIVE

        if operation_type in ("delete_permanent", "mass_delete", "format_drive", "shutdown", "restart"):
            return RiskLevel.DESTRUCTIVE

        # 4. Check sensitive patterns
        for pattern in SENSITIVE_PATTERNS:
            if pattern.search(clean_cmd):
                return RiskLevel.SENSITIVE

        if operation_type in ("software_install", "software_uninstall", "service_control", "system_config", "registry_edit"):
            return RiskLevel.SENSITIVE

        return RiskLevel.LOW

    def evaluate_plan(self, plan: ActionPlan) -> tuple[bool, Optional[str]]:
        """Evaluate action plan against execution mode and risk level.

        Returns:
            (is_allowed, confirmation_message_or_block_reason)
        """
        # Re-verify risk level
        risk = self.classify_risk(plan.command, plan.operation_type, plan.target)
        plan.risk_level = risk

        # Rule 1: BLOCKED actions are never executed
        if risk == RiskLevel.BLOCKED:
            return False, f"Action blocked by HAKS security policy: Attempted sensitive/dangerous operation ({plan.objective})."

        # Rule 2: Restricted mode allows only read-only LOW inspection
        if self.mode == ExecutionMode.RESTRICTED:
            if risk != RiskLevel.LOW or plan.operation_type not in ("read", "inspect", "list", "search"):
                return False, "HAKS is running in RESTRICTED mode (read-only). Modifying operations are disabled."

        # Rule 3: Destructive operations ALWAYS require targeted warning confirmation
        if risk == RiskLevel.DESTRUCTIVE:
            plan.confirmation_required = True
            msg = (
                f"⚠️ DESTRUCTIVE OPERATION WARNING:\n"
                f"Target: {plan.target or plan.command}\n"
                f"Objective: {plan.objective}\n"
                f"Consequence: This action will permanently modify or delete data, or affect Windows state. "
                f"Do you explicitly confirm execution?"
            )
            plan.confirmation_message = msg
            return True, msg

        # Rule 4: Sensitive operations require confirmation in FULL_CONTROL and SAFE_CONFIRMATION
        if risk == RiskLevel.SENSITIVE:
            plan.confirmation_required = True
            msg = (
                f"Confirmation Required:\n"
                f"Action: {plan.objective}\n"
                f"Command: {plan.command}\n"
                f"Privilege: {plan.privilege_level.value.upper()}\n"
                f"Do you approve proceeding with this system change?"
            )
            plan.confirmation_message = msg
            return True, msg

        # Rule 5: Low risk actions in FULL_CONTROL auto-execute without asking
        if self.mode == ExecutionMode.FULL_CONTROL:
            plan.confirmation_required = False
            plan.confirmation_message = None
            return True, None

        # Rule 6: In SAFE_CONFIRMATION mode, any non-read action requires a confirmation
        if self.mode == ExecutionMode.SAFE_CONFIRMATION:
            if plan.operation_type not in ("read", "inspect", "list", "search"):
                plan.confirmation_required = True
                msg = f"Approve execution: {plan.objective} (`{plan.command}`)"
                plan.confirmation_message = msg
                return True, msg

        plan.confirmation_required = False
        return True, None

    def is_process_protected(self, process_name: str) -> bool:
        """Check if process is a protected Windows core service."""
        return process_name.lower().strip() in CRITICAL_WINDOWS_PROCESSES
