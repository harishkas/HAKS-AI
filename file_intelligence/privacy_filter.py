"""Privacy and Security Filter for HAKS File Intelligence (Phase 9.2).

Enforces directory allowlists, blocks credentials, secret files, memory databases,
and validates filesystem path boundaries against path traversal.
"""

import fnmatch
import logging
import os
from pathlib import Path
from typing import Optional

from config.settings import AppConfig, get_settings

logger = logging.getLogger("haks.file_intelligence.privacy")

# Secret and credential file patterns that must NEVER be indexed or searched
BLOCKED_SECRET_PATTERNS = [
    ".env",
    ".env.*",
    "*.env",
    "credentials.json",
    "client_secret*.json",
    "service_account*.json",
    "*id_rsa*",
    "*id_ed25519*",
    "*.pem",
    "*.key",
    "*.pfx",
    "*.p12",
    "*token*.json",
    "*token*.txt",
    "*password*.txt",
    "*secret*.txt",
    "*.kdbx",
]

# Database files that must not be indexed to prevent circular indexing or leaking memory
BLOCKED_DATABASE_NAMES = {
    "personal_memory.db",
    "work_memory.db",
    "settings.db",
    "audit_trail.db",
    "file_metadata.db",
}

# Critical Windows system folders that should never be scanned without explicit intent
BLOCKED_SYSTEM_DIRS = [
    r"C:\Windows\System32",
    r"C:\Windows\SysWOW64",
    r"C:\Windows\WinSxS",
    r"C:\$Recycle.Bin",
    r"C:\System Volume Information",
]


class PrivacyFilter:
    """Validates files and directories against security and privacy boundaries."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        allowed_directories: Optional[list[str | Path]] = None,
    ) -> None:
        self.config = config or get_settings()
        self.allowed_directories: list[Path] = []

        if allowed_directories:
            for d in allowed_directories:
                p = Path(d).resolve()
                if p.exists() and p.is_dir():
                    self.allowed_directories.append(p)
        else:
            # Default allowlist: user root dir, project workspace, and user documents
            self.allowed_directories = [
                self.config.paths.root_dir.resolve(),
                Path(os.environ.get("USERPROFILE", r"C:\Users\Default")).resolve() / "Documents",
                Path(os.environ.get("USERPROFILE", r"C:\Users\Default")).resolve() / "Desktop",
            ]

    def add_allowed_directory(self, dir_path: str | Path) -> bool:
        """Add a directory to the permitted indexing allowlist."""
        p = Path(dir_path).resolve()
        if p.exists() and p.is_dir() and p not in self.allowed_directories:
            self.allowed_directories.append(p)
            return True
        return False

    def is_path_safe_to_access(self, target_path: str | Path) -> tuple[bool, str]:
        """Verify target path is resolved, exists, free from traversal, and not a blocked secret."""
        raw_str = str(target_path)
        if ".." in raw_str:
            return False, "Path traversal sequence '..' is forbidden."

        try:
            p = Path(target_path).resolve()
        except Exception as exc:
            return False, f"Invalid path syntax: {exc}"

        if not p.exists():
            return False, f"Target path does not exist: {p}"

        # 1. Check blocked database files
        if p.name.lower() in BLOCKED_DATABASE_NAMES:
            return False, f"Access blocked: '{p.name}' is a protected internal HAKS database."

        # 2. Check secret and credential patterns
        for pattern in BLOCKED_SECRET_PATTERNS:
            if fnmatch.fnmatch(p.name.lower(), pattern.lower()):
                return False, f"Access blocked: '{p.name}' matches credential pattern '{pattern}'."

        # 3. Check blocked system directories
        str_p = str(p).lower()
        for b_dir in BLOCKED_SYSTEM_DIRS:
            if str_p.startswith(b_dir.lower()):
                return False, f"Access blocked: '{p}' is inside protected system directory '{b_dir}'."

        return True, "Path is safe and permitted."

    def is_file_indexable(self, file_path: str | Path) -> bool:
        """Check whether a file is permitted for document text and vector indexing."""
        safe, _ = self.is_path_safe_to_access(file_path)
        if not safe:
            return False

        p = Path(file_path).resolve()
        # Must be a file
        if not p.is_file():
            return False

        # Exclude hidden files or folders (e.g. .git, .pytest_cache, .venv)
        for part in p.parts:
            if part.startswith(".") and len(part) > 1 and not part.startswith("."):
                return False
            if part in (".git", ".venv", ".pytest_cache", "__pycache__", "node_modules"):
                return False

        return True

    def is_in_allowlist(self, target_path: str | Path) -> bool:
        """Check if target path is inside any allowed directory root."""
        try:
            p = Path(target_path).resolve()
            for allowed in self.allowed_directories:
                try:
                    p.relative_to(allowed)
                    return True
                except ValueError:
                    continue
        except Exception:
            pass
        return False

