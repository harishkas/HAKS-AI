"""Secure Inter-Process Communication for Windows Administrative Helper.

Handles token-authenticated serialization and deserialization of elevated
commands and execution results using temporary filesystem handoff channels.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
from typing import Any, Optional


def get_ipc_staging_dir() -> Path:
    """Return secure temporary staging directory for administrative tasks."""
    temp_root = Path(os.environ.get("TEMP", os.environ.get("TMP", r"C:\Windows\Temp")))
    staging = temp_root / "HAKS_Admin_IPC"
    staging.mkdir(parents=True, exist_ok=True)
    return staging


class AdminIPC:
    """Manages secure request creation and result retrieval for elevated tasks."""

    @staticmethod
    def create_request_package(
        command: str,
        working_dir: Optional[str] = None,
        timeout_seconds: int = 120,
    ) -> tuple[str, Path, Path]:
        """Generate a cryptographically randomized token and write request JSON.

        Returns:
            (token, request_file_path, response_file_path)
        """
        staging = get_ipc_staging_dir()
        token = secrets.token_hex(16)
        cmd_hash = hashlib.sha256(command.encode("utf-8")).hexdigest()

        req_path = staging / f"req_{token}.json"
        res_path = staging / f"res_{token}.json"

        payload = {
            "token": token,
            "command": command,
            "command_hash": cmd_hash,
            "working_dir": working_dir or os.getcwd(),
            "timeout_seconds": timeout_seconds,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(req_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return token, req_path, res_path

    @staticmethod
    def read_and_validate_request(req_path: Path) -> dict[str, Any]:
        """Read and verify request integrity inside the elevated helper."""
        if not req_path.exists():
            raise FileNotFoundError(f"Elevation request file not found: {req_path}")

        with open(req_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        # Integrity verification
        expected_hash = hashlib.sha256(payload["command"].encode("utf-8")).hexdigest()
        if expected_hash != payload.get("command_hash"):
            raise ValueError("Integrity check failed: Command payload was modified.")

        return payload

    @staticmethod
    def write_response_package(res_path: Path, result: dict[str, Any]) -> None:
        """Write execution result from the elevated helper process."""
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    @staticmethod
    def read_response_package(res_path: Path, delete_after: bool = True) -> Optional[dict[str, Any]]:
        """Read result JSON returned by the elevated helper."""
        if not res_path.exists():
            return None

        try:
            with open(res_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        finally:
            if delete_after:
                try:
                    res_path.unlink(missing_ok=True)
                except Exception:
                    pass

    @staticmethod
    def cleanup(req_path: Path, res_path: Path) -> None:
        """Remove temporary IPC request and response files."""
        try:
            req_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            res_path.unlink(missing_ok=True)
        except Exception:
            pass
