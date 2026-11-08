"""Installer and Executable Signature / Integrity Verifier for HAKS.

Verifies SHA-256 checksums and Windows Authenticode digital signatures
on downloaded installation binaries before authorization.
"""

import hashlib
import json
import logging
from pathlib import Path
import subprocess
from typing import Optional

logger = logging.getLogger("haks.windows_control.verifier")


class InstallerVerifier:
    """Verifies file checksums and Windows Authenticode digital signatures."""

    @staticmethod
    def calculate_sha256(file_path: Path) -> str:
        """Calculate the SHA-256 hexadecimal digest of a file."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    @classmethod
    def verify_checksum(cls, file_path: Path, expected_sha256: str) -> bool:
        """Verify whether file matches the expected SHA-256 checksum."""
        actual = cls.calculate_sha256(file_path).lower().strip()
        expected = expected_sha256.lower().strip()
        return actual == expected

    @staticmethod
    def check_authenticode_signature(file_path: Path) -> dict[str, str]:
        """Check Windows Authenticode digital signature using PowerShell."""
        if not file_path.exists():
            return {"status": "FileNotFound", "signer": "None"}

        # Run Get-AuthenticodeSignature in PowerShell
        ps_cmd = (
            f"Get-AuthenticodeSignature -LiteralPath '{file_path}' | "
            f"Select-Object Status, StatusMessage, @{{Name='Subject';Expression={{$_.SignerCertificate.Subject}}}} | "
            f"ConvertTo-Json -Compress"
        )
        try:
            res = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps_cmd],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if res.returncode == 0 and res.stdout.strip():
                data = json.loads(res.stdout.strip())
                status = str(data.get("Status", "Unknown"))
                subject = str(data.get("Subject", "Unknown"))
                return {
                    "status": status,
                    "signer": subject,
                    "is_valid": status == "Valid",
                }
        except Exception as exc:
            logger.warning("Could not verify Authenticode signature for %s: %s", file_path, exc)

        return {"status": "Unknown", "signer": "Unknown", "is_valid": False}
