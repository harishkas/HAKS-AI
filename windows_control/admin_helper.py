"""Windows Elevated Administrator Helper Process for HAKS.

Launched via legitimate Windows UAC ('runas') ShellExecute to perform
only the approved administrative operation in an isolated short-lived process.
"""

import ctypes
import os
from pathlib import Path
import subprocess
import sys
import time

# Ensure parent directory is importable
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from windows_control.admin_ipc import AdminIPC


def is_admin() -> bool:
    try:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except Exception:
        return False


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python admin_helper.py <request_json_path>", file=sys.stderr)
        return 1

    req_path = Path(sys.argv[1])
    staging_dir = req_path.parent
    token_name = req_path.stem.replace("req_", "")
    res_path = staging_dir / f"res_{token_name}.json"

    # 1. Enforce admin privilege requirement
    if not is_admin():
        AdminIPC.write_response_package(
            res_path,
            {
                "exit_code": 5,
                "stdout": "",
                "stderr": "Administrator privileges were not granted.",
                "duration_ms": 0.0,
                "success": False,
            },
        )
        return 5

    # 2. Read and validate request
    try:
        request_data = AdminIPC.read_and_validate_request(req_path)
    except Exception as exc:
        AdminIPC.write_response_package(
            res_path,
            {
                "exit_code": 1,
                "stdout": "",
                "stderr": f"Failed to validate administrative request: {exc}",
                "duration_ms": 0.0,
                "success": False,
            },
        )
        return 1

    command = request_data["command"]
    working_dir = request_data.get("working_dir") or os.getcwd()
    timeout = request_data.get("timeout_seconds", 120)

    # 3. Execute approved command
    start_time = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            cwd=working_dir,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        AdminIPC.write_response_package(
            res_path,
            {
                "exit_code": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "duration_ms": duration_ms,
                "success": proc.returncode == 0,
            },
        )
        return proc.returncode
    except subprocess.TimeoutExpired:
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        AdminIPC.write_response_package(
            res_path,
            {
                "exit_code": 1460,  # ERROR_TIMEOUT
                "stdout": "",
                "stderr": f"Administrative operation timed out after {timeout} seconds.",
                "duration_ms": duration_ms,
                "success": False,
            },
        )
        return 1460
    except Exception as exc:
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        AdminIPC.write_response_package(
            res_path,
            {
                "exit_code": 1,
                "stdout": "",
                "stderr": f"Execution error in elevated helper: {exc}",
                "duration_ms": duration_ms,
                "success": False,
            },
        )
        return 1
    finally:
        # Clean up request package
        try:
            req_path.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
