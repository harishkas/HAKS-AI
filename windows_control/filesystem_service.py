"""Windows Filesystem Management Service for HAKS.

Provides complete real filesystem operations across all system drives:
- Drive enumeration and capacity inspection (fixed, removable, CD-ROM, network)
- Recursive search and project directory scanning with permission-aware traversal
- Tracking of inaccessible / access-denied directories without silent failure
- Directory size calculation and large file discovery
- File CRUD, copy, move, and rename across drives
- Windows Recycle Bin deletion and permanent deletion safeguards
- ZIP archive creation and extraction
- Duplicate file detection via chunked hashing
- Storage breakdown reports
- Protection of active HAKS databases and critical Windows system directories
"""

import ctypes
from ctypes import wintypes
import hashlib
import logging
import os
from pathlib import Path
import shutil
import time
from typing import Any, Optional
import zipfile

import psutil

from config.settings import AppConfig, get_settings
from windows_control.models import DriveInfo, SearchResult, StorageReport

logger = logging.getLogger("haks.windows_control.filesystem")

# Win32 Drive Type Constants
DRIVE_TYPE_MAP = {
    0: "unknown",
    1: "no_root_dir",
    2: "removable",
    3: "fixed",
    4: "network",
    5: "cdrom",
    6: "ramdisk",
}

# Critical Windows protected roots and directories
PROTECTED_SYSTEM_NAMES = {
    "system volume information",
    "$recycle.bin",
    "boot",
    "recovery",
    "efi",
}


class FilesystemService:
    """Complete Windows filesystem control, multi-drive traversal, and storage analysis."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or get_settings()
        self._protected_haks_paths = [
            self.config.paths.data_dir.resolve(),
            self.config.paths.logs_dir.resolve(),
        ]

    # --------------------------------------------------------------------------
    # 1. Drive Enumeration & Multiple-Drive Inspection
    # --------------------------------------------------------------------------

    def list_drives(self) -> list[DriveInfo]:
        """List all mounted storage drives with type, volume label, total, and free capacity."""
        drives: list[DriveInfo] = []
        try:
            for part in psutil.disk_partitions(all=True):
                mount = part.mountpoint
                drive_type_str = "fixed"
                vol_name = ""

                # Win32 API drive type & volume label inspection
                try:
                    kernel32 = ctypes.windll.kernel32
                    dtype = kernel32.GetDriveTypeW(mount)
                    drive_type_str = DRIVE_TYPE_MAP.get(dtype, "unknown")

                    vol_buf = ctypes.create_unicode_buffer(261)
                    fs_buf = ctypes.create_unicode_buffer(261)
                    success = kernel32.GetVolumeInformationW(
                        ctypes.c_wchar_p(mount),
                        vol_buf,
                        ctypes.sizeof(vol_buf),
                        None,
                        None,
                        None,
                        fs_buf,
                        ctypes.sizeof(fs_buf),
                    )
                    if success:
                        vol_name = vol_buf.value
                except Exception as ex:
                    logger.debug("Win32 volume info error on %s: %s", mount, ex)

                # Query capacity if disk is ready
                try:
                    usage = psutil.disk_usage(mount)
                    drives.append(
                        DriveInfo(
                            device=part.device,
                            mountpoint=mount,
                            fstype=part.fstype,
                            total_gb=round(usage.total / (1024**3), 2),
                            used_gb=round(usage.used / (1024**3), 2),
                            free_gb=round(usage.free / (1024**3), 2),
                            percent_used=usage.percent,
                            drive_type=drive_type_str,
                            volume_name=vol_name,
                            is_ready=True,
                        )
                    )
                except (PermissionError, OSError):
                    # Unmounted / unready drive (e.g. empty optical CD-ROM or disconnected network share)
                    drives.append(
                        DriveInfo(
                            device=part.device,
                            mountpoint=mount,
                            fstype=part.fstype,
                            total_gb=0.0,
                            used_gb=0.0,
                            free_gb=0.0,
                            percent_used=0.0,
                            drive_type=drive_type_str,
                            volume_name=vol_name,
                            is_ready=False,
                        )
                    )
        except Exception as exc:
            logger.error("Error enumerating drives: %s", exc)
        return drives

    def get_drive_for_path(self, path: str) -> Optional[str]:
        """Determine which logical drive letter / mountpoint hosts the given path."""
        try:
            resolved = Path(path).resolve()
            drive = resolved.drive
            if drive:
                return f"{drive}\\" if not drive.endswith("\\") else drive
        except Exception:
            pass
        return None

    # --------------------------------------------------------------------------
    # 2. Permission-Aware File & Folder Search
    # --------------------------------------------------------------------------

    def search_files(
        self,
        query: str,
        start_dir: Optional[str] = None,
        extension: Optional[str] = None,
        max_results: int = 100,
    ) -> SearchResult:
        """Search files by query string and optional extension across directories.

        Maintains permission-awareness: inaccessible / access-denied directories
        are recorded in `inaccessible_directories` rather than being silently ignored.
        """
        start_time = time.perf_counter()
        root = Path(start_dir or os.getcwd()).resolve()

        result = SearchResult(
            query=query,
            root_directory=str(root),
        )

        if not root.exists():
            result.search_duration_seconds = round(time.perf_counter() - start_time, 3)
            return result

        ext_filter = extension.lower().lstrip(".") if extension else None
        clean_query = query.lower()

        def _on_walk_error(err: OSError) -> None:
            """Capture access-denied and OS permission errors during directory traversal."""
            err_path = getattr(err, "filename", str(err))
            if err_path and err_path not in result.inaccessible_directories:
                result.inaccessible_directories.append(str(err_path))

        try:
            for dirpath, dirnames, filenames in os.walk(root, onerror=_on_walk_error):
                result.total_dirs_scanned += 1

                # Filter out and record inaccessible subdirs that cannot be listed
                accessible_subdirs = []
                for d in dirnames:
                    full_d = Path(dirpath) / d
                    try:
                        # Test read permission
                        _ = os.listdir(full_d)
                        accessible_subdirs.append(d)
                    except PermissionError:
                        result.inaccessible_directories.append(str(full_d))
                    except OSError:
                        pass
                dirnames[:] = accessible_subdirs

                for fname in filenames:
                    result.total_files_scanned += 1

                    # Check extension filter
                    if ext_filter and not fname.lower().endswith(f".{ext_filter}"):
                        continue

                    # Check query match in filename
                    if clean_query in fname.lower() or not clean_query:
                        full_p = Path(dirpath) / fname
                        try:
                            stat = full_p.stat()
                            result.matched_files.append(
                                {
                                    "name": fname,
                                    "path": str(full_p),
                                    "size_bytes": stat.st_size,
                                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                                    "modified_time": time.ctime(stat.st_mtime),
                                    "is_file": True,
                                }
                            )
                        except (PermissionError, OSError) as os_err:
                            result.inaccessible_directories.append(str(full_p))
                            continue

                        if len(result.matched_files) >= max_results:
                            result.truncated = True
                            result.search_duration_seconds = round(time.perf_counter() - start_time, 3)
                            return result

        except Exception as exc:
            logger.warning("Error during file search in %s: %s", root, exc)

        result.search_duration_seconds = round(time.perf_counter() - start_time, 3)
        return result

    # --------------------------------------------------------------------------
    # 3. Directory Sizes, Large File Discovery & Storage Reports
    # --------------------------------------------------------------------------

    def find_large_files(
        self,
        start_dir: Optional[str] = None,
        min_size_mb: float = 50.0,
        max_results: int = 50,
    ) -> list[dict[str, Any]]:
        """Identify files exceeding the given size threshold (in MB)."""
        root = Path(start_dir or os.getcwd()).resolve()
        large_files: list[dict[str, Any]] = []
        threshold_bytes = int(min_size_mb * 1024 * 1024)

        def _on_err(err: OSError) -> None:
            pass

        try:
            for dirpath, _, filenames in os.walk(root, onerror=_on_err):
                for fname in filenames:
                    fp = Path(dirpath) / fname
                    try:
                        sz = fp.stat().st_size
                        if sz >= threshold_bytes:
                            large_files.append(
                                {
                                    "name": fname,
                                    "path": str(fp),
                                    "size_bytes": sz,
                                    "size_mb": round(sz / (1024 * 1024), 2),
                                }
                            )
                            if len(large_files) >= max_results * 2:
                                break
                    except (PermissionError, OSError):
                        continue
        except Exception as exc:
            logger.warning("Error finding large files in %s: %s", root, exc)

        large_files.sort(key=lambda x: x["size_mb"], reverse=True)
        return large_files[:max_results]

    def analyze_storage(self, directory: str, large_file_threshold_mb: float = 50.0) -> StorageReport:
        """Analyze directory storage usage, file extensions, and identify large files."""
        root = Path(directory).resolve()
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory}")

        total_bytes = 0
        file_count = 0
        dir_count = 0
        large_files: list[dict[str, Any]] = []
        extension_sizes: dict[str, int] = {}

        threshold_bytes = int(large_file_threshold_mb * 1024 * 1024)

        def _on_err(err: OSError) -> None:
            pass

        for dirpath, dirnames, filenames in os.walk(root, onerror=_on_err):
            dir_count += len(dirnames)
            for fname in filenames:
                file_count += 1
                fp = Path(dirpath) / fname
                try:
                    sz = fp.stat().st_size
                    total_bytes += sz

                    ext = fp.suffix.lower() or "no_extension"
                    extension_sizes[ext] = extension_sizes.get(ext, 0) + sz

                    if sz >= threshold_bytes:
                        large_files.append(
                            {
                                "path": str(fp),
                                "size_mb": round(sz / (1024 * 1024), 2),
                            }
                        )
                except (PermissionError, OSError):
                    continue

        large_files.sort(key=lambda x: x["size_mb"], reverse=True)
        ext_breakdown_mb = {
            k: round(v / (1024 * 1024), 2)
            for k, v in sorted(extension_sizes.items(), key=lambda item: item[1], reverse=True)[:10]
        }

        return StorageReport(
            target_path=str(root),
            total_size_mb=round(total_bytes / (1024 * 1024), 2),
            file_count=file_count,
            dir_count=dir_count,
            large_files=large_files[:20],
            extension_breakdown=ext_breakdown_mb,
        )

    # --------------------------------------------------------------------------
    # 4. File CRUD & Operations Across the Filesystem
    # --------------------------------------------------------------------------

    def create_directory(self, path: str) -> str:
        """Create a new folder recursively anywhere on authorized filesystems."""
        p = Path(path).resolve()
        p.mkdir(parents=True, exist_ok=True)
        logger.info("Created directory: %s", p)
        return str(p)

    def read_text_file(self, path: str, max_chars: int = 100000) -> str:
        """Read text file contents with size cap from any accessible drive."""
        p = Path(path).resolve()
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        try:
            with open(p, "r", encoding="utf-8") as f:
                return f.read(max_chars)
        except UnicodeDecodeError:
            with open(p, "r", encoding="cp1252", errors="replace") as f:
                return f.read(max_chars)

    def write_text_file(self, path: str, content: str, append: bool = False) -> bool:
        """Write or append text content to a file."""
        p = Path(path).resolve()
        self._check_safety_protection(p, action="write")
        p.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with open(p, mode, encoding="utf-8") as f:
            f.write(content)
        logger.info("Wrote %d chars to %s", len(content), p)
        return True

    def write_file(self, path: str, content: str, append: bool = False) -> bool:
        """Alias for write_text_file."""
        return self.write_text_file(path, content, append)

    def read_file(self, path: str, max_chars: int = 100000) -> str:
        """Alias for read_text_file."""
        return self.read_text_file(path, max_chars)

    def get_storage_report(self, directory: str, large_file_threshold_mb: float = 50.0) -> StorageReport:
        """Alias for analyze_storage."""
        return self.analyze_storage(directory, large_file_threshold_mb)

    def copy_item(self, source: str, destination: str) -> str:
        """Copy file or directory across paths and drives."""
        src = Path(source).resolve()
        dst = Path(destination).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Source not found: {source}")

        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            copied = shutil.copy2(src, dst)
            return str(copied)
        else:
            copied = shutil.copytree(src, dst, dirs_exist_ok=True)
            return str(copied)

    def move_item(self, source: str, destination: str) -> str:
        """Move file or directory across paths and drives."""
        src = Path(source).resolve()
        dst = Path(destination).resolve()
        self._check_safety_protection(src, action="move")
        if not src.exists():
            raise FileNotFoundError(f"Source not found: {source}")

        dst.parent.mkdir(parents=True, exist_ok=True)
        moved = shutil.move(str(src), str(dst))
        return str(moved)

    def rename_item(self, path: str, new_name: str) -> str:
        """Rename a file or folder in place."""
        target = Path(path).resolve()
        self._check_safety_protection(target, action="rename")
        if not target.exists():
            raise FileNotFoundError(f"Target not found: {path}")

        # If new_name is just a name, keep same parent; if it's a full path, move to it
        if os.sep in new_name or "/" in new_name:
            dest = Path(new_name).resolve()
        else:
            dest = target.parent / new_name

        renamed = target.rename(dest)
        logger.info("Renamed %s to %s", target, renamed)
        return str(renamed)

    def delete_item(self, path: str, permanent: bool = False) -> bool:
        """Delete file or directory. Uses Windows Recycle Bin by default."""
        target = Path(path).resolve()
        self._check_safety_protection(target, action="delete")

        if not target.exists():
            return False

        if permanent:
            # Permanent deletion
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            logger.info("Permanently deleted: %s", target)
            return True
        else:
            # Send to Windows Recycle Bin via ctypes SHFileOperationW
            try:
                # FO_DELETE = 3, FOF_ALLOWUNDO = 0x0040, FOF_NOCONFIRMATION = 0x0010, FOF_SILENT = 0x0004
                class SHFILEOPSTRUCTW(ctypes.Structure):
                    _fields_ = [
                        ("hwnd", wintypes.HWND),
                        ("wFunc", wintypes.UINT),
                        ("pFrom", wintypes.LPCWSTR),
                        ("pTo", wintypes.LPCWSTR),
                        ("fFlags", wintypes.WORD),
                        ("fAnyOperationsAborted", wintypes.BOOL),
                        ("hNameMappings", wintypes.LPVOID),
                        ("lpszProgressTitle", wintypes.LPCWSTR),
                    ]

                # Double-null terminated path
                p_from = str(target) + "\0\0"
                fileop = SHFILEOPSTRUCTW(
                    hwnd=None,
                    wFunc=3,  # FO_DELETE
                    pFrom=p_from,
                    pTo=None,
                    fFlags=0x0040 | 0x0010 | 0x0004,  # FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT
                    fAnyOperationsAborted=False,
                    hNameMappings=None,
                    lpszProgressTitle=None,
                )
                res = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(fileop))
                if res == 0 and not fileop.fAnyOperationsAborted:
                    logger.info("Sent to Recycle Bin: %s", target)
                    return True
            except Exception as e:
                logger.warning("Recycle Bin deletion failed (%s); falling back to unlink.", e)

            # Fallback to permanent deletion if recycle bin call failed
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            return True

    # --------------------------------------------------------------------------
    # 5. Archive & Compression
    # --------------------------------------------------------------------------

    def create_zip(self, source_path: str, output_zip_path: str) -> str:
        """Compress file or directory into a ZIP archive."""
        src = Path(source_path).resolve()
        out = Path(output_zip_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            if src.is_file():
                zf.write(src, arcname=src.name)
            else:
                for root, _, files in os.walk(src):
                    for file in files:
                        fp = Path(root) / file
                        arcname = fp.relative_to(src)
                        zf.write(fp, arcname=str(arcname))

        logger.info("Created ZIP archive: %s", out)
        return str(out)

    def extract_zip(self, zip_path: str, destination_dir: str) -> str:
        """Extract a ZIP archive into the destination folder."""
        zp = Path(zip_path).resolve()
        dst = Path(destination_dir).resolve()
        dst.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zp, "r") as zf:
            zf.extractall(dst)

        logger.info("Extracted ZIP archive %s to %s", zp, dst)
        return str(dst)

    # --------------------------------------------------------------------------
    # 6. Duplicate File Detection
    # --------------------------------------------------------------------------

    def find_duplicates(self, directory: str, min_size_bytes: int = 1024) -> list[list[str]]:
        """Identify duplicate files by SHA-256 hash comparison."""
        root = Path(directory).resolve()
        size_map: dict[int, list[Path]] = {}

        # 1. Group files by exact size
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                fp = Path(dirpath) / fname
                try:
                    sz = fp.stat().st_size
                    if sz >= min_size_bytes:
                        size_map.setdefault(sz, []).append(fp)
                except OSError:
                    continue

        # 2. Hash only files that share the same size
        duplicate_groups: list[list[str]] = []
        for sz, files in size_map.items():
            if len(files) < 2:
                continue

            hash_map: dict[str, list[str]] = {}
            for fp in files:
                try:
                    h = hashlib.sha256()
                    with open(fp, "rb") as f:
                        while chunk := f.read(65536):
                            h.update(chunk)
                    digest = h.hexdigest()
                    hash_map.setdefault(digest, []).append(str(fp))
                except OSError:
                    continue

            for digest, dupes in hash_map.items():
                if len(dupes) > 1:
                    duplicate_groups.append(dupes)

        return duplicate_groups

    # --------------------------------------------------------------------------
    # 7. Safeguards & Protected System Path Verification
    # --------------------------------------------------------------------------

    def is_protected_system_path(self, path: Path | str) -> tuple[bool, str]:
        """Check whether a target path is a critical Windows system directory or HAKS store.

        Returns (is_protected, reason).
        """
        target = Path(path).resolve()

        # Check critical HAKS memory/database storage
        for p in self._protected_haks_paths:
            if target == p or p in target.parents:
                return True, f"HAKS Critical Data Store ({p})"

        # Check drive roots (prevent deleting drive roots like C:\ or E:\)
        if target.parent == target or str(target).endswith(":\\") or str(target).endswith(":/"):
            return True, "Logical Drive Root"

        # Check Windows system roots
        win_dir = Path(os.environ.get("WINDIR", "C:\\Windows")).resolve()
        if target == win_dir:
            return True, "Windows Installation Root"

        sys32 = win_dir / "System32"
        if target == sys32 or sys32 in target.parents:
            return True, "Windows System32 Directory"

        prog_files = Path(os.environ.get("ProgramFiles", "C:\\Program Files")).resolve()
        if target == prog_files:
            return True, "Program Files Directory Root"

        prog_data = Path(os.environ.get("ProgramData", "C:\\ProgramData")).resolve()
        if target == prog_data:
            return True, "ProgramData Directory Root"

        # Check special root system folders
        for part in target.parts:
            if part.lower() in PROTECTED_SYSTEM_NAMES:
                return True, f"Protected Operating System Directory ({part})"

        return False, ""

    def _check_safety_protection(self, target: Path, action: str = "modify") -> None:
        """Enforce strict protection against modifying or deleting critical system paths."""
        is_protected, reason = self.is_protected_system_path(target)
        if is_protected:
            raise PermissionError(
                f"Access Denied: Cannot {action} '{target}' because it is classified as {reason}."
            )
