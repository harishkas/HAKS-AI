"""Safe Filesystem Operations and Storage Analysis Engine (Phase 9.2).

Performs metadata filtering, disk usage analysis, large-file identification,
streaming SHA-256 and BLAKE3 duplicate detection, and safe Recycle Bin operations.
"""

from collections import defaultdict
import fnmatch
import hashlib
import logging
import mimetypes
import os
from pathlib import Path
import shutil
import time
from typing import Any, Optional, Tuple

import blake3
import send2trash

from file_intelligence.models import DiskUsageSummary, DuplicateGroup, FileMetadata
from file_intelligence.privacy_filter import PrivacyFilter

logger = logging.getLogger("haks.file_intelligence.filesystem")


class FilesystemOperationsService:
    """Manages safe file operations, deduplication, and storage analytics."""

    def __init__(self, privacy_filter: Optional[PrivacyFilter] = None) -> None:
        self.privacy = privacy_filter or PrivacyFilter()

    # --------------------------------------------------------------------------
    # 1. File & Metadata Inspection
    # --------------------------------------------------------------------------

    def get_file_info(self, file_path: str | Path) -> Optional[FileMetadata]:
        """Collect comprehensive metadata for a single file."""
        safe, msg = self.privacy.is_path_safe_to_access(file_path)
        if not safe:
            logger.warning("Access to file denied by privacy policy: %s (%s)", file_path, msg)
            return None

        p = Path(file_path).resolve()
        stat = p.stat()
        mime, _ = mimetypes.guess_type(str(p))

        return FileMetadata(
            path=str(p),
            name=p.name,
            extension=p.suffix.lower(),
            size_bytes=stat.st_size,
            created_time=stat.st_ctime,
            modified_time=stat.st_mtime,
            mime_type=mime or "application/octet-stream",
            is_directory=p.is_dir(),
        )

    def search_files_by_metadata(
        self,
        root_dir: str | Path,
        name_pattern: Optional[str] = None,
        extensions: Optional[list[str]] = None,
        min_size_bytes: int = 0,
        max_size_bytes: Optional[int] = None,
        modified_after: Optional[float] = None,
        max_results: int = 100,
    ) -> list[FileMetadata]:
        """Scan directory with customizable metadata filters."""
        root = Path(root_dir).resolve()
        if not root.exists() or not root.is_dir():
            return []

        clean_exts = [e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions] if extensions else None
        results: list[FileMetadata] = []

        for dirpath, dirnames, filenames in os.walk(str(root)):
            # Filter out hidden or blocked directory branches
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in ("node_modules", "__pycache__")]

            for fname in filenames:
                fp = Path(dirpath) / fname
                # Check privacy allowlist/blocklist
                if not self.privacy.is_file_indexable(fp):
                    continue

                if name_pattern and not fnmatch.fnmatch(fname.lower(), name_pattern.lower()):
                    continue

                if clean_exts and fp.suffix.lower() not in clean_exts:
                    continue

                try:
                    st = fp.stat()
                    if st.st_size < min_size_bytes:
                        continue
                    if max_size_bytes is not None and st.st_size > max_size_bytes:
                        continue
                    if modified_after is not None and st.st_mtime < modified_after:
                        continue

                    mime, _ = mimetypes.guess_type(fname)
                    results.append(
                        FileMetadata(
                            path=str(fp),
                            name=fname,
                            extension=fp.suffix.lower(),
                            size_bytes=st.st_size,
                            created_time=st.st_ctime,
                            modified_time=st.st_mtime,
                            mime_type=mime or "application/octet-stream",
                        )
                    )

                    if len(results) >= max_results:
                        return results
                except (OSError, PermissionError):
                    continue

        return results

    def find_large_files(
        self,
        root_dir: str | Path,
        min_size_mb: float = 50.0,
        top_n: int = 20,
    ) -> list[FileMetadata]:
        """Identify files exceeding size threshold sorted descending."""
        min_bytes = int(min_size_mb * 1024 * 1024)
        files = self.search_files_by_metadata(
            root_dir=root_dir,
            min_size_bytes=min_bytes,
            max_results=500,
        )
        files.sort(key=lambda f: f.size_bytes, reverse=True)
        return files[:top_n]

    # --------------------------------------------------------------------------
    # 2. Disk Usage Analytics
    # --------------------------------------------------------------------------

    def analyze_disk_usage(self, root_dir: str | Path, top_large_files: int = 10) -> DiskUsageSummary:
        """Compute hierarchical breakdown of file counts, extensions, and storage consumption."""
        root = Path(root_dir).resolve()
        summary = DiskUsageSummary(
            root_path=str(root),
            total_files=0,
            total_directories=0,
            total_size_bytes=0,
        )

        all_files: list[FileMetadata] = []

        for dirpath, dirnames, filenames in os.walk(str(root)):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in ("node_modules", "__pycache__")]
            summary.total_directories += len(dirnames)

            for fname in filenames:
                fp = Path(dirpath) / fname
                if not self.privacy.is_file_indexable(fp):
                    continue

                try:
                    st = fp.stat()
                    ext = fp.suffix.lower() or ".none"
                    summary.total_files += 1
                    summary.total_size_bytes += st.st_size

                    summary.by_extension[ext] = summary.by_extension.get(ext, 0) + st.st_size
                    summary.by_extension_counts[ext] = summary.by_extension_counts.get(ext, 0) + 1

                    all_files.append(
                        FileMetadata(
                            path=str(fp),
                            name=fname,
                            extension=ext,
                            size_bytes=st.st_size,
                            created_time=st.st_ctime,
                            modified_time=st.st_mtime,
                        )
                    )
                except (OSError, PermissionError):
                    continue

        all_files.sort(key=lambda x: x.size_bytes, reverse=True)
        summary.largest_files = all_files[:top_large_files]
        return summary

    # --------------------------------------------------------------------------
    # 3. Two-Stage Streaming Duplicate Identification (SHA-256 & BLAKE3)
    # --------------------------------------------------------------------------

    def calculate_file_hash(self, file_path: str | Path, algorithm: str = "sha256") -> str:
        """Compute cryptographic hash of a file using chunked streaming buffers (64KB chunks)."""
        p = Path(file_path).resolve()
        hasher = blake3.blake3() if algorithm.lower() == "blake3" else hashlib.sha256()

        with open(p, "rb") as f:
            while True:
                buf = f.read(65536)
                if not buf:
                    break
                hasher.update(buf)

        return hasher.hexdigest()

    def find_duplicates(
        self,
        root_dir: str | Path,
        use_blake3_benchmark: bool = False,
    ) -> Tuple[list[DuplicateGroup], dict[str, float]]:
        """Identify duplicate files using exact file size filtering followed by streaming content hash.

        Rule: Never deletes duplicates automatically. Returns groups for user review.
        """
        root = Path(root_dir).resolve()
        # Stage 1: Group files by exact byte size
        size_groups: dict[int, list[Path]] = defaultdict(list)

        for dirpath, dirnames, filenames in os.walk(str(root)):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in ("node_modules", "__pycache__")]
            for fname in filenames:
                fp = Path(dirpath) / fname
                if not self.privacy.is_file_indexable(fp):
                    continue
                try:
                    sz = fp.stat().st_size
                    if sz > 0:  # Skip zero-byte files
                        size_groups[sz].append(fp)
                except (OSError, PermissionError):
                    continue

        # Keep only sizes with >= 2 candidate files
        candidate_sizes = {sz: paths for sz, paths in size_groups.items() if len(paths) >= 2}

        # Stage 2: Streaming content hashing
        benchmark_metrics: dict[str, float] = {}
        t0 = time.perf_counter()
        total_hashed_bytes = sum(sz * len(paths) for sz, paths in candidate_sizes.items())

        # Group by hash
        hash_groups: dict[str, list[str]] = defaultdict(list)
        file_to_size: dict[str, int] = {}
        blake3_hashes: dict[str, str] = {}

        for sz, paths in candidate_sizes.items():
            for p in paths:
                try:
                    str_p = str(p)
                    file_to_size[str_p] = sz
                    sha = self.calculate_file_hash(p, algorithm="sha256")
                    hash_groups[sha].append(str_p)

                    if use_blake3_benchmark:
                        b3 = self.calculate_file_hash(p, algorithm="blake3")
                        blake3_hashes[sha] = b3
                except Exception as exc:
                    logger.warning("Error hashing %s: %s", p, exc)

        t_elapsed = time.perf_counter() - t0
        mb_hashed = total_hashed_bytes / (1024 * 1024)
        throughput = mb_hashed / (t_elapsed + 1e-6)

        benchmark_metrics["hash_time_seconds"] = round(t_elapsed, 3)
        benchmark_metrics["total_mb_hashed"] = round(mb_hashed, 2)
        benchmark_metrics["throughput_mb_per_sec"] = round(throughput, 2)

        duplicates: list[DuplicateGroup] = []
        for sha, matched_paths in hash_groups.items():
            if len(matched_paths) >= 2:
                sz = file_to_size[matched_paths[0]]
                duplicates.append(
                    DuplicateGroup(
                        size_bytes=sz,
                        sha256_hash=sha,
                        blake3_hash=blake3_hashes.get(sha),
                        file_paths=matched_paths,
                    )
                )

        # Sort by wasted storage descending
        duplicates.sort(key=lambda d: d.wasted_bytes, reverse=True)
        return duplicates, benchmark_metrics

    # --------------------------------------------------------------------------
    # 4. Safe Filesystem Operations (Recycle Bin, Move, Copy, Rename)
    # --------------------------------------------------------------------------

    def send_to_recycle_bin(self, target_path: str | Path) -> Tuple[bool, str]:
        """Safely move a file or directory to the Windows Recycle Bin.

        Rule: Never permanently deletes files as a fallback.
        """
        p = Path(target_path).resolve()
        safe, msg = self.privacy.is_path_safe_to_access(p)
        if not safe:
            return False, f"Recycle Bin operation blocked: {msg}"

        if not p.exists():
            return False, f"Target '{p}' does not exist."

        try:
            send2trash.send2trash(str(p))
            logger.info("Sent file to Windows Recycle Bin: %s", p)
            return True, f"Safely moved '{p.name}' to the Windows Recycle Bin."
        except Exception as exc:
            logger.error("Failed to send '%s' to Recycle Bin: %s", p, exc)
            return False, f"Could not move to Recycle Bin: {exc}. Permanent deletion was NOT attempted."

    def copy_file(self, source: str | Path, destination: str | Path, overwrite: bool = False) -> Tuple[bool, str]:
        """Safely copy a file to a new location."""
        src = Path(source).resolve()
        dst = Path(destination).resolve()

        safe_src, msg_src = self.privacy.is_path_safe_to_access(src)
        if not safe_src:
            return False, f"Copy blocked: {msg_src}"

        if not src.exists() or not src.is_file():
            return False, f"Source file does not exist: {src}"

        if dst.exists() and not overwrite:
            return False, f"Destination '{dst.name}' already exists and overwrite is False."

        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dst))
            logger.info("Copied '%s' -> '%s'", src, dst)
            return True, f"Successfully copied '{src.name}' to '{dst}'."
        except Exception as exc:
            logger.error("Error copying file '%s': %s", src, exc)
            return False, f"Failed to copy file: {exc}"

    def move_file(self, source: str | Path, destination: str | Path, overwrite: bool = False) -> Tuple[bool, str]:
        """Safely move a file to a new directory or rename."""
        src = Path(source).resolve()
        dst = Path(destination).resolve()

        safe_src, msg_src = self.privacy.is_path_safe_to_access(src)
        if not safe_src:
            return False, f"Move blocked: {msg_src}"

        if not src.exists():
            return False, f"Source '{src}' does not exist."

        if dst.exists() and not overwrite:
            return False, f"Destination '{dst.name}' already exists."

        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            logger.info("Moved '%s' -> '%s'", src, dst)
            return True, f"Successfully moved '{src.name}' to '{dst}'."
        except Exception as exc:
            logger.error("Error moving file '%s': %s", src, exc)
            return False, f"Failed to move file: {exc}"

    def rename_file(self, target: str | Path, new_name: str) -> Tuple[bool, str]:
        """Rename a file in its current directory."""
        tgt = Path(target).resolve()
        safe, msg = self.privacy.is_path_safe_to_access(tgt)
        if not safe:
            return False, f"Rename blocked: {msg}"

        if not tgt.exists():
            return False, f"Target '{tgt}' does not exist."

        clean_new = Path(new_name).name  # Prevent path traversal
        new_path = tgt.parent / clean_new
        if new_path.exists():
            return False, f"Target filename '{clean_new}' already exists."

        try:
            tgt.rename(new_path)
            logger.info("Renamed '%s' -> '%s'", tgt.name, clean_new)
            return True, f"Renamed '{tgt.name}' to '{clean_new}'."
        except Exception as exc:
            logger.error("Error renaming '%s': %s", tgt, exc)
            return False, f"Failed to rename file: {exc}"

    def create_directory(self, dir_path: str | Path) -> Tuple[bool, str]:
        """Safely create a new directory."""
        p = Path(dir_path).resolve()
        if ".." in str(dir_path):
            return False, "Path traversal forbidden."

        try:
            p.mkdir(parents=True, exist_ok=True)
            logger.info("Created directory: %s", p)
            return True, f"Created directory: {p}"
        except Exception as exc:
            logger.error("Failed to create directory '%s': %s", p, exc)
            return False, f"Error creating directory: {exc}"

