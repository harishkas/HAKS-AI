"""Software and File Download Manager for HAKS Windows Computer Control.

Handles direct HTTP/HTTPS file downloads, streaming progress monitoring,
destination directory selection, SHA-256 integrity verification, cancellation,
and download history tracking without automatic execution.
"""

import asyncio
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
from typing import Callable, Optional
import uuid

import httpx

from config.settings import AppConfig, get_settings
from windows_control.installer_verifier import InstallerVerifier
from windows_control.models import DownloadItem

logger = logging.getLogger("haks.windows_control.download")


class DownloadManager:
    """Asynchronous HTTP/HTTPS file download manager with progress tracking and verification."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or get_settings()
        self.default_dir: Path = Path(os.environ.get("USERPROFILE", Path.home())) / "Downloads"
        self.default_dir.mkdir(parents=True, exist_ok=True)

        self._active_tasks: dict[str, asyncio.Task] = {}
        self._downloads: dict[str, DownloadItem] = {}

    def get_downloads(self) -> list[DownloadItem]:
        """Return history of all tracked downloads."""
        return list(self._downloads.values())

    def get_download(self, download_id: str) -> Optional[DownloadItem]:
        """Get status of specific download item."""
        return self._downloads.get(download_id)

    async def start_download(
        self,
        url: str,
        destination_dir: Optional[str] = None,
        custom_filename: Optional[str] = None,
        expected_sha256: Optional[str] = None,
        progress_callback: Optional[Callable[[DownloadItem], None]] = None,
    ) -> DownloadItem:
        """Initiate asynchronous file download with streaming progress."""
        dest_folder = Path(destination_dir).resolve() if destination_dir else self.default_dir
        dest_folder.mkdir(parents=True, exist_ok=True)

        # Determine filename
        if custom_filename:
            filename = custom_filename
        else:
            raw_name = url.split("?")[0].rstrip("/").split("/")[-1]
            filename = raw_name if raw_name else f"download_{uuid.uuid4().hex[:8]}.bin"

        dest_file = dest_folder / filename
        dl_id = f"dl_{uuid.uuid4().hex[:10]}"

        item = DownloadItem(
            id=dl_id,
            url=url,
            destination=str(dest_file),
            filename=filename,
            total_bytes=0,
            downloaded_bytes=0,
            progress_percent=0.0,
            status="downloading",
            sha256_expected=expected_sha256,
        )
        self._downloads[dl_id] = item

        # Launch background async task
        task = asyncio.create_task(
            self._download_worker(item, dest_file, progress_callback)
        )
        self._active_tasks[dl_id] = task
        return item

    async def _download_worker(
        self,
        item: DownloadItem,
        dest_file: Path,
        progress_callback: Optional[Callable[[DownloadItem], None]],
    ) -> None:
        """Internal asynchronous worker streaming download chunks to disk."""
        temp_file = dest_file.with_suffix(f"{dest_file.suffix}.part")
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
                async with client.stream("GET", item.url) as response:
                    if response.status_code >= 400:
                        item.status = "failed"
                        item.error = f"HTTP Error {response.status_code}: {response.reason_phrase}"
                        logger.error("Download failed [%s]: %s", item.id, item.error)
                        return

                    total_len = response.headers.get("Content-Length")
                    item.total_bytes = int(total_len) if total_len and total_len.isdigit() else 0

                    with open(temp_file, "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size=65536):
                            f.write(chunk)
                            item.downloaded_bytes += len(chunk)
                            if item.total_bytes > 0:
                                item.progress_percent = round(
                                    (item.downloaded_bytes / item.total_bytes) * 100.0, 1
                                )
                            if progress_callback:
                                try:
                                    progress_callback(item)
                                except Exception:
                                    pass

            # Rename temp file to target upon successful download
            temp_file.replace(dest_file)

            # Checksum verification
            actual_hash = InstallerVerifier.calculate_sha256(dest_file)
            item.sha256_actual = actual_hash

            if item.sha256_expected:
                verified = actual_hash.lower() == item.sha256_expected.lower().strip()
                item.checksum_verified = verified
                if not verified:
                    item.status = "failed"
                    item.error = "Checksum mismatch: Downloaded file did not match expected SHA-256."
                    logger.warning("[%s] %s", item.id, item.error)
                    return
            else:
                item.checksum_verified = True

            item.status = "completed"
            item.progress_percent = 100.0
            logger.info("Download completed successfully [%s]: %s", item.id, dest_file)

        except asyncio.CancelledError:
            item.status = "cancelled"
            item.error = "Download was cancelled by user."
            temp_file.unlink(missing_ok=True)
            logger.info("Download [%s] was cancelled.", item.id)
        except Exception as exc:
            item.status = "failed"
            item.error = str(exc)
            temp_file.unlink(missing_ok=True)
            logger.error("Exception during download [%s]: %s", item.id, exc)
        finally:
            self._active_tasks.pop(item.id, None)
            if progress_callback:
                try:
                    progress_callback(item)
                except Exception:
                    pass

    def cancel_download(self, download_id: str) -> bool:
        """Cancel an in-progress download task."""
        task = self._active_tasks.get(download_id)
        if task and not task.done():
            task.cancel()
            return True
        return False

    def cancel_all(self) -> int:
        """Cancel all active downloads (Emergency Stop)."""
        count = 0
        for task in self._active_tasks.values():
            if not task.done():
                task.cancel()
                count += 1
        return count
