"""Network Connectivity Monitor for HAKS Phase 8.

Monitors real-time internet connectivity, detects offline/online transitions,
notifies listeners, and triggers automatic reconnection workflows.
"""

from datetime import datetime, timezone
import logging
import os
import socket
import time
from typing import Callable, Optional

from PySide6.QtCore import QObject, QTimer

from core.events import ApplicationEvents, get_app_events
from recovery.models import NetworkStatus

logger = logging.getLogger("haks.recovery.network")


class NetworkMonitor(QObject):
    """Monitors host network connectivity and dispatches state transition events."""

    def __init__(
        self,
        events: Optional[ApplicationEvents] = None,
        probe_host: str = "1.1.1.1",
        probe_port: int = 53,
        poll_interval_seconds: int = 10,
    ) -> None:
        super().__init__()
        self.events = events or get_app_events()
        self.probe_host = probe_host
        self.probe_port = probe_port
        self.poll_interval_ms = poll_interval_seconds * 1000

        self._current_status = NetworkStatus()
        self._online_callbacks: list[Callable[[], None]] = []
        self._offline_callbacks: list[Callable[[], None]] = []
        self._is_polling = False
        self._mock_status: Optional[bool] = None

        self._timer = QTimer(self)
        self._timer.setInterval(self.poll_interval_ms)
        self._timer.timeout.connect(self._on_poll)

        # Initial check
        self.check_connectivity(force_probe=True)

        # Auto-start timer only in live UI run (skip during automated test discovery unless requested)
        if not os.environ.get("PYTEST_CURRENT_TEST"):
            self.start_monitoring()

    def start_monitoring(self) -> None:
        """Begin periodic background connectivity polling."""
        if not self._is_polling:
            self._timer.start()
            self._is_polling = True
            logger.info("Network monitoring started (interval=%ss).", self.poll_interval_ms // 1000)

    def stop_monitoring(self) -> None:
        """Stop background connectivity polling."""
        if self._is_polling:
            self._timer.stop()
            self._is_polling = False
            logger.info("Network monitoring stopped.")

    def add_online_callback(self, cb: Callable[[], None]) -> None:
        """Register callback invoked when transitioning to ONLINE."""
        self._online_callbacks.append(cb)

    def add_offline_callback(self, cb: Callable[[], None]) -> None:
        """Register callback invoked when transitioning to OFFLINE."""
        self._offline_callbacks.append(cb)

    def force_status(self, online: Optional[bool]) -> None:
        """Override real connectivity check for testing or manual offline mode."""
        self._mock_status = online
        self.check_connectivity(force_probe=True)

    @property
    def is_online(self) -> bool:
        """Return True if internet is currently reachable."""
        if self._mock_status is not None:
            return self._mock_status
        return self._current_status.is_online

    def get_status(self) -> NetworkStatus:
        """Return cached or latest NetworkStatus."""
        return self._current_status

    def check_connectivity(self, force_probe: bool = False) -> NetworkStatus:
        """Probe network and update status. Detects transitions."""
        prev_online = self._current_status.is_online

        if self._mock_status is not None:
            is_online = self._mock_status
            latency = 12.0 if is_online else None
            msg = "Mocked Online" if is_online else "Mocked Offline"
        else:
            is_online, latency, msg = self._probe_socket()

        new_status = NetworkStatus(
            is_online=is_online,
            last_checked=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            latency_ms=latency,
            status_message=msg,
            probe_host=self.probe_host,
        )
        self._current_status = new_status

        # Transition handling
        if prev_online != is_online:
            logger.info(
                "Network transition detected: %s -> %s (%s)",
                "ONLINE" if prev_online else "OFFLINE",
                "ONLINE" if is_online else "OFFLINE",
                msg,
            )
            self.events.network_connectivity_changed.emit(is_online)

            if is_online:
                for cb in self._online_callbacks:
                    try:
                        cb()
                    except Exception as exc:
                        logger.error("Error in online callback: %s", exc)
            else:
                for cb in self._offline_callbacks:
                    try:
                        cb()
                    except Exception as exc:
                        logger.error("Error in offline callback: %s", exc)

        return new_status

    def _probe_socket(self) -> tuple[bool, Optional[float], str]:
        """Perform a fast non-blocking TCP socket connection to a reliable public DNS address."""
        t0 = time.perf_counter()
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.2)
            sock.connect((self.probe_host, self.probe_port))
            latency = round((time.perf_counter() - t0) * 1000, 2)
            return True, latency, f"Connected to {self.probe_host} in {latency}ms"
        except (socket.timeout, socket.error) as exc:
            # Fallback probe to secondary host if primary timed out
            try:
                if sock:
                    sock.close()
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1.2)
                sock.connect(("8.8.8.8", 53))
                latency = round((time.perf_counter() - t0) * 1000, 2)
                return True, latency, f"Connected to 8.8.8.8 in {latency}ms"
            except Exception:
                return False, None, f"No internet connectivity: {exc}"
        finally:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass

    def _on_poll(self) -> None:
        self.check_connectivity(force_probe=True)
