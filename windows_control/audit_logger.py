"""Audit Logging Service for HAKS Windows Computer Control.

Maintains an immutable, searchable execution trail in settings.db with
automatic credential redaction to ensure compliance, transparency, and safety.
"""

from datetime import datetime, timezone
import json
import logging
from typing import Any, Optional

from config.settings import AppConfig, get_settings
from storage.database import DatabaseManager, get_db_manager
from privacy.redactor import get_data_redactor
from windows_control.models import AuditLogRecord

logger = logging.getLogger("haks.windows_control.audit")

CREATE_AUDIT_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS windows_control_audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source TEXT NOT NULL,
    instruction TEXT NOT NULL,
    plan_id TEXT NOT NULL,
    operation_type TEXT NOT NULL,
    executable TEXT,
    args TEXT,
    working_dir TEXT,
    privilege_level TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    confirmation_status TEXT NOT NULL,
    exit_code INTEGER DEFAULT 0,
    status TEXT NOT NULL,
    duration_ms REAL DEFAULT 0.0,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_win_audit_timestamp ON windows_control_audit_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_win_audit_status ON windows_control_audit_logs(status);
CREATE INDEX IF NOT EXISTS idx_win_audit_plan ON windows_control_audit_logs(plan_id);
"""


class AuditLogger:
    """Records and searches computer control execution records with data redaction."""

    def __init__(
        self,
        config: Optional[Any] = None,
        db_manager: Optional[DatabaseManager] = None,
    ) -> None:
        if isinstance(config, DatabaseManager):
            db_manager = config
            config = None
        self.config = config or get_settings()
        self.db = db_manager or get_db_manager(self.config)
        self.redactor = get_data_redactor()
        self._init_db()

    def _init_db(self) -> None:
        """Ensure audit table and indexes exist in settings database."""
        try:
            with self.db.session("settings") as conn:
                conn.executescript(CREATE_AUDIT_TABLE_SQL)
        except Exception as exc:
            logger.warning("Could not initialize audit table: %s", exc)

    def log_execution(
        self,
        source: str,
        instruction: str,
        plan_id: str,
        operation_type: str,
        executable: str,
        args: str,
        working_dir: str,
        privilege_level: str,
        risk_level: str,
        confirmation_status: str,
        exit_code: int = 0,
        status: str = "completed",
        duration_ms: float = 0.0,
        error: Optional[str] = None,
    ) -> Optional[int]:
        """Record an operation in the audit trail with secret redaction."""
        # Redact secrets from instruction, args, and error
        clean_instr = self.redactor.redact(instruction).sanitized_text
        clean_args = self.redactor.redact(args).sanitized_text
        clean_err = self.redactor.redact(error).sanitized_text if error else None

        sql = """
        INSERT INTO windows_control_audit_logs (
            source, instruction, plan_id, operation_type, executable,
            args, working_dir, privilege_level, risk_level, confirmation_status,
            exit_code, status, duration_ms, error
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        try:
            with self.db.session("settings") as conn:
                cursor = conn.execute(
                    sql,
                    (
                        source,
                        clean_instr,
                        plan_id,
                        operation_type,
                        executable,
                        clean_args,
                        working_dir,
                        privilege_level,
                        risk_level,
                        confirmation_status,
                        exit_code,
                        status,
                        duration_ms,
                        clean_err,
                    ),
                )
                return cursor.lastrowid
        except Exception as exc:
            logger.error("Failed to write to audit log: %s", exc)
            return None

    def query_logs(
        self,
        limit: int = 100,
        source_filter: Optional[str] = None,
        status_filter: Optional[str] = None,
    ) -> list[AuditLogRecord]:
        """Query recent audit log entries."""
        query = "SELECT * FROM windows_control_audit_logs WHERE 1=1"
        params: list[Any] = []

        if source_filter:
            query += " AND source = ?"
            params.append(source_filter)
        if status_filter:
            query += " AND status = ?"
            params.append(status_filter)

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        records: list[AuditLogRecord] = []
        try:
            with self.db.session("settings") as conn:
                rows = conn.execute(query, tuple(params)).fetchall()
                for row in rows:
                    records.append(
                        AuditLogRecord(
                            id=row["id"],
                            timestamp=str(row["timestamp"]),
                            source=row["source"],
                            instruction=row["instruction"],
                            plan_id=row["plan_id"],
                            operation_type=row["operation_type"],
                            executable=row["executable"] or "",
                            args=row["args"] or "",
                            working_dir=row["working_dir"] or "",
                            privilege_level=row["privilege_level"],
                            risk_level=row["risk_level"],
                            confirmation_status=row["confirmation_status"],
                            exit_code=row["exit_code"],
                            status=row["status"],
                            duration_ms=row["duration_ms"],
                            error=row["error"],
                        )
                    )
        except Exception as exc:
            logger.error("Error reading audit logs: %s", exc)

        return records
