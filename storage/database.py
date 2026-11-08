"""Database manager for HAKS SQLite databases.

Ensures independent database storage files, WAL mode, foreign key enforcement,
and clean connection pooling / lifecycle.
"""

import logging
from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Generator, Optional

from config.settings import AppConfig, get_settings
from storage.schema import DATABASE_SCHEMAS

logger = logging.getLogger("haks.storage.database")


class DatabaseManager:
    """Manages independent SQLite databases for HAKS."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or get_settings()
        self.data_dir: Path = self.config.paths.data_dir
        self._db_files: dict[str, str] = {
            "personal_memory": self.config.database.personal_memory_file,
            "work_memory": self.config.database.work_memory_file,
            "conversations": self.config.database.conversations_file,
            "memory_proposals": self.config.database.memory_proposals_file,
            "settings": self.config.database.settings_file,
        }

    def get_db_path(self, db_name: str) -> Path:
        """Get the absolute filesystem path for a named database, ensuring directory exists."""
        if db_name not in self._db_files:
            raise ValueError(f"Unknown database name: '{db_name}'. Available: {list(self._db_files.keys())}")
        target_path = self.data_dir / self._db_files[db_name]
        target_path.parent.mkdir(parents=True, exist_ok=True)
        return target_path

    def _auto_migrate_legacy_db_files(self, db_name: str, target_path: Path) -> None:
        """Automatically migrate legacy flat database files if they exist."""
        flat_db = self.data_dir / f"{db_name}.db"
        if flat_db.exists() and not target_path.exists():
            try:
                logger.info("Migrating legacy flat database %s -> %s", flat_db, target_path)
                flat_db.rename(target_path)
                for ext in ["-wal", "-shm"]:
                    companion = self.data_dir / f"{db_name}.db{ext}"
                    if companion.exists():
                        target_companion = Path(f"{target_path}{ext}")
                        companion.rename(target_companion)
            except Exception as exc:
                logger.warning("Could not auto-migrate flat database %s: %s", db_name, exc)

    def create_connection(self, db_name: str) -> sqlite3.Connection:
        """Create and configure a new SQLite connection with WAL mode and foreign keys."""
        db_path = self.get_db_path(db_name)
        self._auto_migrate_legacy_db_files(db_name, db_path)

        conn = sqlite3.connect(
            str(db_path),
            timeout=self.config.database.busy_timeout_ms / 1000.0,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row

        # Performance & Concurrency PRAGMAs
        if self.config.database.enable_wal:
            conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        if self.config.database.enable_foreign_keys:
            conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(f"PRAGMA busy_timeout = {self.config.database.busy_timeout_ms};")

        return conn

    @contextmanager
    def session(self, db_name: str) -> Generator[sqlite3.Connection, None, None]:
        """Context manager providing a transactional connection."""
        conn = self.create_connection(db_name)
        try:
            yield conn
            conn.commit()
        except Exception as exc:
            conn.rollback()
            logger.error("Database transaction rolled back on '%s': %s", db_name, exc)
            raise
        finally:
            conn.close()

    def initialize(self) -> None:
        """Initialize all 5 databases and apply their schemas."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        for db_name, schema_sql in DATABASE_SCHEMAS.items():
            db_path = self.get_db_path(db_name)
            logger.info("Initializing database: %s -> %s", db_name, db_path)
            with self.session(db_name) as conn:
                conn.executescript(schema_sql)

        # Ensure Phase 4 schema additions exist in pre-existing tables
        for db_name, tbl_name, col_name, col_type in [
            ("personal_memory", "personal_memories", "source_conversation_id", "TEXT"),
            ("work_memory", "work_memories", "source_conversation_id", "TEXT"),
            ("memory_proposals", "memory_proposals", "suggested_title", "TEXT"),
        ]:
            try:
                with self.session(db_name) as conn:
                    cursor = conn.cursor()
                    cursor.execute(f"PRAGMA table_info({tbl_name});")
                    cols = [c[1] for c in cursor.fetchall()]
                    if cols and col_name not in cols:
                        cursor.execute(f"ALTER TABLE {tbl_name} ADD COLUMN {col_name} {col_type};")
            except Exception as e:
                logger.debug("Column migration note for %s.%s: %s", tbl_name, col_name, e)

        logger.info("All 5 HAKS databases initialized successfully.")

    def get_stats(self) -> dict[str, dict]:
        """Return file sizes and table row counts for all databases."""
        stats = {}
        for db_name in self._db_files:
            db_path = self.get_db_path(db_name)
            file_size_bytes = db_path.stat().st_size if db_path.exists() else 0
            tables_info = {}

            if db_path.exists():
                try:
                    with self.session(db_name) as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
                        tables = [r[0] for r in cursor.fetchall()]
                        for table in tables:
                            cursor.execute(f"SELECT COUNT(*) FROM {table};")  # safe: table name from sqlite_master
                            count = cursor.fetchone()[0]
                            tables_info[table] = count
                except Exception as e:
                    logger.warning("Could not read stats for '%s': %s", db_name, e)

            stats[db_name] = {
                "file_path": str(db_path),
                "size_bytes": file_size_bytes,
                "tables": tables_info,
            }
        return stats

    def clear_domain(self, db_name: str) -> None:
        """Clear all records from a specified database domain safely.
        
        Guarantees that clearing one domain (e.g. work_memory) never touches another.
        """
        if db_name not in ("personal_memory", "work_memory", "conversations", "memory_proposals"):
            raise ValueError(f"Domain '{db_name}' cannot be cleared.")
        
        logger.warning("Clearing memory domain: %s", db_name)
        with self.session(db_name) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            tables = [r[0] for r in cursor.fetchall()]
            for table in tables:
                cursor.execute(f"DELETE FROM {table};")
        logger.info("Domain '%s' cleared successfully.", db_name)

    def checkpoint_all(self) -> None:
        """Flush WAL changes into main database files for all active domains."""
        for db_name in self._db_files:
            db_path = self.get_db_path(db_name)
            if db_path.exists():
                try:
                    with self.session(db_name) as conn:
                        conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                except Exception as exc:
                    logger.warning("WAL checkpoint failed on '%s': %s", db_name, exc)

    def vacuum_domain(self, db_name: str) -> None:
        """Run VACUUM on a domain database to defragment and reclaim disk space."""
        if db_name not in self._db_files:
            raise ValueError(f"Unknown database name: '{db_name}'.")
        with self.session(db_name) as conn:
            conn.execute("VACUUM;")
        logger.info("Domain '%s' vacuumed successfully.", db_name)


_db_manager_instance: Optional[DatabaseManager] = None


def get_db_manager(config: Optional[AppConfig] = None) -> DatabaseManager:
    """Return singleton DatabaseManager instance."""
    global _db_manager_instance
    if _db_manager_instance is None:
        _db_manager_instance = DatabaseManager(config)
    return _db_manager_instance
