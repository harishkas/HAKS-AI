"""HAKS Configuration Management.

Defines strongly-typed system configuration using Pydantic.
Supports environment overrides, persistent defaults, and path resolution.
"""

from functools import lru_cache
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load optional .env file if present
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class SystemPaths(BaseModel):
    """Filesystem paths used by HAKS."""

    root_dir: Path = Field(default_factory=lambda: PROJECT_ROOT)
    data_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data")
    logs_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "logs")
    assets_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "assets")
    backups_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "backups")

    def ensure_directories(self) -> None:
        """Create required directories if they do not exist."""
        for path in (self.data_dir, self.logs_dir, self.assets_dir, self.backups_dir):
            path.mkdir(parents=True, exist_ok=True)


class DatabaseConfig(BaseModel):
    """SQLite Database framework configuration."""

    personal_memory_file: str = "personal_memory.db"
    work_memory_file: str = "work_memory.db"
    conversations_file: str = "conversations.db"
    memory_proposals_file: str = "memory_proposals.db"
    settings_file: str = "settings.db"
    enable_wal: bool = True
    busy_timeout_ms: int = 5000
    enable_foreign_keys: bool = True


class LoggingConfig(BaseModel):
    """Application logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default_factory=lambda: os.getenv("HAKS_LOG_LEVEL", "INFO")
    )
    log_file_name: str = "haks.log"
    max_bytes: int = 5 * 1024 * 1024  # 5 MB
    backup_count: int = 5
    enable_console: bool = True


class UIConfig(BaseModel):
    """PySide6 Desktop and Floating UI configuration."""

    app_name: str = "HAKS"
    app_title: str = "HAKS - Personal AI Operating Assistant"
    version: str = "0.1.0"
    window_width: int = 1100
    window_height: int = 750
    min_window_width: int = 850
    min_window_height: int = 580
    theme: Literal["dark", "light"] = "dark"
    enable_high_dpi: bool = True
    start_minimized_to_tray: bool = False


class AIConfig(BaseModel):
    """Local & Cloud AI runtime configuration."""

    default_engine: Literal["local", "cloud"] = "local"
    ollama_base_url: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    )
    local_model: str = Field(
        default_factory=lambda: os.getenv("HAKS_LOCAL_MODEL", "llama3.2:1b")
    )
    cloud_provider: str = Field(
        default_factory=lambda: os.getenv("HAKS_CLOUD_PROVIDER", "gemini")
    )
    request_timeout_seconds: float = 60.0


class VoiceConfig(BaseModel):
    """Voice pipeline configuration."""

    wake_phrase: str = "Hey HAKS"
    supported_languages: list[str] = Field(
        default_factory=lambda: ["ta", "en"]
    )
    preferred_language: Literal["ta", "en", "mixed"] = "mixed"
    continuous_mode: bool = True
    input_sample_rate: int = 16000
    vad_threshold: float = 0.5


class SecurityConfig(BaseModel):
    """Security and Windows OS permission control configuration."""

    permission_enforcement: bool = True
    audit_logging: bool = True
    require_approval_for_level_2: bool = True
    require_approval_for_level_3: bool = True
    approval_timeout_seconds: int = 60


class AppConfig(BaseModel):
    """Root Application Configuration."""

    paths: SystemPaths = Field(default_factory=SystemPaths)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)


@lru_cache(maxsize=1)
def get_settings() -> AppConfig:
    """Return cached singleton instance of AppConfig."""
    config = AppConfig()
    config.paths.ensure_directories()
    return config
