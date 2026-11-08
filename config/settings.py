"""HAKS Configuration Management.

Defines strongly-typed system configuration using Pydantic.
Supports environment overrides, persistent defaults, and path resolution.
"""

from functools import lru_cache
import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load optional .env file if present
load_dotenv()

# Project root directory (contains source code and bundled assets)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Persistent Windows storage root outside source code and .venv
def get_default_persistent_root() -> Path:
    """Resolve persistent OS-level root directory for HAKS."""
    override = os.getenv("HAKS_ROOT_DIR")
    if override:
        return Path(override)
    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "HAKS"
    return Path.home() / "AppData" / "Local" / "HAKS"


def get_default_data_dir() -> Path:
    """Resolve persistent data directory (%LOCALAPPDATA%\\HAKS\\data)."""
    override = os.getenv("HAKS_DATA_DIR")
    if override:
        return Path(override)
    return get_default_persistent_root() / "data"


def get_default_logs_dir() -> Path:
    """Resolve persistent logs directory (%LOCALAPPDATA%\\HAKS\\logs)."""
    override = os.getenv("HAKS_LOGS_DIR")
    if override:
        return Path(override)
    return get_default_persistent_root() / "logs"


class SystemPaths(BaseModel):
    """Filesystem paths used by HAKS with persistent OS-level defaults."""

    root_dir: Path = Field(default_factory=get_default_persistent_root)
    data_dir: Path = Field(default_factory=get_default_data_dir)
    logs_dir: Path = Field(default_factory=get_default_logs_dir)
    assets_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "assets")
    backups_dir: Path = Field(default_factory=lambda: get_default_data_dir() / "backups")

    def ensure_directories(self) -> None:
        """Create required directories and 9 independent storage domains if they do not exist."""
        # Core base directories
        for path in (self.data_dir, self.logs_dir, self.assets_dir, self.backups_dir):
            path.mkdir(parents=True, exist_ok=True)

        # 9 Independent Storage Domains
        domains = [
            self.data_dir / "personal_memory",
            self.data_dir / "personal_memory" / "vector_index",
            self.data_dir / "work_memory",
            self.data_dir / "work_memory" / "vector_index",
            self.data_dir / "conversations",
            self.data_dir / "memory_proposals",
            self.data_dir / "settings",
            self.data_dir / "backups",
            self.logs_dir,
        ]
        for d in domains:
            d.mkdir(parents=True, exist_ok=True)


class DatabaseConfig(BaseModel):
    """SQLite Database framework configuration with domain-isolated database files."""

    personal_memory_file: str = "personal_memory/personal_memory.db"
    work_memory_file: str = "work_memory/work_memory.db"
    conversations_file: str = "conversations/conversations.db"
    memory_proposals_file: str = "memory_proposals/memory_proposals.db"
    settings_file: str = "settings/settings.db"
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
    wake_word_engine: Literal["local", "porcupine"] = "local"
    porcupine_access_key: str = Field(default_factory=lambda: os.getenv("HAKS_PORCUPINE_ACCESS_KEY", ""))
    porcupine_keyword_path: str = Field(default_factory=lambda: os.getenv("HAKS_PORCUPINE_KEYWORD_PATH", ""))
    wake_sensitivity: float = 0.65
    supported_languages: list[str] = Field(
        default_factory=lambda: ["ta", "en"]
    )
    preferred_language: Literal["ta", "en", "mixed"] = "mixed"
    continuous_mode: bool = True
    input_sample_rate: int = 16000
    input_device_index: Optional[int] = None
    output_device_index: Optional[int] = None
    vad_threshold: float = 0.5
    silence_timeout_ms: int = 1200
    conversation_timeout_seconds: int = 45
    whisper_model_size: Literal["tiny", "base", "small"] = "base"
    tts_engine: Literal["edge", "sapi", "piper", "auto"] = "auto"
    tts_tamil_voice: str = "ta-IN-PallaviNeural"
    tts_english_voice: str = "en-IN-NeerjaNeural"
    speech_speed: float = 1.0
    speech_volume: float = 1.0
    enable_barge_in: bool = True


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


AppConfig.model_rebuild()


@lru_cache(maxsize=1)
def get_settings() -> AppConfig:
    """Return cached singleton instance of AppConfig."""
    config = AppConfig()
    config.paths.ensure_directories()
    return config
