"""Data models for HAKS AI Engine integration."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class EngineType(str, Enum):
    """Supported AI engine types."""
    LOCAL = "local"
    CLOUD = "cloud"


class AIMessage(BaseModel):
    """Structured message exchanged with AI engines."""
    role: str = Field(description="Role of the author: 'user', 'assistant', or 'system'")
    content: str = Field(description="Textual content of the message")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    )
    tokens: Optional[int] = Field(default=0, description="Token count if calculated")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Contextual metadata")

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
        }


class TokenUsage(BaseModel):
    """Token consumption statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class AIStreamChunk(BaseModel):
    """Single incremental chunk from streaming generation."""
    delta_text: str = ""
    finish_reason: Optional[str] = None
    is_done: bool = False
    usage: Optional[TokenUsage] = None


class AIResponse(BaseModel):
    """Complete response returned by an AI engine."""
    content: str
    model: str
    finish_reason: Optional[str] = "stop"
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    raw: dict[str, Any] = Field(default_factory=dict)


class AIEngineConfig(BaseModel):
    """Configuration parameters for AI engine instances."""
    engine_type: EngineType = EngineType.CLOUD
    provider: str = "gemini"
    model: str = "gemini-2.5-flash"
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    timeout_seconds: float = 60.0
    temperature: float = 0.3
    max_tokens: Optional[int] = 128
    num_ctx: int = 2048
    num_predict: int = 128
    cpu_threads: int = 2
    repeat_penalty: float = 1.15
    system_prompt: Optional[str] = (
        "You are HAKS, a helpful, private personal AI operating assistant."
    )



def estimate_tokens(text: str) -> int:
    """Hardware-aware rough token estimation (~3.5 characters per token)."""
    if not text:
        return 0
    # Add small overhead for message frame/formatting
    return max(1, int(len(text) / 3.5 + 0.5))


def estimate_messages_tokens(
    messages: list[AIMessage], system_prompt: Optional[str] = None
) -> int:
    """Estimate total prompt tokens for a list of messages and optional system prompt."""
    total = 0
    if system_prompt:
        total += estimate_tokens(system_prompt) + 4
    for msg in messages:
        total += estimate_tokens(msg.content) + 4  # +4 for role/delimiters
    return total



class AIEngineHealth(BaseModel):
    """Diagnostic health state of an AI engine."""
    is_healthy: bool
    engine_name: str
    status_message: str
    available_models: list[str] = Field(default_factory=list)
    latency_ms: Optional[float] = None
    details: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_available(self) -> bool:
        """Alias for is_healthy."""
        return self.is_healthy

    @property
    def message(self) -> str:
        """Alias for status_message."""
        return self.status_message


class EngineCapabilities(BaseModel):
    """Declared capabilities of an AI engine."""
    streaming: bool = True
    is_local: bool = True
    context_window_tokens: int = 4096
    supports_system_prompt: bool = True
    supports_vision: bool = False
    supports_tools: bool = False
