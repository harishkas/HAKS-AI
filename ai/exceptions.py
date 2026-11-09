"""Structured exceptions for the HAKS AI Engine layer."""


class AIEngineError(Exception):
    """Base exception for all AI engine operations."""
    def __init__(self, message: str, engine: str = "", details: dict = None):
        super().__init__(message)
        self.message = message
        self.engine = engine
        self.details = details or {}


class AIEngineConnectionError(AIEngineError):
    """Raised when engine endpoint cannot be reached or connection is refused."""


class AIEngineTimeoutError(AIEngineError):
    """Raised when request times out."""


class ModelNotFoundError(AIEngineError):
    """Raised when requested model is not installed or not available."""


class AuthenticationError(AIEngineError):
    """Raised when API credentials are missing, invalid, or expired."""


class RateLimitError(AIEngineError):
    """Raised when provider returns HTTP 429 Too Many Requests."""


class ProviderNotConfiguredError(AIEngineError):
    """Raised when a cloud provider is selected but credentials/settings are absent."""


class GenerationCancelledError(AIEngineError):
    """Raised when inference is aborted by the user."""


class TokenLimitError(AIEngineError):
    """Raised when the local model reaches its token limit or token repeat limit during prediction."""


class ContextOverflowError(AIEngineError):
    """Raised when the prompt context exceeds the model's context window limit (num_ctx)."""


class EmptyResponseError(AIEngineError):
    """Raised when the AI engine returns an empty completion or stream."""

