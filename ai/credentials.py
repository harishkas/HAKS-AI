"""Secure Credential Manager for HAKS AI Cloud Providers.

Uses Windows Credential Manager via keyring for encrypted, secure OS-level storage.
Falls back to environment variables if present.
Ensures API keys are never written to plain-text log files or UI traces.
"""

import os
from typing import Optional
import keyring
from core.logger import get_logger

logger = get_logger("ai.credentials")

SERVICE_NAME = "HAKS_AI"

# Standard provider key mappings to environment variables
PROVIDER_ENV_VARS: dict[str, list[str]] = {
    "gemini": ["HAKS_GEMINI_API_KEY", "GEMINI_API_KEY"],
    "openai": ["HAKS_OPENAI_API_KEY", "OPENAI_API_KEY"],
    "anthropic": ["HAKS_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"],
    "groq": ["HAKS_GROQ_API_KEY", "GROQ_API_KEY"],
    "openrouter": ["HAKS_OPENROUTER_API_KEY", "OPENROUTER_API_KEY"],
    "custom": ["HAKS_CUSTOM_API_KEY"],
}


class CredentialManager:
    """Interface to Windows Credential Manager and secure environment settings."""

    def __init__(self, service_name: str = SERVICE_NAME) -> None:
        self.service_name = service_name

    def set_api_key(self, provider: str, api_key: str) -> None:
        """Store an API key securely in Windows Credential Manager."""
        provider_norm = provider.lower().strip()
        key_stripped = api_key.strip()
        if not key_stripped:
            self.delete_api_key(provider_norm)
            return

        try:
            keyring.set_password(self.service_name, provider_norm, key_stripped)
            logger.info("Securely stored API key for provider: %s in Windows Credential Manager", provider_norm)
        except Exception as exc:
            logger.error("Failed to store key in keyring for %s: %s", provider_norm, exc)
            raise

    def get_api_key(self, provider: str) -> Optional[str]:
        """Retrieve an API key from Windows Credential Manager or environment variables.

        Never logs the retrieved key.
        """
        provider_norm = provider.lower().strip()

        # 1. Check Windows Credential Manager
        try:
            key = keyring.get_password(self.service_name, provider_norm)
            if key and key.strip():
                return key.strip()
        except Exception as exc:
            logger.warning("Keyring lookup for provider '%s' failed: %s", provider_norm, exc)

        # 2. Check environment variables
        env_names = PROVIDER_ENV_VARS.get(provider_norm, [f"HAKS_{provider_norm.upper()}_API_KEY"])
        for env_var in env_names:
            val = os.getenv(env_var)
            if val and val.strip():
                return val.strip()

        return None

    def delete_api_key(self, provider: str) -> bool:
        """Remove an API key from Windows Credential Manager."""
        provider_norm = provider.lower().strip()
        try:
            keyring.delete_password(self.service_name, provider_norm)
            logger.info("Deleted API key for provider: %s", provider_norm)
            return True
        except keyring.errors.PasswordDeleteError:
            # Not found
            return False
        except Exception as exc:
            logger.warning("Could not delete key for %s: %s", provider_norm, exc)
            return False

    def has_api_key(self, provider: str) -> bool:
        """Check whether an API key is available for the given provider."""
        key = self.get_api_key(provider)
        return bool(key and len(key.strip()) > 0)

    @staticmethod
    def mask_key(key: Optional[str]) -> str:
        """Return masked key for safe UI display (e.g., '••••••••abcd')."""
        if not key or not key.strip():
            return "[Not Configured]"
        k = key.strip()
        if len(k) <= 6:
            return "••••••••"
        return "••••••••" + k[-4:]


_credential_manager: Optional[CredentialManager] = None


def get_credential_manager() -> CredentialManager:
    """Return singleton CredentialManager instance."""
    global _credential_manager
    if _credential_manager is None:
        _credential_manager = CredentialManager()
    return _credential_manager
