"""Unit tests for CredentialManager and Windows Credential Manager integration."""

import os
import pytest
from ai.credentials import CredentialManager


def test_mask_key():
    """Verify that mask_key never exposes full API keys."""
    assert CredentialManager.mask_key(None) == "[Not Configured]"
    assert CredentialManager.mask_key("") == "[Not Configured]"
    assert CredentialManager.mask_key("12345") == "••••••••"
    assert CredentialManager.mask_key("AIzaSyB1234567890abcdef") == "••••••••cdef"


def test_credential_manager_env_fallback(monkeypatch):
    """Verify CredentialManager checks environment variables if not in keyring."""
    cm = CredentialManager(service_name="HAKS_TEST_SERVICE")

    monkeypatch.setenv("HAKS_GEMINI_API_KEY", "test_gemini_key_12345")
    assert cm.get_api_key("gemini") == "test_gemini_key_12345"
    assert cm.has_api_key("gemini") is True

    monkeypatch.delenv("HAKS_GEMINI_API_KEY", raising=False)
    # When missing
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    # May return None if not in keyring
    assert cm.mask_key(cm.get_api_key("gemini")) in ("[Not Configured]", "••••••••12345")


def test_credential_lifecycle():
    """Verify save, retrieve, and delete lifecycle through CredentialManager."""
    cm = CredentialManager(service_name="HAKS_TEST_SERVICE")
    test_provider = "test_provider_haks"
    secret_key = "sk-test-secret-key-9999"

    try:
        cm.set_api_key(test_provider, secret_key)
        retrieved = cm.get_api_key(test_provider)
        assert retrieved == secret_key
        assert cm.has_api_key(test_provider) is True

        # Masking
        masked = cm.mask_key(retrieved)
        assert "secret" not in masked
        assert masked.endswith("9999")

        # Deletion
        deleted = cm.delete_api_key(test_provider)
        assert deleted is True
        assert cm.get_api_key(test_provider) is None
    finally:
        cm.delete_api_key(test_provider)
