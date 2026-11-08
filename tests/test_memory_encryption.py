"""Unit tests for memory encryption at rest and key management."""

from pathlib import Path
import pytest

from config.settings import AppConfig
from memory.encryption import MemoryEncryptionService


def test_encryption_and_decryption(temp_workspace: AppConfig):
    """Verify encryption and decryption of plain text and JSON structures."""
    service = MemoryEncryptionService(temp_workspace)

    secret_text = "Personal confidential note: Patient medical history"
    token = service.encrypt_string(secret_text)
    assert token != secret_text
    assert len(token) > 20

    decrypted = service.decrypt_string(token)
    assert decrypted == secret_text


def test_json_encryption(temp_workspace: AppConfig):
    """Verify serialization, encryption, decryption, and deserialization of dicts."""
    service = MemoryEncryptionService(temp_workspace)
    payload = {
        "title": "Secret Project HAKS",
        "keys": ["api_key_1", "api_key_2"],
        "importance": 5,
    }

    token = service.encrypt_json(payload)
    restored = service.decrypt_json(token)
    assert restored == payload


def test_corrupted_ciphertext_fails(temp_workspace: AppConfig):
    """Verify invalid or tampered ciphertext raises ValueError."""
    service = MemoryEncryptionService(temp_workspace)
    with pytest.raises(ValueError):
        service.decrypt_string("invalid_non_fernet_token")


def test_file_encryption_and_decryption(temp_workspace: AppConfig, tmp_path: Path):
    """Verify encrypting and decrypting full backup files."""
    service = MemoryEncryptionService(temp_workspace)
    orig_file = tmp_path / "raw_backup.json"
    enc_file = tmp_path / "encrypted_backup.enc"
    dec_file = tmp_path / "decrypted_backup.json"

    sample_content = '{"domain": "personal", "count": 42}'
    orig_file.write_text(sample_content, encoding="utf-8")

    service.encrypt_file(orig_file, enc_file)
    assert enc_file.exists()
    assert enc_file.read_bytes() != orig_file.read_bytes()

    service.decrypt_file(enc_file, dec_file)
    assert dec_file.exists()
    assert dec_file.read_text(encoding="utf-8") == sample_content
