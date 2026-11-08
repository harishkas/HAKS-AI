"""Cryptographic service providing encryption at rest for HAKS memory storage and backups.

Uses authenticated AES-256 symmetric encryption (Fernet) with PBKDF2HMAC
key derivation and key persistence in Windows Credential Manager.
"""

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Optional
import keyring
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from config.settings import AppConfig, get_settings
from core.logger import get_logger

logger = get_logger("memory.encryption")

SERVICE_NAME = "HAKS_MEMORY_KEYSTORE"
KEY_ACCOUNT = "master_memory_key"
SALT_ACCOUNT = "master_memory_salt"


class MemoryEncryptionService:
    """Manages encryption and decryption of memory data, backups, and exports."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or get_settings()
        self._fernet: Optional[Fernet] = None
        self._key_b64: Optional[str] = None
        self._salt_b64: Optional[str] = None
        self._init_cipher()

    def _init_cipher(self) -> None:
        """Derive or load the master encryption key from Windows Credential Manager."""
        key_b64 = keyring.get_password(SERVICE_NAME, KEY_ACCOUNT)
        salt_b64 = keyring.get_password(SERVICE_NAME, SALT_ACCOUNT)

        # Fallback file paths for test/headless environments
        keys_dir = self.config.paths.data_dir / ".keys"
        keys_dir.mkdir(parents=True, exist_ok=True)
        key_file = keys_dir / "mem.key"
        salt_file = keys_dir / "mem.salt"
        hash_file = keys_dir / "key_hash.sha256"

        if not key_b64:
            if key_file.exists():
                key_b64 = key_file.read_text(encoding="utf-8").strip()
            if salt_file.exists():
                salt_b64 = salt_file.read_text(encoding="utf-8").strip()

        if not key_b64 or not salt_b64:
            # Generate new 16-byte cryptographically secure salt
            salt = os.urandom(16)
            salt_b64 = base64.b64encode(salt).decode("utf-8")

            # Generate random 32-byte secret
            passphrase = os.urandom(32)

            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100_000,
            )
            derived_key = base64.urlsafe_b64encode(kdf.derive(passphrase))
            key_b64 = derived_key.decode("utf-8")

            # Store in Windows Credential Manager
            try:
                keyring.set_password(SERVICE_NAME, KEY_ACCOUNT, key_b64)
                keyring.set_password(SERVICE_NAME, SALT_ACCOUNT, salt_b64)
                logger.info("Created new master memory encryption key in Windows Credential Manager.")
            except Exception as exc:
                logger.warning("Could not persist encryption key in keyring (%s); using secure file fallback.", exc)

            # File fallback
            key_file.write_text(key_b64, encoding="utf-8")
            salt_file.write_text(salt_b64, encoding="utf-8")

        self._key_b64 = key_b64
        self._salt_b64 = salt_b64
        self._fernet = Fernet(key_b64.encode("utf-8"))

        # Save verification hash if not present
        if not hash_file.exists():
            key_hash = hashlib.sha256(key_b64.encode("utf-8")).hexdigest()
            hash_file.write_text(key_hash, encoding="utf-8")

        logger.info("MemoryEncryptionService initialized successfully.")

    def is_key_healthy(self) -> bool:
        """Verify that encryption cipher is operational."""
        try:
            test_token = self.encrypt_string("healthcheck")
            return self.decrypt_string(test_token) == "healthcheck"
        except Exception:
            return False

    def export_recovery_key(self) -> str:
        """Export user-facing recovery phrase / key for disaster restoration."""
        if not self._key_b64:
            raise RuntimeError("Encryption key is not initialized.")
        return f"HAKS-REC-{self._key_b64}"

    def recover_with_recovery_key(self, recovery_key: str) -> bool:
        """Restore encryption cipher using an exported recovery key."""
        clean_key = recovery_key.strip()
        if clean_key.startswith("HAKS-REC-"):
            clean_key = clean_key[len("HAKS-REC-"):]

        try:
            # Validate Fernet key format
            new_fernet = Fernet(clean_key.encode("utf-8"))
            test_enc = new_fernet.encrypt(b"recovery_test")
            if new_fernet.decrypt(test_enc) != b"recovery_test":
                return False
        except Exception as exc:
            logger.error("Invalid recovery key format: %s", exc)
            return False

        # Verify against hash if hash file exists
        keys_dir = self.config.paths.data_dir / ".keys"
        hash_file = keys_dir / "key_hash.sha256"
        key_hash = hashlib.sha256(clean_key.encode("utf-8")).hexdigest()

        if hash_file.exists():
            expected_hash = hash_file.read_text(encoding="utf-8").strip()
            if key_hash != expected_hash:
                logger.warning("Recovery key does not match previously registered key hash.")
                # We still allow recovery if explicit, but warn

        self._key_b64 = clean_key
        self._fernet = new_fernet

        # Persist restored key in Windows Credential Manager and fallback file
        try:
            keyring.set_password(SERVICE_NAME, KEY_ACCOUNT, clean_key)
        except Exception as exc:
            logger.warning("Could not persist recovered key to keyring: %s", exc)

        keys_dir.mkdir(parents=True, exist_ok=True)
        key_file = keys_dir / "mem.key"
        key_file.write_text(clean_key, encoding="utf-8")
        hash_file.write_text(key_hash, encoding="utf-8")

        logger.info("Master memory encryption key successfully recovered and restored.")
        return True

    def encrypt_field(self, plaintext: str) -> str:
        """Transparently encrypt a sensitive field, adding 'enc:' prefix."""
        if not plaintext:
            return ""
        if plaintext.startswith("enc:"):
            return plaintext  # already encrypted
        token = self.encrypt_string(plaintext)
        return f"enc:{token}"

    def decrypt_field(self, ciphertext: str) -> str:
        """Transparently decrypt a sensitive field. Falls back cleanly to plaintext."""
        if not ciphertext:
            return ""
        if ciphertext.startswith("enc:"):
            token = ciphertext[4:]
            try:
                return self.decrypt_string(token)
            except Exception as exc:
                logger.error("Failed to decrypt field token: %s", exc)
                return ciphertext
        if ciphertext.startswith("gAAAA"):
            # Raw Fernet token
            try:
                return self.decrypt_string(ciphertext)
            except Exception:
                return ciphertext
        # Unencrypted legacy plaintext
        return ciphertext

    def encrypt_string(self, plaintext: str) -> str:
        """Encrypt a UTF-8 string, returning a URL-safe Base64 token."""
        assert self._fernet is not None
        if not plaintext:
            return ""
        return self._fernet.encrypt(plaintext.encode("utf-8")).decode("utf-8")

    def decrypt_string(self, ciphertext: str) -> str:
        """Decrypt a URL-safe Base64 token back to a UTF-8 string."""
        assert self._fernet is not None
        if not ciphertext:
            return ""
        try:
            return self._fernet.decrypt(ciphertext.encode("utf-8")).decode("utf-8")
        except Exception as exc:
            logger.error("Decryption failed: %s", exc)
            raise ValueError(f"Corrupted or invalid ciphertext: {exc}")

    def encrypt_bytes(self, data: bytes) -> bytes:
        """Encrypt raw bytes."""
        assert self._fernet is not None
        return self._fernet.encrypt(data)

    def decrypt_bytes(self, token: bytes) -> bytes:
        """Decrypt raw token."""
        assert self._fernet is not None
        return self._fernet.decrypt(token)

    def encrypt_json(self, data: dict) -> str:
        """Serialize a dict to JSON and encrypt it."""
        serialized = json.dumps(data, ensure_ascii=False)
        return self.encrypt_string(serialized)

    def decrypt_json(self, token: str) -> dict:
        """Decrypt a token and parse the JSON dict."""
        decrypted_str = self.decrypt_string(token)
        return json.loads(decrypted_str)

    def encrypt_file(self, input_path: Path, output_path: Path) -> None:
        """Encrypt an entire file (e.g. for backup bundles)."""
        data = input_path.read_bytes()
        encrypted = self.encrypt_bytes(data)
        output_path.write_bytes(encrypted)
        logger.info("Encrypted file %s -> %s (%d bytes)", input_path.name, output_path.name, len(encrypted))

    def decrypt_file(self, input_path: Path, output_path: Path) -> None:
        """Decrypt an encrypted file."""
        encrypted = input_path.read_bytes()
        decrypted = self.decrypt_bytes(encrypted)
        output_path.write_bytes(decrypted)
        logger.info("Decrypted file %s -> %s (%d bytes)", input_path.name, output_path.name, len(decrypted))


_encryption_service: Optional[MemoryEncryptionService] = None


def get_encryption_service(config: Optional[AppConfig] = None) -> MemoryEncryptionService:
    """Return singleton MemoryEncryptionService instance."""
    global _encryption_service
    if _encryption_service is None:
        _encryption_service = MemoryEncryptionService(config)
    return _encryption_service
