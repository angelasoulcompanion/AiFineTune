"""
Cryptography utilities for secure token storage
Uses Fernet symmetric encryption (AES-128-CBC with HMAC)
"""

import base64
import os
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from ..config import settings


def _get_fernet() -> Fernet:
    """Get Fernet instance with encryption key"""
    encryption_key = settings.ENCRYPTION_KEY

    if not encryption_key:
        # Generate a key from JWT secret if no encryption key is set
        # This is acceptable for development but should use dedicated key in production
        encryption_key = settings.JWT_SECRET_KEY

    # Derive a proper Fernet key from the secret
    # Use PBKDF2 to derive a 32-byte key
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b"aifinetune_salt_v1",  # Fixed salt for deterministic key derivation
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(encryption_key.encode()))
    return Fernet(key)


def encrypt_token(token: str) -> str:
    """
    Encrypt a token for secure storage

    Args:
        token: The plaintext token to encrypt

    Returns:
        Base64-encoded encrypted token
    """
    if not token:
        return ""

    fernet = _get_fernet()
    encrypted = fernet.encrypt(token.encode())
    return encrypted.decode()


def decrypt_token(encrypted_token: str) -> Optional[str]:
    """
    Decrypt an encrypted token

    Args:
        encrypted_token: Base64-encoded encrypted token

    Returns:
        Decrypted plaintext token, or None if decryption fails
    """
    if not encrypted_token:
        return None

    try:
        fernet = _get_fernet()
        decrypted = fernet.decrypt(encrypted_token.encode())
        return decrypted.decode()
    except InvalidToken:
        # Token is invalid or was encrypted with a different key
        return None
    except Exception:
        return None


def is_encrypted(value: str) -> bool:
    """
    Check if a value appears to be Fernet-encrypted

    Fernet tokens start with 'gAAAAA' (base64-encoded version indicator)
    """
    if not value:
        return False
    return value.startswith("gAAAAA")
