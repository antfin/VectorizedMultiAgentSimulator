"""Encrypt/decrypt LLM API keys for OVH AI Training jobs.

Keys are encrypted locally before submission so they appear as opaque
blobs in `ovhai job get` output. The container decrypts at runtime
using a passphrase passed as a separate env var.

Uses Fernet symmetric encryption (AES-128-CBC + HMAC-SHA256).
The passphrase is derived into a key via PBKDF2.

Usage:
    # At submit time (local machine):
    from src.secrets_util import encrypt_env
    encrypted = encrypt_env(dotenv_values(".env"), passphrase="my-secret")
    submit_training_job(..., llm_env=encrypted)

    # At runtime (inside OVH container, called by train.py):
    from src.secrets_util import decrypt_and_load_env
    decrypt_and_load_env()  # reads LERO_ENCRYPTED + LERO_PASSPHRASE from env
"""

import base64
import hashlib
import json
import os
from typing import Dict, Optional


def _derive_key(passphrase: str, salt: bytes = b"lero-ovh-v1") -> bytes:
    """Derive a 32-byte Fernet key from a passphrase via PBKDF2."""
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100_000,
    )
    raw = kdf.derive(passphrase.encode())
    return base64.urlsafe_b64encode(raw)


def encrypt_env(
    env_dict: Dict[str, str],
    passphrase: str,
) -> Dict[str, str]:
    """Encrypt a dict of env vars into two OVH-safe env vars.

    Returns:
        {"LERO_ENCRYPTED": "<base64 blob>", "LERO_PASSPHRASE": "<passphrase>"}

    The passphrase is still visible in `ovhai job get`, but the actual
    API keys are encrypted. For full security, use a short-lived
    passphrase and rotate after each job.
    """
    from cryptography.fernet import Fernet

    # Filter to only LLM-related keys
    llm_keys = {
        k: v for k, v in env_dict.items()
        if v and any(
            pat in k.upper()
            for pat in ["API_KEY", "ACCESS_TOKEN", "ENDPOINTS"]
        )
    }

    if not llm_keys:
        return {}

    key = _derive_key(passphrase)
    f = Fernet(key)
    payload = json.dumps(llm_keys).encode()
    encrypted = f.encrypt(payload)

    return {
        "LERO_ENCRYPTED": encrypted.decode(),
        "LERO_PASSPHRASE": passphrase,
    }


def decrypt_and_load_env(
    passphrase: Optional[str] = None,
) -> Dict[str, str]:
    """Decrypt LLM keys from env vars and set them in os.environ.

    Called at runtime inside the OVH container. Reads LERO_ENCRYPTED
    and LERO_PASSPHRASE from environment.

    Returns the decrypted key-value pairs (also sets them in os.environ).
    """
    encrypted = os.environ.get("LERO_ENCRYPTED")
    if not encrypted:
        return {}

    if passphrase is None:
        passphrase = os.environ.get("LERO_PASSPHRASE", "")

    if not passphrase:
        raise ValueError(
            "LERO_ENCRYPTED is set but LERO_PASSPHRASE is missing. "
            "Cannot decrypt LLM keys."
        )

    from cryptography.fernet import Fernet, InvalidToken

    key = _derive_key(passphrase)
    f = Fernet(key)

    try:
        decrypted = f.decrypt(encrypted.encode())
    except InvalidToken:
        raise ValueError(
            "Failed to decrypt LERO_ENCRYPTED — wrong passphrase?"
        )

    env_dict = json.loads(decrypted)
    for k, v in env_dict.items():
        os.environ[k] = v

    return env_dict
