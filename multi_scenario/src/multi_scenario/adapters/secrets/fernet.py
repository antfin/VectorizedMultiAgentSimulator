"""FernetSecretsAdapter — symmetric encryption for secrets shipped to remote runners.

Fernet (AES-128-CBC + HMAC-SHA256). PBKDF2-HMAC-SHA256 derives the 32-byte key
from a passphrase + fixed salt. The encrypted blob plus passphrase ride along
as two env vars on the remote job (``MS_ENCRYPTED_SECRETS`` / ``MS_SECRETS_PASSPHRASE``);
the runtime side reads them back and decrypts.

**Threat model (be honest about what this protects):** this is *not* protection
against a malicious cloud provider — they can read both the blob and the
passphrase from the job spec / logs. It's protection against bystanders
glancing at job specs, S3 buckets, or `ovhai job get` output. Real security
comes from **rotating the passphrase per job** so a leaked passphrase only
compromises that one job's secrets.

For stronger threat models, plug in a different adapter that talks to a real
KMS — this module's interface is small enough to swap.
"""

import base64
import json
import os

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Fixed salt — defends against generic rainbow tables; passphrase rotation is
# the real defence. Bumping the version here invalidates previously-encrypted
# blobs (intentional — forces re-encryption with the new salt).
_SALT = b"multi_scenario-v1"
_PBKDF2_ITERATIONS = 100_000

# Public env-var names used by the encrypt-for-env / decrypt-from-env helpers.
ENCRYPTED_ENV_VAR = "MS_ENCRYPTED_SECRETS"
PASSPHRASE_ENV_VAR = "MS_SECRETS_PASSPHRASE"


class FernetSecretsAdapter:
    """Encrypt/decrypt arbitrary {str: str} secret dicts using Fernet."""

    def encrypt(self, secrets: dict[str, str], passphrase: str) -> str:
        """Encrypt a secrets dict; returns a UTF-8 string blob (Fernet token)."""
        key = _derive_key(passphrase)
        token = Fernet(key).encrypt(json.dumps(secrets).encode("utf-8"))
        return token.decode("utf-8")

    def decrypt(self, blob: str, passphrase: str) -> dict[str, str]:
        """Decrypt a Fernet token back into the original secrets dict."""
        key = _derive_key(passphrase)
        try:
            payload = Fernet(key).decrypt(blob.encode("utf-8"))
        except InvalidToken as exc:
            raise ValueError(
                "secrets decrypt failed — wrong passphrase or corrupted blob?"
            ) from exc
        return json.loads(payload)

    def encrypt_for_env(
        self, secrets: dict[str, str], passphrase: str
    ) -> dict[str, str]:
        """Wrap an encrypted blob into the two env-var pair shipped to OVH jobs."""
        return {
            ENCRYPTED_ENV_VAR: self.encrypt(secrets, passphrase),
            PASSPHRASE_ENV_VAR: passphrase,
        }

    def decrypt_from_env(self) -> dict[str, str]:
        """Decrypt secrets staged in ``os.environ`` via :meth:`encrypt_for_env`.

        Reads ``MS_ENCRYPTED_SECRETS`` + ``MS_SECRETS_PASSPHRASE`` from env.
        Returns ``{}`` when the encrypted blob is unset (no-op on jobs that
        don't ship secrets). Raises ``ValueError`` if the blob is set but
        the passphrase is missing — that's a misconfigured shipping path.

        **Does NOT mutate ``os.environ``.** Caller decides how to inject the
        decrypted secrets (e.g. into a particular subprocess env or a config).
        """
        blob = os.environ.get(ENCRYPTED_ENV_VAR)
        if not blob:
            return {}
        passphrase = os.environ.get(PASSPHRASE_ENV_VAR)
        if not passphrase:
            raise ValueError(
                f"{ENCRYPTED_ENV_VAR} is set but {PASSPHRASE_ENV_VAR} is missing — "
                "passphrase required to decrypt."
            )
        return self.decrypt(blob, passphrase)


def _derive_key(passphrase: str) -> bytes:
    """Derive a 32-byte Fernet key from a passphrase via PBKDF2-HMAC-SHA256."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=_SALT,
        iterations=_PBKDF2_ITERATIONS,
    )
    return base64.urlsafe_b64encode(kdf.derive(passphrase.encode("utf-8")))
