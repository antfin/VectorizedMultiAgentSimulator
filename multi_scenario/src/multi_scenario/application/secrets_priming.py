"""F8.4 Phase 2.5 — in-container os.environ priming from encrypted secrets.

The OvhRunner ships encrypted ``MS_ENCRYPTED_SECRETS`` +
``MS_SECRETS_PASSPHRASE`` env vars to the container (F6.1 Fernet
plumbing). LERO's LiteLLM client reads ``OPENAI_API_KEY`` /
``ANTHROPIC_API_KEY`` / ``OVH_API_KEY`` from ``os.environ`` at call
time — but the secrets adapter deliberately does NOT auto-set those:
F6.1 was designed with "caller's responsibility" semantics so a future
non-LLM secret consumer doesn't get the LLM keys for free.

This module is the LERO consumer's hook. It:

1. Reads ``MS_ENCRYPTED_SECRETS`` if present.
2. Decrypts via :class:`FernetSecretsAdapter`.
3. Sets each decrypted key/value into ``os.environ`` (no overwrites —
   if the user explicitly exported the key already, that wins).

Idempotent: calling it twice is a no-op on the second call. No-op-safe
when the env vars aren't set (non-OVH local runs).
"""

import logging
import os

from multi_scenario.adapters.secrets.fernet import (
    ENCRYPTED_ENV_VAR,
    FernetSecretsAdapter,
)


_log = logging.getLogger(__name__)
_PRIMED = False


def prime_os_environ_from_encrypted_secrets() -> None:
    """Inject decrypted secrets from ``MS_ENCRYPTED_SECRETS`` into ``os.environ``.

    Idempotent. No-op when the encrypted-blob env var isn't set.

    Existing values in ``os.environ`` are preserved (never overwritten)
    so a local CLI override can take precedence over the shipped
    encrypted blob.
    """
    global _PRIMED  # pylint: disable=global-statement
    if _PRIMED:
        return
    _PRIMED = True
    if not os.environ.get(ENCRYPTED_ENV_VAR):
        return  # non-OVH path; nothing to decrypt
    try:
        secrets = FernetSecretsAdapter().decrypt_from_env()
    except Exception as exc:  # pylint: disable=broad-except
        _log.warning("failed to decrypt MS_ENCRYPTED_SECRETS: %s", exc)
        return
    for key, value in secrets.items():
        if key in os.environ:
            # Caller already set this — respect it.
            continue
        os.environ[key] = value
    # WARNING level so the line surfaces in container logs without
    # requiring a basicConfig(INFO) at the entry point. Phase 5a's logs
    # never showed this confirmation because Python's default root
    # logger is at WARNING — confirming priming worked was guesswork.
    _log.warning(
        "primed os.environ with %d decrypted secrets (keys: %s)",
        len(secrets),
        sorted(secrets.keys()),
    )
