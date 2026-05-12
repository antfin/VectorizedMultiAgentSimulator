"""F8.4 Phase 2.5 — :func:`prime_os_environ_from_encrypted_secrets` contract."""

# pylint: disable=missing-function-docstring,redefined-outer-name

import importlib
import os

import pytest

from multi_scenario.adapters.secrets.fernet import (
    ENCRYPTED_ENV_VAR,
    FernetSecretsAdapter,
    PASSPHRASE_ENV_VAR,
)


@pytest.fixture(autouse=True)
def _reset_module_state():
    """Reset the once-only ``_PRIMED`` flag between tests."""
    import multi_scenario.application.secrets_priming as mod

    importlib.reload(mod)
    yield


def _stage_encrypted_secrets(monkeypatch, secrets: dict[str, str]) -> None:
    """Encrypt ``secrets`` and place them in os.environ via monkeypatch."""
    env = FernetSecretsAdapter().encrypt_for_env(secrets, passphrase="pp")
    monkeypatch.setenv(ENCRYPTED_ENV_VAR, env[ENCRYPTED_ENV_VAR])
    monkeypatch.setenv(PASSPHRASE_ENV_VAR, env[PASSPHRASE_ENV_VAR])


def test_no_op_when_encrypted_env_var_unset(monkeypatch):
    """Without the encrypted blob, os.environ is not touched."""
    monkeypatch.delenv(ENCRYPTED_ENV_VAR, raising=False)
    monkeypatch.delenv(PASSPHRASE_ENV_VAR, raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    from multi_scenario.application.secrets_priming import (
        prime_os_environ_from_encrypted_secrets,
    )

    prime_os_environ_from_encrypted_secrets()
    assert "OPENAI_API_KEY" not in os.environ


def test_decrypts_and_injects_secrets_when_blob_present(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _stage_encrypted_secrets(monkeypatch, {"OPENAI_API_KEY": "sk-test"})

    from multi_scenario.application.secrets_priming import (
        prime_os_environ_from_encrypted_secrets,
    )

    prime_os_environ_from_encrypted_secrets()
    assert os.environ.get("OPENAI_API_KEY") == "sk-test"


def test_preserves_existing_os_environ_values(monkeypatch):
    """Local CLI override must win over the shipped blob."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-local-override")
    _stage_encrypted_secrets(monkeypatch, {"OPENAI_API_KEY": "sk-shipped"})

    from multi_scenario.application.secrets_priming import (
        prime_os_environ_from_encrypted_secrets,
    )

    prime_os_environ_from_encrypted_secrets()
    assert os.environ.get("OPENAI_API_KEY") == "sk-local-override"


def test_idempotent_when_called_twice(monkeypatch):
    """Second call is a no-op (per-process guard)."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _stage_encrypted_secrets(monkeypatch, {"OPENAI_API_KEY": "sk-test"})

    from multi_scenario.application.secrets_priming import (
        prime_os_environ_from_encrypted_secrets,
    )

    prime_os_environ_from_encrypted_secrets()
    # Now reset the env var; second prime() call shouldn't refill it
    # (the per-process guard short-circuits).
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    prime_os_environ_from_encrypted_secrets()
    assert "OPENAI_API_KEY" not in os.environ


def test_broken_passphrase_logs_warning_but_does_not_crash(monkeypatch, caplog):
    """Defensive: decryption failure shouldn't propagate to the orchestrator."""
    import logging

    _stage_encrypted_secrets(monkeypatch, {"OPENAI_API_KEY": "sk-test"})
    # Corrupt the passphrase.
    monkeypatch.setenv(PASSPHRASE_ENV_VAR, "wrong-passphrase")

    from multi_scenario.application.secrets_priming import (
        prime_os_environ_from_encrypted_secrets,
    )

    caplog.set_level(
        logging.WARNING, logger="multi_scenario.application.secrets_priming"
    )
    prime_os_environ_from_encrypted_secrets()  # no raise
    assert any("decrypt" in rec.getMessage().lower() for rec in caplog.records)
