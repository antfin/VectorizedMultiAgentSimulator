"""F6.1 tests: FernetSecretsAdapter — encrypt / decrypt / env round-trip."""

import os

import pytest

from multi_scenario.adapters.secrets.fernet import (
    ENCRYPTED_ENV_VAR,
    PASSPHRASE_ENV_VAR,
    FernetSecretsAdapter,
)


def test_encrypt_decrypt_round_trip() -> None:
    """Encrypted blob + correct passphrase round-trip to the original dict."""
    adapter = FernetSecretsAdapter()
    secrets = {"OPENAI_API_KEY": "sk-test", "ANTHROPIC_API_KEY": "ant-key-xyz"}
    blob = adapter.encrypt(secrets, passphrase="hunter2")
    assert isinstance(blob, str) and blob  # non-empty
    out = adapter.decrypt(blob, passphrase="hunter2")
    assert out == secrets


def test_decrypt_wrong_passphrase_raises() -> None:
    """Decrypting with the wrong passphrase raises a clear error."""
    adapter = FernetSecretsAdapter()
    blob = adapter.encrypt({"K": "v"}, passphrase="right")
    with pytest.raises(ValueError, match="passphrase"):
        adapter.decrypt(blob, passphrase="wrong")


def test_encrypt_empty_dict_round_trips() -> None:
    """Empty input encrypts + decrypts to an empty dict (graceful no-op shape)."""
    adapter = FernetSecretsAdapter()
    blob = adapter.encrypt({}, passphrase="x")
    assert adapter.decrypt(blob, passphrase="x") == {}


def test_encrypt_for_env_returns_two_known_var_names() -> None:
    """Wrapping for env shipment exposes the documented var names."""
    adapter = FernetSecretsAdapter()
    env = adapter.encrypt_for_env({"K": "v"}, passphrase="x")
    assert set(env.keys()) == {ENCRYPTED_ENV_VAR, PASSPHRASE_ENV_VAR}
    assert env[PASSPHRASE_ENV_VAR] == "x"
    assert env[ENCRYPTED_ENV_VAR]  # non-empty blob


def test_decrypt_from_env_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read the two env vars + decrypt → original dict; does NOT mutate os.environ."""
    adapter = FernetSecretsAdapter()
    secrets = {"OPENAI_API_KEY": "sk-abc"}
    env = adapter.encrypt_for_env(secrets, passphrase="pp")
    monkeypatch.setenv(ENCRYPTED_ENV_VAR, env[ENCRYPTED_ENV_VAR])
    monkeypatch.setenv(PASSPHRASE_ENV_VAR, env[PASSPHRASE_ENV_VAR])
    out = adapter.decrypt_from_env()
    assert out == secrets
    # Decrypted secret was NOT injected into os.environ — caller's responsibility.
    assert "OPENAI_API_KEY" not in os.environ


def test_decrypt_from_env_returns_empty_when_blob_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing ``MS_ENCRYPTED_SECRETS`` → returns empty dict (no-op when nothing to ship)."""
    monkeypatch.delenv(ENCRYPTED_ENV_VAR, raising=False)
    monkeypatch.delenv(PASSPHRASE_ENV_VAR, raising=False)
    assert FernetSecretsAdapter().decrypt_from_env() == {}


def test_decrypt_from_env_raises_when_passphrase_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blob set but passphrase missing → clear error (misconfigured shipping)."""
    adapter = FernetSecretsAdapter()
    blob = adapter.encrypt({"K": "v"}, passphrase="pp")
    monkeypatch.setenv(ENCRYPTED_ENV_VAR, blob)
    monkeypatch.delenv(PASSPHRASE_ENV_VAR, raising=False)
    with pytest.raises(ValueError, match="passphrase"):
        adapter.decrypt_from_env()
