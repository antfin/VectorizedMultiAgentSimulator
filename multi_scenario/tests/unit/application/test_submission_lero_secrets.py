"""F8.4 Phase 2.5 — :func:`_collect_lero_secret_env` contract."""

# pylint: disable=missing-function-docstring,redefined-outer-name,protected-access

import pytest

from multi_scenario.application.submission import _collect_lero_secret_env
from multi_scenario.domain.models import ExperimentConfig


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    """Stop the helper's ``.env`` auto-walk from picking up the dev's real keys.

    :func:`_collect_lero_secret_env` walks up from cwd looking for a
    ``.env`` (Phase 2.5 convenience). Without isolation, those tests
    leak the live ``multi_scenario/.env`` / ``rendezvous_comm/.env``
    into ``os.environ`` and assertions go wrong. Pinning cwd to a
    fresh tmp_path (which has no ``.env``) keeps the walk a no-op.
    """
    monkeypatch.chdir(tmp_path)


def _cfg(*, with_lero: bool) -> ExperimentConfig:
    base = {
        "experiment": {"id": "x", "seed": 0},
        "scenario": {"type": "discovery", "params": {}},
        "algorithm": {"type": "mappo", "params": {}},
        "training": {"max_iters": 1, "device": "cpu"},
        "evaluation": {"interval_iters": 1, "episodes": 1},
    }
    if with_lero:
        base["lero"] = {"n_iterations": 1, "n_candidates": 1}
        base["llm"] = {"model": "gpt-4o-mini"}
    return ExperimentConfig.model_validate(base)


def test_non_lero_cfg_returns_no_secret_env(monkeypatch):
    """Non-LERO submissions don't need LLM keys; helper returns ``(None, None)``."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    env, passphrase = _collect_lero_secret_env(_cfg(with_lero=False))
    assert env is None
    assert passphrase is None


def test_lero_cfg_with_no_keys_in_env_returns_no_secret_env(monkeypatch):
    """No local keys → no secret_env. The OVH job will 401 at LLM call
    time (clear signal) rather than ship an empty Fernet blob."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OVH_API_KEY", raising=False)
    env, passphrase = _collect_lero_secret_env(_cfg(with_lero=True))
    assert env is None
    assert passphrase is None


def test_lero_cfg_collects_openai_key_when_present(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OVH_API_KEY", raising=False)
    env, passphrase = _collect_lero_secret_env(_cfg(with_lero=True))
    assert env == {"OPENAI_API_KEY": "sk-openai"}
    # 32-byte urlsafe = ≥ 32 chars
    assert passphrase is not None and len(passphrase) >= 32


def test_lero_cfg_collects_all_provider_keys_when_present(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak-claude")
    monkeypatch.setenv("OVH_API_KEY", "ovh-key")
    env, _passphrase = _collect_lero_secret_env(_cfg(with_lero=True))
    assert env == {
        "OPENAI_API_KEY": "sk-openai",
        "ANTHROPIC_API_KEY": "ak-claude",
        "OVH_API_KEY": "ovh-key",
    }


def test_dotenv_in_cwd_is_picked_up_when_env_unset(tmp_path, monkeypatch):
    """``.env`` autoload reaches the helper before it reads ``os.environ``.

    A user who keeps ``OPENAI_API_KEY`` in ``multi_scenario/.env`` (no
    shell ``export``) should still ship the key to OVH. This pins
    that behaviour: a ``.env`` in cwd, no env-var set, must end up in
    the collected dict.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OVH_API_KEY", raising=False)
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-from-dotenv\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    env, passphrase = _collect_lero_secret_env(_cfg(with_lero=True))
    assert env == {"OPENAI_API_KEY": "sk-from-dotenv"}
    assert passphrase is not None


def test_passphrase_is_fresh_per_submission(monkeypatch):
    """Two consecutive submissions get distinct passphrases.

    Reuse would mean a leaked passphrase from one job decrypts the
    next; per-submission rotation closes that window.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    _, p1 = _collect_lero_secret_env(_cfg(with_lero=True))
    _, p2 = _collect_lero_secret_env(_cfg(with_lero=True))
    assert p1 != p2
