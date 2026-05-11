"""F9.8 — Submit-page LERO preflight check contract."""

# pylint: disable=missing-function-docstring,redefined-outer-name

import pytest

from multi_scenario.domain.models import ExperimentConfig
from multi_scenario.frontend.lero_preflight import (
    check_lero_api_key,
    expected_api_key_var,
)


def _cfg(*, lero: bool, llm: dict | None = None) -> ExperimentConfig:
    payload: dict = {
        "experiment": {"id": "x", "seed": 0},
        "scenario": {"type": "discovery", "params": {}},
        "algorithm": {"type": "mappo", "params": {}},
        "training": {"max_iters": 1, "device": "cpu"},
        "evaluation": {"interval_iters": 1, "episodes": 1},
    }
    if lero:
        payload["lero"] = {"n_iterations": 1, "n_candidates": 1}
        payload["llm"] = llm or {"model": "gpt-4o-mini"}
    elif llm is not None:
        payload["llm"] = llm
    return ExperimentConfig.model_validate(payload)


# ── expected_api_key_var ────────────────────────────────────────────


@pytest.mark.parametrize(
    "model,expected",
    [
        ("gpt-4o-mini", "OPENAI_API_KEY"),
        ("gpt-5", "OPENAI_API_KEY"),
        ("openai/my-custom", "OPENAI_API_KEY"),
        ("claude-sonnet-4-6", "ANTHROPIC_API_KEY"),
        ("anthropic/claude-3-5-sonnet", "ANTHROPIC_API_KEY"),
        ("ovh/mistral", "OVH_API_KEY"),
        ("unknown-model-xyz", "OPENAI_API_KEY"),  # fallback
    ],
)
def test_expected_var_dispatches_by_prefix(model: str, expected: str):
    assert expected_api_key_var(model) == expected


# ── check_lero_api_key ─────────────────────────────────────────────


def test_check_passes_when_cfg_lero_is_none(monkeypatch):
    """Non-LERO YAMLs don't need an LLM key — preflight is a no-op."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = check_lero_api_key(_cfg(lero=False))
    assert result.ok
    assert "not a lero run" in result.detail.lower()


def test_check_passes_when_required_key_present(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    result = check_lero_api_key(_cfg(lero=True))
    assert result.ok
    assert result.required_env_var == "OPENAI_API_KEY"


def test_check_fails_when_required_key_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = check_lero_api_key(_cfg(lero=True))
    assert not result.ok
    assert result.required_env_var == "OPENAI_API_KEY"
    assert "gpt-4o-mini" in result.detail


def test_check_dispatches_to_anthropic_for_claude_model(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = _cfg(lero=True, llm={"model": "claude-sonnet-4-6"})
    result = check_lero_api_key(cfg)
    assert not result.ok
    assert result.required_env_var == "ANTHROPIC_API_KEY"


def test_check_detail_mentions_dot_env_recovery_path(monkeypatch):
    """Error message points the user at the standard fix."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = check_lero_api_key(_cfg(lero=True))
    assert ".env" in result.detail
