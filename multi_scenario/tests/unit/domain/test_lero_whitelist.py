"""F9.5 — :class:`AllowedKeysDict` + fairness-violation contract.

Pure-domain unit tests (no torch, no VMAS). The patched scenario is
exercised separately under tests/integration/scenarios/.
"""

# pylint: disable=missing-function-docstring

import pytest

from multi_scenario.domain.lero import (
    AllowedKeysDict,
    FairnessViolation,
    LeroError,
    LOCAL_ALLOWED_KEYS,
    LOCAL_FORBIDDEN_KEYS,
)


def _state() -> dict:
    return {
        "agent_pos": [0.1, 0.2],
        "agent_vel": [0.0, 0.0],
        "agent_idx": 0,
        "n_agents": 4,
        "n_targets": 4,
        "covering_range": 0.35,
        "agents_per_target_required": 2,
        "lidar_targets": [1.0] * 12,
        # forbidden keys — what an LLM might fish for
        "agents_pos": "GLOBAL_DATA",
        "targets_pos": "GLOBAL_DATA",
    }


# ── allowed lookups pass through ─────────────────────────────────────


def test_allowed_key_returns_wrapped_value():
    d = AllowedKeysDict(_state())
    assert d["agent_pos"] == [0.1, 0.2]
    assert d["n_agents"] == 4


def test_get_returns_value_for_allowed_key():
    d = AllowedKeysDict(_state())
    assert d.get("agent_idx") == 0


# ── forbidden lookups raise loudly ───────────────────────────────────


def test_forbidden_key_raises_fairness_violation():
    d = AllowedKeysDict(_state())
    with pytest.raises(FairnessViolation, match="agents_pos"):
        _ = d["agents_pos"]


def test_fairness_violation_subclasses_key_error_and_lero_error():
    """Both base classes matter: LLM ``except KeyError`` keeps working,
    orchestrator ``except LeroError`` catches it cleanly."""
    d = AllowedKeysDict(_state())
    with pytest.raises(KeyError):
        _ = d["targets_pos"]
    with pytest.raises(LeroError):
        _ = d["targets_pos"]


def test_get_does_not_swallow_fairness_violation():
    """``dict.get`` would silently return the default for missing keys,
    masking a fairness violation as a benign None. AllowedKeysDict.get
    re-raises FairnessViolation so the failure surfaces."""
    d = AllowedKeysDict(_state())
    with pytest.raises(FairnessViolation):
        d.get("agents_pos", default=None)


# ── unknown keys: plain KeyError (not FairnessViolation) ─────────────


def test_unknown_key_raises_plain_key_error():
    """LLM typo distinct from cheating: orchestrator's fail-mode
    taxonomy needs to distinguish them."""
    d = AllowedKeysDict(_state())
    with pytest.raises(KeyError) as exc_info:
        _ = d["typo_xyz"]
    assert not isinstance(exc_info.value, FairnessViolation)


def test_get_unknown_key_returns_default():
    d = AllowedKeysDict(_state())
    assert d.get("typo_xyz", "fallback") == "fallback"


# ── allowed-but-missing-from-state ──────────────────────────────────


def test_allowed_key_missing_from_state_raises_plain_key_error():
    """``lidar_agents`` is in the allowed set but missing from this
    state (use_agent_lidar=False) → plain KeyError, not a violation."""
    state = _state()
    del state["lidar_targets"]  # remove an allowed key
    d = AllowedKeysDict(state)
    with pytest.raises(KeyError) as exc_info:
        _ = d["lidar_targets"]
    assert not isinstance(exc_info.value, FairnessViolation)


# ── Mapping protocol ────────────────────────────────────────────────


def test_iter_yields_only_allowed_keys_present_in_state():
    d = AllowedKeysDict(_state())
    keys = set(d)
    # All allowed keys present in state are yielded.
    assert keys <= LOCAL_ALLOWED_KEYS
    assert "agent_pos" in keys
    # No forbidden keys leak through iteration.
    assert keys.isdisjoint(LOCAL_FORBIDDEN_KEYS)


def test_contains_respects_allowlist():
    d = AllowedKeysDict(_state())
    assert "agent_pos" in d
    assert "agents_pos" not in d
    assert "typo_xyz" not in d


def test_len_counts_allowed_keys_present():
    d = AllowedKeysDict(_state())
    # Manually count: agent_pos, agent_vel, agent_idx, n_agents,
    # n_targets, covering_range, agents_per_target_required, lidar_targets = 8
    assert len(d) == 8


# ── Custom whitelist ────────────────────────────────────────────────


def test_custom_allowed_set_overrides_default():
    d = AllowedKeysDict(_state(), allowed={"agent_pos"}, forbidden={"agents_pos"})
    assert d["agent_pos"] == [0.1, 0.2]
    with pytest.raises(FairnessViolation):
        _ = d["agents_pos"]
    # Previously-allowed key now unknown → plain KeyError
    with pytest.raises(KeyError) as exc_info:
        _ = d["agent_vel"]
    assert not isinstance(exc_info.value, FairnessViolation)


# ── Error message quality ───────────────────────────────────────────


def test_fairness_violation_message_includes_label_and_key():
    d = AllowedKeysDict(_state(), label="custom-label")
    with pytest.raises(FairnessViolation) as exc_info:
        _ = d["agents_pos"]
    msg = str(exc_info.value)
    assert "custom-label" in msg
    assert "agents_pos" in msg


# ── Whitelist constants are stable ──────────────────────────────────


def test_whitelist_contents_match_documented_set():
    """Pin the allowed / forbidden sets so adding a key requires a
    visible PR (and matching prompt-template update)."""
    assert "agent_pos" in LOCAL_ALLOWED_KEYS
    assert "lidar_targets" in LOCAL_ALLOWED_KEYS
    assert "agents_pos" in LOCAL_FORBIDDEN_KEYS
    assert "covered_targets" in LOCAL_FORBIDDEN_KEYS
    # Allowed and forbidden sets must not overlap (would be ambiguous).
    assert LOCAL_ALLOWED_KEYS.isdisjoint(LOCAL_FORBIDDEN_KEYS)
