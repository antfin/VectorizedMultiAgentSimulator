"""F9.5 — :mod:`adapters.scenarios._lero_patch_helpers` unit-level tests.

Pure-helper tests — no VMAS world. The full patched scenario lifecycle
is exercised in ``test_patched_discovery.py`` (slower, opt-in via
``@pytest.mark.slow``).
"""

# pylint: disable=missing-function-docstring,protected-access

import math

import pytest
import torch

from multi_scenario.adapters.scenarios._lero_patch_helpers import (
    compile_llm_function,
    EXEC_NAMESPACE,
    maybe_wrap_obs_state,
    sanitize_reward,
)
from multi_scenario.domain.lero import AllowedKeysDict, FairnessViolation


# ── sanitize_reward ─────────────────────────────────────────────────


def test_sanitize_replaces_nan_with_zero():
    r = torch.tensor([1.0, float("nan"), 2.0])
    out = sanitize_reward(r, clip=None)
    assert out.tolist() == [1.0, 0.0, 2.0]


def test_sanitize_replaces_inf_with_zero():
    r = torch.tensor([1.0, float("inf"), float("-inf"), 2.0])
    out = sanitize_reward(r, clip=None)
    assert out.tolist() == [1.0, 0.0, 0.0, 2.0]


def test_sanitize_clamps_to_pm_clip():
    r = torch.tensor([-100.0, -10.0, 0.0, 10.0, 100.0])
    out = sanitize_reward(r, clip=50.0)
    assert out.tolist() == [-50.0, -10.0, 0.0, 10.0, 50.0]


def test_sanitize_clip_none_skips_clamp():
    r = torch.tensor([1000.0, -1000.0])
    out = sanitize_reward(r, clip=None)
    assert out.tolist() == [1000.0, -1000.0]


def test_sanitize_clip_zero_skips_clamp():
    """``clip=0`` is treated as disabled (avoid degenerate "always zero")."""
    r = torch.tensor([5.0, -5.0])
    out = sanitize_reward(r, clip=0.0)
    assert out.tolist() == [5.0, -5.0]


def test_sanitize_chains_nan_then_clip():
    """NaN replacement happens BEFORE clamp — so a NaN doesn't survive
    as a 0 then get clamped, and an inf doesn't blow up the clamp."""
    r = torch.tensor([float("nan"), float("inf"), 100.0])
    out = sanitize_reward(r, clip=50.0)
    assert out.tolist() == [0.0, 0.0, 50.0]


# ── compile_llm_function ────────────────────────────────────────────


def test_compile_pulls_named_function_from_exec_namespace():
    src = "def compute_reward(s):\n    return s['x'] * 2"
    fn = compile_llm_function(src, "compute_reward")
    assert fn({"x": torch.tensor(3.0)}).item() == 6.0


def test_compile_function_has_access_to_torch_and_math():
    """LLM code is exec'd with ``EXEC_NAMESPACE`` so ``torch`` / ``math``
    / ``F`` are resolvable without an import statement (which the AST
    validator forbids when not in ALLOWED_IMPORTS)."""
    src = "def compute_reward(s):\n    return torch.tensor(math.pi)"
    fn = compile_llm_function(src, "compute_reward")
    assert fn({}).item() == pytest.approx(math.pi)


def test_compile_function_raises_when_name_missing():
    src = "def something_else(s):\n    return s"
    with pytest.raises(ValueError, match="not found"):
        compile_llm_function(src, "compute_reward")


def test_compile_namespace_contains_expected_keys():
    """``EXEC_NAMESPACE`` is the trust boundary — pin its contents so
    adding a new name (which would give LLM code a new capability)
    requires a visible PR."""
    assert set(EXEC_NAMESPACE.keys()) == {"torch", "math", "F"}


# ── maybe_wrap_obs_state dispatch ───────────────────────────────────


def test_global_mode_returns_state_unchanged():
    state = {"x": 1, "y": 2}
    out = maybe_wrap_obs_state(state, mode="global", whitelist_strict=True)
    assert out is state


def test_local_mode_strict_false_returns_state_unchanged():
    """Paper-faithful default (non-strict) is permissive — LLM sees raw dict."""
    state = {"x": 1, "y": 2}
    out = maybe_wrap_obs_state(state, mode="local", whitelist_strict=False)
    assert out is state


def test_local_mode_strict_true_wraps_in_allowed_keys_dict():
    state = {"agent_pos": torch.tensor([0.1])}
    out = maybe_wrap_obs_state(state, mode="local", whitelist_strict=True)
    assert isinstance(out, AllowedKeysDict)
    # Wrapped state enforces forbidden keys.
    state_with_forbidden = {"agents_pos": "ORACLE"}
    wrapped = maybe_wrap_obs_state(
        state_with_forbidden, mode="local", whitelist_strict=True
    )
    with pytest.raises(FairnessViolation):
        _ = wrapped["agents_pos"]


# ── EXEC_NAMESPACE isolation ────────────────────────────────────────


def test_compile_does_not_pollute_caller_namespace():
    """Mutations inside one LLM function's namespace don't leak to the
    next compile_llm_function call."""
    src1 = "def f1(s):\n    s['secret'] = 42\n    return s"
    src2 = "def f2(s):\n    return s.get('secret', None)"
    fn1 = compile_llm_function(src1, "f1")
    fn2 = compile_llm_function(src2, "f2")
    # Both are independent functions — fn1's side effects on its own
    # input dict don't bleed into fn2's namespace.
    assert fn2({}) is None
    fn1({"x": 1})
    assert fn2({}) is None
