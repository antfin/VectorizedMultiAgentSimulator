"""F9.4 — codegen extraction + AST validation contract."""

# pylint: disable=missing-function-docstring

import textwrap

import pytest

from multi_scenario.domain.lero import (
    ALLOWED_IMPORTS,
    extract_candidates,
    validate_function,
)


# ── validate_function ────────────────────────────────────────────────


_GOOD_REWARD = textwrap.dedent(
    """
    import torch
    def compute_reward(scenario_state):
        return scenario_state['agent_pos'].sum()
"""
).strip()


def test_validate_accepts_well_formed_function():
    assert validate_function(_GOOD_REWARD, "compute_reward", ["scenario_state"])


def test_validate_rejects_syntax_error():
    bad = "def compute_reward(scenario_state):\n    return ("
    assert not validate_function(bad, "compute_reward", ["scenario_state"])


def test_validate_rejects_missing_function_name():
    src = "def something_else(scenario_state):\n    return 1"
    assert not validate_function(src, "compute_reward", ["scenario_state"])


def test_validate_rejects_wrong_arg_names():
    src = "def compute_reward(state):\n    return 1"
    assert not validate_function(src, "compute_reward", ["scenario_state"])


def test_validate_rejects_wrong_arg_arity():
    src = "def compute_reward(scenario_state, extra):\n    return 1"
    assert not validate_function(src, "compute_reward", ["scenario_state"])


@pytest.mark.parametrize("module", sorted(ALLOWED_IMPORTS))
def test_validate_accepts_each_allowed_import(module: str):
    src = textwrap.dedent(
        f"""
        import {module}
        def compute_reward(scenario_state):
            return scenario_state['agent_pos']
    """
    ).strip()
    assert validate_function(src, "compute_reward", ["scenario_state"])


@pytest.mark.parametrize("forbidden", ["os", "subprocess", "sys", "requests"])
def test_validate_rejects_forbidden_imports(forbidden: str):
    src = textwrap.dedent(
        f"""
        import {forbidden}
        def compute_reward(scenario_state):
            return scenario_state['agent_pos']
    """
    ).strip()
    assert not validate_function(src, "compute_reward", ["scenario_state"])


def test_validate_rejects_forbidden_from_import():
    src = textwrap.dedent(
        """
        from os import path
        def compute_reward(scenario_state):
            return scenario_state['agent_pos']
    """
    ).strip()
    assert not validate_function(src, "compute_reward", ["scenario_state"])


def test_validate_rejects_relative_import():
    src = textwrap.dedent(
        """
        from . import util
        def compute_reward(scenario_state):
            return scenario_state['agent_pos']
    """
    ).strip()
    assert not validate_function(src, "compute_reward", ["scenario_state"])


def test_validate_finds_function_inside_module_with_helpers():
    """Helper functions defined alongside the target are allowed; the
    check is "is the target present?", not "is the target the only thing?"
    """
    src = textwrap.dedent(
        """
        import torch
        def _helper(x):
            return x ** 2
        def compute_reward(scenario_state):
            return _helper(scenario_state['agent_pos']).sum()
    """
    ).strip()
    assert validate_function(src, "compute_reward", ["scenario_state"])


# ── extract_candidates ───────────────────────────────────────────────


def _wrap(code: str) -> str:
    """Wrap code in a ```python ... ``` fence as the LLM would emit."""
    return (
        f"Here you go:\n```python\n{code}\n```\nLet me know if you want me to iterate."
    )


def test_extract_returns_one_candidate_per_valid_response():
    responses = [_wrap(_GOOD_REWARD)]
    out = extract_candidates(responses)
    assert len(out) == 1
    assert out[0].reward_source == _GOOD_REWARD
    assert out[0].obs_source is None
    assert out[0].raw_response == responses[0]


def test_extract_handles_both_reward_and_obs_in_one_response():
    obs = textwrap.dedent(
        """
        def enhance_observation(scenario_state):
            return scenario_state['agent_pos']
    """
    ).strip()
    resp = (
        "Reward:\n```python\n"
        + _GOOD_REWARD
        + "\n```\nObservation:\n```python\n"
        + obs
        + "\n```"
    )
    out = extract_candidates([resp])
    assert len(out) == 1
    assert out[0].reward_source == _GOOD_REWARD
    assert out[0].obs_source == obs


def test_extract_skips_empty_response():
    out = extract_candidates(["", _wrap(_GOOD_REWARD)])
    assert len(out) == 1


def test_extract_skips_response_without_code_blocks():
    out = extract_candidates(["No code here, sorry.", _wrap(_GOOD_REWARD)])
    assert len(out) == 1


def test_extract_skips_response_where_all_blocks_fail_validation():
    bad = textwrap.dedent(
        """
        import os
        def compute_reward(scenario_state):
            os.system('rm -rf /')
            return 0
    """
    ).strip()
    out = extract_candidates([_wrap(bad)])
    assert out == []


def test_extract_evolve_reward_false_drops_reward_blocks():
    """Even valid reward blocks are skipped when not requested."""
    out = extract_candidates([_wrap(_GOOD_REWARD)], evolve_reward=False)
    assert out == []


def test_extract_evolve_observation_false_drops_obs_blocks():
    obs = textwrap.dedent(
        """
        def enhance_observation(scenario_state):
            return scenario_state['agent_pos']
    """
    ).strip()
    out = extract_candidates([_wrap(obs)], evolve_observation=False)
    assert out == []


def test_extract_accepts_compute_reward_bonus_alias():
    """``bonus`` mode generates ``compute_reward_bonus``; codegen accepts it."""
    bonus = textwrap.dedent(
        """
        def compute_reward_bonus(scenario_state):
            return scenario_state['agent_pos'].sum() * 0.1
    """
    ).strip()
    out = extract_candidates([_wrap(bonus)])
    assert len(out) == 1
    assert "compute_reward_bonus" in out[0].reward_source


def test_extract_preserves_response_order():
    responses = [
        _wrap(_GOOD_REWARD.replace("agent_pos", "agent_vel")),  # variant 1
        _wrap(_GOOD_REWARD),  # variant 2
    ]
    out = extract_candidates(responses)
    assert len(out) == 2
    assert "agent_vel" in out[0].reward_source
    assert "agent_pos" in out[1].reward_source
