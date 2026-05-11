"""F9.4 — byte-parity for :func:`extract_candidates` vs rendezvous_comm.

Feed the same response strings to both implementations and assert the
resulting (reward_source, obs_source, raw_response) tuples match
byte-for-byte. If this drifts, F8.4's S3b-local replication will
silently change.

Skipped when ``rendezvous_comm`` isn't importable (containers /
slim CI environments).
"""

# pylint: disable=missing-function-docstring

import textwrap

import pytest

from multi_scenario.domain.lero import extract_candidates as ours_extract


def _ensure_rendezvous_importable():
    # pylint: disable=import-outside-toplevel
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3].parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from rendezvous_comm.src.lero.codegen import extract_candidates  # noqa: F401
    except Exception as exc:  # pylint: disable=broad-except
        pytest.skip(f"rendezvous_comm not importable: {exc}")


def _theirs_extract(responses, *, evolve_reward=True, evolve_observation=True):
    """rendezvous_comm's extract_candidates with our kwarg shape."""
    # pylint: disable=import-outside-toplevel
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3].parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from rendezvous_comm.src.lero.codegen import (  # type: ignore
        extract_candidates as their_extract,
    )

    return their_extract(
        responses,
        evolve_reward=evolve_reward,
        evolve_observation=evolve_observation,
    )


def _wrap(code: str) -> str:
    return f"```python\n{code}\n```"


_GOOD_REWARD = textwrap.dedent(
    """
    import torch
    def compute_reward(scenario_state):
        return scenario_state['agent_pos'].sum()
"""
).strip()

_GOOD_OBS = textwrap.dedent(
    """
    import torch
    def enhance_observation(scenario_state):
        return scenario_state['lidar_targets']
"""
).strip()


@pytest.mark.parametrize(
    "label,responses",
    [
        ("reward_only", [_wrap(_GOOD_REWARD)]),
        ("obs_only", [_wrap(_GOOD_OBS)]),
        (
            "both_in_one_response",
            [f"Reward:\n{_wrap(_GOOD_REWARD)}\nObs:\n{_wrap(_GOOD_OBS)}"],
        ),
        ("empty_response", ["", _wrap(_GOOD_REWARD)]),
        ("response_without_blocks", ["plain text reply", _wrap(_GOOD_REWARD)]),
        (
            "candidate_with_helper_fn",
            [
                _wrap(
                    textwrap.dedent(
                        """
                        import torch
                        def _h(x):
                            return x * 2
                        def compute_reward(scenario_state):
                            return _h(scenario_state['agent_pos']).sum()
                    """
                    ).strip()
                )
            ],
        ),
        (
            "candidate_with_forbidden_import",
            [
                _wrap(
                    textwrap.dedent(
                        """
                        import os
                        def compute_reward(scenario_state):
                            return 0
                    """
                    ).strip()
                )
            ],
        ),
    ],
)
def test_extract_candidates_byte_parity(label: str, responses: list[str]):
    """Our extract output equals rendezvous_comm's extract output."""
    _ensure_rendezvous_importable()
    theirs = _theirs_extract(responses)
    ours = ours_extract(responses)
    assert len(ours) == len(
        theirs
    ), f"{label}: candidate count differs (ours={len(ours)}, theirs={len(theirs)})"
    for o, t in zip(ours, theirs):
        assert o.reward_source == t.reward_source, f"{label}: reward_source differs"
        assert o.obs_source == t.obs_source, f"{label}: obs_source differs"
        assert o.raw_response == t.raw_response, f"{label}: raw_response differs"


@pytest.mark.parametrize(
    "evolve_reward,evolve_observation",
    [
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_extract_candidates_byte_parity_flag_combinations(
    evolve_reward: bool, evolve_observation: bool
):
    """Flag combinations match rendezvous_comm's gating exactly."""
    _ensure_rendezvous_importable()
    resp = [f"R:\n{_wrap(_GOOD_REWARD)}\nO:\n{_wrap(_GOOD_OBS)}"]
    ours = ours_extract(
        resp, evolve_reward=evolve_reward, evolve_observation=evolve_observation
    )
    theirs = _theirs_extract(
        resp, evolve_reward=evolve_reward, evolve_observation=evolve_observation
    )
    assert len(ours) == len(theirs)
    for o, t in zip(ours, theirs):
        assert o.reward_source == t.reward_source
        assert o.obs_source == t.obs_source
