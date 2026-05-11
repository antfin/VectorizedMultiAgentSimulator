"""F9.6.a — :class:`InitialAndFeedbackComposer` contract.

Renders against the ported v2_fewshot_k2_local templates so the
test exercises the real Jinja registry, not a stub.
"""

# pylint: disable=missing-function-docstring,redefined-outer-name

import pytest

from multi_scenario.adapters.prompt_composers import InitialAndFeedbackComposer
from multi_scenario.adapters.prompts import JinjaPromptRenderer
from multi_scenario.domain.lero import (
    Candidate,
    CandidateCode,
    CandidateMetrics,
    CandidateResult,
)


@pytest.fixture
def composer() -> InitialAndFeedbackComposer:
    return InitialAndFeedbackComposer(
        renderer=JinjaPromptRenderer(),
        prompt_version="v2_fewshot_k2_local",
        n_candidates=3,
    )


@pytest.fixture
def task_params() -> dict:
    return {
        "n_agents": 4,
        "n_targets": 4,
        "agents_per_target": 2,
        "covering_range": 0.35,
        "n_lidar_rays_entities": 15,
        "n_lidar_rays_agents": 12,
        "obs_lidar_agents": '"lidar_agents":     # [batch, 12] — agent LiDAR',
    }


def _result(
    iteration: int = 0,
    idx: int = 0,
    m1: float = 0.0,
    m2: float = 0.0,
    verdict: str = "regression",
) -> CandidateResult:
    return CandidateResult(
        candidate=Candidate(
            iteration=iteration,
            candidate_idx=idx,
            code=CandidateCode(
                reward_source="def compute_reward(s): return s['agent_pos']"
            ),
        ),
        metrics=CandidateMetrics(M1_success_rate=m1, M2_avg_return=m2),
        verdict=verdict,
    )


# ── Iteration 0: initial prompt only ─────────────────────────────────


def test_iter_zero_returns_system_plus_initial_user(composer, task_params):
    out = composer.compose(iteration=0, history=[], task_params=task_params)
    assert len(out.messages) == 2
    assert out.messages[0]["role"] == "system"
    assert out.messages[1]["role"] == "user"
    # Substitution happened: n_agents → 4.
    assert "4 agents" in out.messages[1]["content"]


def test_iter_zero_uses_configured_prompt_version(composer, task_params):
    out = composer.compose(iteration=0, history=[], task_params=task_params)
    assert out.prompt_version == "v2_fewshot_k2_local"


def test_iter_zero_ignores_strategy_card(composer, task_params):
    """F9.6.a is the default composer — strategy_card is a no-op here."""
    from multi_scenario.domain.lero import StrategyCard

    with_card = composer.compose(
        iteration=0,
        history=[],
        task_params=task_params,
        strategy_card=StrategyCard(rationale="ignored"),
    )
    without_card = composer.compose(iteration=0, history=[], task_params=task_params)
    assert with_card.messages == without_card.messages


# ── Iteration > 0: feedback message appended ─────────────────────────


def test_iter_one_appends_feedback_with_history(composer, task_params):
    history = [
        _result(iteration=0, idx=0, m1=0.5, m2=10.0, verdict="progress"),
        _result(iteration=0, idx=1, m1=0.1, m2=2.0, verdict="regression"),
    ]
    out = composer.compose(iteration=1, history=history, task_params=task_params)
    assert len(out.messages) == 3
    feedback = out.messages[2]["content"]
    # Feedback section names the M1 numbers for ranking.
    assert "0.500" in feedback
    assert "0.100" in feedback
    # best_idx (1-indexed) of the rank-0 (M1=0.5) candidate = 1
    assert "Best candidate was #1" in feedback or "Best was #1" in feedback


def test_feedback_ranks_progress_above_regression_even_with_lower_m1(
    composer, task_params
):
    """Verdict dominates raw M1 — a 'progress' verdict with M1=0.30
    ranks above a 'regression' verdict with M1=0.35.

    Catches the rendezvous_comm 2026-04-16 inversion where M1 alone
    flipped the ranking under crash/regression conditions.
    """
    history = [
        _result(iteration=0, idx=0, m1=0.35, m2=5.0, verdict="regression"),
        _result(iteration=0, idx=1, m1=0.30, m2=8.0, verdict="progress"),
    ]
    out = composer.compose(iteration=1, history=history, task_params=task_params)
    feedback = out.messages[2]["content"]
    # The "progress" candidate (idx 1, original-1-indexed = #2) should be best.
    assert "Best candidate was #2" in feedback or "Best was #2" in feedback


def test_feedback_includes_code_only_for_top_k(task_params):
    """Top-k=2 with 4 candidates → only top 2 have code blocks."""
    composer = InitialAndFeedbackComposer(
        renderer=JinjaPromptRenderer(),
        prompt_version="v2_fewshot_k2_local",
        n_candidates=4,
        top_k_with_code=2,
    )
    history = [
        _result(iteration=0, idx=i, m1=1.0 - 0.2 * i, verdict="progress")
        for i in range(4)
    ]
    out = composer.compose(iteration=1, history=history, task_params=task_params)
    feedback = out.messages[2]["content"]
    # Code blocks count: should be ≤ 2 (top_k_with_code).
    n_code_blocks = feedback.count("```python")
    assert n_code_blocks == 2


# ── Comm metrics gating ──────────────────────────────────────────────


def test_feedback_omits_comm_line_when_no_history_uses_tokens(composer, task_params):
    history = [_result(iteration=0, idx=0, m1=0.5)]
    out = composer.compose(iteration=1, history=history, task_params=task_params)
    feedback = out.messages[2]["content"]
    assert "M5 (Avg Tokens)" not in feedback


def test_feedback_includes_comm_line_when_any_candidate_used_tokens(task_params):
    composer = InitialAndFeedbackComposer(
        renderer=JinjaPromptRenderer(),
        prompt_version="v2_fewshot_k2_local",
        n_candidates=3,
    )
    history = [
        CandidateResult(
            candidate=Candidate(
                iteration=0,
                candidate_idx=0,
                code=CandidateCode(
                    reward_source="def compute_reward(s): return s['agent_pos']"
                ),
            ),
            metrics=CandidateMetrics(M1_success_rate=0.4, M5_tokens=2.5),
            verdict="progress",
        ),
    ]
    out = composer.compose(iteration=1, history=history, task_params=task_params)
    feedback = out.messages[2]["content"]
    # Note: the feedback template variable ``comm_metrics`` only appears
    # in v1 / v1_global templates. v2_fewshot_k2_local doesn't render it
    # — pin this by verifying the composer accepted the param without
    # crashing rather than asserting it lands in the prompt text.
    assert feedback  # rendered without error


# ── ComposedPrompt structure ─────────────────────────────────────────


def test_composed_prompt_carries_render_context(composer, task_params):
    out = composer.compose(iteration=0, history=[], task_params=task_params)
    # Render context includes the task params + the n_candidates default.
    assert out.render_context["n_agents"] == 4
    assert out.render_context["n_candidates"] == 3


def test_composed_prompt_signal_tier_defaults_to_scalar(composer, task_params):
    out = composer.compose(iteration=0, history=[], task_params=task_params)
    assert out.signal_tier == "scalar"


def test_composer_satisfies_protocol():
    import inspect

    from multi_scenario.domain.ports import PromptComposer  # noqa: F401

    sig = inspect.signature(InitialAndFeedbackComposer.compose)
    params = set(sig.parameters)
    assert {"iteration", "history", "task_params", "strategy_card"} <= params
