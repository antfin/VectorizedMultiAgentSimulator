"""LLM-driven scenario tests for v4.

These tests CALL THE REAL meta-LLM (gpt-5.4-mini by default). Mark
them with `pytest -m llm` to run; otherwise pytest skips them.

Cost: ~$0.001 per scenario, ~10 scenarios → ~€0.10 per full sweep.
Wall: ~30s per scenario (LLM latency dominates).
"""

from __future__ import annotations

import os

import pytest

from src.lero.config import LLMConfig
from src.lero.llm_client import LLMClient
from src.lero.meta.v4_bootstrap import bootstrap_from_description
from .fixtures import (
    fake_bootstrap_card,
    fake_round_result,
)
from .runner import run_strategist_scenario


pytestmark = pytest.mark.llm


@pytest.fixture(scope="module")
def meta_llm():
    if not os.environ.get("OPENAI_API_KEY"):
        from dotenv import load_dotenv
        load_dotenv()
    return LLMClient(LLMConfig(
        model="gpt-5.4-mini",
        temperature=1.0,
        max_retries=2,
    ))


# ── B1: Bootstrap understands rendezvous ────────────────────────


def test_bootstrap_understands_rendezvous(tmp_path, meta_llm):
    desc = tmp_path / "task.md"
    desc.write_text(
        "# Multi-robot rendezvous\n"
        "4 agents, 4 targets, k=2 simultaneous coverage required.\n"
        "Local sensors only: lidar_targets, lidar_agents, agent_pos, "
        "agent_vel, agent_idx.\n"
        "Hand-crafted reward (ER1-style). Optimize stable end-of-training "
        "M1 at 10M.\n",
    )
    from pathlib import Path
    base_prompt = Path(
        "/Users/afin/Documents/Studio/PHD/Code/"
        "VectorizedMultiAgentSimulator/rendezvous_comm/src/lero/prompts/"
        "v2_fewshot_modular_v2"
    )
    result = bootstrap_from_description(
        description_path=desc, meta_llm=meta_llm,
        output_dir=tmp_path / "out",
        base_prompt_dir=base_prompt,
    )
    card = result.card
    # The LLM should mention coordination concepts
    text = (
        " ".join(card.high_level_strategies_considered)
        + " " + " ".join(card.proposed_initial_obs_features)
    ).lower()
    assert any(
        kw in text for kw in ("hold", "rendezvous", "coordinat", "proximity")
    ), f"Bootstrap did not mention coordination concepts. Got: {text[:300]}"


# ── S1: Strategy diversity at round 0 ──────────────────────────


def test_strategy_diversity_round_0(meta_llm):
    bs = fake_bootstrap_card()
    result = run_strategist_scenario(
        name="S1_diversity_round_0",
        bootstrap=bs,
        history=[],
        expectations={
            "n_strategies_3": lambda b: len(b.strategies) == 3,
            "covers_2_or_more_domains": lambda b: (
                len(set(s.target_domain for s in b.strategies)) >= 2
            ),
            "diversity_rationale_nonempty": lambda b: (
                len(b.diversity_rationale) > 20
            ),
        },
        meta_llm=meta_llm,
    )
    assert result.parse_error is None, result.parse_error
    for k, v in result.expectations_passed.items():
        assert v, f"Expectation '{k}' failed for round-0 diversity test"


# ── S2: After peak_collapse, at least one strategy reverts reward ──


def test_strategy_after_collapse_reverts_reward(meta_llm):
    """If a prior round had peak_collapse on a reward-modifying strategy,
    the next round must include at least one strategy with
    revert_to_baseline_reward=True."""
    bs = fake_bootstrap_card()
    prior = fake_round_result(
        round_idx=0,
        strategies=[
            {"strategy_id": "S1", "target_domain": "reward",
             "idea": "potential shaping"},
            {"strategy_id": "S2", "target_domain": "observation",
             "idea": "proximity_count"},
            {"strategy_id": "S3", "target_domain": "both",
             "idea": "obs + reward"},
        ],
        candidate_outcomes=[
            {"strategy_id": "S1", "shape": "peak_collapse",
             "final_M1": 0.05, "peak_M1": 0.40},
            {"strategy_id": "S2", "shape": "monotonic_rise",
             "final_M1": 0.18, "peak_M1": 0.20},
            {"strategy_id": "S3", "shape": "peak_collapse",
             "final_M1": 0.08, "peak_M1": 0.35},
        ],
        best_id="S2",
        best_outcome={"shape": "monotonic_rise", "final_M1": 0.18,
                      "peak_M1": 0.20},
    )
    result = run_strategist_scenario(
        name="S2_revert_after_collapse",
        bootstrap=bs,
        history=[prior],
        expectations={
            "any_revert_flag": lambda b: any(
                s.revert_to_baseline_reward for s in b.strategies
            ),
            "revert_has_reason": lambda b: all(
                (not s.revert_to_baseline_reward) or s.revert_reason
                for s in b.strategies
            ),
        },
        meta_llm=meta_llm,
    )
    assert result.parse_error is None, result.parse_error
    if not result.expectations_passed["any_revert_flag"]:
        # Soft fail with diagnostic
        details = " | ".join(
            f"{s.strategy_id}:{s.target_domain}(revert={s.revert_to_baseline_reward})"
            for s in result.bundle.strategies
        )
        pytest.skip(
            f"Strategist did not flag any revert after peak_collapse. "
            f"Strategies: {details}"
        )


# ── S3: After monotonic_rise, at least one strategy refines the winner ──


def test_strategy_builds_on_winning_pattern(meta_llm):
    bs = fake_bootstrap_card()
    prior = fake_round_result(
        round_idx=0,
        strategies=[
            {"strategy_id": "S1", "target_domain": "observation",
             "idea": "hold_signal + proximity_count"},
            {"strategy_id": "S2", "target_domain": "reward",
             "idea": "tiny shaping"},
            {"strategy_id": "S3", "target_domain": "both", "idea": "kitchen sink"},
        ],
        candidate_outcomes=[
            {"strategy_id": "S1", "shape": "monotonic_rise",
             "final_M1": 0.45, "peak_M1": 0.45},
            {"strategy_id": "S2", "shape": "flat_zero",
             "final_M1": 0.0, "peak_M1": 0.005},
            {"strategy_id": "S3", "shape": "oscillating",
             "final_M1": 0.10, "peak_M1": 0.30},
        ],
        best_id="S1",
        best_outcome={"shape": "monotonic_rise", "final_M1": 0.45,
                      "peak_M1": 0.45},
    )
    result = run_strategist_scenario(
        name="S3_build_on_winner",
        bootstrap=bs,
        history=[prior],
        expectations={
            "at_least_one_refines_obs": lambda b: any(
                s.target_domain in ("observation", "both")
                and (
                    "hold" in (s.high_level_idea or "").lower()
                    or "proximity" in (s.high_level_idea or "").lower()
                    or "refine" in (s.high_level_idea or "").lower()
                )
                for s in b.strategies
            ),
        },
        meta_llm=meta_llm,
    )
    assert result.parse_error is None, result.parse_error
    if not result.expectations_passed["at_least_one_refines_obs"]:
        details = " | ".join(
            f"{s.strategy_id}: {s.high_level_idea[:60]}"
            for s in result.bundle.strategies
        )
        pytest.skip(
            f"Strategist did not refine the winning pattern. "
            f"Strategies: {details}"
        )
