"""Fixtures for v4 LLM scenario tests."""

from __future__ import annotations

from typing import Any, Dict, List

from src.lero.meta.v4_schemas import (
    BootstrapCard,
    CandidateAnalysis,
    RoundResult,
    StrategyBundle,
    StrategyV4,
)


def fake_bootstrap_card() -> BootstrapCard:
    return BootstrapCard(
        task_summary="4 agents must rendezvous on 4 targets, k=2 simultaneous coverage required",
        success_metric_understanding="M1 = fraction of episodes with all 4 targets simultaneously covered within 200 steps",
        key_difficulty="2 agents must arrive at the same target on the same step — pure coordination",
        failure_modes_anticipated=[
            "ships passing in the night (sequential arrivals miss simultaneity)",
            "anti-crowding learned avoidance",
            "policy collapse via observation exploits at long training",
        ],
        high_level_strategies_considered=[
            "encode hold_signal so an arrived agent stays for partner",
            "encode proximity_count for partner detection",
            "smooth potential-shaping reward (bounded)",
        ],
        proposed_initial_obs_features=[
            "hold_signal = (target_near AND partner_near)",
            "approach_signal = (target_near AND NOT partner_near)",
            "proximity_count = (lidar_targets < cover_r).sum()",
            "gap = sorted(lidar_targets)[1] - sorted(lidar_targets)[0]",
            "agent_idx one-hot for role differentiation",
        ],
        proposed_initial_reward_components=[],
        fairness_audit="Uses lidar_targets, lidar_agents, agent_pos/vel, agent_idx — all in LOCAL_ALLOWED_KEYS. No oracle access.",
        assumptions=[
            "agents are point particles",
            "lidar rays are uniform around the agent",
            "covering_range = 0.35 is the threshold",
        ],
    )


def fake_candidate_analysis(
    strategy_id: str,
    shape: str,
    final_M1: float,
    peak_M1: float,
) -> CandidateAnalysis:
    return CandidateAnalysis(
        candidate_id=f"r0_{strategy_id}_eval",
        strategy_id=strategy_id,
        final_M1=final_M1,
        final_M6=final_M1 + 0.10,
        peak_M1=peak_M1,
        peak_at_frame=120000,
        slope_M6=0.05 if shape == "monotonic_rise" else 0.0,
        noise_std_M1=0.02,
        shape_tag=shape,
        stability_score=(
            0.3 * peak_M1
            + 0.7 * final_M1
            - 0.2 * max(0, peak_M1 - final_M1)
            + 0.05 * (final_M1 + 0.10)
        ),
        qualitative_summary=f"{shape} shape, final_M1={final_M1:.3f}",
    )


def fake_round_result(
    round_idx: int,
    strategies: List[Dict[str, Any]],
    candidate_outcomes: List[Dict[str, Any]],
    best_id: str,
    best_outcome: Dict[str, Any],
) -> RoundResult:
    """Build a synthetic RoundResult.

    `strategies`: list of dicts with strategy_id/target_domain/idea
    `candidate_outcomes`: list of dicts (strategy_id, shape, final_M1, peak_M1)
    `best_outcome`: same shape as one of candidate_outcomes
    """
    strats = [
        StrategyV4(
            strategy_id=s["strategy_id"],
            high_level_idea=s.get("idea", "test"),
            target_domain=s.get("target_domain", "observation"),
            revert_to_baseline_reward=s.get("revert", False),
            revert_reason=s.get("revert_reason"),
            slot_edits=s.get("slot_edits", {}),
            expected_effect=s.get("expected", "x"),
            rationale=s.get("rationale", "x"),
        )
        for s in strategies
    ]
    cands = [
        fake_candidate_analysis(
            o["strategy_id"],
            o["shape"],
            o["final_M1"],
            o["peak_M1"],
        )
        for o in candidate_outcomes
    ]
    bundle = StrategyBundle(
        round_idx=round_idx,
        strategies=strats,
        diversity_rationale="synthetic diverse strategies",
    )
    mid = fake_candidate_analysis(
        best_id,
        best_outcome["shape"],
        best_outcome["final_M1"],
        best_outcome["peak_M1"],
    )
    cross_summary = (
        f"Round {round_idx}: best={best_id} "
        f"shape={best_outcome['shape']} final={best_outcome['final_M1']:.3f}"
    )
    return RoundResult(
        round_idx=round_idx,
        bundle=bundle,
        candidates=cands,
        best_strategy_id=best_id,
        best_candidate_2M=mid,
        cross_round_summary=cross_summary,
    )
