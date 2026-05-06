"""Per-candidate trajectory analysis + stability scoring (LERO-MP v4).

Given a metrics history (one dict per evaluation point during inner /
mid-frames training), produce a typed CandidateAnalysis whose
qualitative_summary feeds the next-round Strategist.

The stability_score formula explicitly favors flat-stable trajectories
over peak-then-collapse, addressing the v3 finding that LLM-engineered
observations can create exploit channels even with hand-crafted reward.
"""

from __future__ import annotations

import logging
import statistics
from typing import Any, Dict, List, Sequence

from .v4_schemas import (
    CandidateAnalysis,
    FitnessWeights,
    ShapeTag,
)

_log = logging.getLogger("rendezvous.lero.mp.v4")


# ── Stability score ─────────────────────────────────────────────


def stability_score(
    peak_M1: float,
    final_M1: float,
    m6_at_end: float,
    weights: FitnessWeights,
) -> float:
    """v4 fitness function.

    Higher = better. Designed to reward flat-stable end-of-training
    over peak-then-collapse:

        score = α·peak + β·final − γ·max(0, peak−final) + δ·M6_end

    With default weights (α=0.3, β=0.7, γ=0.2, δ=0.05):

        peak=0.5 final=0.5     → 0.55  (flat-stable mid-tier)
        peak=0.5 final=0.05    → 0.095 (peak-collapse, penalized)
        peak=0.3 final=0.3     → 0.32  (lower flat-stable)
        peak=0.9 final=0.9     → 1.00  (best-case)

    The β > α weighting biases the loop toward final 10M performance.
    The penalty γ punishes any peak that the policy can't sustain.
    """
    gap = max(0.0, float(peak_M1) - float(final_M1))
    return (
        weights.peak * float(peak_M1)
        + weights.final * float(final_M1)
        - weights.stability_penalty * gap
        + weights.m6_bonus * float(m6_at_end)
    )


# ── Trajectory shape classification ─────────────────────────────


def classify_shape(
    metrics_history: Sequence[Dict[str, Any]],
) -> ShapeTag:
    """Tag a trajectory shape based on M1 evolution.

    Conservative tags — the analyzer prefers ``flat_zero`` or
    ``oscillating`` over over-claiming a strong rise.
    """
    if not metrics_history:
        return "flat_zero"

    m1 = [float(p.get("M1", 0.0)) for p in metrics_history]
    if not m1:
        return "flat_zero"

    peak = max(m1)
    final = m1[-1]

    if peak < 0.02:
        return "flat_zero"

    drop = peak - final
    # peak_collapse: large drop from peak
    if drop > 0.20:
        return "peak_collapse"

    # late_ramp: bottom 80% < 0.05, top 20% > 0.20
    cutoff = max(1, int(0.8 * len(m1)))
    early_max = max(m1[:cutoff], default=0.0) if cutoff < len(m1) else 0.0
    late_max = max(m1[cutoff:], default=0.0)
    if early_max < 0.05 and late_max > 0.20:
        return "late_ramp"

    # monotonic_rise: end ≥ 95% of peak AND peak ≥ 0.05
    if peak >= 0.05 and final >= 0.95 * peak:
        return "monotonic_rise"

    # plateau: peak < 0.05 (just stuck at flat-nonzero)
    if peak < 0.05:
        return "flat_nonzero"

    # plateau-then-modest-drop or just bumpy
    if drop > 0.05:
        return "plateau"

    return "oscillating"


# ── Trajectory regression / noise ───────────────────────────────


def _slope_per_100k(metrics_history: Sequence[Dict[str, Any]]) -> float:
    """Linear-regression slope of M6 per 100k frames.

    Returns 0.0 if fewer than 2 points or no frame info.
    """
    if len(metrics_history) < 2:
        return 0.0
    pts = []
    for p in metrics_history:
        m6 = p.get("M6")
        frame = p.get("frame")
        if m6 is None or frame is None:
            continue
        pts.append((float(frame), float(m6)))
    if len(pts) < 2:
        return 0.0
    n = len(pts)
    mean_x = sum(x for x, _ in pts) / n
    mean_y = sum(y for _, y in pts) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in pts)
    den = sum((x - mean_x) ** 2 for x, _ in pts)
    if den == 0:
        return 0.0
    return num / den * 100_000.0


def _noise_std_M1_last5(metrics_history: Sequence[Dict[str, Any]]) -> float:
    if len(metrics_history) < 2:
        return 0.0
    last5 = [float(p.get("M1", 0.0)) for p in metrics_history[-5:]]
    if len(last5) < 2:
        return 0.0
    return statistics.pstdev(last5)


# ── Qualitative summary text ────────────────────────────────────


def _qualitative_summary(
    shape: ShapeTag,
    slope: float,
    noise: float,
    final_M1: float,
    peak_M1: float,
) -> str:
    """One- to two-sentence prose for the next round's Strategist."""
    drop = peak_M1 - final_M1
    parts: List[str] = []

    if shape == "monotonic_rise":
        parts.append(
            f"Stable monotonic rise — final M1={final_M1:.3f} retained "
            f"{(final_M1/peak_M1*100 if peak_M1 else 0):.0f}% of peak."
        )
    elif shape == "peak_collapse":
        parts.append(
            f"Peak-collapse: M1 reached {peak_M1:.3f} then collapsed to "
            f"{final_M1:.3f} (gap={drop:+.2f}). Likely exploit pattern."
        )
    elif shape == "late_ramp":
        parts.append(
            f"Late-ramp: flat for most of training, then late surge to "
            f"{final_M1:.3f}. May not have converged."
        )
    elif shape == "oscillating":
        parts.append(
            f"Oscillating: final M1={final_M1:.3f} but high variance — "
            f"std(last5)={noise:.3f}. Policy not stable."
        )
    elif shape == "plateau":
        parts.append(f"Plateau: peak {peak_M1:.3f} then small drift to {final_M1:.3f}.")
    elif shape == "flat_zero":
        parts.append("Flat at zero — no useful learning.")
    else:  # flat_nonzero
        parts.append(
            f"Flat at low value (peak={peak_M1:.3f}). Stuck below useful " f"threshold."
        )

    if slope > 0.05:
        parts.append(f"M6 trending up at {slope:+.3f}/100k.")
    elif slope < -0.05:
        parts.append(f"M6 trending DOWN at {slope:+.3f}/100k.")

    return " ".join(parts)


# ── Main entry point ────────────────────────────────────────────


def analyze_candidate_trajectory(
    candidate_id: str,
    strategy_id: str,
    metrics_history: Sequence[Dict[str, Any]],
    weights: FitnessWeights,
) -> CandidateAnalysis:
    """Build a CandidateAnalysis from a metrics history.

    ``metrics_history`` is a list of dicts, one per evaluation step.
    Each dict should have at least: ``M1``, ``M6``, ``frame``.
    """
    if not metrics_history:
        # Empty trajectory → safe defaults
        return CandidateAnalysis(
            candidate_id=candidate_id,
            strategy_id=strategy_id,
            final_M1=0.0,
            final_M6=0.0,
            peak_M1=0.0,
            peak_at_frame=0,
            slope_M6=0.0,
            noise_std_M1=0.0,
            shape_tag="flat_zero",
            stability_score=0.0,
            qualitative_summary="No metrics — candidate failed before eval.",
        )

    m1_vals = [float(p.get("M1", 0.0)) for p in metrics_history]
    m6_vals = [float(p.get("M6", 0.0)) for p in metrics_history]
    frame_vals = [int(p.get("frame", 0)) for p in metrics_history]

    final_M1 = m1_vals[-1]
    final_M6 = m6_vals[-1]
    peak_M1 = max(m1_vals)
    peak_idx = m1_vals.index(peak_M1)
    peak_at_frame = frame_vals[peak_idx] if peak_idx < len(frame_vals) else 0

    shape = classify_shape(metrics_history)
    slope = _slope_per_100k(metrics_history)
    noise = _noise_std_M1_last5(metrics_history)
    score = stability_score(peak_M1, final_M1, final_M6, weights)
    summary = _qualitative_summary(shape, slope, noise, final_M1, peak_M1)

    return CandidateAnalysis(
        candidate_id=candidate_id,
        strategy_id=strategy_id,
        final_M1=final_M1,
        final_M6=final_M6,
        peak_M1=peak_M1,
        peak_at_frame=peak_at_frame,
        slope_M6=slope,
        noise_std_M1=noise,
        shape_tag=shape,
        stability_score=score,
        qualitative_summary=summary,
    )


# ── Aggregate (cross-candidate) analysis ────────────────────────


def aggregate_round_analysis(
    round_idx: int,
    candidates: Sequence[CandidateAnalysis],
) -> str:
    """Cross-candidate comparison text for the next round's Strategist.

    Designed to be paste-ready into a meta-LLM prompt — explicitly
    contrasts strategies and flags collapse / late-ramp patterns.
    """
    if not candidates:
        return f"Round {round_idx}: no candidates produced metrics."

    ranked = sorted(
        candidates,
        key=lambda c: c.stability_score,
        reverse=True,
    )
    lines = [f"Round {round_idx} ({len(candidates)} strategies):"]
    for c in ranked:
        lines.append(
            f"  [{c.strategy_id}] score={c.stability_score:+.3f}  "
            f"peak_M1={c.peak_M1:.3f}  final_M1={c.final_M1:.3f}  "
            f"shape={c.shape_tag}  →  {c.qualitative_summary}"
        )
    best = ranked[0]
    if any(c.shape_tag == "peak_collapse" for c in candidates):
        collapsed = [
            c.strategy_id for c in candidates if c.shape_tag == "peak_collapse"
        ]
        lines.append(
            f"  ⚠ peak_collapse on: {', '.join(collapsed)} — consider "
            f"reverting reward modifications next round."
        )
    if best.stability_score > 0.10:
        lines.append(f"  ✓ {best.strategy_id} is the round winner (mid-train at 2M).")
    else:
        lines.append(
            f"  ⚠ Best round score is low ({best.stability_score:+.3f}); "
            f"consider re-anchoring on bootstrap recipe."
        )
    return "\n".join(lines)
