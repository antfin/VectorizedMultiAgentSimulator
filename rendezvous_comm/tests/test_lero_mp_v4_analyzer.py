"""Tests for v4 analyzer + stability_score."""

from __future__ import annotations

import pytest

from src.lero.meta.v4_analyzer import (
    aggregate_round_analysis,
    analyze_candidate_trajectory,
    classify_shape,
    stability_score,
)
from src.lero.meta.v4_schemas import FitnessWeights


W = FitnessWeights()


# ── stability_score ─────────────────────────────────────────────


def test_stability_flat_strong_beats_peak_collapse():
    flat_strong = stability_score(0.5, 0.5, 0.7, W)
    peak_collapse = stability_score(0.5, 0.05, 0.4, W)
    assert flat_strong > peak_collapse


def test_stability_perfect_beats_partial():
    perfect = stability_score(0.9, 0.9, 1.0, W)
    partial = stability_score(0.5, 0.5, 0.7, W)
    assert perfect > partial


def test_stability_zero_when_zero():
    assert stability_score(0.0, 0.0, 0.0, W) == 0.0


def test_stability_penalty_does_not_make_negative_when_no_gap():
    """Penalty term only fires when peak > final."""
    score = stability_score(0.3, 0.3, 0.3, W)
    expected = 0.3 * 0.3 + 0.7 * 0.3 + 0.05 * 0.3
    assert abs(score - expected) < 1e-9


# ── classify_shape ─────────────────────────────────────────────


def _traj(m1_vals, frames=None):
    if frames is None:
        frames = [(i + 1) * 60_000 for i in range(len(m1_vals))]
    return [
        {"M1": v, "M6": v + 0.1, "frame": f}
        for v, f in zip(m1_vals, frames)
    ]


def test_classify_flat_zero():
    assert classify_shape(_traj([0.0, 0.0, 0.005, 0.0])) == "flat_zero"


def test_classify_monotonic_rise():
    assert classify_shape(_traj([0.0, 0.05, 0.1, 0.2, 0.3])) == "monotonic_rise"


def test_classify_peak_collapse():
    # peak 0.5, final 0.05, gap 0.45 > 0.20 threshold
    assert classify_shape(_traj([0.0, 0.2, 0.5, 0.3, 0.05])) == "peak_collapse"


def test_classify_late_ramp():
    # 80% < 0.05, last 20% > 0.20
    traj = _traj([0.01, 0.02, 0.0, 0.01, 0.02, 0.0, 0.01, 0.02, 0.25, 0.30])
    assert classify_shape(traj) == "late_ramp"


def test_classify_flat_nonzero():
    assert classify_shape(_traj([0.02, 0.03, 0.04, 0.045])) == "flat_nonzero"


def test_classify_oscillating():
    # peak ≥ 0.05, drop > 0.05 from peak
    traj = _traj([0.0, 0.1, 0.05, 0.12, 0.06, 0.10])
    assert classify_shape(traj) in ("oscillating", "plateau")


def test_classify_empty_returns_flat_zero():
    assert classify_shape([]) == "flat_zero"


# ── analyze_candidate_trajectory ────────────────────────────────


def test_analyze_full_pipeline():
    traj = _traj([0.0, 0.05, 0.1, 0.2, 0.3, 0.35])
    a = analyze_candidate_trajectory("c0", "S1", traj, W)
    assert a.candidate_id == "c0"
    assert a.strategy_id == "S1"
    assert a.final_M1 == 0.35
    assert a.peak_M1 == 0.35
    assert a.shape_tag == "monotonic_rise"
    assert a.stability_score > 0.30  # decent score
    assert "monotonic" in a.qualitative_summary.lower()


def test_analyze_peak_collapse_penalized():
    bad = analyze_candidate_trajectory(
        "c0", "S1", _traj([0.0, 0.3, 0.5, 0.3, 0.05]), W,
    )
    good = analyze_candidate_trajectory(
        "c1", "S2", _traj([0.0, 0.1, 0.2, 0.25, 0.3, 0.3]), W,
    )
    assert bad.shape_tag == "peak_collapse"
    assert good.shape_tag == "monotonic_rise"
    assert good.stability_score > bad.stability_score


def test_analyze_empty_trajectory_safe_defaults():
    a = analyze_candidate_trajectory("c0", "S1", [], W)
    assert a.final_M1 == 0.0
    assert a.shape_tag == "flat_zero"
    assert a.stability_score == 0.0


# ── aggregate_round_analysis ────────────────────────────────────


def test_aggregate_ranks_by_score():
    a1 = analyze_candidate_trajectory(
        "c0", "S1", _traj([0.0, 0.1, 0.2, 0.25, 0.3]), W,
    )
    a2 = analyze_candidate_trajectory(
        "c1", "S2", _traj([0.0, 0.3, 0.5, 0.3, 0.05]), W,
    )
    a3 = analyze_candidate_trajectory(
        "c2", "S3", _traj([0.0, 0.0, 0.0, 0.0, 0.005]), W,
    )
    text = aggregate_round_analysis(1, [a1, a2, a3])
    s1_pos = text.find("[S1]")
    s2_pos = text.find("[S2]")
    s3_pos = text.find("[S3]")
    # S1 has highest score → appears first
    assert 0 <= s1_pos < s2_pos
    assert 0 <= s1_pos < s3_pos


def test_aggregate_warns_on_collapse():
    bad = analyze_candidate_trajectory(
        "c0", "S2", _traj([0.0, 0.3, 0.5, 0.3, 0.05]), W,
    )
    text = aggregate_round_analysis(2, [bad])
    assert "peak_collapse" in text.lower() or "reverting reward" in text.lower()


def test_aggregate_marks_round_winner():
    good = analyze_candidate_trajectory(
        "c0", "S1", _traj([0.0, 0.1, 0.3, 0.4, 0.5]), W,
    )
    text = aggregate_round_analysis(0, [good])
    assert "round winner" in text.lower()
