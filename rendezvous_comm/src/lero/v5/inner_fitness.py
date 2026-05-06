"""Weighted multi-metric fitness for v5 inner-eval ranking.

At sub-threshold eval budgets (1M frames on hard task), M1 is often
zero across all candidates so M1-only ranking reduces to noise. v5
uses a weighted sum that exposes partial-progress signal:

    fitness = 1.0 * M1                   (primary when present)
            + 0.5 * M6                   (partial-coverage proxy)
            + 0.2 * M6_slope             (rising vs stagnant)
            - 0.05 * M4 / 10.0           (collision penalty, normalized)
            + shape_bonus[shape_tag]     (trajectory quality)

Weights chosen so that M6=0.17 vs 0.05 (typical sub-threshold spread)
yields a ~0.06 fitness gap — bigger than M2 noise and big enough to
break flat_zero ties.
"""

from __future__ import annotations

from typing import Dict


SHAPE_BONUS: Dict[str, float] = {
    "monotonic_rise": 0.10,
    "late_ramp": 0.05,
    "plateau": 0.00,
    "oscillating": -0.05,
    "peak_collapse": -0.10,
    "flat_nonzero": 0.00,
    "flat_zero": 0.00,
}


def weighted_fitness(
    metrics: Dict[str, float],
    shape_tag: str = "flat_zero",
    m6_slope: float = 0.0,
) -> float:
    m1 = float(metrics.get("M1_success_rate", 0.0))
    m6 = float(metrics.get("M6_coverage_progress", 0.0))
    m4 = float(metrics.get("M4_avg_collisions", 0.0))
    return (
        1.0 * m1
        + 0.5 * m6
        + 0.2 * float(m6_slope)
        - 0.05 * (m4 / 10.0)
        + SHAPE_BONUS.get(shape_tag, 0.0)
    )


def m6_slope(metrics_history) -> float:
    """Last-third minus first-third mean M6 — a coarse trend signal."""
    if not metrics_history or len(metrics_history) < 3:
        return 0.0
    pts = [
        float(p.get("M6", p.get("M6_coverage_progress", 0.0))) for p in metrics_history
    ]
    n = len(pts)
    third = max(1, n // 3)
    early = sum(pts[:third]) / third
    late = sum(pts[-third:]) / third
    return late - early
