"""Statistical analysis utilities for experiment comparison."""
import numpy as np
from typing import Dict, List, Optional, Tuple


def compare_experiments(
    values_a: List[float],
    values_b: List[float],
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Compare two groups of metric values using Mann-Whitney U test.

    Non-parametric — suitable for small sample sizes (typical in RL seeds).

    Returns:
        {"statistic", "p_value", "significant", "effect_size", "n_a", "n_b"}
    """
    from scipy.stats import mannwhitneyu

    a = np.array(values_a)
    b = np.array(values_b)

    if len(a) < 2 or len(b) < 2:
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "significant": False,
            "effect_size": float("nan"),
            "n_a": len(a),
            "n_b": len(b),
        }

    stat, p = mannwhitneyu(a, b, alternative="two-sided")

    # Rank-biserial correlation as effect size
    n = len(a) * len(b)
    effect_size = 1 - (2 * stat / n) if n > 0 else 0.0

    return {
        "statistic": float(stat),
        "p_value": float(p),
        "significant": p < alpha,
        "effect_size": float(effect_size),
        "n_a": len(a),
        "n_b": len(b),
    }


def bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Bootstrap confidence interval for the mean.

    Returns (lower, upper) bounds.
    """
    arr = np.array(values)
    if len(arr) < 2:
        m = float(arr.mean()) if len(arr) == 1 else 0.0
        return (m, m)

    rng = np.random.default_rng(42)
    boot_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (lower, upper)


def pareto_frontier(
    points: List[Tuple[float, float]],
    maximize_x: bool = True,
    maximize_y: bool = True,
) -> List[int]:
    """Return indices of Pareto-optimal points.

    Args:
        points: list of (x, y) tuples
        maximize_x: True if higher x is better
        maximize_y: True if higher y is better

    Returns:
        List of indices into the original list.
    """
    if not points:
        return []

    arr = np.array(points)
    if not maximize_x:
        arr[:, 0] = -arr[:, 0]
    if not maximize_y:
        arr[:, 1] = -arr[:, 1]

    # Sort by x descending
    order = np.argsort(-arr[:, 0])
    frontier = []
    max_y = -np.inf

    for idx in order:
        if arr[idx, 1] > max_y:
            frontier.append(int(idx))
            max_y = arr[idx, 1]

    return sorted(frontier)
