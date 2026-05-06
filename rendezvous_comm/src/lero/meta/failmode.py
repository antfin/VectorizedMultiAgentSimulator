"""Fail-mode taxonomy for LERO-MP outer-loop decisions.

Given the candidate metrics produced by one inner-loop pass under a
prompt template, decide *what went wrong* (or what went right). The
classification drives two outer-loop decisions:

  1. Whether to re-prompt (``trigger.should_meta_iterate``).
  2. Which prompt slot to edit (``slot_policy=failmode_taxonomy``).

Keep this module PURE — no I/O, no globals, no LLM. All inputs are
metric dicts or small result summaries so the module is trivially
unit-testable without GPU or API access.

See docs/lero_metaprompt_plan.md §7.3 for the taxonomy table.
"""

from dataclasses import dataclass
from enum import Enum
from statistics import pstdev
from typing import Any, Dict, Optional, Sequence


class FailMode(str, Enum):
    """The dominant pathology under one template / one outer iter.

    Priority (when multiple apply): fairness > nan > dim_mismatch
    > reward_magnitude_inflation > reward_hack > stuck > healthy.
    Priority is encoded in ``classify_inner_result`` below, not here.
    """

    HEALTHY = "healthy"  # none of the below fired
    FAIRNESS_VIOLATION = "fairness_violation"
    NAN_CRASH = "nan_crash"
    DIM_MISMATCH = "dim_mismatch"
    REWARD_HACK = "reward_hack"  # peak-vs-final gap too large
    REWARD_MAGNITUDE_INFLATION = "reward_magnitude_inflation"
    STUCK = "stuck"  # low M1, low variance, no improvement


@dataclass(frozen=True)
class FailModeThresholds:
    """Knobs for the classifier. Defaults match the plan §6.2, §7.3."""

    peak_vs_final_gap: float = 0.20  # reward-hack trigger
    low_m1: float = 0.10  # "stuck" only if best < this
    stuck_variance: float = 0.05  # across-seed σ(M1) ≤ this → stuck
    inflation_ratio: float = 2.0  # |M2| growth across iters
    min_history_for_inflation: int = 2  # need ≥2 iters to see growth
    # Inflation baseline floor: when historical |M2| is near zero the
    # growth ratio explodes (anything / ~0 is huge) and the classifier
    # would fire on noise. Require the max prior |M2| to exceed this
    # before the inflation rule is considered — 1.0 is the typical
    # |M2| magnitude of a well-behaved, non-hacking reward function.
    inflation_m2_floor: float = 1.0


# ── per-candidate inspection ─────────────────────────────────────


def candidate_error_type(metrics: Dict[str, Any]) -> Optional[str]:
    """Return the ``_error_type`` recorded by the inner loop, if any.

    The inner loop stores a string under ``_error`` and (for errors
    we want to distinguish) ``_error_type``. This helper is a thin
    wrapper so callers don't need to know the schema.
    """
    return metrics.get("_error_type")


# ── outer-loop rollup for one template ───────────────────────────


def classify_inner_result(
    candidate_metrics: Sequence[Dict[str, Any]],
    tier2_metrics: Optional[Dict[str, Any]] = None,
    template_history: Optional[Sequence[Dict[str, Any]]] = None,
    thresholds: FailModeThresholds = FailModeThresholds(),
) -> FailMode:
    """Classify the dominant fail-mode under one prompt template.

    Args:
        candidate_metrics: list of per-candidate metric dicts from the
            inner loop's iter_{k}/candidate_{j}_metrics.json files.
            Each dict may contain M1_success_rate, M6_coverage_progress,
            M2_avg_return, _error, _error_type, peak_M1, etc.
        tier2_metrics: if the template was promoted to Tier-2 full
            training, this is the final_metrics.json payload. Only
            needed to detect REWARD_HACK (peak-vs-final divergence).
        template_history: list of summary dicts for PREVIOUS templates
            in the outer loop; each has ``best_M2``. Used to detect
            REWARD_MAGNITUDE_INFLATION.

    Returns:
        The single dominant FailMode, applying the priority order.
    """
    # 1. Fairness > everything. If any candidate tripped the whitelist
    #    the template is teaching the LLM to cheat; must fix first.
    for m in candidate_metrics:
        if candidate_error_type(m) == "FairnessViolation":
            return FailMode.FAIRNESS_VIOLATION

    # 2. NaN crash — policy divergence under this template's reward.
    for m in candidate_metrics:
        if candidate_error_type(m) == "NaNAction":
            return FailMode.NAN_CRASH
        err = (m.get("_error") or "").lower()
        if err and ("nan" in err or "isnan" in err):
            return FailMode.NAN_CRASH

    # 3. Dim mismatch / KeyError / shape — LLM misunderstood the schema.
    for m in candidate_metrics:
        if candidate_error_type(m) in {"KeyError", "RuntimeError", "DimMismatch"}:
            return FailMode.DIM_MISMATCH
        err = (m.get("_error") or "").lower()
        if any(
            tok in err for tok in ("keyerror", "shape", "dimension", "size mismatch")
        ):
            return FailMode.DIM_MISMATCH

    # 4. Reward-magnitude inflation — |M2| grows across templates but
    #    M1 stays flat. Only detectable with history.
    if (
        template_history
        and len(template_history) >= thresholds.min_history_for_inflation
    ):
        prev_absM2 = [abs(h["best_M2"]) for h in template_history if "best_M2" in h]
        cur_valid = [m for m in candidate_metrics if "_error" not in m]
        # Guard: if historical |M2| is still near zero (typical for
        # short smoke runs where the best M2 is O(−1)), any non-trivial
        # current M2 would dwarf it and trigger a false positive. Wait
        # until there is a real baseline to grow from.
        max_prev = max(prev_absM2) if prev_absM2 else 0.0
        if cur_valid and max_prev >= thresholds.inflation_m2_floor:
            cur_absM2 = max(abs(m.get("M2_avg_return", 0.0)) for m in cur_valid)
            cur_M1 = max(m.get("M1_success_rate", 0.0) for m in cur_valid)
            prev_M1 = max(h.get("best_M1", 0.0) for h in template_history)
            ratio = cur_absM2 / max_prev
            if ratio >= thresholds.inflation_ratio and cur_M1 <= prev_M1 + 1e-6:
                return FailMode.REWARD_MAGNITUDE_INFLATION

    # 5. Reward hack — Tier-2 peak-vs-final divergence.
    if tier2_metrics is not None:
        peak = tier2_metrics.get("peak_M1")
        final = tier2_metrics.get("final_M1")
        if peak is not None and final is not None:
            if (peak - final) >= thresholds.peak_vs_final_gap:
                return FailMode.REWARD_HACK

    # 6. Stuck — best M1 low AND cross-seed variance low (plateau that
    #    is not noise). Needs multi-seed info stored on candidates.
    stuck_candidates = [
        m for m in candidate_metrics if "_error" not in m and "M1_per_seed" in m
    ]
    if stuck_candidates:
        best = max(
            (m for m in stuck_candidates),
            key=lambda m: m.get("M1_success_rate", 0.0),
        )
        per_seed = best.get("M1_per_seed") or []
        if len(per_seed) >= 2:
            variance = pstdev(per_seed)
            best_m1 = best.get("M1_success_rate", 0.0)
            if best_m1 < thresholds.low_m1 and variance <= thresholds.stuck_variance:
                return FailMode.STUCK

    return FailMode.HEALTHY


# ── slot-picking policy ──────────────────────────────────────────

# Mapping from fail-mode → which prompt slot the meta-LLM should edit.
# See plan §7.3. ``fairness`` is never an edit target — the slot is
# FROZEN and the meta-prompt optimizer is told so.
SLOT_POLICY_MAP: Dict[FailMode, str] = {
    FailMode.FAIRNESS_VIOLATION: "state_schema",  # clarify what IS allowed
    FailMode.NAN_CRASH: "output_spec",  # tighten bound on output
    FailMode.DIM_MISMATCH: "state_schema",  # clarify shapes
    FailMode.REWARD_HACK: "guidance",  # add anti-hack constraint
    FailMode.REWARD_MAGNITUDE_INFLATION: "guidance",  # cap magnitudes
    FailMode.STUCK: "examples",  # rotate / add examples
    FailMode.HEALTHY: "guidance",  # round-robin default
}


def pick_slot_to_edit(
    fail_mode: FailMode,
    history: Optional[Sequence[str]] = None,
    policy: str = "failmode_taxonomy",
) -> str:
    """Decide which prompt slot the meta-LLM should propose an edit for.

    Args:
        fail_mode: output of ``classify_inner_result``.
        history: names of previously edited slots, most recent last. Used
            only by ``round_robin``.
        policy: ``failmode_taxonomy`` (default), ``round_robin``, or
            the literal ``fixed:<slot_name>``.
    """
    if policy == "failmode_taxonomy":
        return SLOT_POLICY_MAP[fail_mode]
    if policy == "round_robin":
        order = [
            "guidance",
            "examples",
            "output_spec",
            "state_schema",
            "task_context",
        ]
        if not history:
            return order[0]
        last = history[-1]
        try:
            idx = order.index(last)
        except ValueError:
            return order[0]
        return order[(idx + 1) % len(order)]
    if policy.startswith("fixed:"):
        return policy.split(":", 1)[1]
    raise ValueError(f"Unknown slot policy: {policy!r}")
