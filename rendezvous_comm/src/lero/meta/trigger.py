"""Outer-loop trigger logic for LERO-MP.

Given the history of templates we've tried so far, decide whether to
ask the meta-LLM for a new prompt template (and why). The checks
correspond 1:1 to plan §6.2 + §6.4.

PURE module — no I/O, no side effects. All state lives in the
``TemplateRecord`` objects the caller passes in.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple

from .failmode import FailMode


class TriggerReason(str, Enum):
    """Why the outer loop is (or isn't) firing another mutation."""

    PLATEAU = "plateau"  # keep going, plateau
    SEED_INSTABILITY = "seed_instability"  # keep going, high σ
    REWARD_HACK = "reward_hack"  # keep going, peak-vs-final
    FAIL_MODE_CLUSTER = "fail_mode_cluster"  # keep going, same bug repeating
    COOLDOWN = "cooldown"  # HOLD, too soon
    BUDGET_EXCEEDED = "budget_exceeded"  # STOP
    CYCLE_DETECTED = "cycle_detected"  # STOP
    FAIRNESS_REPEATED = "fairness_repeated"  # STOP (teaching-to-cheat)
    CONVERGED = "converged"  # STOP, we're done
    INITIAL = "initial"  # first run, no history yet


@dataclass
class TemplateRecord:
    """Summary of one outer-loop iteration under one prompt template."""

    template_version: str
    inner_iter_count: int  # how many inner iters used
    best_peak_M1: float  # Tier-2 peak or Tier-1 best
    best_final_M1: Optional[float] = None  # None if not Tier-2'd
    best_M6: float = 0.0
    best_M2: float = 0.0
    seed_M1_std: float = 0.0  # σ of M1 across seeds, at best cand
    fail_mode: FailMode = FailMode.HEALTHY
    mutation_target_slot: Optional[
        str
    ] = None  # which slot was edited to MAKE this version
    mutation_rationale: Optional[str] = None


@dataclass
class TriggerDecision:
    """Outcome of ``should_meta_iterate``."""

    should_iterate: bool  # emit a new template?
    should_stop: bool  # stop the outer loop entirely?
    reason: TriggerReason
    detail: str = ""

    @property
    def keep_going(self) -> bool:
        return not self.should_stop


@dataclass
class TriggerConfig:
    """Mirror of MetaPromptTrigger + MetaPromptBudget, decoupled so
    this module doesn't import from lero.config (no circular deps).
    """

    plateau_iters: int = 2
    plateau_delta: float = 0.03
    variance_threshold: float = 0.15
    peak_vs_final_gap_max: float = 0.20
    cooldown_inner_iters: int = 3
    max_outer_iters: int = 3
    max_total_inner_candidates: int = 200
    fail_mode_cluster_count: int = 3  # ≥N of last 5 share fail_mode
    fail_mode_cluster_window: int = 5
    fairness_abort_count: int = 2  # ≥N fairness violations → abort
    converged_delta: float = 0.02  # best improved < this for N outer iters → done
    converged_iters: int = 3


def _cycle_detected(history: Sequence[TemplateRecord]) -> bool:
    """A cycle is when the same (slot, rationale) pair appears twice
    in the last ~5 mutations. Means we're stuck oscillating.
    """
    if len(history) < 3:
        return False
    fingerprints: List[Tuple[Optional[str], Optional[str]]] = [
        (r.mutation_target_slot, r.mutation_rationale)
        for r in history[-5:]
        if r.mutation_target_slot is not None
    ]
    return len(fingerprints) != len(set(fingerprints))


def _plateau(history: Sequence[TemplateRecord], cfg: TriggerConfig) -> bool:
    """Best-peak-M1 hasn't moved by ≥ plateau_delta in the last
    ``plateau_iters`` records.
    """
    if len(history) < cfg.plateau_iters + 1:
        return False
    window = history[-(cfg.plateau_iters + 1) :]
    improvements = [
        window[i + 1].best_peak_M1 - window[i].best_peak_M1
        for i in range(len(window) - 1)
    ]
    return all(d < cfg.plateau_delta for d in improvements)


def _converged(history: Sequence[TemplateRecord], cfg: TriggerConfig) -> bool:
    """Same as plateau but with a stricter threshold and longer window
    — used to STOP the loop, not to emit another mutation.
    """
    if len(history) < cfg.converged_iters + 1:
        return False
    window = history[-(cfg.converged_iters + 1) :]
    improvements = [
        window[i + 1].best_peak_M1 - window[i].best_peak_M1
        for i in range(len(window) - 1)
    ]
    return all(d < cfg.converged_delta for d in improvements)


def should_meta_iterate(
    history: Sequence[TemplateRecord],
    total_candidates_so_far: int,
    cfg: TriggerConfig,
    inner_iters_since_last_mutation: int = 0,
) -> TriggerDecision:
    """Decide whether to emit a new prompt template.

    Returns a TriggerDecision with two orthogonal flags:
      - should_iterate: if True, caller should call mutation.propose_new_template
      - should_stop: if True, caller should exit the outer loop

    The caller can interleave: stop conditions always dominate iterate conditions.
    """

    if not history:
        return TriggerDecision(
            should_iterate=True,
            should_stop=False,
            reason=TriggerReason.INITIAL,
            detail="No history yet — run the inner loop on the seed template first.",
        )

    # ── HARD STOP conditions (checked first) ─────────────────────

    if len(history) >= cfg.max_outer_iters:
        return TriggerDecision(
            False,
            True,
            TriggerReason.BUDGET_EXCEEDED,
            f"Reached max_outer_iters={cfg.max_outer_iters}.",
        )

    if total_candidates_so_far >= cfg.max_total_inner_candidates:
        return TriggerDecision(
            False,
            True,
            TriggerReason.BUDGET_EXCEEDED,
            f"Spent {total_candidates_so_far} candidates "
            f"(cap {cfg.max_total_inner_candidates}).",
        )

    fairness_count = sum(
        1 for r in history if r.fail_mode == FailMode.FAIRNESS_VIOLATION
    )
    if fairness_count >= cfg.fairness_abort_count:
        return TriggerDecision(
            False,
            True,
            TriggerReason.FAIRNESS_REPEATED,
            f"{fairness_count} fairness violations — template is "
            f"teaching the LLM to cheat; abort for human review.",
        )

    if _cycle_detected(history):
        return TriggerDecision(
            False,
            True,
            TriggerReason.CYCLE_DETECTED,
            "Same (slot, rationale) edit appearing multiple times "
            "in the last 5 mutations — stop to avoid oscillation.",
        )

    if _converged(history, cfg):
        return TriggerDecision(
            False,
            True,
            TriggerReason.CONVERGED,
            f"best_peak_M1 improved < {cfg.converged_delta} for "
            f"{cfg.converged_iters} outer iters — done.",
        )

    # ── COOLDOWN: hold off even if a trigger fires ───────────────

    if inner_iters_since_last_mutation < cfg.cooldown_inner_iters:
        return TriggerDecision(
            False,
            False,
            TriggerReason.COOLDOWN,
            f"Only {inner_iters_since_last_mutation}/"
            f"{cfg.cooldown_inner_iters} inner iters since last mutation.",
        )

    # ── SOFT triggers (emit a new template) ──────────────────────

    last = history[-1]

    # Reward-hack fires regardless of plateau, because peak-vs-final
    # divergence is a correctness signal, not a progress signal.
    if (
        last.best_final_M1 is not None
        and (last.best_peak_M1 - last.best_final_M1) >= cfg.peak_vs_final_gap_max
    ):
        return TriggerDecision(
            True,
            False,
            TriggerReason.REWARD_HACK,
            f"peak-M1={last.best_peak_M1:.3f} vs final-M1="
            f"{last.best_final_M1:.3f}: gap ≥ "
            f"{cfg.peak_vs_final_gap_max}.",
        )

    if last.seed_M1_std > cfg.variance_threshold:
        return TriggerDecision(
            True,
            False,
            TriggerReason.SEED_INSTABILITY,
            f"σ(M1 across seeds)={last.seed_M1_std:.3f} > "
            f"{cfg.variance_threshold}.",
        )

    # Fail-mode clustering: last K records share a non-healthy fail_mode.
    window = list(history[-cfg.fail_mode_cluster_window :])
    bad = [
        r
        for r in window
        if r.fail_mode != FailMode.HEALTHY
        and r.fail_mode != FailMode.FAIRNESS_VIOLATION  # handled above
    ]
    if len(bad) >= cfg.fail_mode_cluster_count:
        mode_counts = {}
        for r in bad:
            mode_counts[r.fail_mode] = mode_counts.get(r.fail_mode, 0) + 1
        dominant = max(mode_counts, key=mode_counts.get)
        return TriggerDecision(
            True,
            False,
            TriggerReason.FAIL_MODE_CLUSTER,
            f"{mode_counts[dominant]} of last {len(window)} records "
            f"share fail_mode={dominant.value}.",
        )

    if _plateau(history, cfg):
        return TriggerDecision(
            True,
            False,
            TriggerReason.PLATEAU,
            f"best_peak_M1 improved < {cfg.plateau_delta} for "
            f"{cfg.plateau_iters} consecutive outer iters.",
        )

    # No trigger fired — continue the inner loop on the same template.
    return TriggerDecision(
        False,
        False,
        TriggerReason.COOLDOWN,
        "No trigger fired; keep the current template.",
    )
