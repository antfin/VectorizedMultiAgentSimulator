"""Code-side diagnosis: given an inner result + active strategy, label
it as achieved / partial / translation_failure / rl_too_hard.

This is the v7 advance over v6: where v6 maps inner outcomes to the
4-class classification (found_good / partial_signal / no_signal_simple
/ no_signal_complex), v7 splits "no signal" into TWO causally distinct
labels based on whether the strategy's structural pattern made it
into the code.

The label drives the next move:
  achieved             → STOP (deep-train candidate)
  partial              → REFINE current strategy (sharpen prompt)
  translation_failure  → REFINE INNER PROMPT (be more explicit so LLM
                                              actually realizes the
                                              strategy)
  rl_too_hard          → SWITCH to next-best strategy (pattern was
                                                       there but PPO
                                                       didn't learn)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from ..v5.inner_loop import InnerResult
from ..v6_prompt_lab.analyzer import analyze_inner_code
from .strategy import DiagnosisLabel, V7Strategy

_log = logging.getLogger("rendezvous.lero.v7.diagnosis")


@dataclass
class V7Diagnosis:
    label: DiagnosisLabel
    pattern_present: bool
    metrics_signature_match: bool
    rationale: str
    inner_M1: float
    inner_M6: float


def diagnose_inner_result(
    inner: Optional[InnerResult],
    strategy: V7Strategy,
) -> V7Diagnosis:
    """Code-side diagnosis. Given an inner result and the strategy
    we tested this round, return a V7Diagnosis."""
    if inner is None or inner.best is None:
        return V7Diagnosis(
            label="too_early",
            pattern_present=False,
            metrics_signature_match=False,
            rationale="no inner result yet",
            inner_M1=0.0,
            inner_M6=0.0,
        )

    metrics = inner.best.metrics
    m1 = float(metrics.get("M1_success_rate", 0.0))
    m6 = float(metrics.get("M6_coverage_progress", 0.0))
    m3 = float(metrics.get("M3_avg_steps", 0.0))
    m4 = float(metrics.get("M4_avg_collisions", 0.0))

    # Pattern detection via AST analyzer.
    code = inner.best.candidate.obs_source or ""
    ast_ana = analyze_inner_code(code)
    # We use touches_both_lidars as the v7 default pattern check; it
    # corresponds to "the inner code combines both LiDAR channels in
    # a single expression" — the structural property required by every
    # coordination-cue strategy. Strategies that need a different
    # pattern can override this in a future extension.
    pattern_present = ast_ana.touches_both_lidars

    sig = strategy.success_signature
    m1_match = m1 >= sig.expected_M1_at_1M
    m6_match = m6 >= sig.expected_M6_at_1M_min
    m3_match = (sig.expected_M3_at_1M_max is None
                or m3 <= sig.expected_M3_at_1M_max)
    m4_match = (sig.expected_M4_at_1M_max is None
                or m4 <= sig.expected_M4_at_1M_max)
    metrics_signature_match = m1_match and m6_match and m3_match and m4_match

    # Decision tree.
    label: DiagnosisLabel
    if m1 >= sig.expected_M1_at_1M and ast_ana.has_cross_source:
        # Real success: meets M1 threshold AND has the pattern.
        label = "achieved"
        rationale = (
            f"M1={m1:.3f} ≥ {sig.expected_M1_at_1M} and pattern present "
            "(strategy realized in code AND PPO learned)"
        )
    elif m6 >= sig.expected_M6_at_1M_min:
        # Partial signal: M6 climbing even if M1 not threshold.
        label = "partial"
        rationale = (
            f"M6={m6:.3f} ≥ {sig.expected_M6_at_1M_min} (partial signal); "
            f"pattern_present={pattern_present}, M1={m1:.3f}"
        )
    elif pattern_present:
        # Pattern is in code but metrics flat → strategy too hard for RL.
        label = "rl_too_hard"
        rationale = (
            f"pattern present (touches both LiDARs) but M1={m1:.3f}, "
            f"M6={m6:.3f} — strategy realized in code but PPO did not "
            f"learn from it at 1M frames"
        )
    else:
        # Pattern missing AND metrics flat → translation failed.
        label = "translation_failure"
        rationale = (
            f"pattern absent (no cross-source op in best inner code) "
            f"AND M1={m1:.3f} M6={m6:.3f} — inner LLM did not realize "
            f"the strategy"
        )

    return V7Diagnosis(
        label=label,
        pattern_present=pattern_present,
        metrics_signature_match=metrics_signature_match,
        rationale=rationale,
        inner_M1=m1,
        inner_M6=m6,
    )
