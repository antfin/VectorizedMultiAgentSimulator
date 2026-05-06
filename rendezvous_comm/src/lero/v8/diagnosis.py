"""v8 diagnosis — extends v7 with `too_many_features` and `over_gated`.

Decision tree:

  if M1 ≥ expected_M1 and pattern_present  → "achieved"
  elif M6 ≥ expected_M6_min                → "partial"
  elif n_features > feature_count_cap      → "too_many_features"   (NEW)
  elif n_gated > gated_cap and n_dense<3   → "over_gated"          (NEW)
  elif pattern_present                     → "rl_too_hard"
  else                                     → "translation_failure"

The two new labels nudge the meta-LLM toward simpler, denser observations
BEFORE giving up on a strategy and switching.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Optional

from ..v5.inner_loop import InnerResult
from ..v6_prompt_lab.analyzer import (
    analyze_inner_code,
    count_dense_features,
    count_gated_features,
)
from ..v7.strategy import V7Strategy

_log = logging.getLogger("rendezvous.lero.v8.diagnosis")


V8DiagnosisLabel = Literal[
    "achieved",
    "partial",
    "too_many_features",
    "over_gated",
    "translation_failure",
    "rl_too_hard",
    "too_early",
]


@dataclass
class V8Diagnosis:
    label: V8DiagnosisLabel
    pattern_present: bool
    metrics_signature_match: bool
    rationale: str
    inner_M1: float
    inner_M6: float
    n_features: int = 0
    n_gated: int = 0
    n_dense: int = 0


def diagnose_inner_result_v8(
    inner: Optional[InnerResult],
    strategy: V7Strategy,
    feature_count_cap: int,
    gated_feature_cap: int,
    min_dense_when_over_gated: int = 3,
) -> V8Diagnosis:
    """v8 diagnosis. Returns a V8Diagnosis with the v7 labels plus
    `too_many_features` and `over_gated`."""
    if inner is None or inner.best is None:
        return V8Diagnosis(
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

    code = inner.best.candidate.obs_source or ""
    ast_ana = analyze_inner_code(code)
    pattern_present = ast_ana.touches_both_lidars

    n_features = ast_ana.n_returned_features
    n_gated = count_gated_features(code)
    n_dense = count_dense_features(ast_ana, code)

    sig = strategy.success_signature
    m1_match = m1 >= sig.expected_M1_at_1M
    m6_match = m6 >= sig.expected_M6_at_1M_min
    metrics_match = m1_match and m6_match

    label: V8DiagnosisLabel
    if m1 >= sig.expected_M1_at_1M and ast_ana.has_cross_source:
        label = "achieved"
        rationale = f"M1={m1:.3f} ≥ {sig.expected_M1_at_1M} and pattern present"
    elif m6 >= sig.expected_M6_at_1M_min:
        label = "partial"
        rationale = (
            f"M6={m6:.3f} ≥ {sig.expected_M6_at_1M_min} (partial signal); "
            f"M1={m1:.3f}, n_features={n_features}, gated={n_gated}"
        )
    elif n_features > feature_count_cap:
        label = "too_many_features"
        rationale = (
            f"n_features={n_features} > cap={feature_count_cap} — "
            f"observation has too many features at 1M-frame eval; "
            f"action: trim_features"
        )
    elif n_gated > gated_feature_cap and n_dense < min_dense_when_over_gated:
        label = "over_gated"
        rationale = (
            f"n_gated={n_gated} > cap={gated_feature_cap} AND "
            f"n_dense={n_dense} < {min_dense_when_over_gated} — too many "
            f"cover-zone-gated features without enough dense signals; "
            f"action: replace_gated_with_dense"
        )
    elif pattern_present:
        label = "rl_too_hard"
        rationale = (
            f"pattern present, n_features={n_features}, gated={n_gated} OK, "
            f"but M1={m1:.3f} M6={m6:.3f} — strategy realized in code but "
            f"PPO did not learn at 1M frames"
        )
    else:
        label = "translation_failure"
        rationale = f"pattern absent in best inner code; M1={m1:.3f} M6={m6:.3f}"

    return V8Diagnosis(
        label=label,
        pattern_present=pattern_present,
        metrics_signature_match=metrics_match,
        rationale=rationale,
        inner_M1=m1,
        inner_M6=m6,
        n_features=n_features,
        n_gated=n_gated,
        n_dense=n_dense,
    )
