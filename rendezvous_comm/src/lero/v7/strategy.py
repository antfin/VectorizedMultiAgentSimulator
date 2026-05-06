"""V7Strategy + V7StrategyBundle — the macro-strategy memory v7 adds
on top of v6's operational slot-edit system.

A V7Strategy encodes:
  - a POLICY-level full-solution description (what should the agents do?)
  - a SUCCESS SIGNATURE the strategy can be measured against
  - LERO-translation hints (how to map this to slot text)
  - LERO-codability + RL-trainability scores (0-10 each)
  - per-attempt history (last_outcome, last metrics)

The bundle ranks strategies and excludes ones with last_outcome ∈
{rl_too_hard} from being picked again until manually retried.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


# Diagnosis labels for the reflection step.
DiagnosisLabel = Literal[
    "achieved",
    "partial",
    "translation_failure",  # pattern absent — inner LLM didn't realize the strategy
    "rl_too_hard",  # pattern present but metrics flat — strategy too complex for PPO at 1M
    "too_early",  # not enough info to judge (no inner result yet)
]


@dataclass
class SuccessSignature:
    """Concrete, machine-checkable expectations for a strategy."""

    # AST pattern the inner code should exhibit when this strategy is realized.
    # Free-text description; the meta-LLM and the AST analyzer interpret it.
    # Examples: "cross-source boolean AND of target_close and agent_close",
    # "geometric difference between target and agent proximities".
    ast_pattern_description: str

    # Metric signatures: what M1/M6/M3/M4 should look like at 1M-frame eval
    # if the strategy is working. Used by code-side diagnosis to label
    # an inner result as achieved / partial / rl_too_hard.
    expected_M1_at_1M: float = 0.05  # threshold for "found_good"
    expected_M6_at_1M_min: float = 0.20  # threshold for "partial_signal"
    expected_M3_at_1M_max: Optional[float] = None  # if set, M3 must be ≤
    expected_M4_at_1M_max: Optional[float] = None  # collisions cap


@dataclass
class V7Strategy:
    """One full-solution policy-level hypothesis."""

    name: str  # short handle, e.g. "pairs_commit"
    full_solution: str  # 2-3 sentences: what should agents do
    success_signature: SuccessSignature

    # How to translate this strategy into operational slot text for the
    # inner LLM. The meta-LLM writes this when it picks the strategy.
    lero_translation_hint: str

    # 0-10 self-reported scores by the meta-LLM
    lero_codability: int  # how easily LERO can prompt for it
    rl_trainability: int  # how easily PPO can learn from it

    # Mutated across outer iters
    attempts: int = 0
    last_outcome: Optional[DiagnosisLabel] = None
    last_inner_M1: float = 0.0
    last_inner_M6: float = 0.0
    last_pattern_present: bool = False

    @property
    def combined_score(self) -> float:
        return (self.lero_codability + self.rl_trainability) / 2.0

    @property
    def excluded(self) -> bool:
        """True if this strategy should NOT be picked next."""
        return self.last_outcome == "rl_too_hard"


@dataclass
class V7StrategyBundle:
    """Ordered set of strategies with bundle-level memory."""

    strategies: List[V7Strategy] = field(default_factory=list)
    chosen_idx: int = 0  # which strategy is currently active
    history: List[Dict[str, Any]] = field(default_factory=list)
    # history entries: {"outer_idx": k, "strategy_name": "...", "outcome": "..."}

    def current(self) -> Optional[V7Strategy]:
        if 0 <= self.chosen_idx < len(self.strategies):
            return self.strategies[self.chosen_idx]
        return None

    def next_best_idx(self, exclude_current: bool = True) -> Optional[int]:
        """Find the highest-scoring non-excluded strategy index.

        If `exclude_current=True`, skip the currently-chosen index too.
        Returns None if no strategy is eligible.
        """
        best_idx: Optional[int] = None
        best_score: float = -1e9
        for i, s in enumerate(self.strategies):
            if s.excluded:
                continue
            if exclude_current and i == self.chosen_idx:
                continue
            if s.combined_score > best_score:
                best_score = s.combined_score
                best_idx = i
        return best_idx

    def record_outcome(
        self,
        outer_idx: int,
        outcome: DiagnosisLabel,
        inner_M1: float,
        inner_M6: float,
        pattern_present: bool,
    ) -> None:
        s = self.current()
        if s is None:
            return
        s.attempts += 1
        s.last_outcome = outcome
        s.last_inner_M1 = inner_M1
        s.last_inner_M6 = inner_M6
        s.last_pattern_present = pattern_present
        self.history.append(
            {
                "outer_idx": outer_idx,
                "strategy_name": s.name,
                "outcome": outcome,
                "M1": inner_M1,
                "M6": inner_M6,
                "pattern_present": pattern_present,
            }
        )

    def format_for_prompt(self) -> str:
        """Compact textual summary of the bundle for the meta-LLM."""
        lines = ["Current strategy bundle (ranked by combined score):"]
        for i, s in enumerate(self.strategies):
            marker = "→" if i == self.chosen_idx else " "
            excl = " [EXCLUDED rl_too_hard]" if s.excluded else ""
            lines.append(
                f"  {marker} #{i} {s.name} (codability={s.lero_codability} "
                f"trainability={s.rl_trainability} score={s.combined_score:.1f}"
                f" attempts={s.attempts} last={s.last_outcome}){excl}"
            )
            lines.append(f"      full_solution: {s.full_solution[:160]}")
        if self.history:
            lines.append("")
            lines.append("Bundle history (oldest first):")
            for h in self.history[-6:]:
                lines.append(
                    f"  outer {h['outer_idx']}: {h['strategy_name']} → "
                    f"{h['outcome']} (M1={h['M1']:.3f} M6={h['M6']:.3f} "
                    f"pattern={h['pattern_present']})"
                )
        return "\n".join(lines)
