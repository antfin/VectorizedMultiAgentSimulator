"""v9 strategy data classes — extends v7's Strategy with CoT block.

A v9 strategy carries:

  - the same identity / scoring fields as v7 (name, full_solution,
    success_signature, lero_codability, rl_trainability)
  - a structured chain_of_thought (why_it_works, what_is_needed,
    failure_modes) authored before the slot text
  - chosen_strategy_artifacts populated only when this strategy is
    selected: inferable_hints_text, examples_text, feedback_template
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class V9SuccessSignature:
    """Predicted metrics + structural pattern at 1M frames if the
    strategy works."""

    ast_pattern_description: str
    expected_M1_at_1M: float = 0.05
    expected_M6_at_1M_min: float = 0.20


@dataclass
class V9ChainOfThought:
    """Reasoning block authored by the meta-LLM before any prompt text."""

    why_it_works: str
    what_is_needed: List[str] = field(default_factory=list)
    failure_modes: List[str] = field(default_factory=list)


@dataclass
class V9Artifacts:
    """The three pieces of inner-prompt text the meta-LLM authors when
    a strategy is chosen.

    inferable_hints_text replaces v8's `guidance_observation`-style
    paragraph with a "What you CAN infer" block modeled on S3b-local's
    v2_fewshot_k2_local prompt.

    examples_text contains 2-3 worked Python fewshots (one of which
    must include role one-hot per task_domain.mandatory_features).

    feedback_template is appended to feedback.txt between iters of the
    inner loop — strategy-specific reminder that lives across iters.
    """

    inferable_hints_text: str = ""
    examples_text: str = ""
    feedback_template: str = ""


@dataclass
class V9Strategy:
    name: str
    full_solution: str
    success_signature: V9SuccessSignature
    chain_of_thought: V9ChainOfThought
    lero_codability: int
    rl_trainability: int
    artifacts: V9Artifacts = field(default_factory=V9Artifacts)
    attempts: int = 0
    last_outcome: Optional[str] = None
    last_M1: float = 0.0
    last_M6: float = 0.0
    last_pattern_present: bool = False
    excluded: bool = False

    @property
    def combined_score(self) -> float:
        return 0.5 * (self.lero_codability + self.rl_trainability)


@dataclass
class V9Bundle:
    strategies: List[V9Strategy]
    chosen_idx: int = 0
    history: List[Dict] = field(default_factory=list)
    task_understanding: str = ""

    def current(self) -> V9Strategy:
        return self.strategies[self.chosen_idx]

    def pending(self) -> List[V9Strategy]:
        """Strategies not yet attempted, in score-rank order."""
        return [s for s in self.strategies if not s.excluded and s.attempts == 0]

    def next_pending_idx(self) -> Optional[int]:
        ranked = sorted(
            range(len(self.strategies)),
            key=lambda i: -self.strategies[i].combined_score,
        )
        for i in ranked:
            s = self.strategies[i]
            if not s.excluded and s.attempts == 0:
                return i
        return None

    def format_for_prompt(self) -> str:
        lines = ["Current strategy bundle (score-ranked):"]
        ranked = sorted(
            range(len(self.strategies)),
            key=lambda i: -self.strategies[i].combined_score,
        )
        for rank, i in enumerate(ranked):
            s = self.strategies[i]
            marker = "→" if i == self.chosen_idx else ("✗" if s.excluded else " ")
            lines.append(
                f"  {marker} #{rank} {s.name} "
                f"(cod={s.lero_codability} train={s.rl_trainability} "
                f"score={s.combined_score:.1f} "
                f"attempts={s.attempts} last={s.last_outcome})"
            )
            lines.append(f"      {s.full_solution[:160]}")
        return "\n".join(lines)
