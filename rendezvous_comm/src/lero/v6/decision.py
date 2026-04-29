"""V6MetaDecision schema + code-side classifier + policy enforcement.

The meta-LLM emits a V6MetaDecision per outer iter. We then run a
code-side classifier independently from raw inner metrics and
cross-check vs the LLM's claim — code wins on disagreement (per
docs/v6_plan.md §10).

Policy enforcement runs after the cross-check:
  - mode unlock (no `evolve_reward=True` in outer iter 0)
  - complexity-level monotone (cannot decrease without `reset_simpler`)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

_log = logging.getLogger("rendezvous.lero.v6.decision")


Classification = Literal[
    "found_good",
    "partial_signal",
    "no_signal_simple",
    "no_signal_complex",
]

NextMode = Literal[
    "stop",
    "refine_current",
    "try_different_simple",
    "add_simple_reward",
    "reset_simpler",
]


@dataclass
class V6MetaDecision:
    classification: Classification
    next_mode: NextMode
    rationale: str
    next_evolve_observation: bool
    next_evolve_reward: bool
    slot_edits: Dict[str, str] = field(default_factory=dict)
    complexity_level: int = 1  # 1..4
    enforcement_notes: List[str] = field(default_factory=list)


def classify_inner_result(
    best_M1: float,
    best_shape: str,
    fitness_trajectory: List[float],
    best_M6: float,
    prior_complexity: int,
) -> Classification:
    """Independent classification from raw inner metrics.

    Thresholds match docs/v6_plan.md §3.2:
      - found_good: best M1 ≥ 0.05 AND shape ∈ {monotonic_rise, late_ramp}
      - partial_signal: best M1 ≥ 0.02 (any shape) OR M6 trajectory shows
        rising trend across iters
      - no_signal_simple: all-flat with prior complexity ≤ 2
      - no_signal_complex: all-flat with prior complexity ≥ 3
    """
    if best_M1 >= 0.05 and best_shape in ("monotonic_rise", "late_ramp"):
        return "found_good"

    if best_M1 >= 0.02:
        return "partial_signal"

    # Detect rising M6 trajectory (proxy for "something is starting to work").
    # Treat as partial_signal if best_M6 ≥ 0.20 — meaningful coverage progress
    # even when M1 stays sub-threshold at 1M frames.
    if best_M6 >= 0.20:
        return "partial_signal"

    # Otherwise, all-flat. Distinguish by prior complexity.
    if prior_complexity >= 3:
        return "no_signal_complex"
    return "no_signal_simple"


_VALID_SLOTS = {"guidance_observation", "guidance_reward", "guidance_shared"}


# Forbidden feature names — names of S3b-local's WINNING coordination
# signals. The meta-LLM is allowed to talk about families and operations
# (e.g. "expose proximity counts", "compute distances to nearest target")
# but NOT to literally name these solutions. This is the runtime
# enforcement of v6 plan §2 anti-cheat.
_FORBIDDEN_TOKENS = (
    "hold_signal",
    "hold_target_signal",
    "approach_signal",
    "approach_target_signal",
    "crowd_signal",
    "sparsity_signal",
    "gap_to_partner",
    "pair_formation_zone",
    "pair-formation zone",
    "nearest_unassigned",
    "nearest unassigned helper",
    "second agent needed",
    "the second agent needed",
    "I am the second",
)


def _redact_forbidden(text: str) -> tuple[str, list[str]]:
    """Replace forbidden tokens in a slot value with <REDACTED>.

    Returns the (possibly modified) text and a list of tokens that
    were redacted. Case-insensitive match.
    """
    if not text:
        return text, []
    found: list[str] = []
    out = text
    low = out.lower()
    for tok in _FORBIDDEN_TOKENS:
        tl = tok.lower()
        if tl in low:
            # Case-insensitive replace via regex with the original case
            import re as _re
            pattern = _re.compile(_re.escape(tok), _re.IGNORECASE)
            out, n = pattern.subn("<REDACTED>", out)
            if n > 0:
                found.append(tok)
                low = out.lower()
    return out, found


def enforce_decision(
    raw: V6MetaDecision,
    outer_idx: int,
    prior_complexity: int,
    prior_classification: Optional[Classification],
    code_classification: Classification,
) -> V6MetaDecision:
    """Apply v6 policy rules to the LLM's raw decision.

    Returns a (possibly amended) decision plus a list of enforcement
    notes describing any modifications. Code-side cross-check on
    classification: if LLM said `found_good` but code disagreed, force
    the code's classification.
    """
    notes: List[str] = []
    classification = raw.classification

    # 1. Cross-check classification — code wins on disagreement.
    if classification != code_classification:
        notes.append(
            f"classification overridden: LLM={classification!r} → "
            f"code={code_classification!r}"
        )
        classification = code_classification

    # 2. If found_good, force stop and clear slot edits.
    if classification == "found_good":
        if raw.next_mode != "stop":
            notes.append(
                f"next_mode forced to 'stop' (was {raw.next_mode!r}) — "
                f"classification=found_good"
            )
        return V6MetaDecision(
            classification="found_good",
            next_mode="stop",
            rationale=raw.rationale,
            next_evolve_observation=raw.next_evolve_observation,
            next_evolve_reward=raw.next_evolve_reward,
            slot_edits={},
            complexity_level=raw.complexity_level,
            enforcement_notes=notes,
        )

    # 3. Outer iter 0: force obs-only (next decision applies to outer iter 1).
    next_evolve_observation = raw.next_evolve_observation
    next_evolve_reward = raw.next_evolve_reward
    # Outer iter 0's decision configures outer iter 1. The first inner
    # call (outer_idx=0) is hardcoded to obs-only via the runner's init,
    # not via the decision system. Subsequent decisions are validated
    # against prior classification.
    # We still guard: if there's no obs evolution AND no reward
    # evolution, that's a no-op — force obs-only.
    if not next_evolve_observation and not next_evolve_reward:
        notes.append(
            "both flags False — forcing next_evolve_observation=True"
        )
        next_evolve_observation = True

    # 4. Reward unlock requires obs-only previously tried with
    # classification ∈ {no_signal_simple, partial_signal}.
    if next_evolve_reward and prior_classification not in (
        "no_signal_simple", "partial_signal", "no_signal_complex",
    ):
        notes.append(
            f"reward unlock denied: prior classification "
            f"{prior_classification!r} doesn't justify it; reverting to "
            f"obs-only"
        )
        next_evolve_reward = False
        next_evolve_observation = True

    # 5. Complexity monotone: only allow increases unless reset_simpler.
    complexity_level = max(1, min(4, int(raw.complexity_level)))
    if (complexity_level < prior_complexity and
            raw.next_mode != "reset_simpler"):
        notes.append(
            f"complexity decrease without reset_simpler: "
            f"{prior_complexity} → {complexity_level}; clamping to "
            f"{prior_complexity}"
        )
        complexity_level = prior_complexity

    # 6. Filter slot_edits: only valid slot names, only enabled flags.
    slot_edits: Dict[str, str] = {}
    for k, v in (raw.slot_edits or {}).items():
        if k not in _VALID_SLOTS:
            notes.append(f"unknown slot {k!r} — dropped")
            continue
        if k == "guidance_reward" and not next_evolve_reward:
            notes.append("guidance_reward edit dropped — flag is False")
            continue
        if k == "guidance_observation" and not next_evolve_observation:
            notes.append("guidance_observation edit dropped — flag is False")
            continue
        if not isinstance(v, str):
            continue
        # 6b. Anti-cheat: redact forbidden feature names from slot text.
        redacted, hits = _redact_forbidden(v)
        if hits:
            notes.append(
                f"ANTI-CHEAT: redacted {len(hits)} forbidden tokens from "
                f"{k}: {hits!r}"
            )
        slot_edits[k] = redacted

    if notes:
        for n in notes:
            _log.warning("v6 enforcement: %s", n)

    return V6MetaDecision(
        classification=classification,
        next_mode=raw.next_mode,
        rationale=raw.rationale,
        next_evolve_observation=next_evolve_observation,
        next_evolve_reward=next_evolve_reward,
        slot_edits=slot_edits,
        complexity_level=complexity_level,
        enforcement_notes=notes,
    )
