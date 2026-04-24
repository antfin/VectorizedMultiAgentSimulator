"""Level 1 Strategist — picks WHAT to improve.

Single LLM call. Reads the current run's history, the cross-run
``mutation_log.jsonl`` summary, top-k candidate evidence, and a
per-seed strategy bias. Outputs a structured ``StrategyCard`` (YAML)
that the Level 2 Editor consumes.

The Strategist is deliberately limited: it does NOT write any prompt
text. Its job is picking:
  1. A domain (reward / observation / shared / both)
  2. A concrete target slot (guidance_reward | guidance_observation |
     guidance_shared)
  3. 1-2 focus ideas
  4. A list of patterns to AVOID (learned from mutation_log)

See docs/lero_metaprompt_v2_plan.md §5 for the card schema.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

from .failmode import FailMode
from .mutation_log import MutationLogEntry, summarize_for_prompt
from .trigger import TemplateRecord


_log = logging.getLogger("rendezvous.lero.mp.strategy")


# Canonical sub-slot names the Strategist may pick. The Editor (Level 2)
# will refuse to edit anything outside this set.
VALID_TARGET_SLOTS = frozenset({
    "guidance_shared",
    "guidance_reward",
    "guidance_observation",
})

VALID_DOMAINS = frozenset({"reward", "observation", "shared", "both"})


VALID_SIGNAL_TIERS = frozenset({"scalar", "fingerprint", "curve_shape"})


@dataclass
class StrategyCard:
    """Output of Level 1. Consumed by Level 2 mutation.editor.

    v3 adds ``include_signals`` — the Strategist's knob for deciding
    which tiers of behavioral feedback get forwarded to the Editor
    and next-iteration inner-loop feedback (LERO-MP v3 §4.1).
    """

    target_domain: Literal["reward", "observation", "shared", "both"]
    target_slot: str  # one of VALID_TARGET_SLOTS
    focus: List[str] = field(default_factory=list)
    avoid: List[str] = field(default_factory=list)
    confidence: Literal["small", "medium", "large"] = "medium"
    rationale: str = ""
    include_signals: List[str] = field(default_factory=lambda: ["scalar"])
    signal_rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_domain": self.target_domain,
            "target_slot": self.target_slot,
            "focus": list(self.focus),
            "avoid": list(self.avoid),
            "confidence": self.confidence,
            "rationale": self.rationale,
            "include_signals": list(self.include_signals),
            "signal_rationale": self.signal_rationale,
        }


class StrategyParseError(ValueError):
    """Raised when the Level 1 LLM output doesn't parse into a
    valid StrategyCard. Callers can retry or abort."""


# ── seed-level bias ────────────────────────────────────────────

# Hard-coded 3-way (reproducible). Set ``seed % 3``; the Strategist
# reads the bias as a *soft* preference, not a constraint.
SEED_STRATEGY_BIAS: Dict[int, str] = {
    0: "observation_first",
    1: "reward_first",
    2: "exploratory",
}


def bias_for_seed(seed: int) -> str:
    return SEED_STRATEGY_BIAS[seed % 3]


def _bias_hint(bias: str) -> str:
    if bias == "observation_first":
        return (
            "This seed's soft preference is to explore OBSERVATION "
            "feature engineering first. Override only if the mutation_log "
            "shows observation attempts already failed AND rewards look "
            "unstable."
        )
    if bias == "reward_first":
        return (
            "This seed's soft preference is to explore REWARD shaping "
            "first. Override only if the mutation_log shows reward "
            "attempts already failed AND observations look impoverished."
        )
    if bias == "exploratory":
        return (
            "This seed's preference is EXPLORATORY: pick whichever domain "
            "has fewer prior attempts in the mutation_log, and prefer "
            "patterns not yet tried."
        )
    return "(no specific bias; pick based on evidence alone)"


# ── Level 1 prompt ────────────────────────────────────────────

def build_strategist_prompt(
    history: Sequence[TemplateRecord],
    mutation_log_entries: Sequence[MutationLogEntry],
    top_candidates: Sequence[Dict[str, Any]],
    seed_bias: str,
    fail_mode: FailMode,
    fairness_slot_excerpt: str = "",
) -> str:
    """Assemble the Level 1 prompt. Pure function, no I/O."""

    # Compact history summary (one line per record)
    if history:
        hist_lines = []
        for r in history[-5:]:
            final = (
                f"final_M1={r.best_final_M1:.3f}  "
                if r.best_final_M1 is not None else ""
            )
            hist_lines.append(
                f"  {r.template_version:<40}  "
                f"peak_M1={r.best_peak_M1:.3f}  {final}"
                f"M6={r.best_M6:.3f}  fail_mode={r.fail_mode.value}"
            )
        hist_block = "\n".join(hist_lines)
    else:
        hist_block = "  (no prior records on this run)"

    # Cross-run memory summary
    log_block = summarize_for_prompt(list(mutation_log_entries))

    # Candidate-aggregate signal (very compact — we don't need full code
    # at Level 1; Level 2 will see it)
    if top_candidates:
        valid = [c for c in top_candidates if "_error" not in c]
        if valid:
            m6_vals = [c.get("M6_coverage_progress", 0.0) for c in valid]
            m1_vals = [c.get("M1_success_rate", 0.0) for c in valid]
            m2_vals = [c.get("M2_avg_return", 0.0) for c in valid]
            cand_block = (
                f"  N valid: {len(valid)}\n"
                f"  M1 range [{min(m1_vals):.3f}, {max(m1_vals):.3f}]\n"
                f"  M6 range [{min(m6_vals):.3f}, {max(m6_vals):.3f}]\n"
                f"  M2 range [{min(m2_vals):+.2f}, {max(m2_vals):+.2f}]"
            )
        else:
            cand_block = "  (all top candidates errored)"
    else:
        cand_block = "  (no candidates)"

    fairness_ref = (
        f"\n[FAIRNESS CONTRACT — already enforced, DO NOT re-encode "
        f"into your card]\n{fairness_slot_excerpt.strip()}\n"
        if fairness_slot_excerpt else ""
    )

    return f"""[ROLE]
You are the LERO-MP Strategist. Your job is ONLY to decide WHAT to
improve next. You do not write prompt text — another LLM will do that.
You output a structured YAML StrategyCard.

[SEED BIAS]
{_bias_hint(seed_bias)}

[DOMINANT FAIL MODE (from the most recent inner run)]
{fail_mode.value}

[CURRENT-RUN HISTORY  —  template records, most recent last]
{hist_block}

[CROSS-RUN MUTATION LOG  —  prior attempts, their deltas, and verdicts]
{log_block}

[CANDIDATE-AGGREGATE STATS  —  numeric evidence from the last inner run]
{cand_block}
{fairness_ref}
[TASK]
Choose ONE of these three sub-slots to have the Editor rewrite:
  - guidance_shared      → applies to BOTH reward + observation code
  - guidance_reward      → reward function shaping / magnitude / stability
  - guidance_observation → observation feature engineering

Pick the ONE whose targeted improvement has the highest expected
delta to peak_M1 given the evidence above. Use the mutation_log to
AVOID patterns that already underperformed: any entry with
verdict ∈ {{regression, collapse}} should be listed in `avoid`.

[SIGNAL SELECTION — keep prompts lean]
You also choose which tiers of behavioral feedback are forwarded
downstream to the Editor and next-iteration inner-LLM feedback:

  - "scalar":       M1/M2/M3/M4/M6/M8/M9 per candidate — cheap, always
                    useful. ALWAYS include this.
  - "fingerprint":  coverage-over-time trace (start/peak/end M1 and
                    shape tag). Include only when training-stability
                    issues are visible (e.g. reward-hack shape, peak
                    collapse).
  - "curve_shape":  a single shape tag of the learning curve.
                    Include when candidates differ in stability
                    (e.g. oscillating vs monotonic) and that
                    distinction should drive the edit.

Default is ["scalar"] only. Add tiers ONLY when they'll help the
Editor pick a concrete feature — not just to "have more info".

Output EXACTLY this YAML block, nothing else, no prose outside it:

```yaml
target_domain: reward | observation | shared | both
target_slot: guidance_reward | guidance_observation | guidance_shared
focus:
  - "<one specific pattern or feature to encourage, 1 line>"
  - "<a second one, only if tightly related to the first>"
avoid:
  - "<pattern that scored regression/collapse previously, if any>"
confidence: small | medium | large
rationale: |
  <2-4 sentences citing specific evidence: which record, which
   verdict, which feature-delta. No generic statements.>
include_signals:
  - scalar
  # optionally also "fingerprint" or "curve_shape" — cite the reason below
signal_rationale: "<1 sentence if include_signals deviates from the default>"
```
"""


# ── response parsing ──────────────────────────────────────────

_YAML_BLOCK_RE = re.compile(
    r"```yaml\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE,
)


def parse_strategy_card(text: str) -> StrategyCard:
    """Extract + validate a StrategyCard from Level 1 output.

    Raises StrategyParseError on any structural problem so the outer
    loop can retry-or-abort rather than proceed with half-parsed state.
    """
    if yaml is None:
        raise RuntimeError("PyYAML is required for StrategyCard parsing")

    m = _YAML_BLOCK_RE.search(text)
    if not m:
        # Be permissive: try parsing the whole text as YAML.
        body = text.strip()
    else:
        body = m.group(1)

    try:
        data = yaml.safe_load(body)
    except yaml.YAMLError as e:
        raise StrategyParseError(f"YAML parse failed: {e}")

    if not isinstance(data, dict):
        raise StrategyParseError(
            f"Expected a YAML mapping, got {type(data).__name__}"
        )

    dom = data.get("target_domain")
    slot = data.get("target_slot")
    if dom not in VALID_DOMAINS:
        raise StrategyParseError(
            f"target_domain={dom!r} not in {sorted(VALID_DOMAINS)}"
        )
    if slot not in VALID_TARGET_SLOTS:
        raise StrategyParseError(
            f"target_slot={slot!r} not in {sorted(VALID_TARGET_SLOTS)}"
        )

    focus = data.get("focus") or []
    avoid = data.get("avoid") or []
    if not isinstance(focus, list) or not all(isinstance(x, str) for x in focus):
        raise StrategyParseError("focus must be a list of strings")
    if not isinstance(avoid, list) or not all(isinstance(x, str) for x in avoid):
        raise StrategyParseError("avoid must be a list of strings")

    conf = data.get("confidence", "medium")
    if conf not in {"small", "medium", "large"}:
        raise StrategyParseError(
            f"confidence={conf!r} must be one of small/medium/large"
        )

    rationale = str(data.get("rationale", "")).strip()

    # v3: include_signals field is OPTIONAL for backward compat with
    # existing LLM outputs that don't yet emit it. Default to
    # ["scalar"] (minimal noise) when absent or invalid.
    raw_tiers = data.get("include_signals") or ["scalar"]
    if not isinstance(raw_tiers, list):
        raw_tiers = ["scalar"]
    include_signals = [
        t for t in raw_tiers
        if isinstance(t, str) and t in VALID_SIGNAL_TIERS
    ]
    if not include_signals:
        include_signals = ["scalar"]
    signal_rationale = str(data.get("signal_rationale", "") or "").strip()

    return StrategyCard(
        target_domain=dom,
        target_slot=slot,
        focus=[f.strip() for f in focus][:3],  # keep ≤ 3
        avoid=[a.strip() for a in avoid][:5],  # keep ≤ 5
        confidence=conf,
        rationale=rationale,
        include_signals=include_signals,
        signal_rationale=signal_rationale,
    )


# ── main entry point ──────────────────────────────────────────

StrategistLLMCallable = Callable[[List[Dict[str, str]]], str]


def strategize(
    history: Sequence[TemplateRecord],
    mutation_log_entries: Sequence[MutationLogEntry],
    top_candidates: Sequence[Dict[str, Any]],
    seed_bias: str,
    fail_mode: FailMode,
    meta_llm_call: StrategistLLMCallable,
    fairness_slot_excerpt: str = "",
) -> StrategyCard:
    """Level 1 → StrategyCard. Encapsulates prompt build + LLM call +
    parse so callers just pass evidence and get a typed decision.
    """
    prompt = build_strategist_prompt(
        history=history,
        mutation_log_entries=mutation_log_entries,
        top_candidates=top_candidates,
        seed_bias=seed_bias,
        fail_mode=fail_mode,
        fairness_slot_excerpt=fairness_slot_excerpt,
    )
    messages = [
        {"role": "system", "content":
            "You are a careful research engineer deciding how to "
            "improve a multi-agent RL prompt template. Output only "
            "the requested YAML block."},
        {"role": "user", "content": prompt},
    ]
    response = meta_llm_call(messages)
    card = parse_strategy_card(response)
    _log.info(
        "Level 1 Strategist: domain=%s slot=%s confidence=%s focus=%r",
        card.target_domain, card.target_slot, card.confidence, card.focus,
    )
    return card
