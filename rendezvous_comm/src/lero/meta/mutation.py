"""Meta-LLM prompt mutation.

Given the current template, a target slot, and the history of past
templates with their inner-loop outcomes, ask a meta-LLM to propose
new text for that single slot. The LLM output is parsed into
(new_slot_content, rationale) and materialized as a fresh prompt
version directory via ``provenance.materialize_mutation``.

The meta-prompt shape mirrors plan §7.1: OPRO-style sorted history,
PromptAgent-style error counts, LCP-style contrastive analysis, and
the single-slot edit contract (with the fairness slot shown hashed as
FROZEN so the meta-LLM never tries to edit it).

Output contract asked of the meta-LLM:

    Rationale: <1-3 sentences>
    Expected-improvement: small | medium | large

    <<<NEW_SLOT_BEGIN>>>
    ... new text for the slot ...
    <<<NEW_SLOT_END>>>

Structured delimiters are more robust than nested YAML for slots that
may contain code fences or indentation.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..prompts.loader import PromptLoader
from .failmode import FailMode
from .provenance import (
    lineage,
    materialize_mutation,
    propose_version_name,
    sha256_text,
)
from .trigger import TemplateRecord

# Imported lazily to avoid circular imports at module load time.
# from .strategy import StrategyCard

_log = logging.getLogger("rendezvous.lero.mp")


# Output delimiters — checked verbatim in parse_mutation_response.
SLOT_BEGIN = "<<<NEW_SLOT_BEGIN>>>"
SLOT_END = "<<<NEW_SLOT_END>>>"

# Reasonable bounds for a slot rewrite. The upper bound defends against
# the meta-LLM echoing the entire prompt back; the lower bound catches
# empty or stub outputs.
MIN_SLOT_CHARS = 10
MAX_SLOT_CHARS = 20_000


@dataclass
class MutationResult:
    """What the outer loop gets back from ``propose_new_template``."""

    new_version: str
    parent_version: str
    target_slot: str
    new_slot_content: str
    rationale: str
    expected_improvement: str  # "small" | "medium" | "large" (free-form fallback)


class MutationParseError(ValueError):
    """Meta-LLM output did not contain the expected slot delimiters
    or violated the size bounds. Callers should retry or abort.
    """


# ── prompt assembly ─────────────────────────────────────────────

def _format_history(history: Sequence[TemplateRecord], limit: int = 5) -> str:
    """OPRO-style sorted list: worst → best. The LLM sees the trend."""
    if not history:
        return "(no prior templates)"
    tail = list(history[-limit:])
    tail.sort(key=lambda r: r.best_peak_M1)
    lines = []
    for r in tail:
        final = (
            f"final_M1={r.best_final_M1:.3f}  "
            if r.best_final_M1 is not None else ""
        )
        lines.append(
            f"  {r.template_version:<40}  "
            f"peak_M1={r.best_peak_M1:.3f}  {final}"
            f"M6={r.best_M6:.3f}  M2={r.best_M2:.2f}  "
            f"σseed={r.seed_M1_std:.3f}  fail_mode={r.fail_mode.value}"
        )
    return "\n".join(lines)


def _contrastive_analysis(history: Sequence[TemplateRecord]) -> str:
    """Pick the best and worst records, describe what differs between
    them *at the slot level*. Gives the meta-LLM a concrete signal."""
    if len(history) < 2:
        return "(need ≥2 prior templates for contrast)"
    ranked = sorted(history, key=lambda r: r.best_peak_M1)
    worst, best = ranked[0], ranked[-1]
    if worst is best:
        return "(all records identical in peak_M1)"
    delta_m1 = best.best_peak_M1 - worst.best_peak_M1
    slot = best.mutation_target_slot or "(unknown)"
    return (
        f"Best ({best.template_version}) vs worst "
        f"({worst.template_version}) differ in slot='{slot}' "
        f"(ΔpeakM1 = {delta_m1:+.3f}). "
        f"Best's rationale: {best.mutation_rationale or '(n/a)'}"
    )


_STATE_KEYS_SIGNAL = [
    "lidar_targets", "lidar_agents", "agent_pos", "agent_vel", "messages",
    "covering_range", "agents_per_target_required",
]


def _feature_signals(code: str) -> Dict[str, int]:
    """Count occurrences of canonical feature-engineering patterns in a
    candidate's source. Gives the meta-LLM concrete evidence about what
    the top candidates are (and aren't) using.
    """
    if not code:
        return {}
    out: Dict[str, int] = {}
    for k in _STATE_KEYS_SIGNAL:
        out[k] = code.count(f'"{k}"') + code.count(f"'{k}'")
    # Coordination-signal patterns from LERO S3b-local (hold/approach/
    # crowd/sparsity) — made k=2 reach M1=0.88.
    out["uses_gap_feature"] = int(
        "gap" in code.lower() or
        "second" in code.lower() or
        (".topk(" in code and "2" in code)
    )
    out["uses_proximity_count"] = int(
        "count" in code.lower() and (
            "lidar" in code.lower() or "< covering_range" in code
        )
    )
    out["uses_hold_or_approach"] = int(
        "hold" in code.lower() or "approach" in code.lower() or
        "crowd" in code.lower()
    )
    out["uses_intensity"] = int(
        "1.0/" in code or "1.0 /" in code or "reciprocal" in code.lower() or
        "torch.exp(-" in code
    )
    return out


def _format_top_candidates(
    top_candidates: Sequence[Dict[str, Any]], limit: int = 3,
) -> str:
    """Compact dump of top-k candidate code snippets + metrics + an
    auto-computed feature-signal summary so the meta-LLM has concrete
    evidence to cite (not just code it may or may not read carefully)."""
    if not top_candidates:
        return "(no candidates reported)"
    blocks = []
    for i, c in enumerate(top_candidates[:limit]):
        m1 = c.get("M1_success_rate", 0.0)
        m2 = c.get("M2_avg_return", 0.0)
        m6 = c.get("M6_coverage_progress", 0.0)
        peak = c.get("peak_M1")
        final = c.get("final_M1")
        code = c.get("obs_code") or c.get("reward_code") or ""
        sig = _feature_signals(code)
        parts = [
            f"### Candidate {i + 1}",
            f"  M1={m1:.3f}  M2={m2:.2f}  M6={m6:.3f}" + (
                f"  peak_M1={peak:.3f}  final_M1={final:.3f}"
                if peak is not None and final is not None else ""
            ),
        ]
        # Feature-signal summary (what the candidate's code references)
        state_refs = {k: sig[k] for k in _STATE_KEYS_SIGNAL if sig.get(k, 0)}
        flags = [
            f"gap_feature={sig.get('uses_gap_feature', 0)}",
            f"proximity_count={sig.get('uses_proximity_count', 0)}",
            f"hold_or_approach={sig.get('uses_hold_or_approach', 0)}",
            f"intensity={sig.get('uses_intensity', 0)}",
        ]
        parts.append(
            f"  state_refs={state_refs}  {' '.join(flags)}"
        )
        if c.get("reward_code"):
            parts.append("```python")
            parts.append(c["reward_code"].strip())
            parts.append("```")
        if c.get("obs_code"):
            parts.append("```python")
            parts.append(c["obs_code"].strip())
            parts.append("```")
        blocks.append("\n".join(parts))
    return "\n\n".join(blocks)


def _candidate_spread(top_candidates: Sequence[Dict[str, Any]]) -> str:
    """Quick aggregate stats the LLM can cite directly in its rationale.

    We do the arithmetic once so the LLM doesn't have to — its track
    record with numerical reasoning in long prompts is shaky.
    """
    if not top_candidates:
        return "(no candidates to aggregate)"
    valid = [c for c in top_candidates if "_error" not in c]
    if not valid:
        return "(all top candidates errored)"
    m6_vals = [c.get("M6_coverage_progress", 0.0) for c in valid]
    m2_vals = [c.get("M2_avg_return", 0.0) for c in valid]
    m1_vals = [c.get("M1_success_rate", 0.0) for c in valid]
    # Which patterns appear in the BEST vs WORST (by M6)?
    ranked = sorted(
        valid, key=lambda c: c.get("M6_coverage_progress", 0.0), reverse=True,
    )
    best, worst = ranked[0], ranked[-1]
    best_sig = _feature_signals(
        best.get("obs_code") or best.get("reward_code") or ""
    )
    worst_sig = _feature_signals(
        worst.get("obs_code") or worst.get("reward_code") or ""
    )
    deltas = {
        k: best_sig.get(k, 0) - worst_sig.get(k, 0)
        for k in ("uses_gap_feature", "uses_proximity_count",
                  "uses_hold_or_approach", "uses_intensity")
    }
    return (
        f"  N valid candidates: {len(valid)}\n"
        f"  M1 range: [{min(m1_vals):.3f}, {max(m1_vals):.3f}]  "
        f"mean={sum(m1_vals) / len(m1_vals):.3f}\n"
        f"  M6 range: [{min(m6_vals):.3f}, {max(m6_vals):.3f}]  "
        f"mean={sum(m6_vals) / len(m6_vals):.3f}\n"
        f"  M2 range: [{min(m2_vals):.2f}, {max(m2_vals):.2f}]  "
        f"mean={sum(m2_vals) / len(m2_vals):.2f}\n"
        f"  Best-M6 vs Worst-M6 feature deltas: {deltas}\n"
        f"  (positive delta = the BEST candidate uses that pattern more "
        f"than the WORST)"
    )


def build_meta_prompt(
    parent_version: str,
    target_slot: str,
    history: Sequence[TemplateRecord],
    top_candidates: Sequence[Dict[str, Any]],
    fail_mode: FailMode,
    loader: Optional[PromptLoader] = None,
) -> str:
    """Assemble the structured prompt sent to the meta-LLM.

    ``loader`` is optional to make this unit-testable without a real
    filesystem. When provided, the current slot text + fairness-hash
    are embedded; when None, we use placeholders.
    """
    if loader is None:
        cur_slot_text = f"(would inline current {target_slot} slot here)"
        fairness_hash = "(unknown)"
        frozen_slots: List[str] = []
    else:
        cur_slot_text = loader.slot_text(target_slot)
        frozen_slots = loader.frozen_slot_names()
        fairness_hash = ""
        if "fairness" in frozen_slots:
            fairness_hash = sha256_text(loader.slot_text("fairness"))

    frozen_line = (
        f"frozen (hash={fairness_hash[:12]}…, DO NOT EDIT)"
        if fairness_hash else "frozen slots: none"
    )

    # Inline the full fairness slot so the LLM can see what rules it
    # must NOT restate. Empirically (2026-04-22) the meta-LLM would
    # otherwise "play safe" by reiterating the fairness contract as
    # generic guidance, producing no real signal for the inner loop.
    if loader is not None and "fairness" in frozen_slots:
        fairness_text = loader.slot_text("fairness")
    else:
        fairness_text = "(fairness slot not available in this context)"

    return f"""[ROLE]
You are a prompt engineer analyzing *actual RL-training evidence* to
improve the instruction template that generates PyTorch reward /
observation code for a multi-agent RL task. Your job is diagnostic,
not advisory. Read the evidence, identify 1-3 *related*, *concrete*
issues visible in the candidate code or metrics, and propose a
targeted edit that addresses them.

[HARD CONSTRAINTS — NEVER VIOLATE]
- The `fairness` slot is FROZEN. You may read it but never edit it:
  {frozen_line}
- The slot you may edit this round is: `{target_slot}`. Do NOT edit
  any other slot.
- Your edit MUST add guidance BEYOND what the fairness slot already
  says. Do NOT restate, paraphrase, or re-enumerate the fairness
  rules — those are already in every prompt, and duplicating them
  wastes context without helping the inner-loop LLM produce better
  code. If your draft reads like a rephrasing of the fairness slot,
  throw it out and think harder.
- Rewards must stay within |r| ≤ 50 (already enforced at runtime).

[FAIRNESS SLOT — already shown to the inner-loop LLM, DO NOT RESTATE]
------------------------------------------------------------
{fairness_text.strip()}
------------------------------------------------------------

[OBJECTIVE]
Maximize peak-M1 primary, M6 (coverage progress) tie-break. Secondary:
avoid reward-hacking (peak-vs-final gap < 0.20), NaN crashes, dim
mismatches.

[DOMINANT FAIL MODE (from the last inner run)]
{fail_mode.value}

[HISTORY — prior templates sorted by peak_M1 ASCENDING]
{_format_history(history)}

[CONTRASTIVE ANALYSIS]
{_contrastive_analysis(history)}

[CANDIDATE-AGGREGATE STATS — the numerical evidence]
{_candidate_spread(top_candidates)}

[TOP CANDIDATES UNDER CURRENT TEMPLATE — code + feature signals]
{_format_top_candidates(top_candidates)}

[REFERENCE TECHNIQUES KNOWN TO HELP ON THIS TASK CLASS]
These are patterns that worked in prior rendezvous experiments. Use
them as *prompts for your reflection*, not text to quote:
  - Nearest vs 2nd-nearest target/agent distances + the gap between
    them (tells the policy "is this target isolated vs contested?")
  - Proximity counts: how many lidar rays return distance < covering_range
  - Intensity features: sum of 1/dist over close rays (sharper signal
    than raw distances)
  - Coordination flags: hold_signal = (target_near AND agent_near) —
    lets an agent that arrived first wait for a partner rather than
    overshooting
  - Crowd / sparsity signals from proximity counts
  - For rewards: smooth shaping on coverage-progress delta instead of
    spiky terminal bonuses

[CURRENT `{target_slot}` SLOT TEXT — what you are replacing]
------------------------------------------------------------
{cur_slot_text}
------------------------------------------------------------

[TASK]
Step 1 — DIAGNOSIS. Read the candidate code and the aggregate stats.
Identify 1-3 *related* issues visible in the evidence. Good issues
are of the form:
  "The top candidates' M6 varies from X to Y, but all of them omit
   <specific feature>. Adding it might narrow that spread."
  "Candidate 2 has M2=... but failed with error ...; the pattern
   suggests <specific cause>."
Bad issues are generic — "avoid NaN" or "use local sensors" — those
are already in the fairness slot.

Step 2 — EDIT. Write a new `{target_slot}` slot whose guidance
addresses those 1-3 issues (and ONLY those). Mention specific
sensors, features, or patterns by name.

Return EXACTLY this format (no other prose, no YAML fences around
the whole thing):

Diagnosis:
1. <issue 1, cite specific metrics or code lines>
2. <issue 2 if related>
3. <issue 3 if related>

Rationale: <1-3 sentences mapping the diagnosis to the edit>
Expected-improvement: small | medium | large

{SLOT_BEGIN}
<the complete new text of the `{target_slot}` slot — targeted,
specific, and non-overlapping with the fairness slot>
{SLOT_END}

Keep the new slot concise ({max(len(cur_slot_text), 200)} chars order
of magnitude; hard cap is 20000). Do not restate fairness rules.
"""


# ── response parsing ────────────────────────────────────────────

_RATIONALE_RE = re.compile(r"Rationale:\s*(.+?)(?=\n[A-Z][a-z]+-?[a-z]*:|\n{2,}|\Z)", re.DOTALL)
_EXPECTED_RE = re.compile(r"Expected-improvement:\s*(small|medium|large)", re.IGNORECASE)
_DIAGNOSIS_RE = re.compile(r"Diagnosis:\s*(.+?)(?=\nRationale:|\n{2,}|\Z)", re.DOTALL)
_SLOT_RE = re.compile(
    rf"{re.escape(SLOT_BEGIN)}\s*\n?(.*?)\n?\s*{re.escape(SLOT_END)}",
    re.DOTALL,
)


# Heuristic fairness-restatement detector. Triggered when the new slot
# is essentially a rephrasing of the frozen fairness contract (which
# the inner-loop LLM already sees separately). See the 2026-04-22
# quick-run finding where 3/3 seeds produced near-identical generic
# restatements.
_FAIRNESS_RESTATEMENT_MARKERS = [
    ("local sensor", "oracle"),          # both ⇒ fairness paraphrase
    ("local sensor", "clamp"),
    ("reward", "|r| <= 50"),
]


def _is_fairness_restatement(new_slot: str) -> bool:
    """True if the new slot looks like a paraphrase of the fairness
    slot — i.e., contains all markers from at least one tuple below
    and adds nothing beyond them."""
    low = new_slot.lower()
    for pair in _FAIRNESS_RESTATEMENT_MARKERS:
        if all(m.lower() in low for m in pair):
            # Count how much non-boilerplate content there is. If the
            # slot is short AND hits every fairness marker, it's a
            # restatement. We keep this permissive — longer slots with
            # specific feature names (lidar_targets, hold_signal, gap,
            # intensity, ...) pass.
            specific_hits = sum(
                low.count(tok) for tok in (
                    "lidar_targets", "lidar_agents", "hold_signal",
                    "approach_signal", "gap", "intensity",
                    "2nd-nearest", "second-nearest", "crowd",
                    "sparsity", "topk", "potential",
                )
            )
            if specific_hits == 0 and len(new_slot) < 1500:
                return True
    return False


def parse_mutation_response(
    text: str, target_slot: str,
) -> Tuple[str, str, str]:
    """Extract (new_slot_content, rationale, expected_improvement).

    Raises MutationParseError on any structural problem so the outer
    loop can either retry or abort — we do not guess partial output.
    """
    m = _SLOT_RE.search(text)
    if not m:
        raise MutationParseError(
            f"Meta-LLM output missing {SLOT_BEGIN}…{SLOT_END} delimiters "
            f"for slot '{target_slot}'."
        )
    new_slot = m.group(1)
    # Normalise: ensure trailing newline (matches file convention).
    if not new_slot.endswith("\n"):
        new_slot = new_slot + "\n"

    n = len(new_slot)
    if n < MIN_SLOT_CHARS:
        raise MutationParseError(
            f"New slot is suspiciously small ({n} chars < {MIN_SLOT_CHARS})."
        )
    if n > MAX_SLOT_CHARS:
        raise MutationParseError(
            f"New slot exceeds size limit ({n} > {MAX_SLOT_CHARS})."
        )
    if _is_fairness_restatement(new_slot):
        raise MutationParseError(
            "New slot reads as a paraphrase of the fairness slot "
            "(contains its markers without any specific feature / "
            "coordination-signal references). The inner-loop LLM "
            "already sees the fairness slot verbatim; restating it "
            "adds no signal. Retry with a targeted edit that cites "
            "specific features (e.g. lidar_targets proximity count, "
            "nearest/2nd-nearest gaps, hold/approach/crowd flags, "
            "intensity) or specific reward-shape patterns."
        )

    r = _RATIONALE_RE.search(text)
    rationale = r.group(1).strip() if r else ""
    if not rationale:
        # A missing rationale is a warning, not fatal — the meta-loop
        # can still proceed, but we log so the provenance trail shows it.
        _log.warning(
            "Meta-LLM output had no parseable rationale for slot '%s'.",
            target_slot,
        )
        rationale = "(no rationale provided)"

    e = _EXPECTED_RE.search(text)
    expected = e.group(1).lower() if e else "unspecified"

    return new_slot, rationale, expected


# ── main entry point ────────────────────────────────────────────

# Type alias for a callable that mimics LLMClient.generate(messages, n=1)[0].
# Keeping this as a plain Callable makes mutation.py trivially mockable.
MetaLLMCallable = Callable[[List[Dict[str, str]]], str]


def build_editor_prompt(
    parent_version: str,
    strategy_card,  # StrategyCard (lazy import)
    top_candidates: Sequence[Dict[str, Any]],
    loader: Optional[PromptLoader] = None,
    prior_slot_versions: Optional[Sequence[Any]] = None,
    behavioral_block: str = "",
) -> str:
    """Level 2 prompt. Receives a StrategyCard and rewrites ONLY the
    slot it names. Does not re-decide domain or strategy — the
    Strategist already did that.

    ``prior_slot_versions`` is an optional list of MutationLogEntry
    instances that previously edited this same slot. When provided,
    the Editor sees what was tried before + the verdicts, and is
    told to DIVERGE rather than duplicate. Fixes the v2 dry-run
    finding that Editor outputs on the same slot converge across
    seeds.
    """
    target_slot = strategy_card.target_slot
    if loader is None:
        cur_slot_text = f"(would inline current {target_slot} slot here)"
        fairness_text = "(fairness slot not available in this context)"
        fairness_hash = "(unknown)"
    else:
        cur_slot_text = loader.slot_text(target_slot)
        fairness_hash = ""
        if "fairness" in loader.frozen_slot_names():
            fairness_text = loader.slot_text("fairness")
            fairness_hash = sha256_text(fairness_text)
        else:
            fairness_text = "(no frozen fairness slot)"

    focus_block = "\n".join(f"  - {f}" for f in strategy_card.focus) or "  (none)"
    avoid_block = "\n".join(f"  - {a}" for a in strategy_card.avoid) or "  (none)"

    # Prior versions of the SAME slot (from mutation_log). If this
    # slot has been edited before, show what was tried and how it
    # scored so the Editor DIVERGES rather than rewriting the same
    # text. Without this, independent seeds that land on the same
    # slot keep proposing near-identical content.
    prior_block = ""
    if prior_slot_versions:
        lines = ["[PRIOR VERSIONS OF THIS SAME SLOT — diverge, don't duplicate]"]
        for i, e in enumerate(prior_slot_versions, 1):
            verdict = getattr(e, "verdict", None) or "pending"
            dp = getattr(e, "delta_peak_M1", None)
            dp_s = f"Δpeak_M1={dp:+.3f}" if dp is not None else "outcome pending"
            excerpt = (getattr(e, "slot_content_excerpt", "") or "")[:300]
            lines.append(
                f"### version {i}  "
                f"({getattr(e, 'new_version', '?')})  "
                f"verdict={verdict}  {dp_s}"
            )
            lines.append(f"    {excerpt}")
        lines.append(
            "\nYour output must be SUBSTANTIVELY different from every "
            "prior version above. If multiple priors tried the same "
            "pattern and scored regression/collapse, avoid that pattern "
            "entirely. If a prior scored marginal/strong improvement, "
            "build on it instead of repeating it verbatim."
        )
        prior_block = "\n".join(lines) + "\n"

    return f"""[ROLE]
You are the LERO-MP Editor. A separate Strategist LLM has already
decided what to improve. Your job is to rewrite ONE specific sub-slot
with targeted, evidence-cited text that implements the Strategist's
focus ideas and avoids the patterns it flagged.

[HARD CONSTRAINTS — NEVER VIOLATE]
- Edit ONLY the `{target_slot}` slot. Do not propose changes to any
  other slot; the outer loop applies single-slot edits.
- The `fairness` slot is FROZEN (hash={fairness_hash[:12]}…, DO NOT
  EDIT, DO NOT RESTATE). It is ALREADY in every rendered prompt.
  Restating it wastes tokens without helping the inner-loop LLM.
- Rewards must stay within |r| ≤ 50 (already enforced at runtime;
  don't need to repeat this).

[FAIRNESS SLOT — shown here so you know what NOT to restate]
------------------------------------------------------------
{fairness_text.strip()}
------------------------------------------------------------

[STRATEGY CARD  —  your instructions from the Strategist]
  target_domain: {strategy_card.target_domain}
  target_slot:   {target_slot}
  confidence:    {strategy_card.confidence}
  rationale:     {strategy_card.rationale}

[FOCUS — implement these specifically, 1-2 ideas only]
{focus_block}

[AVOID — the Strategist has ruled these out based on prior evidence]
{avoid_block}

[CURRENT `{target_slot}` TEXT — what you are replacing]
------------------------------------------------------------
{cur_slot_text}
------------------------------------------------------------

{prior_block}
[TOP CANDIDATES — evidence you may cite]
{_format_top_candidates(top_candidates)}

[BEHAVIORAL SIGNALS — filtered per the Strategist's include_signals]
{behavioral_block or "(no extra behavioral signals forwarded)"}

[OUTPUT FORMAT]
Write the new `{target_slot}` text. Requirements:
  - Target ONLY the Strategist's focus ideas. Don't wander.
  - Name specific features / patterns / signals by exact identifier
    (e.g. `lidar_targets`, `hold_signal`, `2nd-nearest`, `gap`,
    `proximity_count`, `intensity`, `potential shaping`, …).
  - Do NOT restate the fairness contract.
  - Keep the new slot concise (roughly the same order of magnitude
    as the focus — a few short paragraphs, not an essay).

Return EXACTLY this format:

Rationale: <1-3 sentences mapping the focus to your edit>
Expected-improvement: small | medium | large

{SLOT_BEGIN}
<the complete new text of the `{target_slot}` slot — targeted,
specific, non-overlapping with fairness>
{SLOT_END}
"""


def propose_new_template(
    parent_version: str,
    target_slot: str,
    history: Sequence[TemplateRecord],
    top_candidates: Sequence[Dict[str, Any]],
    fail_mode: FailMode,
    meta_llm_call: MetaLLMCallable,
    outer_iter: int,
    prompts_dir: Optional[Path] = None,
    generated_by: str = "",
    strategy_card=None,  # optional StrategyCard for v2 two-level pipeline
    prior_slot_versions: Optional[Sequence[Any]] = None,
    behavioral_block: str = "",
    critic_llm_call: Optional[MetaLLMCallable] = None,
) -> MutationResult:
    """Compose meta-prompt → call LLM → parse → materialize.

    Args:
        parent_version: Current prompt version directory name.
        target_slot: Slot to edit (picked by ``failmode.pick_slot_to_edit``).
        history: Past TemplateRecords in chronological order.
        top_candidates: Top-k inner-loop candidate summaries
            (each has ``M1_success_rate``, ``reward_code``, etc.).
        fail_mode: Dominant fail-mode detected under the current template.
        meta_llm_call: Callable that takes a ``messages`` list and
            returns the assistant text.
        outer_iter: Outer-loop iteration number, used to disambiguate
            the new version name.
        prompts_dir: Override for the prompts root (used in tests).
        generated_by: Free-form identifier of the meta-LLM (audit trail).

    Returns:
        MutationResult whose ``new_version`` is a freshly-written prompt
        directory ready for the next inner loop.

    Raises:
        MutationParseError: if the LLM output does not conform.
        ValueError / FileExistsError / FileNotFoundError: from
        ``materialize_mutation``.
    """
    loader = PromptLoader(parent_version)

    # v2 pipeline: StrategyCard provided → use focused Editor prompt.
    # v1 pipeline: no StrategyCard → use the original single-call
    # meta-prompt (kept for backward compatibility + rollback).
    if strategy_card is not None:
        # Strategist already decided the slot; override the caller's
        # target_slot to match (defensive; caller should pass them
        # consistently, but we don't trust them blindly).
        if strategy_card.target_slot != target_slot:
            _log.warning(
                "propose_new_template: target_slot=%r but "
                "strategy_card.target_slot=%r. Using the card.",
                target_slot, strategy_card.target_slot,
            )
            target_slot = strategy_card.target_slot
        prompt = build_editor_prompt(
            parent_version=parent_version,
            strategy_card=strategy_card,
            top_candidates=top_candidates,
            loader=loader,
            prior_slot_versions=prior_slot_versions,
            behavioral_block=behavioral_block,
        )
    else:
        prompt = build_meta_prompt(
            parent_version=parent_version,
            target_slot=target_slot,
            history=history,
            top_candidates=top_candidates,
            fail_mode=fail_mode,
            loader=loader,
        )
    messages = [
        {"role": "system", "content":
            "You are a careful prompt-engineering assistant. "
            "Follow the output format exactly."},
        {"role": "user", "content": prompt},
    ]
    response = meta_llm_call(messages)
    new_slot, rationale, expected = parse_mutation_response(response, target_slot)

    # v3 §4.2: optional Critic/revise loop around the Editor output.
    # Disabled by default (critic_llm_call=None) to keep v2 pipelines
    # working unchanged. When enabled, the Critic reviews the slot
    # against fairness + focus + priors; "revise" triggers another
    # Editor pass with the critique notes appended.
    critique_outcome = None
    if critic_llm_call is not None and strategy_card is not None:
        from .critique import critique_and_revise  # lazy to avoid cycle

        fairness_txt = (
            loader.slot_text("fairness")
            if "fairness" in loader.frozen_slot_names() else ""
        )

        def _editor_revise(critique, current_slot):
            """Re-invoke the Editor with critique suggestions appended."""
            notes = "\n".join(
                f"  - {s}" for s in critique.suggested_edits
            ) or "  (none)"
            revision_prompt = prompt + (
                f"\n\n[CRITIC FEEDBACK ON YOUR LAST DRAFT]\n"
                f"The critic rated the previous slot as 'revise'.\n"
                f"addresses_focus_reason: "
                f"{critique.addresses_focus_reason}\n"
                f"fairness_restatement_reason: "
                f"{critique.has_fairness_restatement_reason}\n"
                f"suggested_edits:\n{notes}\n\n"
                f"Rewrite the `{target_slot}` slot addressing ALL of "
                f"the above. Return the same OUTPUT FORMAT as before."
            )
            revision_messages = [
                messages[0],
                {"role": "user", "content": revision_prompt},
            ]
            resp2 = meta_llm_call(revision_messages)
            s2, r2, e2 = parse_mutation_response(resp2, target_slot)
            return {"new_slot": s2, "rationale": r2, "expected": e2}

        try:
            critique_outcome = critique_and_revise(
                strategy_card=strategy_card,
                editor_new_slot=new_slot,
                editor_rationale=rationale,
                editor_expected=expected,
                fairness_text=fairness_txt,
                prior_slot_versions=prior_slot_versions or [],
                critic_llm_call=critic_llm_call,
                editor_revise_call=_editor_revise,
                max_revisions=2,
            )
            new_slot = critique_outcome.accepted_slot
            rationale = critique_outcome.rationale
            expected = critique_outcome.expected_improvement
            _log.info(
                "Editor critique: revisions=%d quality=%s "
                "suggested_signal_change=%s",
                critique_outcome.revisions,
                critique_outcome.critique.overall_quality,
                critique_outcome.critique.suggested_signal_change,
            )
        except Exception as e:
            _log.warning(
                "Editor critique pass failed: %s: %s. "
                "Proceeding with original Editor output.",
                type(e).__name__, e,
            )

    # Pick a non-colliding version name. Bump outer_iter on collision —
    # robust against stale runs that left a directory on disk.
    new_version = propose_version_name(parent_version, outer_iter)
    kw = dict(prompts_dir=prompts_dir) if prompts_dir else {}
    attempt = outer_iter
    while True:
        try:
            materialize_mutation(
                parent_version=parent_version,
                new_version=new_version,
                slot_edits={target_slot: new_slot},
                rationale=rationale,
                mutation_operator="rewrite_slot",
                generated_by=generated_by,
                **kw,
            )
            break
        except FileExistsError:
            attempt += 1
            new_version = propose_version_name(parent_version, attempt)
            if attempt > outer_iter + 20:  # hard safety cap
                raise

    _log.info(
        "LERO-MP mutation: %s → %s (slot=%s, expected=%s). Chain len=%d.",
        parent_version, new_version, target_slot, expected,
        len(lineage(new_version, **kw)) if kw else len(lineage(new_version)),
    )

    return MutationResult(
        new_version=new_version,
        parent_version=parent_version,
        target_slot=target_slot,
        new_slot_content=new_slot,
        rationale=rationale,
        expected_improvement=expected,
    )
