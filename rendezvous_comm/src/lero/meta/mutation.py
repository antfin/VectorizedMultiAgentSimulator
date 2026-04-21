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


def _format_top_candidates(
    top_candidates: Sequence[Dict[str, Any]], limit: int = 3,
) -> str:
    """Compact dump of top-k candidate code snippets + metrics.

    Each entry is a dict with keys: ``reward_code`` (str), ``obs_code``
    (str), and metric keys. Missing keys are omitted.
    """
    if not top_candidates:
        return "(no candidates reported)"
    blocks = []
    for i, c in enumerate(top_candidates[:limit]):
        m1 = c.get("M1_success_rate", 0.0)
        m2 = c.get("M2_avg_return", 0.0)
        m6 = c.get("M6_coverage_progress", 0.0)
        peak = c.get("peak_M1")
        final = c.get("final_M1")
        parts = [
            f"### Candidate {i + 1}",
            f"M1={m1:.3f}  M2={m2:.2f}  M6={m6:.3f}" + (
                f"  peak_M1={peak:.3f}  final_M1={final:.3f}"
                if peak is not None and final is not None else ""
            ),
        ]
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

    return f"""[ROLE]
You are a prompt engineer tuning an instruction template used to
generate PyTorch reward/observation code for a multi-agent RL task.

[HARD CONSTRAINTS — NEVER VIOLATE]
- The `fairness` slot is FROZEN. You may read it in context but you
  must never produce edits to it: {frozen_line}
- Any slot you edit must remain consistent with the fairness rules:
  the agent policy only sees local sensors + messages at execution
  time. Do not reintroduce oracle-state references in observation code.
- Rewards must stay within the magnitude bound (|r| ≤ reward_clip=50).
- The slot you may edit this round is: `{target_slot}`. Do NOT edit
  any other slot. The outer loop uses single-slot edits so credit
  assignment is clean.

[OBJECTIVE]
Maximize: peak-M1 primary, M6 (coverage progress) tie-break.
Secondary: do not induce reward-hacking (peak-vs-final gap < 0.20),
NaN crashes, or dimension mismatches.

[DOMINANT FAIL MODE (from the last inner run)]
{fail_mode.value}

[HISTORY — prior templates sorted by peak_M1 ASCENDING]
{_format_history(history)}

[CONTRASTIVE ANALYSIS]
{_contrastive_analysis(history)}

[CURRENT `{target_slot}` SLOT TEXT]
------------------------------------------------------------
{cur_slot_text}
------------------------------------------------------------

[TOP CANDIDATES UNDER CURRENT TEMPLATE]
{_format_top_candidates(top_candidates)}

[TASK]
Propose a new version of the `{target_slot}` slot. Return EXACTLY
this format (no other prose, no YAML fences around the whole thing):

Rationale: <1-3 sentences explaining why this edit should help>
Expected-improvement: small | medium | large

{SLOT_BEGIN}
<the complete new text of the `{target_slot}` slot>
{SLOT_END}

Keep the new slot the same size order-of-magnitude as the current
one ({len(cur_slot_text)} chars). Do not echo surrounding slots.
"""


# ── response parsing ────────────────────────────────────────────

_RATIONALE_RE = re.compile(r"Rationale:\s*(.+?)(?=\n[A-Z][a-z]+-?[a-z]*:|\n{2,}|\Z)", re.DOTALL)
_EXPECTED_RE = re.compile(r"Expected-improvement:\s*(small|medium|large)", re.IGNORECASE)
_SLOT_RE = re.compile(
    rf"{re.escape(SLOT_BEGIN)}\s*\n?(.*?)\n?\s*{re.escape(SLOT_END)}",
    re.DOTALL,
)


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
