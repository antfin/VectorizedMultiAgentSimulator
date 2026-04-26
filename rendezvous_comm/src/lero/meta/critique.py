"""TextGrad-style Editor self-critique loop (LERO-MP v3 §4.2).

After the Editor (Level 2) proposes a new slot, a second LLM call (the
Critic) reviews it against:
  - StrategyCard focus / avoid — does the edit actually implement it?
  - fairness slot — is the edit a sneaky paraphrase?
  - prior slot versions — is the edit substantively different?

The Critic returns a structured ``EditorCritique`` (see schemas.py).
If ``overall_quality == "revise"``, the Editor is re-invoked with the
Critic's ``suggested_edits`` appended to its context. Max 2 revision
rounds. If ``overall_quality == "reject"`` we raise MutationParseError
and let the outer loop graceful-stop.

Inspiration: Yuksekgonul et al. TextGrad, Nature 2024
(https://arxiv.org/abs/2406.07496). LLM critiques act as "textual
gradients" that improve the output in 1–2 rounds on well-scoped tasks.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..schemas import EditorCritique

_log = logging.getLogger("rendezvous.lero.mp.critique")


MAX_REVISIONS = 2

# Markers the Critic looks for to detect a fairness paraphrase.
_FAIRNESS_MARKERS = (
    "local sensor", "oracle", "|r| <= 50", "clamp",
)


def build_critic_prompt(
    strategy_card,  # StrategyCard
    editor_new_slot: str,
    fairness_text: str,
    prior_slot_versions: Sequence[Any] = (),
) -> str:
    """Compose the Critic prompt. Pure function, no I/O."""
    focus_lines = "\n".join(f"  - {f}" for f in strategy_card.focus) or "  (none)"
    avoid_lines = "\n".join(f"  - {a}" for a in strategy_card.avoid) or "  (none)"

    prior_excerpts = []
    for i, e in enumerate(prior_slot_versions, 1):
        verdict = getattr(e, "verdict", None) or "pending"
        excerpt = (getattr(e, "slot_content_excerpt", "") or "")[:300]
        prior_excerpts.append(
            f"  v{i} (verdict={verdict}):\n    {excerpt}"
        )
    priors_block = (
        "\n".join(prior_excerpts)
        if prior_excerpts
        else "  (no prior versions)"
    )

    return f"""[ROLE]
You are the LERO-MP Editor Critic. The Editor LLM has written a new
prompt slot. Your job is to judge whether it's good enough to ship or
needs another revision.

[STRATEGIST'S FOCUS — the Editor was asked to implement these]
{focus_lines}

[STRATEGIST'S AVOID — the Editor was asked to avoid these]
{avoid_lines}

[FAIRNESS SLOT — what the Editor must NOT restate]
------------------------------------------------------------
{fairness_text.strip()}
------------------------------------------------------------

[PRIOR VERSIONS OF THIS SAME SLOT — the Editor was asked to diverge from these]
{priors_block}

[EDITOR'S NEW SLOT TEXT — what you are reviewing]
------------------------------------------------------------
{editor_new_slot.strip()}
------------------------------------------------------------

[TASK]
DEFAULT VERDICT IS "keep". Only escalate to "revise" or "reject"
when there is a CONCRETE, ACTIONABLE problem in the Editor's output.
Phrasing nitpicks, "could be more detailed", or "minor improvements"
are NOT sufficient grounds — they waste outer-loop iterations.

Decide the verdict in this order:

  1. Does the Editor's text mention AT LEAST ONE specific feature
     identifier from the strategist's focus or the reference set
     (e.g. lidar_targets, lidar_agents, hold_signal, gap,
     proximity_count, intensity, agent_idx, potential, partner)?
     If YES → addresses_focus=true.

  2. Does the Editor's text RESTATE the fairness contract verbatim
     or paraphrase its rules ("use local sensors only", "avoid
     oracle state", "clamp rewards to |r|<=50")? Brief allowed
     mentions like "without needing oracle positions" are NOT
     restatements — only flag when the slot's CENTRAL message is
     a fairness reminder. If no restatement → has_fairness_restatement=false.

  3. Is the Editor's text substantively different from EVERY prior
     version listed above? (Different feature focus, different
     reasoning, different concrete advice.) If yes →
     diverges_from_priors=true.

  4. VERDICT:
     - "keep": addresses_focus AND NOT has_fairness_restatement AND
        diverges_from_priors. This is the COMMON case — ship it.
     - "revise": exactly one of those three is wrong AND it's
        clearly fixable in one or two edits.
     - "reject": the output is so off-base that another Editor
        pass would not help (e.g. completely empty, only restates
        fairness, or duplicates a prior verbatim).

  5. suggested_edits MUST be empty list `[]` when overall_quality
     is "keep".

  6. suggested_signal_change defaults to "keep". Only change it
     when the Strategist's tier selection clearly didn't help
     (e.g. "fingerprint" was forwarded but added no signal).

Return EXACTLY this JSON (no prose around it). overall_quality
appears FIRST so you commit to a verdict before listing nitpicks:

{{
  "overall_quality": "keep" | "revise" | "reject",
  "addresses_focus": true | false,
  "addresses_focus_reason": "<1 sentence>",
  "cites_specific_features": ["<feature identifiers literally present in the text>"],
  "has_fairness_restatement": true | false,
  "has_fairness_restatement_reason": "<1 sentence>",
  "diverges_from_priors": true | false,
  "suggested_edits": ["<edit 1>", "<edit 2>"],
  "suggested_signal_change": "keep" | "add_fingerprint" | "drop_fingerprint" | "add_curve_shape" | "drop_curve_shape"
}}
"""


# Match the FIRST balanced {...} block. ``re.DOTALL`` lets it span
# newlines but keep `*?` non-greedy so we don't swallow a trailing }
# from the assistant's prose (e.g. ```json ... ``` followed by text).
_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE,
)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_critique(text: str) -> EditorCritique:
    """Extract an EditorCritique from the Critic LLM's response.

    Tolerant parser: accepts JSON wrapped in ```json fences (Llama
    output style), JSON with trailing prose, and missing optional
    fields. Falls back to safe defaults rather than failing the
    outer loop on a parsing nit.
    """
    blob = None
    # 1. Prefer fenced JSON block (Llama / many open-weights models)
    m = _JSON_FENCE_RE.search(text)
    if m:
        blob = m.group(1)
    # 2. Otherwise grab the first {...} span, balanced
    else:
        m = _JSON_RE.search(text)
        if m:
            blob = m.group(0)
    if not blob:
        raise ValueError(
            f"Critic response did not contain a JSON block. "
            f"Got: {text[:200]!r}"
        )

    # JSON parse — try the raw blob, then strip trailing prose / commas
    data = None
    for candidate in (blob, blob.rstrip(",}\n ") + "}"):
        try:
            data = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    if data is None:
        raise ValueError(
            f"Critic response JSON parse failed. Body: {blob[:200]!r}"
        )

    # Backfill optional fields the model may drop
    data.setdefault("suggested_signal_change", "keep")
    data.setdefault("suggested_edits", [])
    data.setdefault("cites_specific_features", [])
    data.setdefault("addresses_focus_reason", "")
    data.setdefault("has_fairness_restatement_reason", "")

    try:
        return EditorCritique.model_validate(data)
    except Exception as e:
        raise ValueError(
            f"Critic response failed schema validation: {e}. "
            f"Body: {str(data)[:300]!r}"
        )


@dataclass
class CritiqueOutcome:
    """Return value of critique_and_revise."""

    accepted_slot: str
    rationale: str
    expected_improvement: str
    critique: EditorCritique
    revisions: int


def critique_and_revise(
    strategy_card,
    editor_new_slot: str,
    editor_rationale: str,
    editor_expected: str,
    fairness_text: str,
    prior_slot_versions: Sequence[Any],
    critic_llm_call: Callable[[List[Dict[str, str]]], str],
    editor_revise_call: Optional[
        Callable[[EditorCritique, str], Dict[str, str]]
    ] = None,
    max_revisions: int = MAX_REVISIONS,
) -> CritiqueOutcome:
    """Run critic → optional revise loop on Editor output.

    Args:
        strategy_card: StrategyCard driving the edit.
        editor_new_slot: Editor's first-pass slot text.
        editor_rationale: Editor's first-pass rationale.
        editor_expected: Editor's first-pass expected_improvement tag.
        fairness_text: the current frozen fairness slot.
        prior_slot_versions: MutationLogEntry sequence for this slot.
        critic_llm_call: callable that runs the Critic LLM on messages.
        editor_revise_call: optional callable (critique, current_slot) ->
            {"new_slot", "rationale", "expected"} that re-invokes the
            Editor with revision notes. When None, the loop skips
            revision and just returns the Critic verdict alongside the
            original Editor output.
        max_revisions: how many revision rounds to allow.

    Returns:
        CritiqueOutcome with the final accepted slot + rationale.

    Raises:
        ValueError if overall_quality == "reject" on the final round.
    """
    current_slot = editor_new_slot
    current_rationale = editor_rationale
    current_expected = editor_expected
    last_critique: Optional[EditorCritique] = None

    for round_idx in range(max_revisions + 1):
        prompt = build_critic_prompt(
            strategy_card=strategy_card,
            editor_new_slot=current_slot,
            fairness_text=fairness_text,
            prior_slot_versions=prior_slot_versions,
        )
        response = critic_llm_call([
            {
                "role": "system",
                "content": (
                    "You are a careful critic. Return only the JSON "
                    "block requested; no prose around it."
                ),
            },
            {"role": "user", "content": prompt},
        ])
        last_critique = parse_critique(response)
        _log.info(
            "Critic round %d: quality=%s addresses_focus=%s "
            "fairness_restatement=%s diverges=%s",
            round_idx, last_critique.overall_quality,
            last_critique.addresses_focus,
            last_critique.has_fairness_restatement,
            last_critique.diverges_from_priors,
        )

        if last_critique.overall_quality == "keep":
            return CritiqueOutcome(
                accepted_slot=current_slot,
                rationale=current_rationale,
                expected_improvement=current_expected,
                critique=last_critique,
                revisions=round_idx,
            )
        if last_critique.overall_quality == "reject":
            raise ValueError(
                f"Editor output rejected by Critic: "
                f"{last_critique.addresses_focus_reason} | "
                f"{last_critique.has_fairness_restatement_reason}"
            )
        # overall_quality == "revise" — need another Editor pass
        if editor_revise_call is None or round_idx >= max_revisions:
            _log.warning(
                "Critic asked to revise but no revise-callable OR "
                "max_revisions reached. Accepting current slot."
            )
            return CritiqueOutcome(
                accepted_slot=current_slot,
                rationale=current_rationale,
                expected_improvement=current_expected,
                critique=last_critique,
                revisions=round_idx,
            )
        revision = editor_revise_call(last_critique, current_slot)
        current_slot = revision.get("new_slot", current_slot)
        current_rationale = revision.get("rationale", current_rationale)
        current_expected = revision.get("expected", current_expected)

    # Exhausted revisions — accept the last slot even if still "revise".
    assert last_critique is not None  # mypy hint
    return CritiqueOutcome(
        accepted_slot=current_slot,
        rationale=current_rationale,
        expected_improvement=current_expected,
        critique=last_critique,
        revisions=max_revisions,
    )
