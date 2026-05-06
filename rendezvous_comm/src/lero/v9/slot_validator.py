"""v9.1 §2.3 slot-edit structural validator.

Production v9 Phase 6 showed that the meta-LLM under `refine_current`
emits prose-only `slot_edits` that strip the S3b-local-class structured
text. Specifically:

  - `inferable_hints` becomes a single paragraph (no bulleted concept
    list) — the inner LLM has no concept enumeration to mimic.
  - `examples` becomes a few sentences ("Example 1: If agent A...") with
    NO fenced ```python``` blocks — the inner LLM has no working code to
    mimic.

This validator inspects each proposed slot_edit against structural
requirements derived from task_domain.yaml. Failed slots are REJECTED
(the outer loop keeps the previous slot text instead).

Pure function — no side effects — fully unit-testable.

See docs/v9_1_plan.md §2.3 for design rationale.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SlotValidationResult:
    """Per-slot validation outcome.

    Attributes:
        slot: name of the slot validated (e.g., 'inferable_hints')
        passed: True iff slot is structurally adequate
        issues: human-readable rejection reasons (empty if passed)
        metrics: numerical breakdown for telemetry / debugging
    """

    slot: str
    passed: bool
    issues: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


_ROLE_ONE_HOT_PATTERN = re.compile(
    r"(F\.one_hot\s*\(|"
    r"one_hot\s*\(|"
    r"\[:\s*,\s*agent_idx\s*\]\s*=\s*1|"
    r"torch\.zeros\([^)]*n_agents)",
    re.MULTILINE,
)
_PYTHON_FENCE_PATTERN = re.compile(r"```python(.*?)```", re.DOTALL)


def _validate_inferable_hints(
    text: str,
    task_domain: Dict[str, Any],
    prev_text: Optional[str],
    min_concepts: int,
    max_growth_factor: float,
) -> SlotValidationResult:
    """inferable_hints must mention ≥`min_concepts` of the
    task_domain.inferable_concepts AND not blow up by more than
    `max_growth_factor`× the prev text."""
    concepts = task_domain.get("inferable_concepts") or []
    text_lower = (text or "").lower()
    concepts_hit: List[str] = []
    for c in concepts:
        phrase = c["concept"].lower()
        idiom = c["idiom"].lower()
        idiom_keywords = re.findall(r"[a-z_]{4,}", idiom)
        if phrase in text_lower or any(k in text_lower for k in idiom_keywords):
            concepts_hit.append(c["concept"])

    issues: List[str] = []
    if len(concepts_hit) < min_concepts:
        issues.append(
            f"only {len(concepts_hit)}/{len(concepts)} task_domain "
            f"inferable_concepts mentioned (need ≥{min_concepts}); "
            f"missing: {[c['concept'] for c in concepts if c['concept'] not in concepts_hit]}"
        )

    if prev_text and prev_text.strip():
        ratio = len(text or "") / max(1, len(prev_text))
        if ratio > max_growth_factor:
            issues.append(
                f"text length {len(text)} > {max_growth_factor}× "
                f"previous {len(prev_text)} (runaway growth)"
            )

    return SlotValidationResult(
        slot="inferable_hints",
        passed=len(issues) == 0,
        issues=issues,
        metrics={
            "concepts_hit": len(concepts_hit),
            "concepts_total": len(concepts),
            "concepts_matched": concepts_hit,
            "char_count": len(text or ""),
            "prev_char_count": len(prev_text) if prev_text else 0,
        },
    )


def _validate_examples(
    text: str,
    task_domain: Dict[str, Any],
    prev_text: Optional[str],
    min_blocks: int,
    require_role_one_hot: bool,
    max_growth_factor: float,
) -> SlotValidationResult:
    """examples must contain ≥`min_blocks` fenced ```python``` code
    blocks AND (if `require_role_one_hot`) at least one block must
    include a role one-hot pattern."""
    text = text or ""
    blocks = _PYTHON_FENCE_PATTERN.findall(text)
    n_blocks = len(blocks)
    role_in_any = any(bool(_ROLE_ONE_HOT_PATTERN.search(b)) for b in blocks)

    issues: List[str] = []
    if n_blocks < min_blocks:
        issues.append(
            f"only {n_blocks} fenced ```python``` blocks (need ≥{min_blocks})"
        )
    if require_role_one_hot and blocks and not role_in_any:
        issues.append(
            "no example contains role_one_hot pattern "
            "(F.one_hot or torch.zeros + [:, agent_idx] = 1)"
        )
    if require_role_one_hot and not blocks:
        # Already flagged above for "no blocks", don't double-count
        pass

    if prev_text and prev_text.strip():
        ratio = len(text) / max(1, len(prev_text))
        if ratio > max_growth_factor:
            issues.append(
                f"text length {len(text)} > {max_growth_factor}× "
                f"previous {len(prev_text)} (runaway growth)"
            )

    return SlotValidationResult(
        slot="examples",
        passed=len(issues) == 0,
        issues=issues,
        metrics={
            "n_python_blocks": n_blocks,
            "blocks_with_role_one_hot": sum(
                1 for b in blocks if _ROLE_ONE_HOT_PATTERN.search(b)
            ),
            "char_count": len(text),
            "prev_char_count": len(prev_text) if prev_text else 0,
        },
    )


def validate_slot_edits(
    slot_edits: Dict[str, str],
    task_domain: Dict[str, Any],
    prev_slots: Optional[Dict[str, str]] = None,
    min_concepts: int = 5,
    min_python_blocks: int = 2,
    require_role_one_hot_in_examples: bool = True,
    max_growth_factor: float = 2.5,
) -> Dict[str, SlotValidationResult]:
    """Validate each slot_edit. Returns a dict {slot_name: result}.

    Caller behavior: if `result.passed=False`, KEEP the previous slot
    text instead of overwriting with the rejected edit.

    Args:
        slot_edits: dict of {slot_name: new_text} from the meta-LLM
        task_domain: parsed task_domain.yaml (needs `inferable_concepts`)
        prev_slots: optional dict of current slot text (for growth check)
        min_concepts: minimum task_domain.inferable_concepts that must
            appear in `inferable_hints` (default 5 of 7)
        min_python_blocks: minimum fenced ```python``` blocks in
            `examples` (default 2)
        require_role_one_hot_in_examples: if True, at least one example
            must include a role one-hot pattern (default True)
        max_growth_factor: reject if new text > N× previous (default 2.5)

    Returns:
        dict {slot_name: SlotValidationResult} for slots present in
        `slot_edits`. Slots not in `slot_edits` are not validated (caller
        handles via the "keep previous" branch trivially).

    Note: `feedback_template` slot is not structurally validated — it's
    a free-form reminder. Callers can add custom checks if needed.
    """
    prev_slots = prev_slots or {}
    results: Dict[str, SlotValidationResult] = {}

    if "inferable_hints" in slot_edits:
        results["inferable_hints"] = _validate_inferable_hints(
            slot_edits["inferable_hints"],
            task_domain,
            prev_slots.get("inferable_hints"),
            min_concepts,
            max_growth_factor,
        )

    if "examples" in slot_edits:
        results["examples"] = _validate_examples(
            slot_edits["examples"],
            task_domain,
            prev_slots.get("examples"),
            min_python_blocks,
            require_role_one_hot_in_examples,
            max_growth_factor,
        )

    return results


def filter_valid_slot_edits(
    slot_edits: Dict[str, str],
    task_domain: Dict[str, Any],
    prev_slots: Optional[Dict[str, str]] = None,
    **validator_kwargs,
) -> tuple[Dict[str, str], Dict[str, SlotValidationResult]]:
    """Convenience: validate edits, return only the passing ones.

    Returns (filtered_edits, full_results). Caller iterates results to
    log rejections.
    """
    results = validate_slot_edits(
        slot_edits, task_domain, prev_slots, **validator_kwargs
    )
    filtered = {
        k: v for k, v in slot_edits.items() if k not in results or results[k].passed
    }
    return filtered, results
