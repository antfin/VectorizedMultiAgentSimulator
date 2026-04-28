"""v5 meta-refiner — outer-level textual gradient over the metaprompt.

The meta-LLM sees the inner-loop trajectory of the previous outer
iteration plus the cumulative outer registry, and rewrites the
high-level guidance slot files (guidance_observation, guidance_reward,
guidance_shared) for the next inner search.

Operates on the SAME slot files v4_composer touches, but instead of
N parallel strategies, emits a SINGLE refined metaprompt that
incorporates lessons learned from inner.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from ..llm_client import LLMClient
from .inner_loop import InnerResult
from .registry import Registry

_log = logging.getLogger("rendezvous.lero.v5.meta")


_VALID_SLOTS = {
    "guidance_observation",
    "guidance_reward",
    "guidance_shared",
}


_META_SYSTEM = (
    "You are a meta-strategist for an evolutionary code-search system. "
    "Your job is to write the HIGH-LEVEL strategic framing that the "
    "inner code-generation LLM uses to write Python observation/reward "
    "code. You do NOT write Python code. You write English guidance "
    "that nudges the inner LLM toward the right FAMILY of features. "
    "You see what worked and what failed across previous outer "
    "iterations and refine the framing accordingly. You favor "
    "incremental, falsifiable refinements over wholesale rewrites — "
    "unless stagnation forces a pivot."
)


def _read_slot_files(prompt_dir: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for slot in _VALID_SLOTS:
        p = prompt_dir / f"{slot}.txt"
        out[slot] = p.read_text() if p.exists() else ""
    return out


def _format_inner_trajectory(inner: InnerResult) -> str:
    if not inner.registry.fitness_trajectory:
        return "(no inner iterations completed)"
    parts = ["Inner-loop fitness trajectory (one value per inner iter):"]
    parts.append(
        "  " + " → ".join(
            f"{f:+.3f}" for f in inner.registry.fitness_trajectory
        )
    )
    if inner.best:
        parts.append(
            f"Best inner candidate: M1={inner.best.metrics.get('M1_success_rate', 0):.3f} "
            f"shape={inner.best.shape} fitness={inner.best.fitness:+.3f}"
        )
    if inner.worst and inner.worst is not inner.best:
        parts.append(
            f"Worst inner candidate: M1={inner.worst.metrics.get('M1_success_rate', 0):.3f} "
            f"shape={inner.worst.shape} fitness={inner.worst.fitness:+.3f}"
        )
    return "\n".join(parts)


def _build_meta_prompt(
    current_slots: Dict[str, str],
    outer_registry: Registry,
    last_inner: InnerResult,
    pivot: bool,
    task_summary: str,
) -> str:
    pivot_block = ""
    if pivot:
        pivot_block = (
            "\n⚠️ OUTER STAGNATION DETECTED — last 2 outer iterations did "
            "not meaningfully improve.\n"
            "Do NOT make small adjustments. PIVOT: propose a "
            "fundamentally different framing of the task. Identify the "
            "implicit assumption in the prior metaprompt that may be "
            "wrong. Replace at least one slot wholesale.\n"
        )

    return f"""[TASK SUMMARY]
{task_summary}

[CURRENT METAPROMPT — guidance the inner LLM saw last outer iteration]
--- guidance_observation.txt ---
{current_slots.get('guidance_observation', '(empty)')}

--- guidance_reward.txt ---
{current_slots.get('guidance_reward', '(empty)')}

--- guidance_shared.txt ---
{current_slots.get('guidance_shared', '(empty)')}

[INNER LOOP RESULT FROM LAST OUTER ITERATION]
{_format_inner_trajectory(last_inner)}

[CUMULATIVE OUTER REGISTRY — all metaprompt framings tried so far]
{outer_registry.format_for_prompt(max_failures=6)}
{pivot_block}
[YOUR TASK]
Refine the metaprompt slots for the NEXT inner search. Two steps:

STEP 1 — DIAGNOSIS (3-5 sentences):
  - What hypothesis did the current metaprompt encode?
  - Did the inner loop's best candidate confirm or refute that hypothesis?
  - What CATEGORY of feature seems missing from what inner generated?
  - If pivoting: what is the wrong assumption being made?

STEP 2 — SLOT EDITS:
  Emit a JSON object with the slot files you want to OVERWRITE. Only
  include slots you want to change; omitted slots stay as-is. Each
  value is the FULL replacement text (multi-line markdown is fine).

[OUTPUT FORMAT]

### DIAGNOSIS
<diagnosis paragraph>

### SLOT_EDITS
```json
{{
  "guidance_observation": "...",       (omit if no change)
  "guidance_reward": "...",            (omit if no change)
  "guidance_shared": "..."             (omit if no change)
}}
```
"""


def _parse_slot_edits(raw: str) -> Dict[str, str]:
    if "### SLOT_EDITS" not in raw:
        raise ValueError("meta-refiner response missing '### SLOT_EDITS' header")
    body = raw.split("### SLOT_EDITS", 1)[1]
    json_start = body.find("{")
    json_end = body.rfind("}")
    if json_start < 0 or json_end < json_start:
        raise ValueError("SLOT_EDITS section did not contain valid JSON")
    blob = body[json_start:json_end + 1]
    data = json.loads(blob)
    edits = {}
    for k, v in data.items():
        if k not in _VALID_SLOTS:
            _log.warning("meta-refiner emitted unknown slot '%s' — ignored", k)
            continue
        if not isinstance(v, str):
            continue
        edits[k] = v
    return edits


def refine_metaprompt(
    prev_prompt_dir: Path,
    next_prompt_dir: Path,
    outer_registry: Registry,
    last_inner: InnerResult,
    meta_llm: LLMClient,
    task_summary: str,
    pivot: bool = False,
) -> Dict[str, str]:
    """Call the meta-LLM and write the refined slot files into
    next_prompt_dir (assumed to be a copy of prev_prompt_dir).

    Returns the dict of slot edits the meta-LLM produced.
    """
    import shutil
    if next_prompt_dir.exists():
        shutil.rmtree(next_prompt_dir)
    shutil.copytree(prev_prompt_dir, next_prompt_dir)

    current_slots = _read_slot_files(prev_prompt_dir)
    prompt = _build_meta_prompt(
        current_slots=current_slots,
        outer_registry=outer_registry,
        last_inner=last_inner,
        pivot=pivot,
        task_summary=task_summary,
    )

    last_err: Optional[Exception] = None
    edits: Optional[Dict[str, str]] = None
    raw = ""
    t0 = time.monotonic()
    for attempt in range(1, 4):
        raw = meta_llm.generate(
            [
                {"role": "system", "content": _META_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            n=1,
        )[0]
        try:
            edits = _parse_slot_edits(raw)
            break
        except (ValueError, json.JSONDecodeError) as e:
            last_err = e
            _log.warning("meta-refiner parse failed attempt %d/3: %s",
                         attempt, e)

    if edits is None:
        raise ValueError(
            f"meta-refiner failed after 3 attempts. Last error: {last_err}"
        )

    elapsed = time.monotonic() - t0
    for slot, text in edits.items():
        normalized = (text or "").rstrip() + "\n"
        (next_prompt_dir / f"{slot}.txt").write_text(normalized)

    (next_prompt_dir / "_refiner_response.txt").write_text(raw)
    diagnosis_match = re.search(r"### DIAGNOSIS\s*\n(.*?)(?=### |$)",
                                raw, re.DOTALL)
    if diagnosis_match:
        (next_prompt_dir / "_refiner_diagnosis.md").write_text(
            diagnosis_match.group(1).strip()
        )

    _log.info(
        "v5 meta-refiner edited %d slots in %.1fs: %s",
        len(edits), elapsed, ", ".join(edits.keys()),
    )
    return edits
