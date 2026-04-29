"""v6 meta-strategist — single-strategy, simplicity-first textual gradient.

Differences vs v5_meta_refiner:

  * Emits ONE strategy per outer iter (not 3).
  * Outputs structured V6MetaDecision (classification, next_mode,
    flags, slot_edits, complexity_level, rationale).
  * Forces SIMPLICITY-FIRST in outer iter 0 (only obs-only mode allowed).
  * Forbids feature-level prescriptions in slot text — talk about
    families and operations, not solution names.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Dict, List, Optional

from ..llm_client import LLMClient
from .decision import (
    V6MetaDecision,
    classify_inner_result,
    enforce_decision,
)

_log = logging.getLogger("rendezvous.lero.v6.meta")


_META_SYSTEM = """You are a meta-strategist for an LLM-driven evolutionary search. Your job is to write the high-level English guidance that an INNER LLM will use to write Python observation/reward code for a multi-agent task.

You MUST follow these rules:

1. SIMPLICITY FIRST.
   In outer iter 0, propose the simplest possible strategy. Examples of the level you should aim for: "expose one or two summary statistics from the local sensors that capture where the closest task-relevant thing is and how close it is." Do NOT enumerate features. Do NOT describe complex multi-agent coordination logic. Trust that the inner LLM can take simple guidance and write functioning code.

2. ESCALATE ONLY WHEN JUSTIFIED.
   You may increase complexity_level (1=simplest, 4=most complex) only if the prior round's classification was "no_signal_simple" or "partial_signal" AND you can name in writing the specific failure mode you're trying to address.

3. CLASSIFY BEFORE DECIDING.
   Each round, you produce a classification of the inner search result in {found_good, partial_signal, no_signal_simple, no_signal_complex} based on the inner trajectory. The classification, not your taste, drives the next move. The runner ALSO computes the same classification independently from raw metrics and will override your claim if you're wrong, so be honest.

4. STOP WHEN GOOD.
   If best inner M1 ≥ 0.05 with rising shape, classification = found_good. You output next_mode=stop and write empty slot_edits. Do not chase marginal improvements.

5. MODE SELECTION: OBS / REWARD / BOTH.
   In outer iter 0 you may only set next_evolve_observation=true and next_evolve_reward=false. You may unlock next_evolve_reward=true in a later round only if obs-only has been tried and the classification suggests obs alone won't suffice. Each outer iter produces ONE strategy — the inner LERO loop will explore it thoroughly via its own 4×3×1M iterative search. Don't try to do the inner loop's job at the outer level. Justify any mode change in writing.

6. NO PRE-CANNED ANSWERS.
   You do NOT know the optimal feature set. You do NOT name specific features by handle. Talk about families and operations, not solutions. Describe HOW to think about the problem, not WHAT to compute.

You will be given:
- The task definition (number of agents, targets, k, sensor description).
- The current metaprompt (slot files).
- The inner-loop result from the prior round (best+worst code excerpts, fitness trajectory, M1/M6 trajectories, shapes).
- The cumulative outer registry (what was tried, what worked, what didn't).

You will output a single JSON object with the fields below."""


def _format_inner_summary(inner) -> str:
    """One-block textual summary of an InnerResult for the meta prompt."""
    if inner is None:
        return "(no prior inner round — this is outer iter 0)"
    parts: List[str] = []
    parts.append(
        "Inner-loop fitness trajectory (one value per inner iter): "
        f"{[round(f, 3) for f in inner.registry.fitness_trajectory]}"
    )
    if inner.best:
        b = inner.best
        parts.append(
            f"Best inner candidate: M1={b.metrics.get('M1_success_rate', 0):.3f} "
            f"M6={b.metrics.get('M6_coverage_progress', 0):.3f} "
            f"shape={b.shape} fitness={b.fitness:+.3f}"
        )
        if b.candidate.obs_source:
            excerpt = "\n".join(b.candidate.obs_source.splitlines()[:30])
            parts.append("Best candidate observation code (first ~30 lines):")
            parts.append(f"```python\n{excerpt}\n```")
        if b.candidate.reward_source:
            excerpt = "\n".join(b.candidate.reward_source.splitlines()[:30])
            parts.append("Best candidate reward code (first ~30 lines):")
            parts.append(f"```python\n{excerpt}\n```")
    if inner.worst and inner.worst is not inner.best:
        w = inner.worst
        parts.append(
            f"Worst inner candidate: M1={w.metrics.get('M1_success_rate', 0):.3f} "
            f"shape={w.shape} fitness={w.fitness:+.3f}"
        )
    return "\n".join(parts)


def _format_outer_registry(outer_registry) -> str:
    if not outer_registry.entries:
        return "(no prior outer iters)"
    parts = ["Prior outer iters and their inner outcomes:"]
    for e in outer_registry.entries:
        parts.append(
            f"  outer {e.iter_idx}: M1={e.M1:.3f} shape={e.shape} "
            f"fitness={e.fitness:+.3f}"
        )
        parts.append(f"    summary: {e.summary[:200]}")
    return "\n".join(parts)


def _format_current_slots(slots: Dict[str, str]) -> str:
    parts = []
    for slot in ("guidance_observation", "guidance_reward", "guidance_shared"):
        text = slots.get(slot, "")
        body = text.strip() or "(empty)"
        parts.append(f"--- {slot}.txt ---")
        parts.append(body)
        parts.append("")
    return "\n".join(parts)


def _build_meta_prompt(
    task_summary: str,
    current_slots: Dict[str, str],
    last_inner,
    outer_registry,
    outer_idx: int,
    prior_complexity: int,
    prior_classification: Optional[str],
) -> str:
    obs_only_lock = (
        "**OUTER ITER 0 LOCK: next_evolve_observation must be true, "
        "next_evolve_reward must be false. Slot edits limited to "
        "guidance_observation and guidance_shared.**"
        if outer_idx == 0 else
        "Outer iter > 0: you may unlock next_evolve_reward=true with "
        "justification."
    )
    return f"""[TASK SUMMARY]
{task_summary}

[CURRENT METAPROMPT — what the inner LLM saw last round]
{_format_current_slots(current_slots)}

[INNER LOOP RESULT FROM LAST OUTER ITER]
{_format_inner_summary(last_inner)}

[CUMULATIVE OUTER REGISTRY]
{_format_outer_registry(outer_registry)}

[CONTEXT]
This is outer iter {outer_idx}.
Prior complexity_level: {prior_complexity} (1=simplest, 4=most complex).
Prior classification: {prior_classification or "n/a (this is the first iter)"}.
{obs_only_lock}

[OUTPUT FORMAT]
Reply with two sections.

### REASONING
<3-6 sentences: what the inner result implies, what hypothesis you'll test next, and why your complexity_level choice is justified>

### DECISION
```json
{{
  "classification": "found_good" | "partial_signal" | "no_signal_simple" | "no_signal_complex",
  "next_mode": "stop" | "refine_current" | "try_different_simple" | "add_simple_reward" | "reset_simpler",
  "rationale": "<2-3 sentences>",
  "next_evolve_observation": true | false,
  "next_evolve_reward": true | false,
  "complexity_level": 1 | 2 | 3 | 4,
  "slot_edits": {{
    "guidance_observation": "<full replacement text or omit>",
    "guidance_reward": "<full replacement text or omit>",
    "guidance_shared": "<full replacement text or omit>"
  }}
}}
```

Remember: talk about families and operations, not specific feature names. Trust the inner LLM to do the implementation."""


def _parse_decision(raw: str) -> V6MetaDecision:
    if "### DECISION" not in raw:
        raise ValueError("missing '### DECISION' section header")
    body = raw.split("### DECISION", 1)[1]
    js = body.find("{")
    je = body.rfind("}")
    if js < 0 or je < js:
        raise ValueError("no JSON object in DECISION section")
    blob = body[js:je + 1]
    data = json.loads(blob)

    return V6MetaDecision(
        classification=data["classification"],
        next_mode=data["next_mode"],
        rationale=str(data.get("rationale", "")),
        next_evolve_observation=bool(data.get("next_evolve_observation", True)),
        next_evolve_reward=bool(data.get("next_evolve_reward", False)),
        slot_edits=dict(data.get("slot_edits") or {}),
        complexity_level=int(data.get("complexity_level", 1)),
    )


def call_meta_strategist(
    meta_llm: LLMClient,
    task_summary: str,
    current_slots: Dict[str, str],
    last_inner,
    outer_registry,
    outer_idx: int,
    prior_complexity: int,
    prior_classification: Optional[str],
) -> tuple[V6MetaDecision, str]:
    """Call meta-LLM, parse with retry, return raw decision (NOT yet enforced)
    plus the LLM response text for logging."""
    prompt = _build_meta_prompt(
        task_summary=task_summary,
        current_slots=current_slots,
        last_inner=last_inner,
        outer_registry=outer_registry,
        outer_idx=outer_idx,
        prior_complexity=prior_complexity,
        prior_classification=prior_classification,
    )
    last_err: Optional[Exception] = None
    raw_text = ""
    decision: Optional[V6MetaDecision] = None
    t0 = time.monotonic()
    for attempt in range(1, 4):
        raw_text = meta_llm.generate(
            [
                {"role": "system", "content": _META_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            n=1,
        )[0]
        try:
            decision = _parse_decision(raw_text)
            break
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            last_err = e
            _log.warning(
                "v6 meta-strategist parse failed attempt %d/3: %s",
                attempt, e,
            )
    if decision is None:
        raise ValueError(
            f"v6 meta-strategist failed after 3 attempts. "
            f"Last error: {last_err}"
        )
    elapsed = time.monotonic() - t0
    _log.info(
        "v6 meta-strategist outer=%d: classification=%s next_mode=%s "
        "evolve_obs=%s evolve_rew=%s complexity=%d (%.1fs)",
        outer_idx, decision.classification, decision.next_mode,
        decision.next_evolve_observation, decision.next_evolve_reward,
        decision.complexity_level, elapsed,
    )
    return decision, raw_text


def decide_and_enforce(
    meta_llm: LLMClient,
    task_summary: str,
    current_slots: Dict[str, str],
    last_inner,
    outer_registry,
    outer_idx: int,
    prior_complexity: int,
    prior_classification: Optional[str],
) -> tuple[V6MetaDecision, str]:
    """End-to-end: call meta-LLM, run code-side classifier, enforce policy.
    Returns the enforced V6MetaDecision and the raw LLM response text."""
    raw_decision, raw_text = call_meta_strategist(
        meta_llm=meta_llm,
        task_summary=task_summary,
        current_slots=current_slots,
        last_inner=last_inner,
        outer_registry=outer_registry,
        outer_idx=outer_idx,
        prior_complexity=prior_complexity,
        prior_classification=prior_classification,
    )

    # Code-side classifier — runs only when we have a prior inner result.
    if last_inner is None or last_inner.best is None:
        # First outer iter, no prior inner. Trust LLM's classification
        # but force it to "no_signal_simple" if it claimed "found_good"
        # (impossible without a prior inner).
        code_classification = (
            "no_signal_simple"
            if raw_decision.classification == "found_good"
            else raw_decision.classification
        )
    else:
        b = last_inner.best
        code_classification = classify_inner_result(
            best_M1=float(b.metrics.get("M1_success_rate", 0.0)),
            best_shape=b.shape,
            fitness_trajectory=list(last_inner.registry.fitness_trajectory),
            best_M6=float(b.metrics.get("M6_coverage_progress", 0.0)),
            prior_complexity=prior_complexity,
        )

    enforced = enforce_decision(
        raw=raw_decision,
        outer_idx=outer_idx,
        prior_complexity=prior_complexity,
        prior_classification=prior_classification,
        code_classification=code_classification,
    )
    return enforced, raw_text
