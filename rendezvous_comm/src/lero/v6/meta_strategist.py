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


_META_SYSTEM = """## Role

You are a prompt engineer iterating on the high-level English guidance that an INNER LLM uses to write Python observation/reward code for a multi-agent reinforcement learning task. Each round, the inner LLM runs a 4 × 3 × 1M iterative-refinement search on the guidance you write. You see the result and decide what to change next.

## Goal

Reach `classification = found_good` (best inner M1 ≥ 0.05 with rising shape) using as few outer rounds as possible, by writing guidance that helps the inner LLM converge quickly on a working code recipe — without prescribing the recipe yourself.

## Inputs you will receive

- TASK_DEFINITION — agents, targets, k, sensors, episode budget.
- CURRENT_METAPROMPT — the slot files the inner LLM saw last round.
- INNER_RESULT — best/worst candidate code, M1 and M6 trajectories, shape tag, fitness.
- OUTER_REGISTRY — every prior round's outcome.
- CONTEXT — outer iter index, prior complexity_level, prior classification.

## Decision rules (use judgement, not absolutes)

- **Start simplest.** In outer iter 0, propose the simplest single direction you can articulate (one or two sentences of guidance, no enumeration). Trust the inner LLM to pick reasonable code from a sparse hint.
- **Escalate only on evidence.** Increase `complexity_level` (1..4) only when the prior round's measured classification justifies it. State the specific failure mode you observed before adding any complexity.
- **One strategy per round.** You write one English direction; the inner loop explores it via 12 candidate codes. Do not pre-write multiple strategies. Do not micromanage the implementation.
- **Stop when done.** If best inner M1 ≥ 0.05 with monotonic_rise / late_ramp shape: emit `classification=found_good`, `next_mode=stop`, empty `slot_edits`. The runner cross-checks your classification against raw metrics — be honest, your claim will be overridden if it disagrees with the data.
- **Obs before reward.** Outer iter 0 must set `next_evolve_observation=true, next_evolve_reward=false`. Reward unlocks only when obs-only has been tried and classification was `no_signal_simple` or `partial_signal`. Justify any mode change.
- **No pre-canned answers.** You do not know the optimal feature set. Talk about families and operations, not solution names. Describe how to think about the problem, not what to compute.

## What "done" looks like for the GUIDANCE you write

Before finalising your slot_edits, check:

- Does the guidance name a single direction, not a menu?
- Is it concrete enough that the inner LLM can decide what to write, but loose enough that 12 candidates can vary meaningfully?
- Does it avoid feature names, function names, exact PyTorch operations?
- Does it leave the question open at the right level of abstraction (families and operations, not recipes)?
- If reward is unlocked, is the reward direction a single shaping idea, not a multi-term recipe?

If any answer is no, revise once, minimally, before emitting.

## Stop conditions

- `found_good` reached → emit `next_mode=stop` and exit the loop.
- Two consecutive `no_signal_complex` rounds → emit `next_mode=reset_simpler` and explicitly retreat to a simpler hypothesis you have not yet tested.
- MAX_OUTER reached → the runner stops you regardless.

## Operational guidance (V4 — added 2026-04-30 after Phase 2 sweep)

When you write `slot_edits` text describing operations, prefer ones that COMBINE information across multiple input channels. The inner LLM defaults to writing within-channel statistics (target-only summaries OR agent-only summaries); your guidance should explicitly nudge it toward operations that touch BOTH the target and the agent sensor channels in a single expression. Patterns to invite (without naming features): products, boolean conjunctions, ratios, differences between channel-derived quantities.

Phrase operations as DECISIONS the policy needs to make, not as descriptive statistics. Examples of decision-shaped phrasing (template only, do not copy verbatim): "stay vs move", "alone vs paired", "scout vs converge", "this target or another one". Each operation you mention in slot_edits should map to a decision the agent could plausibly make from the resulting feature.

When you describe operations in `guidance_observation`, USE PARALLEL BULLETS for the target sensor channel and the agent sensor channel. Whenever you describe an operation on one channel, immediately describe the analogous operation on the other channel.

Place a 3-5 line "operations palette" sub-section at the top of `guidance_observation` listing 3-5 PATTERN TEMPLATES — each pattern is one sentence, names a generic operation (product, mask, ratio, gating), and points at how it relates to a decision. Do not name specific features.

## V5 addition (2026-04-30) — pseudo-PyTorch example expressions

You MAY include 1–3 short example expressions (pseudo-PyTorch, ≤3 lines each) inline within `guidance_observation`, demonstrating one or two patterns from your operations palette in code form. Rules:

- Examples are SNIPPETS, not a full `enhance_observation` function. One line per pattern, like:
  `joint_close = (lidar_targets.min(-1).values < r) * (lidar_agents.min(-1).values < r)`
- Choose variable names that describe the DECISION the feature encodes (e.g. `joint_close`, `target_density_minus_agent_density`, `proximity_ratio`), NOT specific solution-feature handles. Runtime grep redacts forbidden tokens; pick neutral names.
- Use 1–3 examples maximum. If you've already given a working pattern, do not pile on more.
- Wrap example expressions in markdown ``` fenced blocks so the inner LLM treats them as templates, not the answer.
- The examples are PATTERNS to inspire — do NOT instruct the inner LLM to copy them. Tell it to design something for THIS task using the same shape.

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
