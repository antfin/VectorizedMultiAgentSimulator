"""v7 meta-strategist — strategy bundle enumeration + grounded reflection.

Two distinct meta-LLM call modes:

  1. enumerate_bundle()   — cold-start. Asks the meta-LLM to propose
                            3-5 full-solution policy-level strategies,
                            score each by lero_codability + rl_trainability,
                            and translate the chosen one to slot text.

  2. reflect_and_decide() — per outer iter. Given inner result + active
                            strategy + automatically-computed diagnosis,
                            asks the meta-LLM to either:
                              - confirm "achieved" → stop
                              - confirm "partial" → refine same strategy
                              - confirm "translation_failure" → sharpen
                                inner prompt for current strategy
                              - confirm "rl_too_hard" → switch to next-
                                best strategy from bundle

The code-side diagnosis (V7Diagnosis) is computed BEFORE the meta-LLM
call and FED to the meta-LLM as a fact. The meta-LLM's job is to
write the next slot text + (optionally) propose bundle updates,
NOT to re-classify the result.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..llm_client import LLMClient
from ..v5.inner_loop import InnerResult
from .diagnosis import V7Diagnosis, diagnose_inner_result
from .strategy import (
    DiagnosisLabel,
    SuccessSignature,
    V7Strategy,
    V7StrategyBundle,
)

_log = logging.getLogger("rendezvous.lero.v7.meta")


# ── System prompts ─────────────────────────────────────────────


_BUNDLE_SYSTEM = """## Role

You are a research strategist for a multi-agent reinforcement learning experiment. Your job is to enumerate POLICY-LEVEL hypotheses about how the agents could solve the task, rank them by feasibility, and translate the best one into operational guidance for an inner LLM that writes Python observation code.

## What you produce

A STRATEGY BUNDLE of 3-5 full-solution hypotheses. Each strategy is a *macro* description of what the agents do (e.g. "pairs commit on shared target"), NOT a feature list. Score each on:

- **lero_codability** (0-10): how easily this strategy can be expressed via observation features (or reward shaping) the inner LLM can write in PyTorch from local LiDAR + own pos/vel
- **rl_trainability** (0-10): how easily PPO with shared-policy MAPPO can learn the resulting policy in 10M frames given the features

Strategies that require centralized communication, multi-step planning, or non-local information should be filtered out — the inner LLM only has lidar_targets, lidar_agents, agent_pos, agent_vel, agent_idx.

The best strategy (highest combined score) becomes the active strategy for outer iter 0. Write `lero_translation_hint` for it: a paragraph of operational vocabulary the inner LLM will use, in V4-V5 style (cross-source operations, decision framing, optional 1-3 pseudo-PyTorch example expressions). Do NOT name specific solution features by handle (no `hold_signal`, `approach_signal`, etc. — runtime grep redacts).

## Critical: the lero_translation_hint must request a COMPLETE feature stack

Prior runs showed that emphasizing only cross-source decision features causes the inner LLM to drop spatial primitives (direction, role, motion) that the policy needs to act on the decision signal. A working observation function for this task includes ALL of:

1. **Spatial primitives**: directional encoding such as `cos(angle_of_smallest_target_ray)` and `sin(angle_of_smallest_target_ray)` — without these, the policy must learn directional extraction from the raw 15-ray LiDAR, which is hard at 1M frames.
2. **Role differentiation**: `agent_idx` as one-hot over `n_agents` so the shared policy can break symmetry between agents.
3. **Motion / state**: scalar speed or velocity components, optional boundary distance.
4. **Proximity summaries**: nearest-target distance, nearest-agent distance, count of close rays.
5. **Cross-source decision features**: products / boolean conjunctions / signed differences combining target-derived and agent-derived quantities, GATED at `covering_range` (the actual covering threshold), NOT at `lidar_range` (the sensor max).

Always instruct the inner LLM to include items 1-5 alongside the strategy's specific cross-source pattern. Use `covering_range` (typically 0.25) as the threshold for boolean masks, not `lidar_range` (typically 0.35).

## Output format

```json
{
  "strategies": [
    {
      "name": "<short_handle>",
      "full_solution": "<2-3 sentences: what should agents do at the policy level>",
      "ast_pattern_description": "<one-line: structural pattern the inner code should exhibit>",
      "expected_M1_at_1M": <float, default 0.05>,
      "expected_M6_at_1M_min": <float, default 0.20>,
      "lero_translation_hint": "<paragraph of operational guidance for the chosen strategy ONLY (or empty for non-chosen)>",
      "lero_codability": <int 0-10>,
      "rl_trainability": <int 0-10>
    },
    ...
  ],
  "chosen_idx": <index of highest-scoring strategy>
}
```

Talk about HOW the policy decides, not WHAT specific feature to compute. The translation hint is operational; the strategies themselves are policy-level."""


_REFLECT_SYSTEM = """## Role

You are reflecting on a single inner-LERO outcome to decide what to do next. The diagnosis has already been COMPUTED for you from raw metrics + AST pattern detection — you do NOT re-classify, you respond to the diagnosis.

## Inputs

You will receive:
  - the active V7Strategy
  - the V7StrategyBundle (with all alternative strategies + history)
  - the inner result (best+worst code excerpts, metrics)
  - the V7Diagnosis (label, pattern_present, metrics_signature_match, rationale)

## What you decide

Given the diagnosis label, take exactly one of these actions:

  - **achieved** → emit `next_action="stop"`. We're done.
  - **partial** → emit `next_action="refine_current_strategy"`. Sharpen the current strategy's slot text; do NOT switch.
  - **translation_failure** → emit `next_action="refine_inner_prompt_for_current"`. The strategy is right but inner LLM didn't realize it; rewrite the slot text more concretely (more parallel bullets, more pseudo-PyTorch examples, more explicit cross-source instruction).
  - **rl_too_hard** → emit `next_action="switch_to_next_strategy"`. The pattern is in the code but PPO didn't learn at 1M. Pick the next-best non-excluded strategy from the bundle. Mark current as `rl_too_hard` (the bundle handles this).

You may ALSO propose bundle updates: add a new strategy you've thought of after observing this round, or demote a strategy that's clearly never going to work. Bundle updates are OPTIONAL.

## Anti-cheat

- Do NOT name S3b-local-style winning features (hold_signal, approach_signal, settle_signal, etc.) — runtime grep redacts.
- Talk about families and operations, not solution names.
- Include 1-3 short pseudo-PyTorch example expressions inline if it helps the inner LLM see the pattern.

## Output format

```json
{
  "next_action": "stop" | "refine_current_strategy" | "refine_inner_prompt_for_current" | "switch_to_next_strategy",
  "rationale": "<2-3 sentences>",
  "slot_edits": {
    "guidance_observation": "<full replacement>",
    "guidance_shared": "<full replacement>"
  },
  "bundle_update": {
    "demote": ["<strategy_name>", ...],
    "add": [<new V7Strategy in same JSON shape as bundle entries>]
  }
}
```"""


# ── Bundle enumeration call ────────────────────────────────────


def _build_bundle_prompt(task_summary: str) -> str:
    return f"""[TASK SUMMARY]
{task_summary}

[OBSERVATION CONSTRAINTS]
The inner LLM may read only:
  - agent_pos, agent_vel (own proprioception)
  - agent_idx, n_agents, n_targets, covering_range, agents_per_target_required
  - lidar_targets [B, n_rays] — distance to nearest TARGET per ray
  - lidar_agents  [B, n_rays] — distance to nearest other AGENT per ray

Filter strategies that require: explicit comms, oracle access to other agents' positions, centralized planning, multi-step memory beyond single-step velocity, or supervised role assignment.

[YOUR TASK]
Emit a STRATEGY BUNDLE of 3-5 full-solution hypotheses ranked by combined feasibility. For the highest-scoring one, write a complete operational `lero_translation_hint`. Output the JSON object per the format in your instructions."""


def _parse_strategy(blob: dict) -> V7Strategy:
    sig = SuccessSignature(
        ast_pattern_description=str(blob.get("ast_pattern_description", "")),
        expected_M1_at_1M=float(blob.get("expected_M1_at_1M", 0.05)),
        expected_M6_at_1M_min=float(blob.get("expected_M6_at_1M_min", 0.20)),
    )
    return V7Strategy(
        name=str(blob["name"]),
        full_solution=str(blob.get("full_solution", "")),
        success_signature=sig,
        lero_translation_hint=str(blob.get("lero_translation_hint", "")),
        lero_codability=int(blob.get("lero_codability", 5)),
        rl_trainability=int(blob.get("rl_trainability", 5)),
    )


def enumerate_bundle(
    meta_llm: LLMClient,
    task_summary: str,
) -> Tuple[V7StrategyBundle, str]:
    """Cold-start: ask the meta-LLM to enumerate strategies and rank.
    Returns (bundle, raw_text). Raises on parse failure after 3 attempts."""
    prompt = _build_bundle_prompt(task_summary)
    last_err: Optional[Exception] = None
    raw = ""
    for attempt in range(1, 4):
        raw = meta_llm.generate(
            [
                {"role": "system", "content": _BUNDLE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            n=1,
        )[0]
        try:
            js = raw.find("{")
            je = raw.rfind("}")
            data = json.loads(raw[js:je + 1])
            strategies = [_parse_strategy(s) for s in data["strategies"]]
            chosen = int(data.get("chosen_idx", 0))
            chosen = max(0, min(len(strategies) - 1, chosen))
            bundle = V7StrategyBundle(
                strategies=strategies, chosen_idx=chosen,
            )
            _log.info(
                "v7 bundle: %d strategies, chosen='%s' (score=%.1f)",
                len(strategies),
                strategies[chosen].name,
                strategies[chosen].combined_score,
            )
            return bundle, raw
        except Exception as e:  # noqa: BLE001
            last_err = e
            _log.warning("bundle parse fail attempt %d: %s", attempt, e)
    raise ValueError(f"v7 bundle enum failed: {last_err}")


# ── Reflection / decision call ─────────────────────────────────


@dataclass
class V7ReflectionDecision:
    next_action: str         # stop | refine_current_strategy | refine_inner_prompt_for_current | switch_to_next_strategy
    rationale: str
    slot_edits: Dict[str, str] = field(default_factory=dict)
    bundle_demote: List[str] = field(default_factory=list)
    bundle_add: List[V7Strategy] = field(default_factory=list)


def _build_reflect_prompt(
    bundle: V7StrategyBundle,
    inner: InnerResult,
    diagnosis: V7Diagnosis,
) -> str:
    cur = bundle.current()
    best_code = (inner.best.candidate.obs_source if inner.best else "")[:1500]
    worst_code = (inner.worst.candidate.obs_source
                  if inner.worst and inner.worst is not inner.best
                  else "")[:800]
    return f"""[ACTIVE STRATEGY]
name: {cur.name}
full_solution: {cur.full_solution}
success_signature_pattern: {cur.success_signature.ast_pattern_description}
expected_M1_at_1M: {cur.success_signature.expected_M1_at_1M}
expected_M6_at_1M_min: {cur.success_signature.expected_M6_at_1M_min}

[BUNDLE STATE]
{bundle.format_for_prompt()}

[INNER RESULT THIS ROUND]
best M1 = {diagnosis.inner_M1:.3f}
best M6 = {diagnosis.inner_M6:.3f}
best obs code:
```python
{best_code}
```
worst obs code:
```python
{worst_code}
```

[CODE-SIDE DIAGNOSIS — TREAT AS FACT]
label: {diagnosis.label}
pattern_present (auto AST check): {diagnosis.pattern_present}
metrics_signature_match: {diagnosis.metrics_signature_match}
rationale: {diagnosis.rationale}

[YOUR TASK]
Given the diagnosis label is **{diagnosis.label}**, choose `next_action`:

  - achieved → next_action = "stop", empty slot_edits
  - partial → next_action = "refine_current_strategy", sharpen guidance
  - translation_failure → next_action = "refine_inner_prompt_for_current", make slot text MORE concrete (more parallel bullets, more pseudo-PyTorch examples, more explicit cross-source instruction)
  - rl_too_hard → next_action = "switch_to_next_strategy", emit empty slot_edits (the next strategy's hint will be used). Optionally demote current via bundle_update.

Output the JSON per your system instructions."""


def reflect_and_decide(
    meta_llm: LLMClient,
    bundle: V7StrategyBundle,
    inner: InnerResult,
    diagnosis: V7Diagnosis,
) -> Tuple[V7ReflectionDecision, str]:
    prompt = _build_reflect_prompt(bundle, inner, diagnosis)
    last_err: Optional[Exception] = None
    raw = ""
    for attempt in range(1, 4):
        raw = meta_llm.generate(
            [
                {"role": "system", "content": _REFLECT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            n=1,
        )[0]
        try:
            js = raw.find("{")
            je = raw.rfind("}")
            data = json.loads(raw[js:je + 1])
            slot_edits = dict(data.get("slot_edits") or {})
            bu = data.get("bundle_update") or {}
            demote = list(bu.get("demote") or [])
            add_blobs = bu.get("add") or []
            add: List[V7Strategy] = []
            for b in add_blobs:
                try:
                    add.append(_parse_strategy(b))
                except Exception as e:  # noqa: BLE001
                    _log.warning("v7 reflect: bundle.add parse skipped: %s", e)
            return V7ReflectionDecision(
                next_action=str(data["next_action"]),
                rationale=str(data.get("rationale", "")),
                slot_edits=slot_edits,
                bundle_demote=demote,
                bundle_add=add,
            ), raw
        except Exception as e:  # noqa: BLE001
            last_err = e
            _log.warning("v7 reflect parse fail attempt %d: %s", attempt, e)
    raise ValueError(f"v7 reflect failed: {last_err}")
