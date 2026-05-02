"""v8 meta-strategist — density-first prompts + meta-authored fewshot.

Inherits v7's structure (strategy bundle, two-call mode: enumerate +
reflect) and adds:

  - Configurable feature count target / cap (15 default)
  - Configurable gated-feature cap (2 default)
  - Density-first preference instruction
  - Meta-LLM authored 3-4 feature WORKING fewshot inside slot text
  - Two new next_action values: trim_features, replace_gated_with_dense

The bundle data class is imported from v7 unchanged.
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
from ..v7.meta_strategist import _parse_strategy
from ..v7.strategy import V7Strategy, V7StrategyBundle
from .diagnosis import V8Diagnosis

_log = logging.getLogger("rendezvous.lero.v8.meta")


def _bundle_system(
    feature_target_min: int,
    feature_target_max: int,
    feature_cap: int,
    gated_cap: int,
) -> str:
    return f"""## Role

You are a research strategist for a multi-agent reinforcement learning experiment. Enumerate POLICY-LEVEL hypotheses, rank by feasibility, translate the best one into operational guidance for an inner LLM that writes Python observation code.

## What you produce

A STRATEGY BUNDLE of 3-5 full-solution hypotheses. Each strategy is a *macro* description of what agents do (e.g. "pairs commit on shared target"), NOT a feature list. Score each on:

- **lero_codability** (0-10): how easily this strategy can be expressed via observation features the inner LLM can write in PyTorch from local LiDAR + own pos/vel
- **rl_trainability** (0-10): how easily PPO with shared-policy MAPPO can learn the resulting policy in 10M frames

Strategies that require explicit comms, oracle access, centralized planning, or supervised role assignment must be filtered.

## Density-first feature design (v8)

The inner LLM at 1M frames learns better from FEW, DENSE features than MANY, COVER-ZONE-GATED features. PPO needs gradient signal everywhere, not just inside the cover zone where the agent rarely visits early.

Rules for the chosen strategy's `lero_translation_hint`:

1. **Target {feature_target_min}-{feature_target_max} returned features. Hard cap: {feature_cap}.** More features = more noise. Quality over quantity.

2. **Limit cover-zone-gated features to at most {gated_cap}** — features that are zero outside cover_r are useful as a "stay vs go" trigger but multiplying many of them adds no information.

3. **Prefer DENSE signals**: features producing informative values everywhere in state space:
   - mean of LiDAR rays below a threshold (still informative when nothing in cover zone)
   - std/variance over rays (uncertainty proxy)
   - count-asymmetry between target and agent ray-counts NORMALIZED by their respective totals (a continuous signed value, NOT gated by cover_r)
   - boundary distance from arena edge
   - directional encoding (cos/sin of argmin ray angle)

4. **Include a 3-4 feature WORKING fewshot inside `lero_translation_hint`** — a fenced ```python``` block with a partial `enhance_observation` showing 3-4 representative features ending with `return torch.cat(...)`. Make at least one feature DENSE. Use neutral variable names that describe the role, NOT S3b-local-style coordination handles. Example shape (do NOT copy verbatim, adapt to your strategy):

```python
def enhance_observation(scenario_state: dict) -> torch.Tensor:
    lidar_t = scenario_state["lidar_targets"]
    cover_r = float(scenario_state["covering_range"])
    nearest_target_dist = lidar_t.min(dim=-1).values            # DENSE
    count_close = (lidar_t < cover_r).float().sum(dim=-1)        # SEMI-DENSE
    std_targets = lidar_t.std(dim=-1)                            # DENSE
    # ... 1 more feature relevant to your strategy ...
    return torch.cat([nearest_target_dist.unsqueeze(-1),
                      count_close.unsqueeze(-1),
                      std_targets.unsqueeze(-1)], dim=-1)
```

## Output format

```json
{{
  "strategies": [
    {{
      "name": "<short_handle>",
      "full_solution": "<2-3 sentences: macro policy intent>",
      "ast_pattern_description": "<one-line: structural pattern in code>",
      "expected_M1_at_1M": <float, default 0.05>,
      "expected_M6_at_1M_min": <float, default 0.20>,
      "lero_translation_hint": "<paragraph + 3-4 feature working fewshot in fenced python block>",
      "lero_codability": <int 0-10>,
      "rl_trainability": <int 0-10>
    }},
    ...
  ],
  "chosen_idx": <index>
}}
```

Anti-cheat: do NOT use S3b-local-style winning feature names (hold_signal, approach_signal, settle_signal, etc.). Runtime grep redacts forbidden tokens."""


_REFLECT_SYSTEM_V8 = """## Role

You reflect on a single inner-LERO outcome and decide what to do next. The diagnosis is COMPUTED for you from raw metrics + AST pattern detection — you do NOT re-classify.

## v8 next_action values

Given the diagnosis label, pick exactly one:

  - **achieved** → `next_action="stop"`. Done.
  - **partial** → `next_action="refine_current_strategy"`. Sharpen guidance.
  - **too_many_features** → `next_action="trim_features"`. The candidate returned too many features. Identify 5-10 to remove (by name from the analyzer report). Rewrite slot text requesting a tighter feature set.
  - **over_gated** → `next_action="replace_gated_with_dense"`. Too many cover-zone-gated features; not enough dense. Rewrite slot text emphasizing mean / std / normalized-count-asymmetry / boundary distance.
  - **translation_failure** → `next_action="refine_inner_prompt_for_current"`. Strategy is right, inner didn't realize it. Make slot text more concrete.
  - **rl_too_hard** → `next_action="switch_to_next_strategy"`. Pattern present, simple enough, but PPO can't learn.

## Constraints when refining (v8)

- The CHOSEN strategy's success_signature.expected_features still applies; respect the feature count cap.
- When trimming, prefer keeping: directional encoding (cos/sin), role one-hot, 1-2 dense density signals (mean/std), 1 cover-zone-gated decision feature, 2-3 nearest distances.
- When replacing gated with dense, prefer ungated `count_target_normalized − count_agent_normalized` over `(min_t < cover_r) * (count_a ≥ 1)`.

Anti-cheat: do NOT name S3b-local handles. Talk about families and operations, not solution names.

## Output format

```json
{
  "next_action": "stop" | "refine_current_strategy" | "trim_features" | "replace_gated_with_dense" | "refine_inner_prompt_for_current" | "switch_to_next_strategy",
  "rationale": "<2-3 sentences>",
  "slot_edits": {
    "guidance_observation": "<full replacement text — INCLUDE a 3-4 feature pseudo-PyTorch fewshot if action is refine/trim/replace>",
    "guidance_shared": "<full replacement or omit>"
  },
  "bundle_update": {
    "demote": ["<strategy_name>", ...],
    "add": [<new V7Strategy in JSON shape from system instructions>]
  }
}
```"""


# ── Bundle enumeration ──────────────────────────────────────────


def _build_bundle_prompt_v8(
    task_summary: str,
    feature_target_min: int,
    feature_target_max: int,
    feature_cap: int,
    gated_cap: int,
) -> str:
    return f"""[TASK SUMMARY]
{task_summary}

[OBSERVATION CONSTRAINTS]
The inner LLM may read only:
  - agent_pos, agent_vel (own proprioception)
  - agent_idx, n_agents, n_targets, covering_range, agents_per_target_required
  - lidar_targets [B, n_rays] — distance to nearest TARGET per ray
  - lidar_agents  [B, n_rays] — distance to nearest other AGENT per ray

[FEATURE BUDGET FOR THIS RUN]
  - Target output features: {feature_target_min}-{feature_target_max}
  - Hard cap: {feature_cap}
  - Cover-zone-gated features cap: {gated_cap}

[YOUR TASK]
Emit a STRATEGY BUNDLE of 3-5 full-solution hypotheses ranked by combined feasibility. For the highest-scoring one, write a complete operational `lero_translation_hint` INCLUDING a 3-4 feature working pseudo-PyTorch fewshot. Output the JSON object."""


def enumerate_bundle_v8(
    meta_llm: LLMClient,
    task_summary: str,
    feature_target_min: int,
    feature_target_max: int,
    feature_cap: int,
    gated_cap: int,
) -> Tuple[V7StrategyBundle, str]:
    system = _bundle_system(
        feature_target_min, feature_target_max, feature_cap, gated_cap,
    )
    user = _build_bundle_prompt_v8(
        task_summary, feature_target_min, feature_target_max,
        feature_cap, gated_cap,
    )
    last_err: Optional[Exception] = None
    raw = ""
    for attempt in range(1, 4):
        raw = meta_llm.generate(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
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
            bundle = V7StrategyBundle(strategies=strategies, chosen_idx=chosen)
            _log.info(
                "v8 bundle: %d strategies, chosen='%s' (score=%.1f)",
                len(strategies), strategies[chosen].name,
                strategies[chosen].combined_score,
            )
            return bundle, raw
        except Exception as e:  # noqa: BLE001
            last_err = e
            _log.warning("v8 bundle parse fail attempt %d: %s", attempt, e)
    raise ValueError(f"v8 bundle enum failed: {last_err}")


# ── Reflection ──────────────────────────────────────────────────


@dataclass
class V8ReflectionDecision:
    next_action: str
    rationale: str
    slot_edits: Dict[str, str] = field(default_factory=dict)
    bundle_demote: List[str] = field(default_factory=list)
    bundle_add: List[V7Strategy] = field(default_factory=list)


def _build_reflect_prompt_v8(
    bundle: V7StrategyBundle,
    inner: InnerResult,
    diagnosis: V8Diagnosis,
    feature_target_min: int,
    feature_target_max: int,
    feature_cap: int,
    gated_cap: int,
) -> str:
    cur = bundle.current()
    best_code = (inner.best.candidate.obs_source if inner.best else "")[:1500]
    worst_code = (inner.worst.candidate.obs_source
                  if inner.worst and inner.worst is not inner.best
                  else "")[:800]
    return f"""[ACTIVE STRATEGY]
{cur.name}: {cur.full_solution}
success_signature: {cur.success_signature.ast_pattern_description}

[BUNDLE STATE]
{bundle.format_for_prompt()}

[INNER RESULT THIS ROUND]
M1={diagnosis.inner_M1:.3f} M6={diagnosis.inner_M6:.3f}
n_features={diagnosis.n_features} (cap={feature_cap}, target {feature_target_min}-{feature_target_max})
n_gated={diagnosis.n_gated} (cap={gated_cap})
n_dense_estimate={diagnosis.n_dense}

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
pattern_present: {diagnosis.pattern_present}
metrics_signature_match: {diagnosis.metrics_signature_match}
rationale: {diagnosis.rationale}

[YOUR TASK]
Given the diagnosis label is **{diagnosis.label}**, choose the corresponding `next_action` per your system instructions. Output the JSON object."""


def reflect_and_decide_v8(
    meta_llm: LLMClient,
    bundle: V7StrategyBundle,
    inner: InnerResult,
    diagnosis: V8Diagnosis,
    feature_target_min: int,
    feature_target_max: int,
    feature_cap: int,
    gated_cap: int,
) -> Tuple[V8ReflectionDecision, str]:
    user = _build_reflect_prompt_v8(
        bundle, inner, diagnosis,
        feature_target_min, feature_target_max,
        feature_cap, gated_cap,
    )
    last_err: Optional[Exception] = None
    raw = ""
    for attempt in range(1, 4):
        raw = meta_llm.generate(
            [
                {"role": "system", "content": _REFLECT_SYSTEM_V8},
                {"role": "user", "content": user},
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
                except Exception:  # noqa: BLE001
                    pass
            return V8ReflectionDecision(
                next_action=str(data["next_action"]),
                rationale=str(data.get("rationale", "")),
                slot_edits=slot_edits,
                bundle_demote=demote,
                bundle_add=add,
            ), raw
        except Exception as e:  # noqa: BLE001
            last_err = e
            _log.warning("v8 reflect parse fail attempt %d: %s", attempt, e)
    raise ValueError(f"v8 reflect failed: {last_err}")
