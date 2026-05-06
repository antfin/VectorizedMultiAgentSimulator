"""v9 meta-strategist — CoT bundle enumeration + combined diagnose+reflect.

Two LLM call types:

  1. enumerate_bundle_v9 — one-shot at outer 0. Reads task_domain.yaml
     (inferable_concepts, mandatory_features, coordination_challenges,
     forbidden_tokens, feature_budget) + a task summary. Outputs:

       {
         "task_understanding": "...",
         "strategies": [
           {"name", "full_solution",
            "chain_of_thought": {"why_it_works", "what_is_needed",
                                  "failure_modes"},
            "expected_M1_at_1M", "expected_M6_at_1M_min",
            "lero_codability", "rl_trainability",
            "ast_pattern_description"},
           ...
         ],
         "chosen_idx": <int>,
         "chosen_strategy_artifacts": {
           "inferable_hints_text": "...",   # "What you CAN infer" block
           "examples_text": "...",           # 2-3 fenced ```python``` examples
           "feedback_template": "..."        # strategy-specific reminder
         }
       }

  2. reflect_decide_v9 — one combined call replacing v8's separate
     diagnose + reflect. Receives:
       - bundle state (current chosen, pending, excluded)
       - inner_result analyzer facts (n_features, n_gated, n_dense,
         touches_both_lidars, role_one_hot_present, M1, M6)
       - last N memory rows (per §5)

     Outputs (JSON):
       {
         "memory_recall": "...",
         "current_outcome_reading": {
           "label": "achieved|partial|translation_failure|rl_too_hard|too_early",
           "diff_vs_predicted": "..."
         },
         "reflection_chain_of_thought": {
           "what_went_right": [...],
           "what_went_wrong": [...],
           "remaining_uncertainty": [...]
         },
         "next_action": "stop|refine_current|switch_to_next",
         "rationale": "...",
         "slot_edits": {
           "inferable_hints": "...",
           "examples": "...",
           "feedback_template": "..."
         },
         "bundle_update": {"demote": [...], "add": [...]}
       }

Anti-cheat: forbidden_tokens from task_domain.yaml are listed in the
system message and the runtime grep redacts any leaks.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..llm_client import LLMClient
from ..prompts.loader import PromptLoader
from .strategy import (
    V9Artifacts,
    V9Bundle,
    V9ChainOfThought,
    V9Strategy,
    V9SuccessSignature,
)

_log = logging.getLogger("rendezvous.lero.v9.meta")


# ── System prompts ───────────────────────────────────────────────


def _bundle_system(td: Dict[str, Any]) -> str:
    """Bundle-enumeration system message, parameterized by task_domain."""
    inferable = td.get("inferable_concepts") or []
    mandatory = td.get("mandatory_features") or []
    forbidden = td.get("forbidden_tokens") or []
    budget = td.get("feature_budget") or {}
    target_min = budget.get("target_min", 12)
    target_max = budget.get("target_max", 17)
    hard_cap = budget.get("hard_cap", 20)
    gated_max = budget.get("gated_max", 3)

    inferable_lines = "\n".join(f"  - {c['concept']} → {c['idiom']}" for c in inferable)
    mandatory_lines = "\n".join(
        f"  - **{m['name']}** ({m.get('idiom','')}): {m['reason'].strip()}"
        for m in mandatory
    )
    forbidden_line = ", ".join(f"`{t}`" for t in forbidden)

    return f"""## Role

You are a research strategist for a multi-agent reinforcement learning experiment. Your job is to enumerate POLICY-LEVEL hypotheses for how agents could solve this task, reason about each via chain-of-thought, then translate the best hypothesis into operational text for an inner LLM that writes Python observation code.

## What you produce — bundle of 3-5 strategies + artifacts for the chosen one

For each strategy:
  - **name** (short_handle), **full_solution** (2-3 sentences)
  - **chain_of_thought**: explicit reasoning BEFORE you author any prompt text
      - `why_it_works`: link the macro to the reward + observation
      - `what_is_needed`: structural requirements (e.g., "agents need role differentiation")
      - `failure_modes`: what would prevent PPO at 1M frames from learning this
  - **lero_codability** (0-10): how easily features can be written in PyTorch from local LiDAR + own pos/vel
  - **rl_trainability** (0-10): how easily PPO with shared-policy MAPPO can learn it in 10M
  - **ast_pattern_description**: one line capturing the structural code signature
  - **expected_M1_at_1M**, **expected_M6_at_1M_min**

Then choose ONE strategy and author three artifact texts for it (these become inner-prompt slots):

### `inferable_hints_text` — "What you CAN infer" block

A bulleted markdown section modeled on the shape:

  ## What you CAN infer from the local sensors
  - Direction to nearest target: argmin over lidar_targets → cos/sin of 2π·idx/n_rays
  - Distance to nearest target: lidar_targets.min(dim=-1).values
  - ...
  - Agent role under shared-policy MAPPO: F.one_hot(agent_idx, n_agents)
  - ...

EVERY entry from the task_domain inferable_concepts list below MUST appear:

{inferable_lines}

In addition, mention the mandatory_features (with the structural reasoning, not just the name):

{mandatory_lines}

### `examples_text` — 2-3 worked PyTorch examples

Each example is a complete function in a fenced ```python``` block.

**Function signature is FIXED (do not change)**:

```python
def enhance_observation(scenario_state: dict) -> torch.Tensor:
    lidar_t = scenario_state["lidar_targets"]
    lidar_a = scenario_state["lidar_agents"]
    agent_pos = scenario_state["agent_pos"]
    agent_vel = scenario_state["agent_vel"]
    agent_idx = scenario_state["agent_idx"]
    n_agents = int(scenario_state["n_agents"])
    cover_r = float(scenario_state["covering_range"])
    ...
```

**SHAPE WARNING — READ CAREFULLY**:

`lidar_targets` has shape `[B, n_target_rays]` (15 rays). `lidar_agents` has shape `[B, n_agent_rays]` (12 rays). **These are NOT the same shape.** Do NOT write any element-wise op between them — it WILL crash at runtime with a broadcast error. Specifically these patterns are FORBIDDEN and will fail:

  - `(lidar_targets < r) & (lidar_agents < r)`     ← broadcast error
  - `lidar_targets * lidar_agents`                  ← broadcast error
  - `(lidar_targets - lidar_agents)`                ← broadcast error
  - any op that puts the two raw arrays on opposite sides of an op

**Correct cross-source pattern**: reduce each LiDAR to a SCALAR per batch first, then combine the scalars:

  - `t_count = (lidar_targets < r).float().sum(dim=-1)`   # [B]
  - `a_count = (lidar_agents < r).float().sum(dim=-1)`   # [B]
  - `crowd_diff = t_count - a_count`                      # [B] ← this is fine

Together the examples should:
  - Cover the strategy from a different angle each time
  - At least ONE example MUST include role one-hot (`F.one_hot(agent_idx, n_agents)` or `one_hot[:, agent_idx] = 1.0`)
  - Together include at least one cross-source feature (combining target-derived and agent-derived SCALARS — see pattern above)
  - Stay within feature budget: target {target_min}-{target_max} features per example, hard cap {hard_cap}, ≤{gated_max} cover-zone-gated features per example

Style: descriptive variable names (no S3b handles), brief comments.

### `feedback_template` — strategy-specific reminder

A 2-3 sentence reminder that gets appended to feedback.txt between inner iters. Should encode the strategy's `what_is_needed` so the inner LLM stays on track when refining. Example shape: "If your candidate has M1=0 at this iter, check that role one-hot is present AND that you have at least one cross-source feature combining target and agent counts. The strategy `<name>` requires both."

## Anti-cheat

Do NOT use these S3b-local-style winning feature names ANYWHERE in any text you author: {forbidden_line}. Runtime grep redacts these tokens and replaces them with `<REDACTED>`.

## Output format — STRICT

A SINGLE JSON object with EXACTLY these top-level keys (no others):

```json
{{
  "task_understanding": "<2-3 sentences about the task>",
  "strategies": [
    {{ "name": "...", "full_solution": "...",
       "chain_of_thought": {{
          "why_it_works": "...",
          "what_is_needed": ["...", "..."],
          "failure_modes": ["...", "..."]
       }},
       "lero_codability": <int>, "rl_trainability": <int>,
       "ast_pattern_description": "...",
       "expected_M1_at_1M": <float>, "expected_M6_at_1M_min": <float>
    }},
    ...3-5 strategies total...
  ],
  "chosen_idx": <integer index into strategies>,
  "chosen_strategy_artifacts": {{
    "inferable_hints_text": "...",
    "examples_text": "...",
    "feedback_template": "..."
  }}
}}
```

Strict JSON: NO trailing commas. NO comments. NO prose outside the JSON object. The three artifact texts MUST be nested under `chosen_strategy_artifacts`, not at the top level.
"""


def _reflect_system(td: Dict[str, Any]) -> str:
    """Combined diagnose+reflect system message."""
    forbidden = td.get("forbidden_tokens") or []
    forbidden_line = ", ".join(f"`{t}`" for t in forbidden)
    return f"""## Role

You reflect on a single inner-LERO outcome and decide what to do next. The AST analyzer has computed structural facts about the best inner candidate; treat them as given. You diagnose the outcome AND choose the action AND author any prompt edits in a SINGLE response.

## Diagnosis labels (pick one)

  - **achieved**:           M1 ≥ expected_M1 AND has_cross_source AND role_one_hot_present
  - **partial**:            M6 ≥ expected_M6_min but M1 below
  - **translation_failure**: pattern absent (no cross-source op, OR no role one-hot when mandatory)
  - **rl_too_hard**:        pattern present, simple enough, but PPO did not learn at 1M
  - **too_early**:          inner result missing or empty

## Next-action map

  - achieved              → stop
  - partial               → refine_current (sharpen current strategy's slot text)
  - translation_failure   → refine_current (rewrite slot text more concretely)
  - rl_too_hard           → switch_to_next (move to next pending strategy in bundle)
  - too_early             → refine_current

## Memory — REQUIRED to drive your decision (v9.1 §2.6)

You will receive the last N=3 memory rows (your prior CoT + actual outcome).
Begin your `memory_recall` by quoting the SPECIFIC predicted vs actual numbers (e.g. "outer 0: predicted M1=0.18, actual M1=0.01"). Memory is not just descriptive — it MUST drive your decision per these rules:

1. **Falsification rule** — if the same strategy has been attempted ≥2 times AND every attempt's actual M1 is below 0.5× the strategy's expected_M1_at_1M, you MUST recommend `next_action = "switch_to_next"`. This is non-negotiable. Do NOT pick `refine_current` to give the strategy "one more chance"; the runtime will override that choice anyway (v9.1 §2.7 falsification gate).

2. **Slot-edit conservation rule** — if `refine_current` is chosen, the new `inferable_hints` and `examples` text MUST preserve every concept and Python example that was in the prior outer's slot text. Specifically: if a prior `what_is_needed` item said "role differentiation" or "cross-source feature", your new slot text must continue to surface those, not drop them. The runtime will reject slot_edits that strip mandatory features (v9.1 §2.3 slot-edit validator).

3. **Concrete-fix rule** — when `refine_current`, propose ONE specific code-level change (e.g. "use `exp(-α·d)` instead of `mean(d)`"), not vague rewording ("sharpen the wording"). Production CoT showed three consecutive outers proposing "sharpen the wording" with no improvement.

## Slot edits (when refine_current or switch_to_next)

When refining or switching, REWRITE the affected slots:
  - `inferable_hints` (full text replacement) — must still cover all task_domain inferable_concepts and mandatory_features.
  - `examples` (full text replacement) — at least one example with role one-hot, ≥2 examples total.
  - `feedback_template` (full text replacement) — strategy-specific reminder.

When switching, also update bundle: `bundle_update.demote` excludes the current strategy and a new chosen_idx is implied by next_action.

## Anti-cheat

Do NOT use these tokens in any text: {forbidden_line}.

## Output format

A SINGLE JSON object with this shape:

```json
{{
  "memory_recall": "...",
  "current_outcome_reading": {{
    "label": "achieved|partial|translation_failure|rl_too_hard|too_early",
    "diff_vs_predicted": "..."
  }},
  "reflection_chain_of_thought": {{
    "what_went_right": ["..."],
    "what_went_wrong": ["..."],
    "remaining_uncertainty": ["..."]
  }},
  "next_action": "stop|refine_current|switch_to_next",
  "rationale": "...",
  "slot_edits": {{
    "inferable_hints": "<text or empty string>",
    "examples": "<text or empty string>",
    "feedback_template": "<text or empty string>"
  }},
  "bundle_update": {{"demote": ["<name>", "..."], "add": []}}
}}
```
"""


# ── Parsing ───────────────────────────────────────────────────────


def _redact_forbidden(text: str, forbidden: List[str]) -> str:
    if not text:
        return text
    out = text
    for tok in forbidden:
        out = re.sub(re.escape(tok), "<REDACTED>", out, flags=re.IGNORECASE)
    return out


def _parse_strategy_v9(blob: Dict[str, Any], forbidden: List[str]) -> V9Strategy:
    cot = blob.get("chain_of_thought") or {}
    sig = V9SuccessSignature(
        ast_pattern_description=blob.get("ast_pattern_description", ""),
        expected_M1_at_1M=float(blob.get("expected_M1_at_1M", 0.05)),
        expected_M6_at_1M_min=float(blob.get("expected_M6_at_1M_min", 0.20)),
    )
    return V9Strategy(
        name=str(blob["name"]),
        full_solution=_redact_forbidden(blob.get("full_solution", ""), forbidden),
        success_signature=sig,
        chain_of_thought=V9ChainOfThought(
            why_it_works=_redact_forbidden(cot.get("why_it_works", ""), forbidden),
            what_is_needed=[
                _redact_forbidden(s, forbidden) for s in cot.get("what_is_needed", [])
            ],
            failure_modes=[
                _redact_forbidden(s, forbidden) for s in cot.get("failure_modes", [])
            ],
        ),
        lero_codability=int(blob.get("lero_codability", 5)),
        rl_trainability=int(blob.get("rl_trainability", 5)),
    )


def _extract_json(raw: str) -> Dict:
    """Find the outermost JSON object in `raw` and parse it.

    LLMs (gpt-5.4-mini in particular) frequently emit trailing commas
    in arrays/objects that strict JSON rejects. We strip those before
    parsing. We also handle the common pattern where the response is
    wrapped in ```json ... ``` fenced blocks.
    """
    text = raw
    # Prefer a ```json ... ``` fenced block when present
    fence = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        js = text.find("{")
        je = text.rfind("}")
        if js < 0 or je < js:
            raise ValueError("no JSON object in LLM response")
        text = text[js : je + 1]
    # Strip trailing commas: `,}` and `,]` (with optional whitespace).
    cleaned = re.sub(r",(\s*[}\]])", r"\1", text)
    return json.loads(cleaned)


# ── Public API ────────────────────────────────────────────────────


@dataclass
class V9ReflectDecision:
    next_action: str
    rationale: str
    label: str
    diff_vs_predicted: str
    memory_recall: str
    reflection_cot: Dict[str, List[str]]
    slot_edits: Dict[str, str] = field(default_factory=dict)
    bundle_demote: List[str] = field(default_factory=list)
    bundle_add: List[V9Strategy] = field(default_factory=list)


def enumerate_bundle_v9(
    meta_llm: LLMClient,
    loader: PromptLoader,
    task_summary: str,
) -> Tuple[V9Bundle, str]:
    """One LLM call. Returns the parsed bundle + the chosen strategy's
    artifacts populated. Raw response returned for telemetry."""
    td = loader.task_domain() or {}
    forbidden = td.get("forbidden_tokens") or []
    system = _bundle_system(td)
    user = _build_bundle_user_prompt(td, task_summary)

    raw = ""
    last_err: Optional[Exception] = None
    for attempt in range(1, 4):
        raw = meta_llm.generate(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            n=1,
        )[0]
        try:
            data = _extract_json(raw)
            strategies = [_parse_strategy_v9(s, forbidden) for s in data["strategies"]]
            # Resolve chosen index: either chosen_idx (int) or
            # chosen_strategy (name string) — LLM uses both shapes.
            chosen = data.get("chosen_idx")
            if chosen is None and "chosen_strategy" in data:
                name = str(data["chosen_strategy"])
                chosen = next(
                    (i for i, s in enumerate(strategies) if s.name == name),
                    0,
                )
            if chosen is None:
                chosen = 0
            chosen = max(0, min(len(strategies) - 1, int(chosen)))

            # Artifacts may be nested under chosen_strategy_artifacts OR
            # at top-level (LLM frequently flattens despite the prompt
            # asking for nesting). Accept both.
            arts_blob = data.get("chosen_strategy_artifacts") or {}
            if not arts_blob and any(
                k in data
                for k in (
                    "inferable_hints_text",
                    "examples_text",
                    "feedback_template",
                )
            ):
                arts_blob = data
            strategies[chosen].artifacts = V9Artifacts(
                inferable_hints_text=_redact_forbidden(
                    arts_blob.get("inferable_hints_text", ""), forbidden
                ),
                examples_text=_redact_forbidden(
                    arts_blob.get("examples_text", ""), forbidden
                ),
                feedback_template=_redact_forbidden(
                    arts_blob.get("feedback_template", ""), forbidden
                ),
            )
            bundle = V9Bundle(
                strategies=strategies,
                chosen_idx=chosen,
                task_understanding=str(data.get("task_understanding", "")),
            )
            _log.info(
                "v9 bundle: %d strategies, chosen='%s' (score=%.1f)",
                len(strategies),
                strategies[chosen].name,
                strategies[chosen].combined_score,
            )
            return bundle, raw
        except Exception as e:  # noqa: BLE001
            last_err = e
            _log.warning("v9 bundle parse fail attempt %d: %s", attempt, e)
    raise ValueError(f"v9 bundle enum failed: {last_err}")


def _author_artifacts_system(td: Dict[str, Any]) -> str:
    """System message for the lazy artifact-authoring LLM call (§2.10).

    Mirrors the bundle-system message but scoped to a SINGLE strategy:
    the LLM must produce ONLY the three artifact texts for one strategy,
    not a full bundle.
    """
    inferable = td.get("inferable_concepts") or []
    mandatory = td.get("mandatory_features") or []
    forbidden = td.get("forbidden_tokens") or []
    budget = td.get("feature_budget") or {}
    target_min = budget.get("target_min", 12)
    target_max = budget.get("target_max", 17)
    hard_cap = budget.get("hard_cap", 20)

    inferable_lines = "\n".join(f"  - {c['concept']} → {c['idiom']}" for c in inferable)
    mandatory_lines = "\n".join(
        f"  - **{m['name']}** ({m.get('idiom','')}): " f"{(m['reason'] or '').strip()}"
        for m in mandatory
    )
    forbidden_line = ", ".join(f"`{t}`" for t in forbidden)

    return f"""## Role

You are authoring inner-prompt slot text for ONE strategy that has just been activated by a switch_to_next decision. Output the THREE artifacts (inferable_hints_text, examples_text, feedback_template) for the strategy described in the user message.

## Hard requirements (will be runtime-validated and rejected if violated)

### `inferable_hints_text` — bulleted "What you CAN infer" block

EVERY entry from this list MUST appear:

{inferable_lines}

In addition, mention the mandatory_features:

{mandatory_lines}

### `examples_text` — 2-3 worked PyTorch examples

Each example is a complete `def enhance_observation(scenario_state: dict) -> torch.Tensor:` in a fenced ```python``` block.

Function signature is FIXED:

```python
def enhance_observation(scenario_state: dict) -> torch.Tensor:
    lidar_t = scenario_state["lidar_targets"]
    lidar_a = scenario_state["lidar_agents"]
    agent_pos = scenario_state["agent_pos"]
    agent_vel = scenario_state["agent_vel"]
    agent_idx = scenario_state["agent_idx"]
    n_agents = int(scenario_state["n_agents"])
    cover_r = float(scenario_state["covering_range"])
    ...
```

SHAPE: `lidar_targets` has shape `[B, 15]`, `lidar_agents` has `[B, 12]`. Reduce each to scalars first before combining.

Together the examples MUST:
  - Include ≥1 example with role one-hot (`F.one_hot(agent_idx, n_agents)` or `torch.zeros(B, n_agents); one_hot[:, agent_idx] = 1.0`)
  - Include ≥1 cross-source feature (e.g., `t_count - a_count`)
  - Stay within {target_min}-{target_max} features per example, hard cap {hard_cap}

### `feedback_template` — strategy-specific reminder

A 2-3 sentence reminder tied to this strategy's `what_is_needed`.

## Anti-cheat

Do NOT use these tokens: {forbidden_line}.

## Output format — STRICT

A SINGLE JSON object with exactly these keys (no others):

```json
{{
  "inferable_hints_text": "...",
  "examples_text": "...",
  "feedback_template": "..."
}}
```

NO trailing commas. NO comments. NO prose outside the JSON object.
"""


def author_artifacts_for_strategy(
    meta_llm: LLMClient,
    loader: "PromptLoader",
    strategy: V9Strategy,
) -> V9Artifacts:
    """v9.1 §2.10 — lazy artifact authoring.

    Called when a strategy that was previously NOT chosen at bundle-enum
    time becomes the active strategy via `switch_to_next`. The original
    bundle-enum LLM only authored artifacts for the chosen strategy
    (#chosen_idx); the others have empty `V9Artifacts()` and would
    produce empty slot files → empty inner prompt → garbage candidates.

    This function fires ONE additional LLM call to author all three
    artifact texts for the new chosen strategy.

    Returns a populated V9Artifacts. The caller assigns it to
    `strategy.artifacts` and writes the slot files.
    """
    td = loader.task_domain() or {}
    forbidden = td.get("forbidden_tokens") or []
    system = _author_artifacts_system(td)
    user = (
        "[STRATEGY TO AUTHOR ARTIFACTS FOR]\n"
        f"name: {strategy.name}\n"
        f"full_solution: {strategy.full_solution}\n"
        f"chain_of_thought.why_it_works: "
        f"{strategy.chain_of_thought.why_it_works}\n"
        f"chain_of_thought.what_is_needed: "
        f"{strategy.chain_of_thought.what_is_needed}\n"
        f"chain_of_thought.failure_modes: "
        f"{strategy.chain_of_thought.failure_modes}\n"
        f"success_signature.ast_pattern_description: "
        f"{strategy.success_signature.ast_pattern_description}\n"
        f"\n"
        f"Output the three artifact texts as a JSON object."
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
            data = _extract_json(raw)
            arts = V9Artifacts(
                inferable_hints_text=_redact_forbidden(
                    data.get("inferable_hints_text", ""),
                    forbidden,
                ),
                examples_text=_redact_forbidden(
                    data.get("examples_text", ""),
                    forbidden,
                ),
                feedback_template=_redact_forbidden(
                    data.get("feedback_template", ""),
                    forbidden,
                ),
            )
            _log.info(
                "v9.1 §2.10 artifacts authored for '%s' "
                "(hints=%dB, examples=%dB, feedback=%dB)",
                strategy.name,
                len(arts.inferable_hints_text),
                len(arts.examples_text),
                len(arts.feedback_template),
            )
            return arts
        except Exception as e:  # noqa: BLE001
            last_err = e
            _log.warning(
                "v9.1 §2.10 artifact parse fail attempt %d: %s",
                attempt,
                e,
            )
    raise ValueError(f"v9.1 §2.10 artifact authoring failed: {last_err}")


def reflect_decide_v9(
    meta_llm: LLMClient,
    loader: PromptLoader,
    bundle: V9Bundle,
    facts: Dict[str, Any],
    memory_rows: List[Dict],
) -> Tuple[V9ReflectDecision, str]:
    """One combined diagnose+reflect+decide call. `facts` carries the
    AST analyzer output for the inner result. `memory_rows` is the
    last-N rows from MemoryStore."""
    td = loader.task_domain() or {}
    forbidden = td.get("forbidden_tokens") or []
    system = _reflect_system(td)
    user = _build_reflect_user_prompt(td, bundle, facts, memory_rows)

    raw = ""
    last_err: Optional[Exception] = None
    for attempt in range(1, 4):
        raw = meta_llm.generate(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            n=1,
        )[0]
        try:
            data = _extract_json(raw)
            reading = data.get("current_outcome_reading") or {}
            cot = data.get("reflection_chain_of_thought") or {}
            slot_edits_raw = data.get("slot_edits") or {}
            slot_edits = {
                k: _redact_forbidden(v, forbidden)
                for k, v in slot_edits_raw.items()
                if isinstance(v, str)
            }
            bu = data.get("bundle_update") or {}
            add_blobs = bu.get("add") or []
            add = [_parse_strategy_v9(b, forbidden) for b in add_blobs]
            return (
                V9ReflectDecision(
                    next_action=str(data["next_action"]),
                    rationale=_redact_forbidden(
                        str(data.get("rationale", "")), forbidden
                    ),
                    label=str(reading.get("label", "")),
                    diff_vs_predicted=_redact_forbidden(
                        str(reading.get("diff_vs_predicted", "")), forbidden
                    ),
                    memory_recall=_redact_forbidden(
                        str(data.get("memory_recall", "")), forbidden
                    ),
                    reflection_cot={
                        k: [_redact_forbidden(s, forbidden) for s in (cot.get(k) or [])]
                        for k in (
                            "what_went_right",
                            "what_went_wrong",
                            "remaining_uncertainty",
                        )
                    },
                    slot_edits=slot_edits,
                    bundle_demote=list(bu.get("demote") or []),
                    bundle_add=add,
                ),
                raw,
            )
        except Exception as e:  # noqa: BLE001
            last_err = e
            _log.warning("v9 reflect parse fail attempt %d: %s", attempt, e)
    raise ValueError(f"v9 reflect failed: {last_err}")


# ── User prompt builders ──────────────────────────────────────────


def _build_bundle_user_prompt(td: Dict[str, Any], task_summary: str) -> str:
    challenges = td.get("coordination_challenges") or []
    challenges_block = "\n".join(f"  - {c}" for c in challenges)
    return f"""[TASK]
{td.get('short_label', '')}

{td.get('task_framing', task_summary)}

[COORDINATION CHALLENGES]
{challenges_block}

[OBSERVATION CONSTRAINTS]
The inner LLM may read only:
  - agent_pos, agent_vel (own proprioception)
  - agent_idx, n_agents, n_targets, covering_range, agents_per_target_required
  - lidar_targets [B, n_rays] — distance to nearest TARGET per ray
  - lidar_agents  [B, n_rays] — distance to nearest other AGENT per ray

[YOUR TASK]
Enumerate 3-5 strategies with chain_of_thought reasoning, choose the strongest, and author the three artifact texts (inferable_hints_text, examples_text, feedback_template) for the chosen strategy. Output the JSON object."""


def _build_reflect_user_prompt(
    td: Dict[str, Any],
    bundle: V9Bundle,
    facts: Dict[str, Any],
    memory_rows: List[Dict],
) -> str:
    cur = bundle.current()
    sig = cur.success_signature
    memory_block = (
        "\n".join(json.dumps(r, default=str) for r in memory_rows)
        if memory_rows
        else "  (no prior outers — this is the first reflection)"
    )
    facts_block = json.dumps(facts, indent=2, default=str)
    return f"""[ACTIVE STRATEGY]
{cur.name}: {cur.full_solution}
ast_pattern_description: {sig.ast_pattern_description}
expected_M1_at_1M: {sig.expected_M1_at_1M}
expected_M6_at_1M_min: {sig.expected_M6_at_1M_min}

chain_of_thought:
  why_it_works: {cur.chain_of_thought.why_it_works}
  what_is_needed: {cur.chain_of_thought.what_is_needed}
  failure_modes: {cur.chain_of_thought.failure_modes}

[BUNDLE STATE]
{bundle.format_for_prompt()}

[MEMORY — LAST 3 OUTERS]
{memory_block}

[ANALYZER FACTS — THIS OUTER]
{facts_block}

[YOUR TASK]
Diagnose the outcome (label), reflect on what went right/wrong with chain-of-thought, choose next_action, and author slot_edits if applicable. Reference the memory in `memory_recall`. Output the JSON object."""
