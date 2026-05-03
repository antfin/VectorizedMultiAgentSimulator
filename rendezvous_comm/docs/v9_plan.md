# v9 plan — task-domain framing, CoT meta-prompt, memory, simplification

**Date:** 2026-05-02
**Status:** plan
**Authoring context:** v8 Phase-3 outcome (M1=0.010 / 18 cands, vs S3b-local M1=0.022 mean / 36 cands at 1M). Diagnosed in `docs/v8_vs_s3blocal_prompt_comparison.md`: v8's auto-authored prompt drops role one-hot, drops the inferable-hints block, and replaces S3b-local's rich worked example with a 1-feature anchor. v9 restructures the meta-prompt and base-prompt slots to recover the S3b-local prompt density without re-introducing handcrafted task-specific text in the meta layer.

---

## 1. Design goals (in priority order)

1. **Match S3b-local's inner-LLM consistency** — role one-hot ≥80%, "what you CAN infer"-style hints generated each outer iter, multi-feature worked example.
2. **Task-portability** — the meta-prompt itself stays task-agnostic. Task framing (rendezvous semantics, k-requirement, coordination challenge) lives in a single editable file the meta-prompt reads. Swap the file → swap the task.
3. **Chain-of-thought in the meta-prompt** — each strategy gets a "why it works / what's needed to make it work" reasoning block authored before any prompt text.
4. **Cross-outer memory** — meta-LLM reads its own prior CoT + observed outcomes when reflecting and deciding next-action.
5. **Simplification** — kill redundant slots and outer/inner state; merge what overlaps.

---

## 2. Task-domain file (NEW) — `src/lero/prompts/task_domains/<name>.yaml`

Single editable YAML capturing everything task-specific that the meta-prompt currently has to discover or that lives in the inner base prompt.

```yaml
# src/lero/prompts/task_domains/rendezvous_k2.yaml
name: rendezvous_k2
short_label: "Multi-agent rendezvous (k=2 per target)"

# Used in inner system message and meta-prompt task summary
task_framing: |
  This is a RENDEZVOUS task. ${n_agents} agents must collectively cover
  ${n_targets} targets. A target is covered when EXACTLY ${agents_per_target}
  agents are simultaneously within ${covering_range} distance of it. Once
  covered, a target is removed (no respawn). The challenge is implicit
  coordination from local sensors alone — agents do NOT know other agents'
  positions, target positions, or coverage status.

# What the LLM should think about during feature design.
# NOT a feature list — just the conceptual handles.
coordination_challenges:
  - "Distinguishing 'I should approach' vs 'someone else is already here'"
  - "Avoiding all 4 agents collapsing onto the same target"
  - "Knowing when to commit vs keep searching, with only LiDAR"
  - "Differentiating roles when policy params are shared across agents"

# Used by meta-prompt to author 'inferable hints' AND to author worked
# examples. Each entry is (concept, generic_pytorch_idiom).
inferable_concepts:
  - concept: "Direction to nearest target"
    idiom: "argmin over lidar_targets → ray index → cos/sin of 2π·idx/n_rays"
  - concept: "Distance to nearest target"
    idiom: "lidar_targets.min(dim=-1).values"
  - concept: "Number of nearby targets"
    idiom: "(lidar_targets < threshold).float().sum(dim=-1)"
  - concept: "Local agent crowdedness"
    idiom: "(lidar_agents < threshold).float().sum(dim=-1)"
  - concept: "Agent role under shared-policy MAPPO"
    idiom: "F.one_hot(agent_idx, n_agents)  # MANDATORY for k>1 rendezvous"
  - concept: "Self-motion state"
    idiom: "agent_vel.norm(dim=-1)"
  - concept: "Boundary distance"
    idiom: "1 - agent_pos.abs().max(dim=-1).values"

# Mandatory features the meta-prompt MUST surface in its inferable_hints
# and worked examples (with structural reason).
mandatory_features:
  - name: role_one_hot
    reason: "Shared-policy MAPPO needs role differentiation; without it no implicit role assignment is possible for k=2."
    idiom: "F.one_hot(agent_idx, n_agents)"
  - name: cross_source_signal
    reason: "Coordination requires combining target and agent sensor streams (e.g. nearby-targets vs nearby-agents)."
    idiom: "(lidar_targets < r).sum() − (lidar_agents < r).sum()"

# Anti-cheat: tokens forbidden from meta-prompt / inner prompt output (these
# are S3b-local handles already in the bench).
forbidden_tokens:
  - hold_signal
  - approach_signal
  - settle_signal
  - rendezvous_pressure
  - t_close_mean
  - t_dispersion
```

**Why this addresses your bullet 1:** changing the task = swap one YAML. The meta-prompt and base prompt stay generic. Rendezvous-specific framing is loaded from the file.

**Open question:** should `mandatory_features` also live somewhere visible at run-time as a hard whitelist (so we can fail-fast a candidate that omits role one-hot), or only as a soft hint? Tradeoff: hard fail = guarantees presence but constrains exploration; soft hint = lets meta-LLM still drop it. **Recommendation:** soft hint v9.0, escalate to hard whitelist v9.1 only if soft fails.

---

## 3. Restructured base prompt — `v3_modular_taskdomain`

Replace `v2_fewshot_modular_v2_local`. Slot list:

| slot | source | content | filled by |
|---|---|---|---|
| `system` | static + task_framing | base SYSTEM message + RENDEZVOUS framing block from task_domain | render time, no meta edit |
| `task_context` | task_framing | rendezvous semantics from task_domain.yaml | render time |
| `state_schema` | static | schema dict + KeyError warning | static, frozen-by-hash |
| `fairness` | static | forbidden keys list | static, frozen-by-hash |
| `inferable_hints` | meta-authored | "What you CAN infer" block, generated by meta-prompt CoT each outer | meta each outer |
| `examples` | meta-authored | 2-3 short worked examples or 1 rich example, generated by meta-prompt CoT | meta each outer |
| `output_spec` | static | function signature + N range | static |

### What changes vs v8

- **DELETED `guidance_observation`** — it was redundant with what `examples` should carry. The auto-fewshot from v8's `guidance_observation` is just a worked example, so put it in `examples`.
- **DELETED `guidance_shared` and `guidance_reward`** — they were always empty in obs-only runs.
- **NEW `inferable_hints`** — recovers the S3b-local "What you CAN infer" section, auto-generated each outer iter.
- **`examples` becomes meta-authored, multi-example** instead of static 1-feature anchor.
- **`system` and `task_context` now read from `task_domain.yaml`** — task-specific text lives in one file.

### Why this addresses your bullets 2-4

- Bullet 2 ("can infer session generated by metaprompt"): new `inferable_hints` slot.
- Bullet 3 ("worked example multiple observations"): `examples` slot now generates 2-3 examples or 1 rich example with multiple features.
- Bullet 4 ("guidance_observation needed?"): **NO**, removed. Same info distributed into `inferable_hints` (the conceptual hints) and `examples` (the operational fewshot). Simpler structure.

---

## 4. Meta-prompt v9 — CoT + multi-output

The v8 meta-prompt outputs one JSON: `{strategies: [...], chosen_idx: ...}` where each strategy has a `lero_translation_hint` paragraph.

v9 meta-prompt outputs structured JSON with explicit reasoning steps.

### 4.1 Bundle enumeration with CoT

```json
{
  "task_understanding": "<2-3 sentence read of the task framing from task_domain.yaml>",
  "strategies": [
    {
      "name": "<short_handle>",
      "full_solution": "<macro intent>",
      "chain_of_thought": {
        "why_it_works": "<2-3 sentences linking the macro to the reward + observation>",
        "what_is_needed": [
          "<requirement 1: e.g. 'agents must distinguish their role'>",
          "<requirement 2: ...>",
          "..."
        ],
        "failure_modes": [
          "<what would make this not work in PPO at 1M frames>"
        ]
      },
      "expected_M1_at_1M": <float>,
      "expected_M6_at_1M_min": <float>,
      "lero_codability": <int 0-10>,
      "rl_trainability": <int 0-10>
    },
    ...
  ],
  "chosen_idx": <int>,
  "chosen_strategy_artifacts": {
    "inferable_hints_text": "<full text for inferable_hints slot — bulleted, written like S3b-local's 'What you CAN infer' block, mentioning each item from task_domain.inferable_concepts including the mandatory ones>",
    "examples_text": "<full text for examples slot — 2-3 worked examples in fenced ```python``` blocks, each illustrating a different aspect of the strategy. At least one example MUST include role one-hot.>",
    "feedback_template": "<text appended to feedback.txt for this strategy — strategy-specific reminder, e.g. 'If your candidate has M1=0 at 1M, check that role one-hot is present and that you have at least one cross-source feature combining lidar_targets and lidar_agents'>"
  }
}
```

### 4.2 Reflect + decide with memory + CoT

```json
{
  "memory_recall": "<1-2 sentences referencing what the prior CoT predicted vs what actually happened>",
  "current_outcome_reading": {
    "label": "<from analyzer, treat as fact>",
    "diff_vs_predicted": "<expected M1 vs actual; expected pattern vs actual>"
  },
  "reflection_chain_of_thought": {
    "what_went_right": ["..."],
    "what_went_wrong": ["..."],
    "remaining_uncertainty": ["..."]
  },
  "next_action": "stop|refine_current|switch_to_next|escalate_features",
  "rationale": "<2-3 sentences>",
  "slot_edits": {
    "inferable_hints": "<full text or omit>",
    "examples": "<full text or omit>",
    "feedback_template": "<full text or omit>"
  },
  "bundle_update": { "demote": [...], "add": [...] }
}
```

### Why this addresses your bullets 5, 7, 8

- Bullet 5 ("feedback message more important like S3b-local, strategy-defined"): meta authors `feedback_template` that is strategy-specific and gets injected into `feedback.txt`.
- Bullet 7 ("chain of thought"): explicit `chain_of_thought` block in every strategy and every reflection.
- Bullet 8 ("memory across outer iters"): `_memory.jsonl` (next section) + `memory_recall` field in reflection.

---

## 5. Memory — `_meta_memory.jsonl` + `memory_recall` in reflection

After every outer iteration, the v9 outer loop appends a row to `<run>/_meta_memory.jsonl`:

```jsonl
{"outer_idx": 0, "ts": "...", "strategy_name": "...",
 "predicted": {"M1": 0.14, "M6": 0.33, "what_is_needed": [...]},
 "actual":    {"M1": 0.010, "M6": 0.177, "diagnosis_label": "rl_too_hard"},
 "delta":     {"M1": -0.13, "M6": -0.15},
 "chain_of_thought": {...},
 "post_hoc_reflection": {...}}
```

Before the next outer iter's reflection LLM call, the outer loop reads the last N entries (default N=3) and injects them as a "memory" block in the user prompt. The meta-LLM is instructed to start its reflection by referencing the memory (the `memory_recall` field).

This gives v9 cross-outer learning that v8 lacked: at outer 1, the meta-LLM can see "in outer 0 I predicted M1=0.14 but got 0.01, and the diagnosis said rl_too_hard despite features being all dense — what did I miss?"

**Implementation:** new file `src/lero/v9/memory.py` with `MemoryStore` (append-only JSONL, last-N reader). Outer loop calls `memory.append(...)` after each iter and `memory.read_recent(N)` before each reflection.

### Why this addresses your bullet 8 fully

The memory is read AFTER the inner-iter loop completes (you said "so all these previous thoughts after the end of inner iterations loop can be analyzed respect to the results"). The `post_hoc_reflection` field captures what the meta-LLM thinks went well vs badly, comparing the inner-iter trajectory to the prior CoT's predictions. The N>1 lookback lets it spot patterns across outers.

---

## 6. Simplification — kill redundancy

You said: "consider also that simplicity seems the key both inner but i suspect also outer so try to analyze if there's redundant part and try to merge and simplify them". Audit:

### 6.1 Redundant in v8 (concrete list)

| redundant pair / dead code | merge / drop |
|---|---|
| `guidance_observation` (auto) + `examples.txt` (static) | merge into `examples` (auto, multi-example) |
| `guidance_shared` + `guidance_reward` (always empty in obs-only) | drop both |
| Outer-level diagnosis labels (`too_many_features`, `over_gated`) + inner-level registry stagnation detection | keep both BUT drop labels that never fire (in v8 Phase-3 the new labels never triggered because of the cap) — investigate whether they earn their complexity |
| `V8OuterConfig.feature_count_target_min/max/cap/gated_cap` | move into `task_domain.yaml`'s `mandatory_features` + a single `feature_budget` block — config DRY |
| `_v8_checkpoint.pkl` + `_bundle_history.json` + `_bundle_state_*.json` | one of these is enough; investigate why all three exist |
| `_inner_legacy_outdir` directory passed to `LeroLoop` then unused | drop |
| `LeroLoop` wrapper around `v5.inner_loop` (used only for back-compat) | call `v5.inner_loop.run` directly |

### 6.2 Inner loop: keep mostly as-is, drop stagnation pivot

Inner loop (v5) is genuinely doing different work from the outer (per-candidate fitness within an iter, registry pruning across iters). Don't merge with outer. The 3 candidate × 3 iter loop is the proven LERO-paper structure.

**Confirmed drop:** the inner stagnation detector (`STAGNATION → pivot prompt`) is removed in v9. It double-counts the outer-level reflection signal — when fitness stalls, the outer's CoT-based reflect+decide already picks the right action (refine_current vs switch_to_next). Removing the inner pivot means the inner loop runs to completion (n_inner_iter iters) before the outer reflects. Cleaner control flow.

### 6.3 Outer loop: collapse diagnose + reflect into one LLM call (Option B)

In v8, diagnose runs locally (AST analyzer, no LLM), and the result is fed into a separate reflect LLM call:

```
   inner loop done → ast_analyze → diagnose label → reflect LLM → decide action
```

**Confirmed v9 design:** one LLM call that does diagnose+reflect+CoT in a single shot, with the AST signature passed in as a "facts" block. Saves one LLM round-trip per outer (~25% LLM cost reduction).

**Telemetry kept:** the AST analyzer output (n_features, n_gated, n_dense, touches_both_lidars, role_one_hot_present) is computed locally and embedded into the reflection user prompt as a `facts` block. The LLM's response includes a `current_outcome_reading.label` field that we record. No separate `_diagnosis.json` file — the label lives inside the saved reflection response.

### 6.4 Slots that vanish in v9

After the merges/drops above, the v9 base prompt has 7 slots (down from v8's 8). The v9 outer-iter LLM call count drops from 2 (bundle / reflect) to 1.5 average (bundle once at outer 0; one combined reflect+decide per subsequent outer). About 25% fewer LLM calls.

---

## 7. Prompt-lab convergence — iterate before RL

You said: "work on metaprompt and try several time just prompting to see that we can reach something more similar to s3b-local". Concrete loop:

```
Phase 1.5b: meta-prompt convergence sweep
  - 5 trials × {meta_prompt_v9.0, v9.1, v9.2}
  - For each trial: enumerate bundle → fill prompt slots → generate 9 inner candidates
  - Measure: role_one_hot rate, inferable_hints text quality (LLM-judge),
             example_count, mandatory_feature_present rate
  - Compare to S3b-local benchmark (97% role one-hot, ~80-line inferable hints)
  - Iterate meta-prompt text until: ≥80% role one-hot AND inferable_hints
    text mentions all 7 task_domain.inferable_concepts AND examples slot
    has ≥2 worked examples
  - Cost: ~5 × 3 × ~$2 = ~$30, ~10-15 min wall
```

Only after this convergence passes do we proceed to Mac smoke + Mac full run.

---

## 8. Implementation phases

| phase | what | wall | LLM cost | gating criterion to next phase |
|---|---|---|---|---|
| 0 | Write `task_domain.yaml` for rendezvous_k2 + this plan reviewed | 30 min | $0 | user sign-off on plan |
| 1 | Build `v3_modular_taskdomain` base prompt + `loader.py` task-domain reader | 2 h | $0 | loader unit tests pass |
| 2 | Build v9 meta-prompt (CoT bundle + reflect+decide combined) — `src/lero/v9/meta_strategist.py` | 3 h | $0 | dry-run validates JSON shape |
| 3 | Build v9 outer loop with `MemoryStore` — `src/lero/v9/outer_loop.py` | 3 h | $0 | dry-run round-trips memory |
| 4 | Phase 1.5b: meta-prompt convergence sweep (Section 7) | 30 min | ~$30 | role_one_hot ≥80% in inner cands, inferable_hints quality matches S3b-local |
| 5 | Mac smoke (200k frames, 1 outer × 1 inner × 1 cand) | 10 min | $1 | end-to-end pipeline runs, memory file written |
| 6 | Mac full RL (1M frames, **bundle_size outer × 3 inner × 3 cand**, 1 seed). Bundle typically 3-5 strategies tried serially per §9.6 — wall scales linearly. | ~5-8h | $8-12 | M1 best ≥ 0.030 (matches S3b-local s2 — the worst seed) |
| 7 | If phase 6 passes: 3-seed Mac OR OVH sweep for stat significance | 15-24h Mac OR 1.5h OVH × 3 | $25 | publish v9 vs S3b-local 1M result |

**Total to phase 6 success criterion:** ~7-8 h work + ~$36 LLM. Fail-fast at phase 4 (cheap) before committing 3h of RL.

---

## 9. Decisions (resolved)

1. **`mandatory_features` enforcement** → **soft hint** (v9.0). Re-evaluate after Phase 6 if role one-hot rate < 80%.
2. **Drop inner-level stagnation pivot** → **yes**. Outer-level reflection handles the same signal.
3. **Diagnose + reflect** → **collapse into single LLM call** (Option B). Telemetry: keep AST analyzer output as a "facts" block in the saved reflection JSON; no separate `_diagnosis.json` file.
4. **Memory lookback** → **N=3** outers in the prompt; full history persists in `_meta_memory.jsonl` for post-hoc analysis.
5. **Prompt-lab convergence judge** → **AST/regex first**. Escalate to LLM-judge only if structural metrics pass but inner-LLM rate still misses target.
6. **Bundle strategy coverage** → **serial, all strategies tried**. Accuracy over speed: `max_outer = bundle_size` (3-5), no early-stop unless `achieved`. Each strategy gets one shot at refinement before switching.

---

## 10. What v9 deliberately does NOT change

- Reward function: still `evolve_reward: false`, scenario reward unchanged.
- Inner loop algorithm: still v5 (3 cands × 3 iters with registry).
- PPO hyperparams, BenchMARL setup, scenario patching: unchanged.
- Anti-cheat boundary: kept (S3b-local handles forbidden tokens).
- `obs_state_mode: local`: unchanged.
- The fairness slot is still frozen-by-hash.

The hypothesis is that the **prompt structure** is the M1 lever, not the algorithm or environment. Phase 6 falsifies this if M1 stays ≤ 0.020.

---

## TL;DR

v9 = **task_domain.yaml** (portable framing) + **CoT meta-prompt** (reasoning before prompts) + **memory** (cross-outer learning) + **slot consolidation** (drop guidance_observation/shared/reward, beef up examples + new inferable_hints) + **prompt-lab convergence** before any RL. Predicted: M1 0.030+ on 1 seed (matches S3b-local worst seed) at the cost of ~$36 LLM and ~7 h dev work.
