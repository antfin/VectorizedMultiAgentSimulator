# LERO-MP: Meta-Prompting Extension Plan

> **Status:** Draft for review — 2026-04-21.
> **Author:** afin (with Claude Code research assist).
> **Relationship to LERO:** Strict superset. Current LERO (`src/lero/*`) stays the *inner loop* that evolves reward/observation code under a fixed prompt template. LERO-MP adds an *outer loop* that evolves the prompt template itself when downstream RL results signal the template is the bottleneck.

---

## 1. Why a new experiment type (WHY)

LERO today closes one loop: LLM → code → RL metrics → textual feedback → LLM. It already implements an OPRO-style in-context evolution over **code artifacts**, using candidates-with-scores in the prompt context. See `loop.py:621–708` and `codegen.py:170–239`.

But the **prompt template itself is frozen per run**. Prompt version is chosen in YAML (`LlmConfig.prompt_version`, `config.py:48`) and every candidate in every iteration of that run uses the same `system.txt` + `initial_user.txt` + `feedback.txt` files.

This matters because LERO's Phase-4 results show the prompt template is often the *actual* bottleneck, not the LLM or the RL optimizer:

- `v2_min` (3-line prompt) → unstable candidates.
- `v2_fewshot` (v2 + 2 MPE reference examples) → k=1 M1=1.0, k=2 local M1=0.88.
- `v2_twofn` (asks for `agent_reward + global_reward` decomposition) → M1=0.01, reward-hacked.

Moving from `v2` → `v2_fewshot` → `v2_twofn` was a **manual** act of meta-prompting. LERO-MP automates that act: when the inner loop plateaus, an LLM meta-optimizer edits the prompt template using the history of (prompt, candidate_code, downstream metrics).

This also addresses a concrete open issue in `lero.md` §6 — "reward design for k≥2 needs either stronger reward shaping or a better prompt." The meta-loop is the systematic answer.

## 2. Literature anchors (what we're borrowing)

Research sources collected 2026-04-21 (see §11 for URLs):

| Paper / system | Year | What we use from it |
| --- | --- | --- |
| **Eureka** (Nvidia, ICLR 2024) | 2023 | Direct blueprint: LLM writes reward code, RL evaluates, textual *reward-reflection* summary drives next generation. 16 candidates × 5 iters in paper. We already do a scaled-down version. |
| **OPRO** (DeepMind, ICLR 2024) | 2023 | Meta-prompt = (task description) + (solution, score) pairs sorted ascending. The LLM is told to propose the *next* solution. LERO's `feedback.txt` already follows this shape — extend it to the prompt template level. |
| **EvoPrompt** (ICLR 2024) | 2023 | EA operators on prompts: mutation (rewrite a section), crossover (combine two parents). Useful as the mutation operator inside the outer loop. |
| **PromptAgent** (ICLR 2024) | 2023 | MCTS over prompts with **error-based feedback** at each node. Key idea: don't just evaluate prompts — extract *what went wrong* and attach that as edge labels. |
| **TextGrad** (Nature 2024) | 2024 | LLM-generated *textual gradients* on each variable in a compound system, composed PyTorch-style. Gives us the abstraction: each prompt *slot* is a variable, each gets its own gradient. |
| **DSPy / MIPROv2** | 2024 | Signatures + bootstrap + Bayesian discrete search over (instructions, few-shot demos). The loader already has a design-note (`loader.py:1–8`) that the file layout maps to DSPy Signatures — we keep forward-compat. |
| **APE** (Zhou 2022) | 2022 | UCB over candidate prompts, ~5 samples per prompt. Useful for the *selection* step in the outer loop (not all meta-prompts should run a full inner loop). |
| **LCP** (Learning from Contrastive Prompts) | 2024 | After N prompts ranked, LLM is asked to *contrast* top vs bottom and articulate the difference. Strong signal for mutation. |

**Why this mix and not just "run Eureka":** Eureka only evolves code; the prompt is a fixed template. Our results say the template matters as much as the code, so we stack an OPRO/TextGrad outer loop *on top of* the existing Eureka-like inner loop, and we borrow PromptAgent's error-feedback structure to make outer-loop feedback informative rather than just scalar.

## 3. Two-level architecture

```text
┌─ OUTER LOOP (meta-prompt, slow) ─────────────────────────────────────┐
│                                                                      │
│  prompt template T_i  ──┐                                            │
│                         │                                            │
│                         ▼                                            │
│            ┌─ INNER LOOP (code evolution, fast) ──────────────┐      │
│            │   for it in 1..N_inner:                          │      │
│            │     candidates = LLM(T_i, feedback_{it-1})       │      │
│            │     metrics    = RL_eval(candidates)  # M1,M6,M2 │      │
│            │     feedback   = top-k summary                   │      │
│            │   return best_M1_under_T, trajectory             │      │
│            └────────────────────────────────────────────────┘        │
│                         │                                            │
│                         ▼                                            │
│   trajectory_i = (best_M1, best_M6, fail-modes, chosen candidates)   │
│   templates[i] = (T_i, trajectory_i)                                 │
│                                                                      │
│   if should_meta_iterate(templates):                                 │
│       T_{i+1} = MetaLLM(templates, slot_to_edit)                     │
│   else:                                                              │
│       stop                                                           │
└──────────────────────────────────────────────────────────────────────┘
```

The inner loop is **unchanged LERO** (with the fixes in `lero.md` §3). Everything new lives in the outer loop.

## 4. Fair information constraint (the "don't cheat" rules)

This section is the non-negotiable contract. It is derived from `lero.md` §2.3–§2.6 (the S3b-global vs S3b-local finding) and from the ER1/ER2/ER3 fairness convention in `docs/comparison_gnn_vs_minimal_team.md`.

### 4.1 What "cheating" means concretely

LERO exposed two distinct ways the LLM can produce results that look good but aren't comparable to our baselines:

1. **Observation cheating (oracle input).** The `enhance_observation` function gets called with a state dict; in `obs_state_mode=global` that dict contains `targets_pos`, `covered_targets`, `agents_per_target`, etc. — information no agent sees at execution time. S3b-global used these and hit M1=1.000 on k=2. **This is not a fair result** (`lero.md:114, 133, 200–213`). The fair baseline, S3b-local, uses only LiDAR + own pos/vel and reached M1=0.88 — still SOTA, but comparable.
2. **Reward hacking (exploitable incentive).** Even with a fair observation, an LLM reward function can have exploitable gaps. S3a_gpt5 showed eval M1=0.86 at 1M frames collapsing to final M1=0.09 at 10M — the policy found the exploit (`lero.md:215–233`). The reward is allowed to see global state (rewards run only during centralized training), but the **produced behavior** must survive 10M-frame training.

### 4.2 Information each function is allowed to use

Codifying the distinction explicitly so meta-prompts and reviewers can check compliance:

| Function | Allowed keys at RL *execution* | Allowed at RL *training* only | Forbidden |
| --- | --- | --- | --- |
| `enhance_observation` (fair mode) | `agent_pos`, `agent_vel`, `agent_idx`, `lidar_targets`, `lidar_agents`, `messages` (if dim_c>0), `covering_range`, `agents_per_target_required`, `n_agents`, `n_targets` (these last are static scenario constants — fine) | — | `targets_pos`, `agents_pos`, `covered_targets`, `agents_per_target`, `all_time_covered`, `agents_targets_dists`, `collision_rew`, `time_penalty` |
| `compute_reward` | (not called at execution) | All of the above global keys | N/A — but the *behavior induced* must not reward-hack |

This mirrors the ER1/ER2/ER3 information envelope (LiDAR + optional messages). It is what makes S3b-local comparable to ER3 GNN, and why S3b-global is reported but excluded from headline claims.

### 4.3 How we enforce fairness in code

Two mechanisms, both cheap:

1. **Default mode pinned in config.** LERO-MP sets `obs_state_mode: local` as the default in `configs/lero_mp/*.yaml`. Changing it to `global` requires an explicit `fairness_waiver: oracle_obs_for_comparison_only` field (logged in `final_metrics.json`). Keeps the capability for diagnostic runs while making misuse visible.
2. **Runtime state-dict whitelist.** In `scenario_patch.py` (next to `_build_obs_state`), wrap the dict passed to `enhance_observation` in a thin `AllowedKeysDict` that raises on any forbidden key lookup when `obs_state_mode=local`. Today LERO trusts the LLM to self-restrict; `v1` failed this and errored at runtime anyway. A whitelist converts silent cheating (if we ever flipped to global with the local-feature prompt) into a loud crash and a logged fail-mode (§6.3).

### 4.4 How we enforce non-exploitable reward

Reward-hacking is a behavior property, not a syntactic one — we can't statically forbid it. Four layers, in order of cost:

1. **Magnitude bound.** Keep `reward_clip=50` (already in `LeroConfig`). Meta-LLM is told this exists so it does not design around it (inflating magnitudes to saturate). A new outer-loop fail-mode, `reward_magnitude_inflation`, fires when successive templates' mean |M2| grows by >2× with no M1 gain — observed historically in S3ac's reward-inflation spiral (`lero.md:251`, M2 1190→6462 over 4 iters, M1 stuck at 0).
2. **Peak-M1 checkpointing during Tier 2** (10M full training). `lero.md` §5.1 marks this as the highest-priority code change; LERO-MP ships it because the meta-loop uses the **peak-M1 policy**, not the final one, as the template's fitness. This recovers gpt-5.4's 0.86 policy from S3a_gpt5 even when the reward degrades later. A gap > 0.20 between peak-M1 and final-M1 is itself a reward-hacking fail-mode (`reward_hack_divergence`).
3. **Multi-seed Tier-1.** Three seeds (§5.3) — a reward that happens to work on one seed but is unstable across seeds is usually exploiting seed-specific local optima.
4. **Default to obs-only on k≥2.** `lero.md` §2.6 central thesis: for k≥2, let the LLM design observations and keep the reward hand-crafted (ER1). LERO-MP ships `configs/lero_mp/mp_k2_obsonly.yaml` with `evolve_reward: false, evolve_observation: true, obs_state_mode: local` as the reference config. The meta-loop then only evolves prompt slots relevant to observation design. Reward-evolution mode is supported but flagged (`mode: reward_evolution` → extra warnings in logs).

### 4.5 Parity checklist with prior experiments

Before any LERO-MP result enters a headline table, it must be audited against this list (mirrors ER1/ER2/ER3 setup):

- [ ] `max_steps` and `covering_range` identical to the baseline row (`max_steps=400, covering_range=0.25` for current comparisons — note the historical `max_steps` bug from 2026-03-22).
- [ ] `targets_respawn=False` (required for M1/M3 correctness; memory "Key Design Decisions").
- [ ] `shared_reward=True` if comparing to ER1/ER2/ER3; `info()` override retained so M8 is computable.
- [ ] `obs_state_mode=local` unless a `fairness_waiver` is present.
- [ ] Three seeds at 10M frames (not a single seed).
- [ ] Peak-M1 reported alongside final-M1. Gap > 0.20 flagged in the writeup.
- [ ] Information envelope: no forbidden keys accessed (whitelist log clean).

## 5. Modular prompt schema

For meta-prompting to be sample-efficient, the meta-LLM must be able to edit **small, typed slots** instead of rewriting monolithic text. This is the TextGrad insight — one gradient per variable — combined with DSPy Signatures.

Current templates are already partly modular (`system.txt` / `initial_user.txt` / `feedback.txt`), but `initial_user.txt` is one long string with 23 `$variable` substitutions. We'll decompose it into explicit named slots with a manifest.

### 5.1 Proposed slot layout (stored on disk per template version)

```text
prompts/<version>/
  meta.yaml             # version, parent, lineage, slot hashes
  system.txt            # [SLOT:role]      — role + constraints
  task_context.txt      # [SLOT:task]      — domain-specific task description
  state_schema.txt      # [SLOT:schema]    — scenario_state dict description
  fairness.txt          # [SLOT:fairness]  — explicit info-envelope rules (§4)
  guidance.txt          # [SLOT:guidance]  — coordination hints per-k
  examples.txt          # [SLOT:examples]  — few-shot blocks (optional)
  output_spec.txt       # [SLOT:output]    — signature + I/O shape rules
  feedback_header.txt   # [SLOT:fb_hdr]    — how to present ranked candidates
  feedback_footer.txt   # [SLOT:fb_ftr]    — "now propose next" instruction
```

A template's final `initial_user.txt` is assembled at render time by concatenating the slots in a fixed order — no free-form surgery required on the outer loop's part. Each slot is independently mutable. Note the new dedicated `fairness.txt` slot: it carries the §4 constraints verbatim and is **not** included in the set of slots the meta-LLM is allowed to edit (see §7.3).

### 5.2 Slot metadata (meta.yaml)

```yaml
version: v3_mp_001
parent: v2_fewshot            # which template this was mutated from
mutation:
  operator: rewrite_slot      # or: add_example, remove_example, crossover
  target_slot: guidance
  rationale: |
    Parent's best-M1 at k=2 was 0.88. Failure mode in the top-3 candidates
    was reward hacking via always-positive terms. Guidance did not warn
    against monotonically increasing bonuses. Added explicit constraint.
provenance:
  generated_by: meta_llm (claude-opus-4-7)
  generated_at: 2026-04-22T09:14:00Z
  slot_hashes:                # sha256 of each slot for change detection
    role: ...
    task: ...
    fairness: ...             # must match the frozen canonical hash
    guidance: <NEW>
fitness_summary:              # filled in after inner loop runs
  inner_iters: 5
  best_peak_M1: null          # Tier 2 peak-M1 (fairness §4.4)
  best_final_M1: null
  best_M6: null
  seeds: [0, 1, 2]
  fairness_waivers: []
```

### 5.3 Why slots (not whole-prompt rewrite)

- **Credit assignment.** If only the `guidance` slot changes between `v2_fewshot` and `v3_mp_001`, and M1 drops from 0.88→0.31, we *know* the guidance edit caused it. With monolithic rewrites we can't tell.
- **Smaller edit = lower variance.** Consistent with TextGrad's per-variable gradients.
- **Human-auditable.** Each prompt version has a diff that reviewers can read in <60s.
- **Compositional.** Crossover of two templates becomes: take slot A from parent1, slot B from parent2.
- **Fairness lock.** A pinned-hash `fairness.txt` slot is the mechanical guarantee that meta-evolution cannot remove the information-envelope rules.

## 6. When to trigger the meta-loop (the N-trials question)

This is the question you asked about. Three anchors from the literature:

- **Eureka:** 16 inner candidates per iter × 5 outer iters (hardware-rich Isaac Gym; ~minutes per candidate).
- **OPRO:** 8 candidates per step, dozens of steps (cheap GSM8K eval).
- **APE:** 5 samples per prompt per round.

Our constraint is **RL training is expensive**: 1M-frame eval ≈ O(10–20 min) on a GPU; 10M-frame full training ≈ O(2–4 h). Ignoring this killed prior attempts at automating prompt search.

### 6.1 Two-tier evaluation budget

| Tier | Frames | Purpose | Use at |
| --- | --- | --- | --- |
| **Tier 0 (smoke)** | 100k | Compilation + not-NaN check + fairness-whitelist check | every candidate, all templates |
| **Tier 1 (screen)** | 1M × 3 seeds | Rank candidates within a template | inner loop, existing LERO |
| **Tier 2 (confirm)** | 10M × 3 seeds, peak-M1 checkpointed | Gate a template for meta-promotion | only on templates whose Tier-1 best is in the top-2 so far |

Rationale: running Tier 2 on every candidate is infeasible; running Tier 1 only once (single seed) was a root cause of reward-hacking escaping detection (`lero.md` §2.5 "eval-vs-final degradation"). Three seeds at Tier 1 is the minimum to get a standard error on M1. Tier 2's peak-M1 checkpointing (§4.4) means the meta-loop reads the *best stable* policy, not the final one.

### 6.2 Trigger conditions for the outer loop

Run the meta-LLM to emit `T_{i+1}` when **any** of these fire for the current template `T_i`:

1. **Plateau.** `best_peak_M1` has not improved by ≥ 0.03 for 2 consecutive inner iterations *and* Tier-1 seed variance is low (σ ≤ 0.05). "The prompt is saturating, not the RL."
2. **Fail-mode clustering.** ≥ 3 of the last 5 top-k candidates share a symptom from the taxonomy in §7.3 (reward-hacking, NaN clip, dim-mismatch, identity-obs, fairness-violation, reward-magnitude-inflation).
3. **Seed instability.** Tier-1 M1 std across 3 seeds > 0.15 at best candidate. Template is producing high-variance code → needs stricter guidance / output_spec.
4. **Peak-vs-final divergence** (Tier 2 only). `peak_M1 − final_M1 > 0.20` → reward hacking confirmed → template is not safe even if final metrics look OK.
5. **Budget.** Hard cap: run meta-LLM at most every `meta_cooldown=3` inner iters regardless.

### 6.3 Number of trials per template before meta-promotion

Recommended default per template (Tier 1):

```text
N_trials(T) = n_candidates × n_inner_iters × n_seeds
            = 5           × 3              × 3
            = 45 training runs of 1M frames each
            ≈ 7–10 GPU-hours on a V100/A100
```

This is **one-third** of Eureka's 80 (16 × 5) and matches what LERO's Phase-4 already does in practice. Meta-promotion (Tier 2) only fires on templates whose Tier-1 best-peak-M1 is within 0.05 of the current champion, so the 10M full training runs stay rare (≤ 2 per outer iter).

### 6.4 Outer-loop stopping

Stop when *any* of:

- `best_peak_M1_T_i` hasn't improved by ≥ 0.02 for 3 outer iters,
- budget ≥ 200 total inner candidates,
- slot-edit history starts cycling (same slot edited with same rationale → we're stuck),
- fairness-violation fail-mode count > 2 (template is repeatedly trying to cheat → abort, reviewer loop).

## 7. The meta-prompt (how the outer LLM is asked)

The meta-LLM gets a structured bundle, not free text. OPRO-style sorted pairs, PromptAgent-style error feedback, LCP-style contrast.

### 7.1 Meta-prompt template (sketch)

```text
[ROLE]
You are a prompt engineer tuning an instruction template used to generate
PyTorch reward/observation code for a multi-agent RL scenario.

[HARD CONSTRAINTS — NEVER VIOLATE]
- The `fairness` slot is FROZEN. You may read it but never rewrite it.
- Any slot you edit must remain consistent with `fairness`: the agent
  policy only sees local sensors + messages at execution time.
- Rewards must stay within the magnitude bound (|r| ≤ reward_clip).

[OBJECTIVE]
Maximize: peak-M1 primary, M6 (coverage progress) tie-break.
Constraints: outputs must compile, not NaN, not reward-hack (peak-vs-final
gap < 0.20), not read forbidden state keys.

[PROMPT UNDER OPTIMIZATION]
Slots (current values hashed for brevity; full text inlined for the slot
you are editing):
  - role:      hash=abc123
  - task:      hash=...
  - fairness:  hash=FROZEN:sha256:...   # DO NOT EDIT
  - guidance:  <<< FULL TEXT >>>         # ← the slot you may edit
  - examples:  hash=...
  - output:    hash=...

[HISTORY — last 3 templates, sorted by best_peak_M1 ascending]
  T_v2_fewshot  peak_M1=0.88  final_M1=0.88  fail_modes=[hack:1, nan:0]
  T_v3_mp_001   peak_M1=0.72  final_M1=0.31  fail_modes=[hack:2, dim:1]
  T_v3_mp_002   peak_M1=0.91  final_M1=0.91  fail_modes=[hack:0, nan:0]  ← best

[CONTRASTIVE ANALYSIS (auto-generated)]
T_v3_mp_002 differs from T_v2_fewshot only in slot=guidance. The new text
added: "avoid rewards that grow monotonically over the episode without
bound". Top candidates under T_v3_mp_002 use tanh/exp-decay shaping; top
candidates under T_v2_fewshot used unbounded potential functions.

[TOP-3 CANDIDATES UNDER CURRENT TEMPLATE]
(code blocks + peak_M1/final_M1/M6/M2 scores + failure summary)

[TASK]
Propose a new version of the `guidance` slot (and ONLY that slot). Explain
in 2 sentences why this edit should help. Return:

```yaml
new_guidance: |
  ...
rationale: |
  ...
expected_improvement: one of {small, medium, large}
```

(End of meta-prompt template.)

### 7.2 Why this shape

- **OPRO:** "history sorted ascending by score" — forces the model to see improvement direction.
- **PromptAgent:** fail-mode counts per template — the "error feedback" at the edge.
- **LCP:** contrastive analysis — spells out the *delta* between winning and losing templates.
- **Single-slot edit:** lower variance (credit assignment).
- **Structured YAML output:** parseable, auditable.
- **Frozen fairness slot:** the "don't cheat" rule is a hard constraint the meta-LLM sees on every call.

### 7.3 Fail-mode taxonomy (determines which slot is edited)

Deterministic policy based on the fail-mode taxonomy (cheap, interpretable):

| Dominant failure | Detection | Edit slot |
| --- | --- | --- |
| Reward hacking (M1↓ late, M6↑ early, peak-vs-final > 0.20) | Tier-2 peak-vs-final check | `guidance` (add bounded-shaping constraint) |
| Reward-magnitude inflation (|M2| grows 2× across iters, M1 flat) | cross-iter M2 trend | `guidance` + `output_spec` (add bound) |
| NaN / crash (reward_clip triggers) | loop fallback chain | `output_spec` (bounded-output rule) |
| Fairness violation (forbidden key lookup) | whitelist-dict trap (§4.3) | `state_schema` — emphasize which keys exist in local mode; abort if repeated |
| Dim mismatch, wrong keys | AST + shape check | `state_schema` (clarify shapes) |
| Low variance, low M1 (stuck) | seed σ low + best_peak_M1 low | `examples` (add or rotate few-shot) |
| Over-general instructions | cross-k M1 gap | `task_context` (add k-specific hints) |
| Otherwise | round-robin | — |

`fairness` is **never** an edit target — only a source of evidence when fairness-violation fires.

## 8. Concrete implementation plan

### 8.1 Code layout (new files only)

```text
rendezvous_comm/src/lero/
  meta/
    __init__.py
    slot_loader.py        # assembles initial_user.txt from slot files
    mutation.py           # calls meta-LLM, parses YAML, writes new version dir
    failmode.py           # candidate-code → fail-mode taxonomy bucket
    trigger.py            # plateau/variance/budget/fairness checks
    provenance.py         # meta.yaml read/write + lineage graph
    fairness.py           # AllowedKeysDict + whitelist enforcement (§4.3)
  loop.py                 # + outer_loop() that wraps existing run()
  config.py               # + MetaPromptConfig
rendezvous_comm/src/lero/prompts/
  v2_fewshot_modular/     # v2_fewshot re-sliced into slots, behavior-identical
    meta.yaml  system.txt  task_context.txt  state_schema.txt
    fairness.txt                # NEW — pinned-hash §4 contents
    guidance.txt  examples.txt  output_spec.txt
    feedback_header.txt  feedback_footer.txt
rendezvous_comm/configs/lero_mp/
  mp_dryrun.yaml          # 1 outer iter, 2 inner iters, 2 candidates (sanity)
  mp_k2_obsonly.yaml      # reference config: §4.4 recommended setup
  mp_phase1.yaml          # 3 outer × 3 inner × 5 cand × 3 seeds
```

### 8.2 Config surface (additive)

```yaml
lero:
  # existing fields unchanged
  prompt_version: v2_fewshot_modular
  obs_state_mode: local               # §4.3 default
  evolve_reward: false                # §4.4 default on k≥2
  evolve_observation: true
  meta_prompt:
    enabled: true
    meta_llm:
      provider: anthropic             # can differ from inner-loop LLM
      model: claude-opus-4-7
      temperature: 0.3
    trigger:
      plateau_iters: 2
      plateau_delta: 0.03
      variance_threshold: 0.15
      peak_vs_final_gap_max: 0.20
      cooldown_inner_iters: 3
    budget:
      max_outer_iters: 3
      max_total_inner_candidates: 200
      tier2_promotion_gap: 0.05
    seeds: [0, 1, 2]
    slot_policy: failmode_taxonomy    # or: round_robin | fixed:guidance
    fairness:
      whitelist_strict: true          # AllowedKeysDict raises on forbidden
      waiver: null                    # set to "oracle_obs_for_comparison_only"
                                      # only for diagnostic runs
```

### 8.3 LERO touch-points (minimal edits, reversible)

- `config.py`: add `MetaPromptConfig` dataclass. ~20 lines.
- `loop.py`: `outer_loop()` calls existing `run()` in a for-loop, threading template dir + returning trajectory. ~80 lines. Existing single-template behavior preserved when `meta_prompt.enabled=false`.
- `prompts/loader.py`: already modular in spirit (`string.Template`). Extend `PromptLoader.render` to optionally concatenate slot files in a declared order. ~15 lines.
- `scenario_patch.py`: wrap `_build_obs_state` output in `AllowedKeysDict` when `obs_state_mode=local` and `whitelist_strict=true`. ~20 lines. Fires `FairnessViolation` fail-mode on forbidden key lookup.
- **Peak-M1 checkpointing** in BenchMARL callback (Tier 2 full training). Saves `best_policy_peak.pt` alongside `best_policy.pt` and records `peak_M1`, `peak_at_frame` in `final_metrics.json`. Closes `lero.md` §5.1 side-quest.
- No changes to `codegen.py`, `llm_client.py`.

### 8.4 Safeguards the codebase already teaches us we need

From the memory and `lero.md` §3, these are not optional:

- `reward_clip=50` on every inner candidate (already there). Meta-LLM is told this exists so it doesn't "design around it".
- Fallback chain on NaN/crash (already there) — outer loop must use the *best stable* candidate, not the crashed nominal-best.
- Tier 1 with 3 seeds (new) — single-seed evaluation is how `v2_twofn` looked viable before it collapsed at 10M.
- Peak-M1 checkpointing during Tier 2 (new — see §4.4, §8.3) — without it, reward hacking in the last 20% of training can mask a good intermediate policy.
- Fairness whitelist (new — §4.3). Cheap. Converts silent cheating into loud crash.
- Provenance graph — every template has `parent` field; we can replay and audit.

## 9. Phased rollout

| Phase | Scope | Gate to next phase |
| --- | --- | --- |
| **P0 Dry run** (0.5 day) | `v2_fewshot_modular` behavior-identical to `v2_fewshot`; outer loop disabled; fairness whitelist active. | Inner-loop M1 matches Phase-4 ±0.02; no fairness violations in the re-baseline. |
| **P1 Single mutation** (1 day) | Outer loop enabled; 1 outer iter; meta-LLM edits `guidance` only; 1 task (k=1). | New template runs end-to-end; `meta.yaml` + lineage written; `fairness.txt` hash unchanged. |
| **P2 Trigger + fail-modes** (1 day) | Plateau + fail-mode triggers; `failmode.py` taxonomy; peak-M1 checkpointing. | Triggers fire on known-bad templates (`v2_twofn` replay), don't fire on `v2_fewshot`; peak-vs-final gap detected on S3a_gpt5 replay. |
| **P3 Phase-1 sweep** (2–3 days GPU) | 3 outer × 3 inner × 5 cand × 3 seeds on k=1 and k=2 tasks, obs-only default config. | Beats `v2_fewshot` by ≥ 0.03 peak-M1 on k=2, or negative result documented. |
| **P4 Ablation** (1–2 days GPU) | Compare: free-form meta vs slot-only meta; plateau-trigger vs every-iter; obs-only vs reward-evolution modes. | Writeup added to `lero.md`. |

## 10. Risks and open questions

1. **Meta-LLM cost.** Each outer iter is 1 API call but history grows — cap at last 5 templates in the prompt (OPRO does 20; we're smaller-scale).
2. **Prompt drift.** Templates could diverge from human-readable. Mitigation: after every N outer iters, require the next edit to be "small" (hard length delta cap on the slot).
3. **Seed correlation.** 3 seeds is the minimum — may need 5 for Phase-2 claims. Budget accordingly.
4. **Is this really better than adding more examples?** Open. H8 (`v2_fewshot`) already showed examples help a lot. Meta-prompting's value may be bounded if the example pool is right. Phase-3 is the falsifier.
5. **Reward-evolution mode is risky on k≥2** (`lero.md` §2.6). Default config turns it off; keep it opt-in, clearly marked in logs, and don't let it influence the "headline" comparison tables.
6. **DSPy migration.** `loader.py:1–8` documents a future move to DSPy Signatures. LERO-MP's slot schema maps to DSPy 1:1 (role = Signature docstring, output_spec = OutputField typing, fairness = a fixed pre-instruction). Not blocking, but keep field names aligned.
7. **Tier-2 dependency on targets_respawn=False.** Known LERO gotcha (memory "Key Design Decisions"). Carry forward in the outer-loop config schema too.
8. **Whitelist false positives.** If a prompt-slot edit causes the LLM to legitimately reference a newly-added local key (e.g. we later expose `last_action`), the whitelist must be updated. Treat `AllowedKeysDict` as a source-of-truth that lives with `fairness.txt`.

## 11. Sources (primary literature consulted 2026-04-21)

- Yang et al., *Large Language Models as Optimizers (OPRO)* — <https://arxiv.org/abs/2309.03409>
- Guo et al., *EvoPrompt* — <https://arxiv.org/abs/2309.08532>
- Wang et al., *PromptAgent* — <https://arxiv.org/abs/2310.16427>
- Ma et al., *Eureka: Human-Level Reward Design via Coding LLMs* — <https://arxiv.org/abs/2310.12931>
- Yüksekgönül et al., *TextGrad* (Nature 2024) — <https://arxiv.org/abs/2406.07496>
- Khattab et al., *DSPy / MIPROv2* — <https://dspy.ai/api/optimizers/MIPROv2/>
- Zhou et al., *Automatic Prompt Engineer (APE)* — <https://sites.google.com/view/automatic-prompt-engineer>
- *A Systematic Survey of Automatic Prompt Optimization Techniques* (EMNLP 2025) — <https://aclanthology.org/2025.emnlp-main.1681/>
- Learning from Contrastive Prompts (LCP) — surveyed in above.

Internal references:

- `rendezvous_comm/docs/lero.md` — §2.3 (S3b-local breakthrough), §2.5 (eval-vs-final), §2.6 (feature-vs-incentive thesis), §3 (implementation lessons), §5.1 (peak-M1 TODO).
- `rendezvous_comm/docs/comparison_gnn_vs_minimal_team.md` — ER1/ER2/ER3 fairness envelope.

## 12. Decision checklist for review

Please mark each as approve / change / reject before I start coding P0:

- [ ] Two-loop architecture (§3).
- [ ] **Fair information constraint (§4)** — `obs_state_mode=local` as default, forbidden-keys whitelist, obs-only default on k≥2, peak-M1 checkpointing as fitness.
- [ ] Frozen `fairness.txt` slot (§5.1) — meta-LLM may read but never edit.
- [ ] Slot schema and file layout (§5.1). Any slots missing or redundant?
- [ ] Trial budget: 5 cand × 3 inner × 3 seeds = 45 Tier-1 runs per template (§6.3).
- [ ] Trigger conditions — plateau + fail-modes + variance + peak-vs-final (§6.2).
- [ ] Slot-picking policy from fail-mode taxonomy (§7.3) — or prefer round-robin / let meta-LLM choose?
- [ ] Phased rollout gates (§9).
- [ ] Config name: `meta_prompt` inside `lero:` vs separate experiment type `lero_mp`.
