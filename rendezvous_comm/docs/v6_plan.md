# LERO v6 Plan — Simplicity-First Meta-Strategy, Inner-Only Validation

**Status:** Draft, awaiting sign-off before implementation
**Date:** 2026-04-29
**Predecessor:** v5 (failed: M1=0.010 single-seed Mac), B' (failed: 0.238 mean), v4 (failed: 0.117 mean)

---

## 1. The premise

S3b-local proved that with the right prompt and the right problem framing, this task is **consistently solvable** — 3 seeds clustered at M1=0.820–0.885 (mean 0.845). The S3b-local prompt is hand-curated by a human with task knowledge. v6's goal is to have the **meta-LLM rediscover such a prompt** without human seeding of the answer features, validated by reaching S3b-local-level inner-loop signal.

If v6 succeeds: we have a meta-prompting recipe that re-derives S3b-local's quality from scratch. If it fails after a fair attempt: the textual-gradient meta-layer is fundamentally not a substitute for human prompt curation on hard coordination tasks.

---

## 2. Definition of "without cheating"

**Allowed**:
- The meta-LLM may use the same *kind* of language S3b-local's prompt uses (LiDAR-aware hints, k-aware framing, code-level operations) — because that's just English describing the problem space, not the answer.
- The meta-LLM is told the task definition (n_agents, n_targets, k, lidar geometry).
- The base inner prompt template provides the runtime state schema and an output-format example.

**Not allowed (this is what would constitute cheating)**:
- Naming `hold_signal`, `approach_signal`, `crowd_signal`, `sparsity_signal`, `gap_to_target`, etc. anywhere in v6's source code, configs, or seed metaprompts.
- Hand-writing the working observation function as a fewshot in the base template.
- Pre-populating `guidance_observation.txt` with anything beyond an empty string at outer iter 0.

**Verification**: a grep over `src/lero/v6/`, `configs/lero_v6/`, and any v6 prompt templates must not contain the feature-name list above. The meta-LLM has to rediscover them (or equivalents) on its own.

---

## 3. Design principles

### 3.1 Simplicity-first
The meta-LLM is instructed to start with the **simplest possible** strategy — usually "expose one or two obvious local-sensor summaries to the policy" — and only escalate complexity when reflection on inner feedback shows simpler approaches plateau.

### 3.2 Reflection-driven escalation
After each inner search, the meta-LLM classifies results into one of four categories and chooses the next move accordingly:

| classification | trigger | next move |
|---|---|---|
| **found_good** | best inner M1 ≥ 0.05 AND `monotonic_rise` OR `late_ramp` | **STOP** outer loop. Save winners. |
| **partial_signal** | best inner M1 ≥ 0.02 but shape unstable (oscillating / plateau) OR rising M6 trajectory across iters | **refine_current** — keep direction, add focus / refine specific obs feature |
| **no_signal_simple** | all flat_zero, simple obs strategy was tried this round | **try_different_simple** OR **add_simple_reward** — different obs angle OR add a single reward shaping term |
| **no_signal_complex** | all flat_zero AND prior rounds already tried complex strategies | **reset_simpler** — explicitly retreat to a simpler hypothesis; meta-LLM must articulate which prior assumption to drop |

The meta-LLM emits both the classification and the next-mode choice, with a 2-3 sentence justification. Both are saved for postmortem.

### 3.3 Mode selection: obs / reward / both
The meta-LLM picks which dimension(s) to evolve **per outer iter**, not all-at-once like v5 did. **The decision is two-layered**:

1. **Flag-level mode** — meta-LLM sets `evolve_observation: bool` and `evolve_reward: bool` for the next inner LERO call. These flags propagate into `LeroConfig` and control whether the inner LLM emits `enhance_observation`, `compute_reward`, or both. This is more than just slot-text editing — it actually changes what the inner LERO loop is allowed to evolve.
2. **Slot-text** — meta-LLM writes the corresponding `guidance_*` text only for slots that the flag-level mode enables. Disabled slots are left empty.

**Mode progression rules**:

- Outer iter 0 default: `evolve_observation=true, evolve_reward=false` (the simplicity-first prior; matches the S3b-local mode that worked).
- `evolve_reward=true` unlocked **only when** obs-only has been tried and classification is `no_signal_simple` or `partial_signal`.
- The meta-LLM justifies each mode change in writing (`rationale` field).
- Both flags can be toggled in either direction across rounds — e.g., if reward evolution is tried and produces `no_signal_complex`, the meta-LLM can revert to obs-only via `reset_simpler`.

This implements the user's instruction: "metaprompt should suggest simple strategy first... and based on results known, unknown or not sure evolve and try other simple or more complex strategy".

### 3.3.1 One strategy per outer iter — inner explores it
Each outer iter emits **a single strategy** (not 3 like v4 / v5's strategy bundle). That single strategy becomes the metaprompt for the inner LERO loop, which then explores it via its standard S3b-local-style mechanism: 4 inner-iters × 3 candidates × 1M frames each, with iterative-refinement feedback between inner-iters. The "exploration" happens at the inner level (12 candidates per outer); the outer level focuses on **strategy selection**, not parallel-strategy fanout.

This is the cleanest test of the simplicity-first hypothesis: each outer round commits to exactly one English-level direction, lets the inner LLM thoroughly explore it via 12 code attempts with feedback, then reflects on the outcome.

### 3.4 Early stop
When classification = `found_good`, the outer loop **terminates immediately**. No further metaprompt evolution; no 10M deep-train (gated to a separate step). The output is the set of inner candidates that satisfied the threshold.

### 3.5 No deep-train until inner is solid
v4/v5/B' burned ~25% of their per-seed budget on a 10M deep-train of an inner-stage random-pick. v6 spends zero on deep-train inside the loop. The deep-train is a separate, gated step that runs only if inner-stage produces a candidate with `M1 ≥ 0.05` + `monotonic_rise`. This isolates the variable we're testing (can the meta-layer reach S3b-local-quality inner signal) from the confounder (did the deep-train get lucky).

---

## 4. Architecture changes vs v5

### 4.1 Base prompt repair (prerequisite)
Per §4 of `prompt_evolution_analysis.md`, the v5 base template (`v2_fewshot_modular_v2`) is misaligned for `obs_state_mode=local`:

- **`state_schema.txt`** lists oracle keys (`agents_pos`, `targets_pos`, `agents_targets_dists`, `covered_targets`, etc.) the runtime patched scenario doesn't provide. → Fix: gate global keys behind `obs_state_mode=global`; in local mode, schema lists only `agent_pos`, `agent_vel`, `agent_idx`, `lidar_targets`, `lidar_agents`, `n_agents`, `n_targets`, `covering_range`, `agents_per_target_required`.
- **`examples.txt`** has only reward examples using global keys. → Fix: add a minimal local-obs example showing PyTorch vectorization patterns (NOT the answer features). Concretely: a single example that returns one feature like `min(lidar_targets)` — just enough to anchor the output format, well below the operational density of S3b-local's prompt.

These repairs are not cheating — they fix the prompt to match runtime, removing a footgun. Without them, even a smart meta-LLM is fighting a misaligned floor.

### 4.2 Meta-strategist rewrite (`src/lero/v6/meta_strategist.py`)

The meta-LLM call now produces a **structured decision** per outer iter, not just slot text:

```python
@dataclass
class V6MetaDecision:
    classification: Literal["found_good", "partial_signal",
                            "no_signal_simple", "no_signal_complex"]
    next_mode: Literal["stop", "refine_current", "try_different_simple",
                       "add_simple_reward", "reset_simpler"]
    rationale: str                # 2-3 sentence why
    # Flag-level mode (drives LeroConfig.evolve_*):
    next_evolve_observation: bool
    next_evolve_reward: bool
    # Slot-text edits — only for slots enabled by the flags above.
    # Slots not in the dict carry over from prior round.
    slot_edits: Dict[str, str]
    # Complexity ladder — used to enforce monotone-only-when-justified escalation.
    complexity_level: int         # 1 (simplest) … 4 (most complex)
```

The meta system prompt enforces:
1. **Start at complexity_level=1** in outer iter 0.
2. **Only increase complexity when classification justifies it** — not "let me try something fancier."
3. **STOP and recommend deep-train when classification = `found_good`**.
4. **No specific feature-name suggestions in `slot_edits`** — meta-LLM must talk about families ("expose nearest-target direction") not handles ("`hold_signal`").

### 4.3 Outer loop changes (`src/lero/v6/outer_loop.py`)

- `MAX_OUTER = 5` (vs v5's fixed 3) — gives more room to escalate if needed, but typical run terminates much earlier via early-stop.
- Before each inner call: apply the prior decision's `next_evolve_observation` / `next_evolve_reward` flags to the inner `LeroConfig` (`base_loop.lero.evolve_observation = ...; base_loop.lero.evolve_reward = ...`). Each outer iter therefore runs with the flag combo the meta-LLM chose.
- After each inner: call `classify_and_decide()` → `V6MetaDecision`. The single strategy produced becomes the metaprompt for the **next** outer iter.
- If `decision.next_mode == "stop"`: break, save winners. (Deep-train is **not** invoked — see §4.5.)
- If complexity_level monotone-broken (e.g., went from 3 to 1 without `reset_simpler` justification): log warning + force the agreed mode anyway (don't argue with the LLM, but record).
- Mode-unlock enforcement: in outer iter 0, `next_evolve_reward=true` is rejected programmatically and replaced with `false`; logged as a violation.
- Checkpoint after every inner (already wired in v5; reuse).

### 4.5 Deep-train: kept commented out for now

v6's first run does **not** invoke deep-train. The `deep_train_winner()` callable (or separate `run_v6_deep_train.py` script) is implemented but kept commented in the runner. Once v6 produces a `found_good` candidate that we want to validate against S3b-local's 0.845, we uncomment / call it manually. This keeps v6 cheap during the architecture-validation phase.

### 4.4 Inner loop reuse
v5's `run_inner_loop` is unchanged. The textual-gradient feedback (best+worst, registry, stagnation) already works at the inner level. The fitness function (weighted multi-metric) is preserved.

---

## 5. The meta system prompt — exact text scaffold

Below is the meta system prompt v6 will use. Note: it does NOT name LiDAR features, does NOT name coordination signals, does NOT say "use min ray" etc. It only encodes the **methodology** of simplicity-first reflection.

```
You are a meta-strategist for an LLM-driven evolutionary search. Your job
is to write the high-level English guidance that an INNER LLM will use
to write Python observation/reward code for a multi-agent task.

You MUST follow these rules:

1. SIMPLICITY FIRST.
   In outer iter 0, propose the simplest possible strategy. Examples of
   the level you should aim for: "expose one or two summary statistics
   from the local sensors that capture where the closest task-relevant
   thing is and how close it is." Do NOT enumerate features. Do NOT
   describe complex multi-agent coordination logic. Trust that the
   inner LLM can take simple guidance and write functioning code.

2. ESCALATE ONLY WHEN JUSTIFIED.
   You may increase complexity_level (1=simplest, 4=most complex) only
   if the prior round's classification was "no_signal_simple" or
   "partial_signal" AND you can name in writing the specific failure mode
   you're trying to address.

3. CLASSIFY BEFORE DECIDING.
   Each round, you produce a classification of the inner search result
   in {found_good, partial_signal, no_signal_simple, no_signal_complex}
   based on the inner trajectory. The classification, not your taste,
   drives the next move.

4. STOP WHEN GOOD.
   If best inner M1 ≥ 0.05 with rising shape, classification = found_good.
   You output next_mode=stop and write nothing further. Do not chase
   marginal improvements.

5. MODE SELECTION: OBS / REWARD / BOTH.
   In outer iter 0 you may only set next_evolve_observation=true and
   next_evolve_reward=false. You may unlock next_evolve_reward=true
   in a later round only if obs-only has been tried and the
   classification suggests obs alone won't suffice. Each outer iter
   produces ONE strategy — the inner LERO loop will explore it
   thoroughly via its own 4×3×1M iterative search. Don't try to do
   the inner loop's job at the outer level. Justify any mode change
   in writing.

6. NO PRE-CANNED ANSWERS.
   You do NOT know the optimal feature set. You do NOT name specific
   features by handle. Talk about families and operations, not solutions.

You will be given:
- The task definition (number of agents, targets, k, sensor description).
- The current metaprompt (slot files).
- The inner-loop result from the prior round (best+worst code, fitness
  trajectory, M1/M6 trajectories, shapes).
- The cumulative outer registry (what was tried, what worked, what didn't).

You will output a JSON object with:
  classification, next_mode, rationale, slot_edits, complexity_level
```

The classification rules + complexity_level enforcement + mode-unlock gating are checked **programmatically in the outer loop** after the meta-LLM responds. If the meta-LLM violates a rule (e.g., emits guidance_reward in outer 0), the outer loop strips the violation and logs a warning. The meta-LLM's output is a *suggestion*; the policy is enforced by code.

---

## 6. Concrete budget

### 6.1 Per outer iter
- Inner: 4 inner-iters × 3 candidates × 1M = 12M frames
- Meta-LLM call: ~10-15s (negligible vs frames)

### 6.2 Per seed
- 1-5 outer iters depending on early-stop
- Best case (early-stop at iter 1 or 2): 12-24M frames, ~3-6h on V100S
- Typical case (iter 3): 36M frames, ~7h on V100S
- Worst case (max iter, no convergence): 60M frames, ~12h on V100S
- **No 10M deep-train** in v6 itself

### 6.3 Sweep cost (3 seeds)
- Best/typical case: ~€20-€60 on OVH; or 9-21h sequential on Mac with checkpoint-resume
- Worst case: ~€100 on OVH

### 6.4 Comparison to v5/B'
- v5 was 46M/seed (with deep-train) at €0 (Mac) or €22 (OVH per seed)
- v6 typical 36M/seed at similar cost — but the budget is spent inside the loop instead of on deep-train
- Net: **same or smaller** total spend than v5

---

## 7. Stopping & success criteria

### 7.1 Inner-loop success (v6 own goal)
- **Primary**: classification = `found_good` in any outer iter, i.e. **best inner M1 ≥ 0.05 with monotonic_rise / late_ramp shape**, replicated in ≥ 2/3 seeds.
- **Secondary**: best M6 trajectory shows climbing (top inner candidate has M6 ≥ 0.20 by frame 800k), even if M1 stays just below 0.05 — partial-signal pathway.

### 7.2 Downstream validation (separate experiment, only if 7.1 holds)
- Run 10M deep-train of v6's best inner candidate (the one that triggered `found_good`).
- Compare post-eval M1 to S3b-local's 0.845 mean.
- Success = within 10pp of S3b-local across 3 seeds.

### 7.3 Failure mode interpretations
- **Inner never escapes flat_zero across all simplicity levels**: meta-LLM textual gradient cannot rediscover a working coordination prompt from natural-language reflection alone, even when methodology is correct. Strong negative result; thesis pivot needed.
- **Inner reaches partial_signal but never `found_good`**: meta-LLM gets close but plateau; investigate whether more outer iters / different complexity ladder would close the gap.
- **`found_good` but deep-train collapses to <0.5**: inner signal was misleading (e.g., reward-hacking that wasn't visible at 1M). Suggests inner-eval budget needs to grow past 1M for v6.

---

## 8. Implementation steps (proposed order)

1. **Repair base template** (`src/lero/prompts/v2_fewshot_modular_v2_local/`)
   - Branch from `v2_fewshot_modular_v2`
   - Fix `state_schema.txt` to local-only when `obs_state_mode=local`
   - Add minimal local-obs example (single feature, intentionally weak — anchors output format only)
   - Verify with diff against original
2. **Write `src/lero/v6/meta_strategist.py`**
   - `V6MetaDecision` schema (Pydantic + JSON-parse retry like v5_strategist)
   - Meta system prompt (the §5 text)
   - Decision classifier helper (so we can compute the same classification ourselves and cross-check the LLM's claim)
3. **Write `src/lero/v6/outer_loop.py`**
   - Reuse v5's checkpoint, inner_loop, registry
   - Replace meta_refiner with the new V6MetaDecision-driven flow
   - Add early-stop on `found_good`
   - Add complexity-level monotone enforcement
   - Add mode-unlock gating (obs in iter 0, reward unlocked iter 1+)
   - **Skip** deep-train inside the loop; expose a separate `deep_train_winner()` callable post-hoc
4. **Configs**
   - `configs/lero_v6/rendezvous_k2_smoke.yaml` — Mac smoke (1 outer × 2 inner × 1 cand × 100k, ~10 min)
   - `configs/lero_v6/rendezvous_k2.yaml` — full (5 outer max × 4 × 3 × 1M, no deep)
5. **Smoke on Mac** — verify pipeline + classification logic + early-stop behavior
6. **Full run**
   - 3 seeds on OVH parallel (clean apples-to-apples vs S3b-local)
   - OR: 1 seed Mac for first sanity check then commit to OVH sweep
7. **Anti-cheat audit**
   - `grep` v6/ for forbidden feature names → must return zero matches
   - Read v6 metaprompt history of the winning seed → confirm meta-LLM-derived (not human-seeded) phrasing
8. **Compare to S3b-local**
   - If `found_good` triggered: run deep-train on best inner, compare to S3b-local's 0.845
   - Update `all_experiments_analysis.md` with v6 entry

---

## 9. Risks & mitigations

| risk | severity | mitigation |
|---|---|---|
| Meta-LLM ignores SIMPLICITY-FIRST and writes complex guidance in iter 0 | medium | Code-side enforcement: strip slot_edits violating mode-unlock or complexity-level rules; log warning |
| Meta-LLM's classification is wrong (says `found_good` when it isn't) | medium | Compute classification independently in outer code from raw inner metrics; cross-check LLM's claim; trust code |
| v5 base prompt repair masks an issue we can't yet see | low | Keep the repair minimal (schema gate + 1-feature example); document as a separate fix not v6 contribution |
| Inner stays flat_zero across all complexity levels even with simplicity-first | medium | Negative result is informative — a real datapoint that meta textual-gradient ≠ human prompt-engineering on hard coord. Document and pivot. |
| `evolve_reward=true` once unlocked still poisons inner like in B'/v5 | medium | Mode-selection gates this. Reward only unlocks if obs-only is exhausted and meta justifies. If it still poisons → keep reward-evolution off entirely (matches S3b-local) |
| Anti-cheat audit fails (forbidden features leak in) | high | Pre-commit grep + manual review of meta system prompt before any run |

---

## 10. Open questions — RESOLVED (2026-04-29 sign-off)

1. **Complexity-level numeric scale** → **1–4** (defaults).
2. **`add_simple_reward` simplicity-constrained on first edit** → **yes** (defaults).
3. **Code wins on classification disagreement** → **yes** (defaults).
4. **Hard MAX_OUTER cap** → **yes, 5** (defaults).
5. **Deep-train code path** → **kept commented / disabled** in the v6 runner. The function is implemented for later use but not invoked by default v6 runs. Once v6 produces a `found_good` candidate we want to validate against S3b-local's 0.845, we uncomment + run it as a separate, gated step.

## 11. New v6-specific feature added 2026-04-29

- **Meta-LLM controls `evolve_observation` / `evolve_reward` flags directly** (not just guidance slot text). Each outer iter's `V6MetaDecision` carries `next_evolve_observation: bool` and `next_evolve_reward: bool`, which are applied to `LeroConfig` before the next inner call. This is the operationalization of "metaprompt should decide first just to change obs and based on feedback... continue obs only or also reward."
- **One strategy per outer iter, inner LERO explores it.** The outer level emits exactly one English-level direction per round; the inner level explores it via its standard 4×3×1M iterative-refinement loop. No parallel-strategy fanout at outer.

Ready to implement following the order in §8 (base prompt repair → meta_strategist → outer_loop → smoke).
