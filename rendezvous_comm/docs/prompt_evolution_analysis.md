# Prompt Evolution Analysis — S3b-local vs v5

**Date:** 2026-04-29
**Purpose:** Trace what the LLM actually wrote each iteration, how scores moved in response, and what the textual feedback (or meta-refiner) said to drive each change. Two regimes side-by-side: the simple inner-loop S3b-local (1 LERO loop, obs-only) vs the v5 two-level architecture (outer metaprompt + inner code).

**Why this matters:** the head-to-head numbers (S3b-local 0.845 mean post-eval vs v5 0.010 single-seed) showed the architecture comparison clearly. This doc shows the *content* of the LLM's iteration to make the failure mode legible — not just "v5 didn't escape flat_zero" but "v5's inner LLM was given progressively sharper strategic guidance and still didn't escape flat_zero".

---

## 1. S3b-local seed 0 — single LERO loop, obs-only

**Config:** `evolve_reward=false`, `evolve_observation=true`, prompt `v2_fewshot_k2_local`, temperature 0.8, 4 inner iters × 3 candidates × 1M frames each. Reward stays = ER1 baseline (hand-crafted) throughout.

### Per-iteration evolution

| iter | Best candidate's observation features (key additions) | Reward | Best M1 @ 1M | Best M6 @ 1M | Feedback the LLM saw before this iter (drove change) |
|---|---|---|---:|---:|---|
| 0 | **Baseline feature set** — `min_t_dist`, target direction `(t_dx, t_dy)` from nearest-ray angle, `n_close_targets` (rays<0.25), nearest-agent equivalents, motion `(speed, vx, vy)`, position `(px, py)`, role one-hot. ~13 features. | ER1 baseline (unchanged) | 0.020 (cand 1) | 0.223 | (initial prompt only — no prior feedback) |
| 1 | **Adds density/dispersion features**: `t_close_mean` (mean distance of close-target rays), `t_dispersion` (std over rays). Drops some redundant motion features. Same role one-hot. ~9 features. | ER1 baseline (unchanged) | **0.050 (cand 2 — global winner)** | **0.425** | LLM saw iter 0's per-candidate metrics; cand 1 (M1=0.02) had richer geometry features, cand 0/2 had fewer. Feedback: "candidate 1 was best; generate 3 improved versions, keep what works, fix what doesn't." → LLM hypothesized that **summarizing density** (not just nearest-ray) would expose multi-target structure. |
| 2 | **Adds angular concentration**: weighted vector-mean of `cos(θ)`, `sin(θ)` over rays using inverse-distance weights `(1 / lidar)`, then `t_conc = √(cx²+cy²)`. Adds same for agents. ~12 features. | ER1 baseline (unchanged) | 0.040 (cand 1) | 0.285 | LLM saw iter 1 cand 2 (the M1=0.05 winner) — its strength was target-density via close-mean. Feedback nudged toward "infer whether approaching a single target or a cluster". → LLM's response: encode **angular concentration as a polar-mean of inverse distances** (a more principled density measure). It didn't beat iter 1, but explored a richer representation. |
| 3 | **Adds target-agent alignment**: cosine similarity between nearest-target direction and nearest-agent direction (`align = t_dx·a_dx + t_dy·a_dy`). Keeps prior density features. ~14 features. | ER1 baseline (unchanged) | 0.040 (cand 2) | 0.263 | Feedback after iter 2 emphasized "rendezvous/hold behavior" — agents need to know if a partner is *aligned* with the same target. → LLM added explicit cosine-similarity feature. Did not surpass iter 1 globally; iter 1 cand 2 remained the deep-train winner. |

**Global best for deep-train: iter 1 cand 2 (M1=0.050).**
After 10M deep-train: peak M1=**0.930** @ frame 8.04M, post-eval M1=**0.885**.

### What the textual gradient actually did

- **Plateaued fast (iter 0→1: +0.030 M1), then flat at 0.04–0.05 across iters 2–3.** The LLM's iterative refinement after iter 1 explored alternative representations (angular concentration, alignment) but didn't find a strictly better recipe in the remaining 2 iters.
- **0.05 at 1M was enough.** Although the inner loop didn't keep climbing, 0.05 sits clearly above the 0.02 flat_zero threshold, so the inner-eval signal was real and the downstream 10M deep-train converted it into 0.93 peak.
- **Reward never changed.** All the search budget went into observation — the only knob the inner LLM was allowed to turn.

---

## 2. v5 seed 0 (Mac full run) — two-level evolution

**Config:** `evolve_reward=true` AND `evolve_observation=true`, prompt `v2_fewshot_modular_v2`, temperature 1.0, 3 outer iters × 4 inner iters × 3 candidates × 1M frames each + 10M deep-train. Same task as S3b-local (cr=0.25 ms=400).

### 2a. Outer level — metaprompt evolution by meta-refiner

The metaprompt is three slot files (`guidance_observation.txt`, `guidance_reward.txt`, `guidance_shared.txt`) that get prepended to the inner LLM's system prompt. v5's meta-refiner reads inner-loop results and rewrites them.

| outer iter | guidance_observation (one-line summary) | guidance_reward (one-line summary) | guidance_shared (one-line summary) | Best inner M1 / fitness | Meta-refiner diagnosis (drove next outer's edits) |
|---|---|---|---|---|---|
| 0 | **(empty — bootstrap)** | **(empty — bootstrap)** | **(empty — bootstrap)** | M1=0.000  /  fitness +0.066 | (no prior outer iter — bootstrap state) |
| 1 | "Bias toward features that make local neighborhood structure actionable: relative geometry, nearest-agent/target relationships, signals separating 'I am alone near a target' from 'I am the second agent needed here.'" | "Reward `exact-2` coverage; distinguish under/correct/over-coverage; encourage early dispersion across targets, then stable pairing; discourage all 4 agents collapsing onto one target." | "Frame as pair-formation + role-symmetry-breaking + assignment-stability. Soft division of labor: spread first, pair, hold." | M1=0.000  /  fitness +0.001 (catastrophic dip on inner iter 4: −5.391) | "Empty metaprompt encoded a weak hypothesis that default policy could find structure. Refuted: best candidate flat_zero with only transient gain. **Missing: explicit pairwise coordination / target-sharing structure.** Wrong assumption: local sensing alone yields partner formation." |
| 2 | "Shift to explicit coordination state, not generic neighborhood summaries: assignment pressure, plausible pair-formation zone, the agent is *the second needed*, redundant 3rd/4th, or free to scout. Prefer compact thresholded indicators + local rank comparisons." | "Reward transitions creating exactly-two occupancy; penalize over-crowding beyond 2 and all-converging; favor early dispersion → 1-agent presence → 2-agent coverage → stability. Reward becoming the *missing partner* for an already-engaged target." | "Treat as distributed-matching problem with stability + anti-crowding. Discrete coordination stages: spread, identify singly-occupied targets, add missing partner, avoid excess." | M1=0.000  /  fitness +0.044 | "Outer 1's neighborhood-summary hypothesis was refuted (still flat_zero). **Missing: pairing state / assignment pressure.** Wrong assumption: better local geometry summaries are sufficient. Need sharper, more discrete coordination primitives separating 'join,' 'hold,' and 'divert.'" |

**Global best across all 3 outer iters: outer iter 0 (the empty bootstrap), fitness +0.066.**
The two refined metaprompts both scored *worse* than the bootstrap. After 10M deep-train of outer-0's best inner: peak M1 = **0.130** @ frame 9.72M, post-eval M1 = **0.010**.

### 2b. Inner level — code evolution within outer iter 0

Per outer iter, 4 inner iters × 3 candidates each. The inner LLM saw v5's enhanced feedback (best+worst code shown, cumulative tried-and-failed registry, stagnation/pivot warning).

| outer 0 inner iter | Best candidate observation (key features) | Best candidate reward (mechanism) | Best M1 | Best fitness | shape | What the v5 feedback said |
|---|---|---|---:|---:|---|---|
| 0 | **Generic baseline**: agent_pos, agent_vel, role-fraction `idx_feat`, raw lidar_targets/agents pass-through. No coordination-specific features. | "maximize coverage rate" — generic, dense reward proportional to mean(1/dist) terms. | 0.000 | +0.025 | flat_zero | (initial prompt only) |
| 1 | Adds `nearest_target_dist`, `second_target_dist` (top-2), nearest agent. Slightly more decision-oriented. | Reward shifts to `exact_cover` term + `under`/`over` penalties (mirroring guidance_reward). | 0.000 | +0.066 (best of outer 0) | flat_zero | "Best=cand0 with M6=0.05 flat_zero. Worst's lidar-dump in failed registry. Generate 3 improved." |
| 2 | Tries angular features (atan2 to nearest target). | Adds time penalty + small partial-coverage bonus. | 0.000 | +0.025 | flat_zero | Similar feedback; pivot not yet triggered. |
| 3 | Reverts toward simpler features (LLM started looping). | Reward gets noisier — squared-distance penalties. | 0.000 | −0.016 | flat_zero | **STAGNATION → pivot prompt**: "Last 2 iters did not improve. State explicit hypothesis why current approach stalls, OR pivot to fundamentally different feature family." |

**Outer iter 0 best is iter_1 cand_0** — and across all of outer 0–2, *this* candidate is what got deep-trained. Outer 1's refined metaprompt produced a candidate that scored fitness **−5.391** on inner iter 3 (mostly collision penalty + flat M1) — meaning the meta-refiner's edits actually *hurt* when the inner LLM tried to operationalize them in code.

### 2c. Why each outer iter's inner search couldn't differentiate candidates

Across all 12 inner iters of v5 × all 3 candidates per iter = **36 candidates**, **every single one scored M1 = 0.000 at 1M frames**. Fitness ranged across ~5.5 points (−5.39 to +0.066) but that range was driven by:
- Collision penalty (some reward variants caused agents to clump → high M4)
- Shape penalty (some hit M1=0.0 with non-rising trajectories)
- M6 micro-differences (0.001 vs 0.133)

**Not by any candidate actually solving any episode.** The inner-fitness ranking was effectively ranking *failure modes*, not progress.

---

## 3. Side-by-side: why one converged and the other didn't

### Per-candidate signal at 1M frames

| | S3b-local (3 seeds × 12 cands = 36 cands) | v5 Mac (1 seed × 36 cands) |
|---|---|---|
| Candidates with M1 ≥ 0.02 (escape flat_zero) | many — best per iter ranged 0.02–0.07 | **0** |
| Candidates with M1 = 0 | most | **all 36** |
| Best fitness range across iters | clear monotone signal | noise (−5.39 to +0.066) |
| Could the textual gradient grade meaningfully? | **yes** — iter 1's M1=0.05 vs iter 0's M1=0.02 is a real gap | **no** — every candidate ties at M1=0; differentiation is from collision/shape penalties, not progress |

### What changed between regimes (single-variable diff)

| factor | S3b-local | v5 |
|---|---|---|
| **`evolve_reward`** | **`false`** (ER1 reward unchanged) | **`true`** (LLM rewrites every candidate) |
| `evolve_observation` | true | true |
| inner candidates per outer | 4 × 3 = 12 | 4 × 3 = 12 (same) |
| inner eval frames | 1M | 1M (same) |
| outer meta-layer | none | 3 outer iters with textual gradient |
| prompt template | `v2_fewshot_k2_local` | `v2_fewshot_modular_v2` |
| temperature | 0.8 | 1.0 |

The hand-typed observation prompt + frozen reward let the inner LLM reach M1=0.02–0.07 at 1M. Switching reward-on let the inner LLM emit reward variants that *might* be exploitable but don't have time to express it within 1M frames — every candidate looks identical (M1=0) to the fitness function. Even the best meta-refinement of strategic framing can't recover a meaningful score gradient out of zero variance.

### The meta-refiner's diagnoses were good — they just had nothing to grade

Outer 1's diagnosis correctly identified that the bootstrap's "let local sensing find structure" hypothesis is wrong, and that **explicit pair-formation primitives** are needed. Outer 2's diagnosis correctly escalated to **discrete coordination stages: spread, identify, add partner, avoid excess.** Both diagnoses are exactly the kind of strategic insight you'd want from a meta-LLM.

But: outer 1's metaprompt made things *worse* (best inner fitness +0.001, with one iter scoring −5.391). The inner LLM, given sharper guidance, emitted code with sharper failure modes — the reward variants it produced when guided toward "exact-2 coverage rewards" must have penalized states the policy commonly reaches in early training, giving worse signal than a vague guidance. With only 1M frames inner eval, a reward that's "right in the limit" but "very negative early" looks identical to a useless reward.

### The takeaway

**The architecture isn't broken — the optimization target is.** v5's two-level textual gradient, weighted multi-metric fitness, best+worst feedback, registry, stagnation+pivot all worked as designed (visible in `_refiner_diagnosis.md` and per-iter feedback files). What was missing is signal in the inner-fitness function, and that's a function of `evolve_reward=true` interacting with 1M-frame inner-eval — neither of which v5's meta-layer can fix from above.

The next experiment is **"v5 with `evolve_reward=false`"**: keep the meta-layer, freeze the reward. Hypothesis: inner candidates start escaping flat_zero (like S3b-local), the textual gradient gets real signal, and the meta-layer has a chance to add value over S3b-local's hand-crafted prompt.

---

## 4. The prompt the inner LLM actually receives — S3b-local vs v5

So far we've compared what the *outer* layer evolves. But what does the inner LLM literally read each iteration? In both regimes there's a base template, plus (for v5) the meta-evolved slots prepended on top. This section opens both prompts side-by-side.

### 4a. Static structure

| | S3b-local (`v2_fewshot_k2_local`) | v5 base (`v2_fewshot_modular_v2`) + meta-evolved slots |
|---|---|---|
| Layout | **Monolithic** — `system.txt` (690 B) + `initial_user.txt` (3.6 kB) + `feedback.txt` | **Slot-decomposed** — 9 slots concatenated: `task_context → current_code → state_schema → fairness → guidance_shared → guidance_reward → guidance_observation → examples → output_spec` |
| Hand-curated for this task | **Yes** — written specifically for k=2 rendezvous + local-only sensors | **No** — generic LERO-MP template inherited from MPE Simple Spread examples |
| Editable by meta-LLM | No (fixed) | The 3 `guidance_*` slots are editable; rest is frozen |
| Total prompt size | ~4.3 kB | ~6 kB base + ~1.9 kB meta-edits (outer 1) ≈ 7.9 kB |

### 4b. Task framing — what the LLM is told the problem is

| | S3b-local `system.txt` (verbatim) | v5 `task_context.txt` (verbatim) |
|---|---|---|
| **Wording** | "**CRITICAL: This is a RENDEZVOUS task where k=2 agents must simultaneously occupy each target. Agents can only see via LiDAR — they do NOT know other agents' positions, target positions, or which targets are covered. Design features that help agents infer coordination state from sensor readings alone.**" | "$n_agents agents must cover $n_targets targets. A target is covered when $agents_per_target agent(s) are within $covering_range distance. Arena: [-1,1] x [-1,1]." |
| Specificity | Names the failure mode (RENDEZVOUS, k=2 simultaneous), names the constraint (lidar only), names the goal (infer coordination state from sensors) | Generic templated definition; doesn't emphasize k=2, doesn't say "rendezvous", doesn't mention the local sensor restriction |

### 4c. State schema — what variables the LLM is told it can use

| | S3b-local `initial_user.txt` (excerpt) | v5 `state_schema.txt` (verbatim) |
|---|---|---|
| Listed keys | **`agent_pos`, `agent_vel`, `agent_idx`, `lidar_targets`, `lidar_agents` only** — followed by an explicit "**You do NOT have access to:** `agents_pos`, `targets_pos`, `agents_targets_dists`, `covered_targets`, `agents_per_target`, `all_time_covered`, or any global state." | **All keys including global state** — `agents_pos`, `targets_pos`, `agents_targets_dists`, `covered_targets`, `agents_per_target`, `all_time_covered`, then local ones |
| Match to runtime when `obs_state_mode=local` | ✅ matches — schema = what LLM gets at runtime | ❌ **misleading** — schema lists oracle keys the runtime patched scenario does NOT provide |

This is a real footgun: v5's inner LLM is shown a schema that includes `agents_pos`, `targets_pos`, `covered_targets`, etc., and is encouraged by the schema to use them — but at runtime those keys aren't in `scenario_state`. The validator catches it (KeyError → candidate fails AST/sig check), but it costs retries and fragments search.

### 4d. Examples shown to the LLM

| | S3b-local | v5 base (`examples.txt`) |
|---|---|---|
| Example domain | **Observation enhancement** — a complete `enhance_observation` showing dir_x/dir_y from min-ray angle, n_close count, one-hot role | **Reward functions only** — two `compute_reward` examples (dense shaping + coverage bonus) |
| Uses local-only state? | ✅ uses `lidar_targets`, `agent_idx` only | ❌ uses `targets_pos`, `agents_targets_dists`, `covered_targets` (global keys) |
| Anchored to this task | ✅ "rendezvous", k=2-aware, lidar-aware | ❌ "MPE Simple Spread tasks. Use them as references, not templates" — explicitly cross-task |

The v5 inner LLM sees zero observation examples, only reward examples — and those reward examples are written for a different task with global state. To produce a useful local-obs function, the inner LLM must extrapolate **across both task and modality** every iteration. S3b-local's inner LLM gets a working local-obs example to refine.

### 4e. Operational guidance — "what features to compute"

S3b-local `initial_user.txt` includes a section "**What you CAN infer from LiDAR**" with explicit suggestions:
- "Direction to nearest target: the ray with smallest value points toward the closest target"
- "Distance to nearest target: min of lidar_targets rays"
- "Number of nearby targets: count of rays below a threshold (e.g. < covering_range)"
- "Agent density nearby: from lidar_agents, count of rays below threshold"
- "Angular distribution: which directions have targets vs agents"
- "Role differentiation: agent_idx as one-hot so the shared policy can assign different roles"

v5 outer 1's `guidance_observation.txt` (after meta-refine) is purely strategic-abstract:
> "Bias observation feature ideas toward making local neighborhood structure actionable. Emphasize relative geometry to nearby agents and targets, especially cues that help detect when one target already has a teammate and when another target is unclaimed. Useful observation-family ideas include local counts or proximity summaries, nearest-agent / nearest-target relationships, and compact signals that separate 'I am alone near a target' from 'I am the second agent needed here.'"

v5 outer 2 escalates further into strategic-abstract:
> "Shift observation ideas toward explicit coordination state rather than generic neighborhood summaries. Prioritize features that let an agent infer assignment pressure: whether any target is already occupied by one teammate, whether the agent is the nearest unassigned helper to a target, and whether it is inside a plausible pair-formation zone relative to a target."

The S3b-local prompt names operations (`min`, `count of rays below threshold`, `angle from ray index`). The v5 metaprompt names abstractions (`assignment pressure`, `pair-formation zone`, `unassigned helper`). Both are coherent — but only one bottoms out at a recipe the LLM can immediately translate to PyTorch.

### 4f. Quantitative summary

| dimension | S3b-local | v5 base + outer 1 meta-edits |
|---|---|---|
| Code-level operational hints | **6 named operations + working code example** | 0 operations; 0 obs examples |
| Task-specific framing (k=2, rendezvous, local) | explicit + "CRITICAL" emphasis | absent in base, partially recovered by meta-edits |
| State schema accuracy under `obs_state_mode=local` | matches | mismatched (lists oracle keys) |
| Example domain match (obs vs reward) | obs example | reward examples only |
| Total prompt the inner LLM sees | ~4.3 kB, single voice | ~7.9 kB, multiple voices (base templates + meta-LLM additions) |

### 4g. Why this matters for the v5 result

v5 was set up as if the meta-refiner's job was to **discover** the strategic framing. But the base template the meta-edits sit on top of is:
1. Reward-centric (no obs example)
2. Misaligned to local-obs-mode (state_schema lists oracle keys)
3. Generic on task framing (no rendezvous emphasis)

So even when the meta-LLM's diagnoses are sharp ("missing pair-formation primitives"), the inner LLM has to translate that sharpness into PyTorch code without any nearby example of what the right TYPE of code looks like, while being shown a misleading schema. That's a hard ask even at temperature 0.8 — at temperature 1.0 it produces high variance and zero convergence.

S3b-local's success isn't *just* `evolve_reward=false`. It's also that the human-curated prompt closes the gap between strategic intent and code recipe in a way the meta-LLM didn't.

### 4h. Implications for the next experiment

If you want a **clean** test of "does the meta-LLM textual gradient add value over a hand-crafted prompt?", the v5 base template needs three repairs first:

1. **Use a local-obs base template** (or fix `state_schema.txt` to gate on `obs_state_mode`) — the inner LLM should not be shown oracle keys it can't actually use.
2. **Add a local-obs example to `examples.txt`** alongside the reward examples (or use S3b-local's `enhance_observation` example directly).
3. **Bake k=2 / rendezvous / "infer coordination from sensors" into `task_context.txt`** instead of leaving it to the meta-LLM to discover.

Then `evolve_reward=false` + the v5 meta-layer can be tested without confounds. Hypothesis: with the prompt floor raised to S3b-local's level, the v5 meta-layer's only job is to evolve *strategic refinements on top of a working base*, which is the regime where textual-gradient methods empirically work.

---

## 5. Relation to the original LERO paper (arXiv:2503.21807)

S3b-local is the LERO paper algorithm with three substantive deviations.

### What's faithful

| element | paper | S3b-local |
|---|---|---|
| Iterative LLM refinement loop | K=4 iters | `n_iterations=4` ✓ |
| Multiple candidates per iter | N=2 | `n_candidates=3` (close) |
| Top-k feedback selection | "Selector ranks; best code returned to LLM" | `top_k=2` in `build_feedback` ✓ |
| Code+metrics feedback to LLM | "best HRFs/OEF code + performance metrics" | `feedback.txt` includes top-2 code + per-cand metrics ✓ |
| `replace` reward mode | ✓ | `reward_mode: "replace"` ✓ |
| Joint deep-train of best | ✓ | `full_frames=10M` ✓ |

The iterative-refinement loop is **paper-derived**, not a S3b-local novelty.

### Three substantive deviations

| # | element | paper | S3b-local | nature |
|---|---|---|---|---|
| 1 | **`evolve_reward`** | **true** (HRF + OEF jointly) | **false** (only OEF) | **scientific** |
| 2 | eval frames per candidate | **30k** | **1M (33×)** | scale |
| 3 | conversation memory | "fresh prompt with concatenated historical best code" | sliding window: last iter's best + feedback (`loop.py:762-780`) | implementation |

Plus task-fairness deviations: `obs_state_mode=local` (vs paper's global oracle), task = VMAS Discovery k=2 (vs paper's MPE Simple Spread k=1), `bonus_scale=0.5` and `reward_clip=50.0` engineering safeguards (because Discovery rewards are unbounded, MPE were [0,5]).

### The single load-bearing deviation

`evolve_reward=false`. With the paper-faithful `evolve_reward=true` configuration on the same hard k=2 task: S3, S3a, S3a_gpt5, S3ac all hit M1 = 0.08–0.105. With `evolve_reward=false`: S3b-local hits M1 = 0.845. The paper's joint-evolve was viable on simple-coordination MPE Simple Spread; it reward-hacks on hard-coordination Discovery k=2.

### Thesis framing

> **"The LERO paper showed iterative LLM-driven evolution works on simple coordination (k=1 MPE). We show that on hard coordination (k=2 VMAS Discovery), the paper's joint reward+obs evolution fails (M1 ≤ 0.10) due to reward hacking, but the obs-only restriction succeeds (M1 = 0.845)."**

This restates the Feature-Engineering-vs-Incentive-Design thesis already in `all_experiments_analysis.md`. The novelty is not new search machinery — it is a principled restriction of what the LLM is allowed to evolve, validated empirically on a harder task than the paper used.

---

## Appendix: artifact pointers

**S3b-local s0 run** (cited above for code excerpts):
`results/s3b_local_replicate/lero_s3b_local_s0/lero/runs/lero/20260429_0508/`
- `iter_{0..3}/candidate_{0,1,2}_obs.py` — per-candidate code
- `iter_{1..3}/feedback.txt` — what the LLM saw before each iter
- `evolution_history.json` — per-iter best M1/M2/M6
- `final_metrics.json` — 10M post-eval (M1=0.885)

**v5 Mac s0 run**:
`results/lero_v5/lero_v5_rendezvous_k2/full_mac_seed0_20260428_2159/`
- `prompts/v5_outer_{0,1,2}_seed0/guidance_*.txt` — metaprompt slot files per outer iter
- `prompts/v5_outer_{1,2}_seed0/_refiner_diagnosis.md` — meta-refiner's diagnosis paragraph
- `prompts/v5_outer_{1,2}_seed0/_refiner_response.txt` — full meta-LLM response (includes JSON SLOT_EDITS)
- `outer_{00,01,02}_inner/iter_{0..3}/feedback.txt` — v5 inner feedback (best+worst code excerpts, registry, stagnation flag)
- `_v5_checkpoint.pkl` — full state including registry + iter records
- `v5_summary.json` — 10M post-eval (M1=0.010)

For additional context see `docs/all_experiments_analysis.md` § "LERO-MP v4 / B' / v5 — Architectural Complexity Made It Worse (2026-04-29)".
