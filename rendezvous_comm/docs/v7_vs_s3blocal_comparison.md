# v7 vs S3b-local — Generated Code & Inner Prompt Comparison

**Date:** 2026-05-01
**Purpose:** Side-by-side feature-by-feature comparison of the observation function and inner prompt that produced S3b-local's M1=0.93 (deep-trained from iter-1 winner with M1=0.05 at 1M-eval) vs v7 run 1's best inner candidate (M1=0.000 at 1M-eval, 35 features, feature_stack_score=5/5).

Runs compared:

- **S3b-local** seed 0, iter 1, candidate 2: `results/s3b_local_replicate/lero_s3b_local_s0/lero/runs/lero/20260429_0508/iter_1/candidate_2_obs.py`. M1@1M = **0.050**, deep-train M1@10M = **0.885**.
- **v7 run 1**, outer 0, iter 2, candidate 0: `results/lero_v7/lero_v7_rendezvous_k2_2x3/run1_20260430_2111/outer_00/inner/iter_2/candidate_0_obs.py`. M1@1M = **0.000**, M6 = 0.166, fitness = +0.069. (Best feature_stack_score=5/5 candidate from outer 0.)

---

## 1. Reward function — IDENTICAL

Both runs use `evolve_reward: false`. The reward is the ER1 hand-crafted scenario reward:

```
covering_rew_coeff = 1.0      # +1.0 per target newly covered this step
agent_collision_penalty = -0.01  # per collision per step
time_penalty = -0.01          # per step
shared_reward = true          # team-summed
```

No LLM-generated reward function in either run. **No reward-side difference.**

---

## 2. Observation function — feature-by-feature side-by-side

### Total feature counts

| | S3b-local iter-1 winner | v7 run-1 outer-0 iter-2 best |
|---|---:|---:|
| Returned features | **19** | **35** |
| feature_stack_score (cross + dir + role + motion + cover_thresh) | **5/5** | **5/5** |
| Source LiDAR rays used | both | both |
| Threshold used for boolean masks | `covering_range` (0.25) | `covering_range` (0.25) |

Both functions use the same input keys, same threshold, and produce a complete 5/5 feature stack. The structural floor is the same. The differences are in *which* specific features each chooses and how dense the cross-source layer is.

### Feature-by-feature table

Every feature in either function is listed once. Columns indicate presence + the variable name in each.

#### Spatial primitives — target side

| feature concept | S3b-local | v7 run-1 best | both? |
|---|---|---|---|
| Distance to nearest target | `min_t_dist = lidar_t.min(dim=-1).values` | `t_min = lidar_t.min(dim=-1).values` | ✓ both |
| Direction-x to nearest target | `t_dx = cos(min_t_ray * 2π / T)` | `t_dir[..., 0]` (same formula) | ✓ both |
| Direction-y to nearest target | `t_dy = sin(min_t_ray * 2π / T)` | `t_dir[..., 1]` (same formula) | ✓ both |
| 2nd-nearest target distance | — | `t_2 = sorted_lidar_t[:, 1:2]` | only v7 |
| 3rd-nearest target distance | — | `t_3 = sorted_lidar_t[:, 2:3]` | only v7 |
| Count of target rays < cover_r | `t_nclose = (lidar_t < cover_r).sum()` | `t_count = (lidar_t < cover_r).sum()` | ✓ both |
| Count of target rays < 0.5·cover_r | — | `t_count_half = (lidar_t < 0.5*cover_r).sum()` | only v7 |
| Mean of close-target rays | `t_close_mean = ...` | — | only S3b |
| Std/dispersion of target rays | `t_dispersion = lidar_t.std()` | — | only S3b |

#### Spatial primitives — agent side

| feature concept | S3b-local | v7 run-1 best | both? |
|---|---|---|---|
| Distance to nearest agent | `min_a_dist = lidar_a.min(dim=-1).values` | `a_min = lidar_a.min(dim=-1).values` | ✓ both |
| Direction-x to nearest agent | `a_dx = cos(min_a_ray * 2π / A)` | `a_dir[..., 0]` (same formula) | ✓ both |
| Direction-y to nearest agent | `a_dy = sin(min_a_ray * 2π / A)` | `a_dir[..., 1]` (same formula) | ✓ both |
| 2nd-nearest agent distance | — | `a_2 = sorted_lidar_a[:, 1:2]` | only v7 |
| 3rd-nearest agent distance | — | `a_3 = sorted_lidar_a[:, 2:3]` | only v7 |
| Count of agent rays < cover_r | `a_nclose = (lidar_a < cover_r).sum()` | `a_count = (lidar_a < cover_r).sum()` | ✓ both |
| Count of agent rays < 0.5·cover_r | — | `a_count_half = (lidar_a < 0.5*cover_r).sum()` | only v7 |
| Mean of close-agent rays | `a_close_mean = ...` | — | only S3b |

#### Self-motion / state

| feature concept | S3b-local | v7 run-1 best | both? |
|---|---|---|---|
| Speed (norm of agent_vel) | `speed = torch.linalg.norm(agent_vel)` | `speed = torch.linalg.norm(vel)` | ✓ both |
| Velocity unit vector | — | `vel_u = vel / (speed+ε)` (2 features) | only v7 |
| Boundary distance | `boundary_dist = min(1−|px|, 1−|py|)` | — | only S3b |
| Velocity-target alignment | — | `t_align = (vel_u · t_dir).sum()` | only v7 |
| Velocity-agent alignment | — | `a_align = (vel_u · a_dir).sum()` | only v7 |

#### Role differentiation

| feature concept | S3b-local | v7 run-1 best | both? |
|---|---|---|---|
| Agent index one-hot (n_agents=4) | `one_hot = F.one_hot(agent_idx, 4)` | `role = F.one_hot(agent_idx, n_agents)` | ✓ both |

#### Cross-source / coordination decision features

| feature concept | S3b-local | v7 run-1 best | both? |
|---|---|---|---|
| Stay-here boolean: target near AND agent near | `settle_signal = (min_t_dist<cover_r) * (a_nclose≥1)` | `joint = near_t * near_a` (where `near_t/a = (min < cover_r)`) | ✓ both |
| Signed asymmetry: agent count − target count | `rendezvous_pressure = a_nclose/A − t_nclose/T` | `signed_gap = (a_min − t_min) * near_t` (different formulation) | both have it, different shape |
| "I'm alone, target is near" boolean | — | `commit = near_t * (1 − tanh(a_min))` | only v7 |
| "Both close, slow down" boolean | — | `stay_put = joint * (1 − tanh(speed))` | only v7 |
| "Approach" gradient | — | `approach = (1 − near_t) * t_align` | only v7 |
| "Pair coordination" signed | — | `pair_pressure = joint * (t_align − a_align)` | only v7 |
| Crowd ratio | — | `crowd_ratio = (t_count + ε) / (a_count + ε)` | only v7 |
| Boolean "target near" / "agent near" | implicit (via settle_signal multiplicand) | `near_t`, `near_a` (separate features) | both have logic; v7 exposes them explicitly |

### Summary of feature differences

**S3b-local has, v7 doesn't (3 features):**
- `t_close_mean` — mean distance of within-cover-r target rays (additional density signal)
- `t_dispersion` — std over target rays (uncertainty proxy)
- `boundary_dist` — distance from arena edge (helps avoid corners)

**v7 has, S3b-local doesn't (19 features):**
- `t_2, t_3, a_2, a_3` — 2nd and 3rd nearest target/agent distances (4 features)
- `t_count_half, a_count_half` — counts at half-cover threshold (2 features)
- `vel_u` — velocity unit vector (2 features)
- `t_align, a_align` — alignment between velocity and target/agent direction (2 features)
- `commit, stay_put, approach, pair_pressure` — additional decision-shaped features (4 features)
- `near_t, near_a` — target/agent near booleans separately (2 features, S3b-local has these implicitly)
- `crowd_ratio` — target-vs-agent count ratio (1 feature)
- `signed_gap` — alternative asymmetry encoding (1 feature)
- v7's `joint` ≈ S3b's `settle_signal` (same concept, both functions have it)

**Both have (16 features in common conceptually):**
- min target/agent distances (2)
- target/agent direction (cos/sin) (4)
- target/agent counts within cover (2)
- speed (1)
- one-hot role (4)
- "stay here" boolean (settle_signal vs joint)
- asymmetry signal (rendezvous_pressure vs signed_gap)
- target/agent close-detection logic

### Two key structural differences

**1. v7's `signed_gap` vs S3b-local's `rendezvous_pressure`** — these are DIFFERENT formulations of the same concept:

```python
# S3b-local — count-based asymmetry, ALWAYS active
rendezvous_pressure = (a_nclose / float(A)) - (t_nclose / float(T))

# v7 — distance-based asymmetry, gated by near_t
signed_gap = (a_min - t_min) * near_t   # zero unless target is within cover
```

S3b-local's signal is dense (always non-zero, smooth gradient over both rays' counts). v7's signal is gated (zero outside cover zone, sharp transition). For PPO at 1M frames, the dense signal may give earlier gradient than the gated one — v7's `signed_gap` gives ZERO learning signal until the agent stumbles within cover_r of a target, which is rare in early training (mean separation 1.0 vs cover_r 0.25).

**2. v7 produces more decision-shaped features but loses density signals** — `t_close_mean` and `t_dispersion` give the policy a sense of "how DENSE / how UNCERTAIN is the target neighborhood" — useful early-training signals. v7 replaces these with more decision-shaped booleans (`commit`, `stay_put`, `approach`, `pair_pressure`) that ALL fire only inside the cover zone. **At 1M frames in early training, the agent rarely reaches cover zone; v7's decision features therefore stay near zero, while S3b-local's density signals provide gradient even far from cover_r.**

This is likely the difference that explains v7 M1=0 vs S3b-local M1=0.05 at 1M eval.

---

## 3. Inner prompt comparison — what each inner LLM literally read

### Length & structure

| | S3b-local (`v2_fewshot_k2_local`) | v7 (`v2_fewshot_modular_v2_local` + slots) |
|---|---|---|
| Prompt template style | **Monolithic** (`system.txt` + `initial_user.txt`) | **Slot-decomposed** (9 sections concatenated) |
| Total bytes (rendered) | ~4.3 KB | ~5.5 KB (+ ~1.5 KB v7-meta-LLM-written guidance) |
| Hand-curated for THIS task | yes | no — meta-LLM-curated at runtime |

### Section-by-section content

#### Role / system identity

| | S3b-local | v7 |
|---|---|---|
| One-line role | "You are a reward engineer designing observation enhancement functions for MARL tasks. Your objective is to create an observation enhancement function that helps agents coordinate using ONLY their LOCAL sensor data." | "You design observation enhancement and/or reward functions for MARL. The output specification below tells you which function(s) to produce in this round." |
| **Critical task framing** | `**CRITICAL: This is a RENDEZVOUS task where k=2 agents must simultaneously occupy each target. Agents can only see via LiDAR — they do NOT know other agents' positions, target positions, or which targets are covered. Design features that help agents infer coordination state from sensor readings alone.**` | (Now in `task_context.txt` after my v6 base-prompt repair: "This is a multi-agent rendezvous task. ... A target is covered when exactly 2 agents are simultaneously within covering_range of it...") |

Both now have rendezvous + k=2 + LiDAR-only framing. **Equivalent at this layer.**

#### Available state schema

Both list ONLY local keys (verified in v6 base-prompt repair). **Equivalent at this layer.**

#### Operational hints / "what you CAN compute"

| | S3b-local | v7 |
|---|---|---|
| Style | **Bulleted human-curated list** | **Bulleted meta-LLM-curated list (V4 ops palette)** |
| Bullets included | *Direction to nearest target / Distance to nearest target / Number of nearby targets / Agent density nearby / Angular distribution / Role differentiation* (6 operational hints) | (V4) *product-like gate / ratio-like comparison / difference-like contrast / conjunction-like mask / gating-like decision* (5 operations); (V5) *3 pseudo-PyTorch examples like `joint_close = target_proximity * partner_proximity`*; (v7 additional) *include directional encoding (cos/sin) + role one-hot + motion + covering_range threshold alongside cross-source patterns* |

Both have an operational layer. v7's is more verbose and includes pseudo-code examples.

#### Concrete fewshot example (full `enhance_observation`)

| | S3b-local | v7 |
|---|---|---|
| Fewshot present | **Yes** — 13 lines, 4 features (`min_dist`, `dir_x`, `dir_y`, `n_close`, `one_hot`) | **No** — only 1-feature anchor (`nearest_target_dist.unsqueeze(-1)`) |
| Cover threshold in example | `0.25` (matches `covering_range`) | absent in example |

**This is the largest remaining structural gap.** S3b-local's prompt teaches by example with a working multi-feature starter; v7 teaches by description (operations palette + pseudo-PyTorch snippets) but doesn't show a full integrated example.

#### Output specification

Both end with a `def enhance_observation(scenario_state) -> Tensor` signature spec. **Equivalent.**

### Inner-loop iterative refinement

| | S3b-local | v7 |
|---|---|---|
| Iters per outer | 4 | 3 (v7 default) |
| Cands per iter | 3 | 3 |
| Feedback shows code | top_k=2 codes per iter | best+worst codes per iter |
| Cumulative registry | no | yes (since v5) |
| Sliding window memory | yes | yes |

v7's feedback is slightly richer than S3b-local's. Not a meaningful difference for this task.

---

## 4. The v7 outer / meta layer — has no S3b-local equivalent

S3b-local has no outer layer. v7 adds one for prompt evolution. In this run, v7's outer:

- Cold-start: enumerated 4 strategies, picked `paired_sector_split` (cod=9, train=8, score=8.5)
- Outer 0 → diagnosed `rl_too_hard` (pattern present, M1 still 0) → switched to `nearest_target_dual_attach`
- Outer 1 → diagnosed `translation_failure` (pattern absent in candidates) → would refine

The meta-layer worked correctly, but neither chosen strategy yielded M1>0 at 1M.

---

## 5. Bottom-line analysis

### What v7 successfully replicated from S3b-local

1. **All 5 feature-stack dimensions** (cross-source ops + directional encoding + role one-hot + motion + covering_range threshold) — feature_stack_score = 5/5 in 9/9 outer-0 candidates ✓
2. **Boolean conjunction of "target near AND agent near"** — `joint = near_t * near_a` ≈ S3b-local's `settle_signal` ✓
3. **Direction-to-target encoding** via `cos/sin(min_ray * 2π / n_rays)` ✓
4. **Role one-hot over n_agents** ✓
5. **`covering_range` as boolean mask threshold** ✓

### What v7 did *differently* and likely hurt RL outcome

1. **`signed_gap = (a_min - t_min) * near_t` GATED by `near_t` instead of S3b-local's UNGATED `rendezvous_pressure = a_nclose/A − t_nclose/T`** — v7's signal is zero outside cover zone; S3b-local's is always informative. At 1M frames, the policy rarely reaches cover zone, so v7's asymmetry signal stays near zero and provides no learning gradient.
2. **v7 dropped `t_close_mean` and `t_dispersion`** (density signals that work outside cover zone) and added more cover-zone-gated decision features (`commit`, `stay_put`, `approach`, `pair_pressure`). This shifts more signal density into the cover zone — exactly the region the agent rarely visits early in training.
3. **v7 has no `boundary_dist`** — S3b-local's boundary feature helps the policy avoid wall-hugging which is a common failure mode of early-training MAPPO in this arena.

### Hypothesis for the M1=0 vs M1=0.05 gap at 1M frames

**Both functions have the same NUMBER of features (35 vs 19) and the same structural primitives (cross-source + directional + role + motion + threshold). The likely difference is signal density in the early-training phase**:

- S3b-local's 19 features are mostly DENSE — `min_t_dist`, `t_dx/dy`, `t_close_mean`, `t_dispersion`, etc. all produce non-zero values everywhere, providing gradient even when far from any target
- v7's 35 features include 4-7 cover-zone-GATED decision booleans (commit / stay_put / approach / pair_pressure / signed_gap) that are zero outside cover zone. These are excellent in the cover regime but provide ZERO gradient when the agent is exploring early

A 1M-frame budget at 600 envs × 200 steps/episode = ~833 episodes per env × 600 envs = ~500K episodes total. Most of these episodes the agents are far from targets. **S3b-local's dense signals provide gradient throughout; v7's gated signals provide gradient only after the agent stumbles into the cover zone.**

### Concrete recommendation for v8

If we run another iteration:

1. **Restore `t_close_mean` and `t_dispersion`** (or equivalents) — dense density signals that work outside cover zone
2. **Replace `signed_gap = (a_min - t_min) * near_t` with the ungated count-asymmetry `rendezvous_pressure = a_nclose/A - t_nclose/T`** — provides gradient everywhere
3. **Add `boundary_dist`** — helps avoid wall-hugging
4. **Drop redundant decision features** like `pair_pressure` and `crowd_ratio` that double-encode information already in `joint` and `signed_gap`

Or simpler: **prefer fewer, denser features over more, gated features.** S3b-local's 19-feature set is empirically the bar; v7's 35-feature set has more decision-shaped layers but loses density.

### Anti-cheat audit on v7's generated code

Searched all 9 outer-0 candidates + all 9 outer-1 candidates for forbidden tokens (`hold_signal`, `approach_signal`, `crowd_signal`, `sparsity_signal`, `gap_to_partner`, `pair_formation_zone`, `nearest_unassigned`, `second agent needed`):

```
$ grep -lE "(hold_signal|approach_signal|crowd_signal|sparsity_signal|gap_to_partner|pair_formation_zone|nearest_unassigned|second agent needed)" $RUN_DIR/outer_*/inner/iter_*/candidate_*_obs.py
(no matches)
```

**Zero forbidden tokens.** v7's inner LLM authored the cross-source coordination logic from V4-V5-V7 prompt nudges, not from leaked answer features.

---

## 6. Files referenced

- v7 best candidate: `results/lero_v7/lero_v7_rendezvous_k2_2x3/run1_20260430_2111/outer_00/inner/iter_2/candidate_0_obs.py`
- S3b-local winner: `results/s3b_local_replicate/lero_s3b_local_s0/lero/runs/lero/20260429_0508/iter_1/candidate_2_obs.py`
- v7 bundle: `results/lero_v7/.../run1_20260430_2111/_bundle_init.json`
- v7 cold-start raw response: `results/lero_v7/.../run1_20260430_2111/_bundle_init_response.txt`
- v7 outer 0 diagnosis: `results/lero_v7/.../run1_20260430_2111/outer_00/_diagnosis.json`
- S3b-local prompt: `src/lero/prompts/v2_fewshot_k2_local/`
- v7 base prompt: `src/lero/prompts/v2_fewshot_modular_v2_local/`
- v7 meta system: `src/lero/v7/meta_strategist.py:_BUNDLE_SYSTEM`
- AST analyzer (verifies feature_stack_score): `src/lero/v6_prompt_lab/analyzer.py`
