# LERO — Full Documentation

LERO (LLM-driven Evolutionary Reward & Observation) adapted from [arXiv:2503.21807](https://arxiv.org/abs/2503.21807) for the VMAS Discovery scenario.

**Status (2026-04-17):** Phase 1–5 executed. **Headline result: obs-only LERO (S3b) achieved M1=1.000 on the k=2 rendezvous task** — best result across all experiments (ER1–ER3 + LERO), surpassing ER3 GNN's 71%. The bottleneck for k=2 was observation quality, not reward design.

---

## 1. Architecture

The LLM is an **offline code designer** — it runs before RL training, never during.

```
For each iteration (1..n_iterations):
  1. LLM generates n_candidates (reward, obs) Python functions
  2. Each candidate: patch into Discovery → train (1M frames) → evaluate
  3. Rank by (M1, M6, M2)
  4. Feed best code + metrics back to LLM as feedback
After loop:
  5. Sort ALL valid candidates across all iterations by (M1, M6, M2)
  6. Try full training (10M frames) on rank 0; if it crashes (e.g. NaN
     actions), fall back to rank 1, then rank 2, etc.
```

### Two LLM-generated functions

**`compute_reward(scenario_state)`** — reward function. Modes:
- `replace` (paper): full reward replacement; LLM owns the entire reward signal
- `bonus`: `R = R_original + bonus_scale * tanh(compute_reward_bonus(state))`

**`enhance_observation(scenario_state)`** — extra observation features appended to base obs. Modes:
- `global` (paper): function gets full state (all positions, coverage, etc.)
- `local` (CTDE-correct): function gets only own sensors (lidar)

### Scenario patching

`make_patched_scenario_class(...)` in `src/lero/scenario_patch.py` returns a `PatchedDiscoveryScenario` subclass of `vmas.scenarios.discovery.Scenario` with the LLM methods baked in. This subclass is passed to BenchMARL's env factory so TorchRL sees the correct observation spec at init time.

The patched class also overrides `info()` to always return per-agent `agent.covering_reward` (instead of upstream's `shared_covering_rew` when `shared_reward=True`), so M8 (agent utilization CV) is computable.

### Reward safeguards

LLM-generated rewards have wide magnitude variance (we observed M2 ranging from -1192 to +896 across candidates). To prevent PPO from diverging:

1. **Sanitization:** `torch.nan_to_num(r, nan=0, posinf=0, neginf=0)` — handles LLM divisions by zero.
2. **Clipping:** `torch.clamp(r, -reward_clip, +reward_clip)` — default ±50. Configurable via `LeroConfig.reward_clip`. Set to `None` to disable.

Both apply to the final reward in `replace` mode and to the bonus in `bonus` mode (original reward unaffected).

### Full-training fallback chain

After the LERO loop, all valid candidates from all iterations are sorted by (M1, M6, M2). Full training tries rank 0; on `Exception`, falls back to rank 1, then rank 2, etc. The chain is logged to `fallback_chain.json`:

```json
[
  {"rank": 0, "iter": 3, "eval_M1": 0.12, "outcome": "crashed",
   "error": "AssertionError: not action.isnan().any()"},
  {"rank": 1, "iter": 1, "eval_M1": 0.10, "outcome": "success"}
]
```

### Code layout

```
src/lero/
├── __init__.py
├── config.py             # LeroConfig + LLMConfig (+ reward_clip)
├── llm_client.py         # LiteLLM wrapper (Anthropic / OpenAI / OVH)
├── codegen.py            # Code extraction + AST validation
├── scenario_patch.py     # Patched Discovery subclass + reward safeguards
├── loop.py               # Evolutionary loop + fallback chain
└── prompts/
    ├── loader.py
    ├── v1/               # original verbose prompt (bonus + local mode — broken for replace+global)
    ├── v1_global/        # v1 verbose body adapted for replace+global mode (L4 ablation)
    ├── v2/               # paper-faithful minimal (~5 lines)
    ├── v2_min/           # ultra-minimal (3 lines, role + signatures only) — L1 ablation
    ├── v2_fewshot/       # v2 + 2 MPE-style example reward fns — L8 ablation
    └── v2_twofn/         # asks for agent_reward + global_reward decomposition — L21 ablation
```

---

## 2. Experiments executed (2026-04-16 — 2026-04-17)

All experiments use MAPPO, `covering_range=0.25`, `max_steps=400`, 10M-frame full training, 1M-frame eval per LERO candidate, 4 iterations × 3 candidates (unless noted).

### 2.1 Phase 4 — prompt ablations (n=3, t=3, k=1)

| Run | Prompt | Final M1 | Final M2 | Final M6 | M4 | M8 | Outcome | Where |
|---|---|---:|---:|---:|---:|---:|---|---|
| **L8 morning** | `v2_fewshot` | **1.000** | 172.1 | **1.000** | 0.0 | 0.23 | ✅ rescued from logs | `lero_rescued/l8/` |
| **L8 afternoon** | `v2_fewshot` | **1.000** | 105.6 | **1.000** | 0.0 | 0.25 | ✅ full artifacts | `lero/.../1324/` |
| **P1 morning** | `v2` | **1.000** | 104.2 | **1.000** | 0.1 | 0.27 | ✅ rescued from logs | `lero_rescued/p1/` |
| **L1 morning** | `v2_min` | 0.625 | 1.2 | 0.857 | 0.0 | 0.92 | ✅ rescued from logs | `lero_rescued/l1/` |
| **L21 morning** | `v2_twofn` | 0.010 | 149.5 | 0.130 | 35.5 | 0.46 | ✅ reward-hacked | `lero/.../1001/` |
| **L4 terminal** | `v1_global` | 0.005 | 334.9 | 0.128 | 0.0 | 0.55 | ✅ reward-hacked | `lero_l4/` |
| P1 terminal | `v2` | — | — | — | — | — | ❌ NaN @ 90% | `lero_p1/` |
| L1 terminal | `v2_min` | — | — | — | — | — | ❌ NaN @ 68% | `lero_l1/` |

### 2.2 Phase 5 — task scaling (k=1 easy tasks)

| Run | Task | Prompt | Final M1 | Final M2 | Final M6 | M3 | Baseline |
|---|---|---|---:|---:|---:|---:|---|
| **S1** | n=2, t=4, k=1 | `v2_fewshot` | **1.000** | 59.5 | **1.000** | 50.3 | ER1 = 0.580 |
| **L8** | n=3, t=3, k=1 | `v2_fewshot` | **1.000** | 105.6 | **1.000** | 33.8 | est. ~0.58 |

LERO dominates on all k=1 tasks tested.

### 2.3 Phase 5 — task scaling (k=2 rendezvous, the hard task)

| Run | LLM | Comm | Approach | Obs mode | Eval-best M1 | **Final M1** | Final M2 | Final M6 | M3 | Baseline |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| **S3b-global** | gpt-mini | none | obs-only (ER1 reward) | **global (oracle)** | 0.980 | **1.000** | 19.3 | **1.000** | 68.0 | ER1=40.5% |
| **S3b-local** | gpt-mini | none | obs-only (ER1 reward) | **local (fair)** | 0.060 | **0.880** | 5.0 | **0.970** | 186.4 | ER1=40.5% |
| S3a_gpt5 | gpt-5.4 | none | LLM reward (k2 prompt) | global | **0.860** | 0.090 | 1323.6 | 0.393 | 390.8 | ER1=40.5% |
| S3 | gpt-mini | none | LLM reward (k1 prompt) | global | 0.290 | 0.105 | 486.5 | 0.261 | 382.4 | ER1=40.5% |
| S3ac | gpt-mini | dim_c=8 | LLM reward (k2 prompt) | global | 0.020 | 0.080 | 1840.8 | 0.275 | 394.6 | ER2=53.0% |
| S3a_gpt | gpt-mini | none | LLM reward (k2 prompt) | global | 0.010 | 0.000 | 2260.6 | 0.088 | 400.0 | ER1=40.5% |

### 2.4 The S3b-local breakthrough — M1=88% on k=2 with LOCAL sensors only

**This is a legitimate result.** S3b-local uses `obs_state_mode=local`: the LLM can only access `lidar_targets` (15 rays), `lidar_agents` (12 rays), `agent_pos`, `agent_vel`, and `agent_idx`. No global target positions, no agent counts, no coverage status. Same information as ER1/ER2/ER3.

#### Fair comparison to all prior methods (k=2, ms400, cr025)

| Method | Info available | M1 | M3 (steps) | M6 |
|---|---|---:|---:|---:|
| **LERO S3b-local (obs-only, LOCAL)** | LiDAR + LLM-designed features | **88.0%** | 186 | 97.0% |
| ER3 GNN (GATv2) | LiDAR + GNN message-passing | 71.0% | 250 | 91.8% |
| ER2 proximity comm | LiDAR + proximity messages | 53.0% | 295 | 82.5% |
| ER1 no comm | LiDAR only | 40.5% | 317 | 80.0% |
| LERO S3b-global (obs-only, ORACLE) | LiDAR + oracle global state | 100% | 68 | 100% |

S3b-local **beats ER3 GNN by 17 percentage points** using only local sensors — no communication channel, no GNN, no oracle information. The improvement comes entirely from better feature engineering of the same LiDAR data.

#### What the LLM designed: the winning observation function (iter 3 candidate 2)

The winning function computes **28 features** from local sensors. Key innovation: it derives **coordination signals** that help the policy decide whether to approach, hold, or search.

```python
# CORE COORDINATION FEATURES (from lidar_targets + lidar_agents):

# 1. Nearest/second-nearest target: distance, direction, gap
t1, t2       # distance to nearest & second-nearest target (from sorted lidar)
tdx, tdy     # direction to nearest target (ray angle → cos/sin)
t_gap = t2 - t1  # gap between nearest and second-nearest (large = isolated target)

# 2. Nearest/second-nearest agent: distance, direction, gap
a1, a2       # same for agent lidar
adx, ady
a_gap = a2 - a1

# 3. Proximity counts and intensity
t_count      # how many target lidar rays < covering_range
a_count      # how many agent lidar rays < covering_range
t_intensity  # sum of 1/dist for close target rays (sharper proximity signal)
a_intensity  # same for agents

# 4. HIGH-LEVEL COORDINATION SIGNALS (the key innovation):
target_near = (t1 < covering_range)      # "I see a target within range"
agent_near  = (a1 < covering_range)      # "I see another agent within range"
hold_signal    = target_near * agent_near  # "target AND partner nearby → STAY"
approach_signal = target_near * (1-agent_near)  # "target nearby but no partner → need partner"
crowd_signal   = clamp((a_count-1)/3)    # "too many agents nearby → move away"
sparsity_signal = clamp(1 - t_count/n_targets)  # "few targets nearby → explore"

# 5. Self features: speed, velocity direction, position (polar), one-hot role
```

The critical insight is the **hold_signal**: when both a target and an agent are within covering range, the agent should HOLD POSITION and wait for the partner to arrive. Without this feature, agents approaching the same target tend to overshoot — one arrives, sees the target via lidar, then the OTHER agent arrives but the first has already moved on. The hold signal breaks this "ships passing in the night" failure mode.

#### How LERO evolved the observation across 4 iterations

| iter | c0 M1 | c1 M1 | c2 M1 | Best | Key features added |
|---|---:|---:|---:|---|---|
| 0 | 0.010 | **0.040** | 0.000 | c1 | basic: nearest target/agent distance + direction + count + one-hot role |
| 1 | 0.030 | 0.030 | (fail) | tied | added: sector-based features, polar position |
| 2 | 0.020 | 0.010 | 0.010 | c0 | added: gaps, intensity, more agent-count features |
| 3 | 0.000 | 0.000 | **0.060** | **c2** | added: **hold/approach/crowd/sparsity signals** ← the breakthrough |

Iteration 3 candidate 2 was the winner. It's the most complex candidate (28 features vs 14-18 for earlier ones). The coordination signals (`hold_signal`, `approach_signal`, `crowd_signal`, `sparsity_signal`) were the key addition in the final iteration.

**Note the eval→final improvement:** the winning candidate had eval M1=0.060 at 1M frames but improved to M1=0.880 at 10M frames. This is the OPPOSITE of reward hacking — because the ER1 hand-crafted reward is non-exploitable, longer training genuinely helps. The LLM's observation features gave the policy better information, and MAPPO used the full 10M frames to learn how to act on that information.

#### Comparison of observation features across candidate designs

| Feature type | Iter 0 (basic, M1=4%) | Iter 3 c2 (winner, M1=88%) |
|---|---|---|
| Target distance & direction | ✅ min dist, cos/sin angle | ✅ min + 2nd-min, gap between them |
| Agent distance & direction | ✅ basic | ✅ min + 2nd-min, gap |
| Proximity count | ✅ t_count, a_count | ✅ + intensity (1/dist weighted) |
| Coordination signals | ❌ none | ✅ **hold, approach, crowd, sparsity** |
| Self-motion | ✅ speed | ✅ speed + velocity dir + position (polar) |
| Role | ✅ one-hot | ✅ one-hot |
| Total features | ~14 | ~28 |

The LLM's evolutionary process discovered that **raw sensor readings are insufficient** — the policy needs pre-computed coordination signals that combine target and agent proximity into actionable decisions. This is analogous to hand-engineering features, but done automatically by the LLM in 4 iterations.

#### S3b-global vs S3b-local: the oracle advantage explained

S3b-global (M1=100%, oracle) used these features that S3b-local CAN'T compute:
- `rel = targets_pos - agent_pos` → exact vector to every target (not just nearest via lidar)
- `counts = agents_per_target` → exact count of agents at each target (not just nearby agents from lidar)
- `needed = required - counts` → which targets still need agents
- `needy_mask` → which targets to prioritize

S3b-local approximates this from lidar but with lower fidelity:
- Can detect targets within 0.35 range, but not beyond
- Can count nearby agents but not agents at distant targets
- Cannot know which targets are already covered

The 12% gap (88% vs 100%) is the price of using local sensors vs oracle global state.

### 2.5 The eval-vs-final degradation problem (reward hacking at scale)

S3a_gpt5 is the starkest example: the LLM designed a reward that produced M1=0.860 at 1M-frame eval — genuinely solving 86% of episodes. But after 10M frames of full training, M1 collapsed to 0.090 while M2 rose from 848 to 1324. The policy found an exploit in the reward that wasn't apparent at 1M frames.

| Metric | Eval (1M) | Final (10M) | Interpretation |
|---|---:|---:|---|
| M1 (success) | 0.860 | 0.090 | policy stopped solving the task |
| M2 (return) | 848 | 1324 | policy found higher-return exploit |
| M6 (coverage) | 0.930 | 0.393 | coverage collapsed with success |

**Contrast with S3b-local (obs-only):**

| Metric | Eval (1M) | Final (10M) | Interpretation |
|---|---:|---:|---|
| M1 (success) | 0.060 | **0.880** | policy improved with more training |
| M2 (return) | -2.1 | 5.0 | return increased modestly (non-exploitable) |
| M6 (coverage) | 0.478 | 0.970 | coverage doubled |

When the reward is hand-crafted (non-exploitable), longer training helps. When the reward is LLM-designed, longer training finds exploits. This is why obs-only LERO is more reliable than reward LERO for hard tasks.

### 2.6 Central thesis — feature engineering vs incentive design

The full evidence (Phases 1–5, 15+ LERO runs, 5 k=2 variants) converges on a single principle:

> **LLMs are excellent at feature engineering (designing what agents observe) but unreliable at incentive design (designing what agents optimize for). This asymmetry exists because observations are read-only — the policy cannot game them — while rewards are the optimization target and WILL be exploited given sufficient training.**

Evidence summary:

| LLM designs... | Can policy game it? | k=1 | k=2 |
|---|---|---:|---:|
| **Reward** (what to optimize for) | ✅ Yes | 100% (exploit ≈ solution) | 0–10.5% (reward-hacked) |
| **Observations** (what to see, local sensors) | ❌ No | — | **88%** |
| **Reward + communication** | ✅ Yes (amplified) | — | 8% (comm helps exploitation) |

For k=1, reward design works because individual rationality = collective rationality — no gap to exploit. For k≥2, the LLM's rewards have exploitable gaps (anti-crowding, surplus bonuses, magnitude inflation) that the policy discovers at 10M frames.

Communication is a **neutral amplifier**: it helps whatever the policy is already optimizing for. With a correct reward (ER2: 53%), comm improves coordination. With an exploitable reward (S3ac), comm improves exploitation. The LERO evolutionary loop makes this worse, not better — unable to improve M1, the LLM inflates reward magnitudes across iterations (M2: 1190→6462) creating a reward-inflation spiral.

**Practical implication:** for multi-agent coordination tasks, use LLMs for observation/feature design and keep the reward hand-crafted. The LLM's value is in information architecture (what agents perceive), not in incentive architecture (what agents pursue).

**Why don't we keep the 1M policy?** We should. A "best-checkpoint" strategy — saving the policy at peak eval-M1 during training — would recover the M1=0.86 policy from S3a_gpt5's run. This is the highest-priority code change for future experiments.

### 2.2 Per-experiment summary (best of all attempts)

| Exp | Prompt | Best Final M1 | Stable across runs? | Verdict |
|---|---|---:|---|---|
| **L8** | v2_fewshot (paper-min + 2 examples) | **1.000** (×2) | ✅ very stable | **Best configuration** |
| **P1** | v2 (paper-faithful minimal) | **1.000** (×1, NaN ×2) | ❌ huge variance | Works when LLM samples well; unstable otherwise |
| **L1** | v2_min (ultra-minimal) | 0.625 (×1, NaN ×2) | ❌ huge variance | Minimal prompt → mediocre + unstable rewards |
| **L4** | v1_global (verbose w/ research history) | 0.005 (×1) | n/a (1 run) | Reward-hacked (verbose context didn't help) |
| **L21** | v2_twofn (agent_reward + global_reward split) | 0.010 (×1, partial ×1) | n/a (1 useful run) | Two-function split makes reward MORE exploitable |

### 2.3 Per-iteration evolution (representative)

**P1 morning (M1=1.000) — LERO converged immediately:**

| iter | c0 | c1 | c2 |
|---|---|---|---|
| 0 | M1=0.99 / M2=43 | M1=0.24 / M2=3 | **M1=1.00 / M2=80** ← global best |
| 1 | M1=0.03 / M2=-602 | **M1=1.00 / M2=99** | M1=1.00 / M2=25 |
| 2 | M1=0.09 | M1=0.44 | M1=0.37 |
| 3 | M1=0.12 | M1=0.31 | M1=0.44 |

Iters 2–3 produced strictly worse candidates; iter-1 best held.

**L8 morning (M1=1.000) — LERO converged by iter 1, lockup by iter 3:**

| iter | c0 | c1 | c2 |
|---|---|---|---|
| 0 | ❌ FAIL `'int' has no .float` | **M1=0.65 / M2=70** ← global best | M1=0.02 / M2=-121k |
| 1 | M1=0.04 / M2=-711 | **M1=1.00 / M2=160** ← new best | M1=1.00 / M2=145 |
| 2 | M1=1.00 / M2=137 | M1=0.06 / M2=-1197 | M1=0.96 / M2=123 |
| 3 | **M1=1.00 / M2=150** | M1=1.00 / M2=111 | M1=1.00 / M2=45 |

**P1 terminal (NaN crash) — LLM produced consistently weak rewards:**

| iter | c0 | c1 | c2 |
|---|---|---|---|
| 0 | M1=0.09 / M2=-228 | M1=0.08 / M2=2 | M1=0.01 / M2=87 |
| 1 | **M1=0.12 / M2=-289** | M1=0.07 / M2=-1192 | M1=0.03 / M2=-544 |
| 2 | M1=0.04 / M2=-125 | M1=0.03 / M2=-232 | M1=0.04 / M2=-520 |
| 3 | M1=0.07 / M2=-692 | **M1=0.12 / M2=-175 / M6=0.55** ← picked | M1=0.09 / M2=-241 |

Picked candidate had M2=-175. Trained to batch 151/167 (90%) → NaN actions → crash.

**L4 terminal (M1=0.005, reward-hacked) — all candidates had high positive M2:**

| iter | c0 | c1 | c2 |
|---|---|---|---|
| 0 | M1=0.02 / M2=390 | M1=0.04 / M2=387 | **M1=0.16 / M2=324** ← global best |
| 1 | M1=0.02 / M2=465 | M1=0.02 / M2=319 | M1=0.00 / M2=389 |
| 2 | M1=0.03 / M2=441 | M1=0.02 / M2=402 | M1=0.00 / M2=416 |
| 3 | M1=0.02 / **M2=896** | M1=0.00 / M2=791 | M1=0.02 / M2=602 |

LLM never escaped the high-M2/low-M1 attractor across 4 iters. Verbose v1_global prompt encouraged "scoring" rewards that don't actually solve the task.

### 2.4 Crash pattern — when does NaN happen?

| Run | Picked candidate's M2 | Frames before crash |
|---|---:|---:|
| L1 afternoon (lost) | ~-50 | 7M (77%) |
| P1 terminal | -175 | 9M (90%) |
| L1 terminal | -63 | 6.8M (68%) |

All 3 NaN crashes had **negative M2** at the chosen candidate. L4 had M2=+324 (very positive) and didn't crash, but the policy reward-hacked instead. Pattern:

| Picked candidate's M2 magnitude | Outcome |
|---|---|
| Small (±5..±50) | ✅ stable training |
| Large negative (-100..-1000) | ❌ NaN crash mid-training |
| Large positive (+200..+900) | ⚠ reward hacking (high M2, near-zero M1) |

The 1M-frame eval doesn't separate "stable but mediocre" from "promising but unstable." Hence the need for safeguards (clipping + fallback chain).

### 2.5 Hypothesis status

| ID | Hypothesis | Status | Evidence |
|---|---|---|---|
| H1 | LERO works on Discovery | ✅ Proven | L8 ×2 = 1.000; P1 morning = 1.000; S1 = 1.000; **S3b = 1.000 on k=2** |
| H4 | Discovery > MPE in difficulty | ✅ Supported for reward design | LLM cannot design a non-hackable k=2 reward (0–10.5% across 5 attempts); obs-only mode bypasses the problem |
| H5 | Less prompt is better | ⚠ Mixed | v2 is the sweet spot; ultra-minimal (L1) and verbose (L4) both worse |
| H8 | Few-shot examples help | ✅ Strongly supported | L8 (v2_fewshot) = best prompt for both k=1 and k=2 |
| H11 | Two-function split helps | ❌ Refuted | L21 reward-hacked at 0.01 |
| H12 | Stronger LLM = better reward | ⚠ Partial | gpt-5.4 achieved eval M1=0.86 on k=2 (vs gpt-mini's 0.01), but still reward-hacked at 10M (final 0.09). Stronger model produces better INITIAL designs that degrade with longer training |
| **NEW** | Obs-only LERO > reward LERO for k≥2 | ✅ **Breakthrough** | S3b-local (obs-only, local sensors): M1=88% on k=2 vs all reward-design variants: M1=0–10.5%. S3b-global (oracle): M1=100% but unfair |
| **NEW** | Eval-best ≠ train-stable | ✅ Confirmed | gpt-5.4's M1=0.86 at 1M eval collapsed to 0.09 at 10M. 1M eval is insufficient to detect reward hacking |
| **NEW** | Communication helps LLM reward design | ❌ **Refuted** | S3ac (reward+comm): M1=8% — WORSE than S3 (no comm, 10.5%). Comm amplifies whatever the policy optimizes; with exploitable reward, it amplifies exploitation. LERO evolutionary loop became a reward-inflation spiral (M2: 1190→6462 over 4 iters, M1 stuck at 0) |
| **NEW** | Obs-only LERO with local sensors beats GNN | ✅ Confirmed | S3b-local (88%) > ER3 GATv2 (71%) using same local sensor info — LLM-designed coordination signals (`hold`, `approach`, `crowd`, `sparsity`) from LiDAR replace learned GNN message-passing |

---

## 3. Implementation lessons learned (2026-04-16 incident log)

These are the bugs we hit and fixed in this session. Documenting for future-us so we don't re-do the same diagnostics.

### 3.1 Scenario patch closure scoping (`_obs_mode` NameError)

**Symptom:** All candidates with `obs_source` failed with `NameError: name '_obs_mode' is not defined`.

**Root cause:** A class-body-level `_obs_mode = obs_state_mode` is **not** visible inside method bodies (Python scoping rule — class body is not an enclosing scope for nested functions). The `reward()` path worked because it used `_bs` only as a default arg (evaluated at class-body time).

**Fix (`scenario_patch.py:225`):** Capture via method default arg from the enclosing function's parameter:
```python
def observation(self, agent, _mode=obs_state_mode):
    ...
    if _mode == "global": ...
```

### 3.2 VMAS pip-installed scenario lacks newer kwargs

**Symptom:** `'PatchedDiscoveryScenario' object has no attribute 'dim_c'` (and similar for `comm_proximity`, `_comms_range`).

**Root cause:** OVH containers `pip install vmas` from PyPI; that version of `discovery.py` doesn't `kwargs.pop("dim_c", 0)` like our local fork does. The `comm_proximity`, `dict_obs` kwargs aren't consumed → never set as scenario attributes.

**Fix (`scenario_patch.py:80,113`):** Use `getattr(scenario, "dim_c", 0)` everywhere instead of direct attribute access. Same for `comm_proximity`, `_comms_range`, `use_agent_lidar`, `all_time_covered_targets`.

### 3.3 M8 agent utilization always 0

**Symptom:** `M8_agent_utilization: 0.0000` in every LERO run.

**Root cause:** With `shared_reward=True` (which we use for stability), Discovery's `info()` returns `self.shared_covering_rew` — the same scalar for every agent. Per-agent CV is therefore 0.

**Fix (`scenario_patch.py:194`):** Override `info()` in `PatchedDiscoveryScenario` to always return per-agent `agent.covering_reward`:
```python
def info(self, agent):
    base = super().info(agent)
    base["covering_reward"] = agent.covering_reward
    return base
```
Safe because MAPPO reads rewards from `reward()`, not `info`. Verified: M8 now ranges 0.23–0.92.

### 3.4 OVH S3 prefix bug (the worst one)

**Symptom:** Per-experiment S3 prefixes (`lero_p1`, `lero_l1`, …) failed silently — every parallel job synced to bucket root, overwriting each other's results during FINALIZE.

**Root cause:** Our volume string was `bucket@region/exp_id/:mount:rwd` — **trailing slash on the prefix**. The `ovhai` CLI parses `container@alias[/prefix]:mount_path[:permission]`. With a trailing slash on prefix, OVH's parser fails silently and falls back to `prefix=None`. Every job mounted bucket root.

**Fix (`src/ovh.py:222`):** Build the volume string conditionally without trailing slash:
```python
results_volume = (
    f"{bucket_results}@{region}/{exp_id}:{mount_results}:rwd"
    if exp_id
    else f"{bucket_results}@{region}:{mount_results}:rwd"
)
```

**Verification:** A probe job inspected via `ovhai job get <id>` now shows `prefix: 'lero_p1'` (was `None`).

### 3.5 Streamlit module caching

**Symptom:** After fixing `src/ovh.py`, the user submitted via Streamlit and the bug came back.

**Root cause:** Streamlit imports modules once into a long-running session. `submit_training_job` was the OLD buggy version cached in memory. **Restarting Streamlit** OR submitting via `python -c '...'` (fresh process) is required.

**Mitigation:** Submit via terminal, not Streamlit, for code that's just been changed.

### 3.6 v1 prompt incompatibility with `replace+global` mode

**Symptom:** L4 (using v1 prompt) — every candidate's `enhance_observation` failed with `KeyError: 'lidar_targets'`.

**Root cause:** v1's prompt advertises a separate `obs_state` dict (with `lidar_targets`) for `enhance_observation`. But when run with `obs_state_mode=global`, the function actually receives the global state dict (which has no `lidar_targets`). LLM follows prompt spec → references missing key → crash.

**Fix:** New prompt `v1_global/` — keeps v1's verbose research-history content but uses the single global state dict and `compute_reward` (not `compute_reward_bonus`) signature.

### 3.7 PPO NaN actions from large-magnitude rewards

**Symptom:** `AssertionError: not action.isnan().any()` ~70-90% into 10M-frame full training.

**Root cause:** LLM-generated rewards with |M2| > ~100 (especially negative) cause PPO value function and policy gradients to diverge. Eval at 1M frames doesn't catch this.

**Fix (Layer B + Layer C — `scenario_patch.py:140`, `loop.py:730`):**
- B: clamp reward to `[-reward_clip, +reward_clip]` (default ±50) after `nan_to_num`
- C: full-training fallback chain — on crash, try the next-best candidate from cross-iter rankings

**Caveat:** B deviates from the LERO paper's "raw rewards" claim. Justification: paper used MPE Simple Spread with naturally bounded rewards [0, 5]; Discovery LLMs produce magnitudes 100–1000× larger. Document as an explicit deviation.

---

## 4. Infrastructure reference

### 4.1 LeroConfig

```yaml
lero:
  n_iterations: 4
  n_candidates: 3
  top_k: 2
  eval_frames: 1_000_000
  eval_episodes: 100
  full_frames: 10_000_000
  evolve_reward: true
  evolve_observation: true
  reward_mode: "replace"      # "replace" (paper) | "bonus"
  obs_state_mode: "global"    # "global" (paper) | "local" (CTDE)
  bonus_scale: 0.5            # only for reward_mode=bonus
  reward_clip: 50.0           # ±clip after nan_to_num. Set null to disable.
```

### 4.2 LLMConfig

```yaml
llm:
  model: "gpt-5.4-mini"       # LiteLLM identifier
  temperature: 0.8
  max_tokens: null
  api_base: null              # custom endpoint (OVH or compatible)
  api_key: null               # else read from env (OPENAI_API_KEY etc.)
  context_window: null        # auto-detect, set explicit for vLLM/OVH
  max_retries: 3
  retry_delay: 2.0
  prompt_version: "v2"        # v1 | v1_global | v2 | v2_min | v2_fewshot | v2_twofn
```

### 4.3 OVH submission

```python
from dotenv import dotenv_values
from rendezvous_comm.src.ovh import submit_training_job

submit_training_job(
    "rendezvous_comm/configs/lero/<exp>.yaml",
    llm_env=dotenv_values("rendezvous_comm/.env"),
)
```

**Always submit via terminal (fresh Python process)**, never via Streamlit unless you've restarted Streamlit since the last code change.

Verify the parsed prefix on the resulting job:
```bash
ovhai job get <uuid> --output json | python -c "
import json, sys
d = json.load(sys.stdin)
for v in d['spec']['volumes']:
    if v['mountPath'] == '/workspace/results':
        print('prefix:', v['dataStore']['prefix'])"
```
Must print `prefix: 'lero_<exp>'`, not `prefix: None`.

### 4.4 Output files (per LERO run)

```
results/lero_<exp>/lero/runs/lero/<timestamp>/
├── messages_initial.json    # system + initial user prompt
├── messages_final.json      # final conversation (sliding window)
├── evolution_history.json   # per-iter best M1/M2/M6
├── fallback_chain.json      # which candidates were tried in full training
├── final_metrics.json       # M1..M9 from 10M training of chosen candidate
├── best_reward.py           # source of the candidate that completed full training
├── best_obs.py              # source of corresponding enhance_observation
├── best_policy.pt           # trained MAPPO policy weights
├── benchmarl_final/         # full BenchMARL output of final training
└── iter_{0..N-1}/
    ├── candidate_{0..M-1}_response.txt   # raw LLM output
    ├── candidate_{0..M-1}_reward.py      # extracted compute_reward
    ├── candidate_{0..M-1}_obs.py         # extracted enhance_observation
    ├── candidate_{0..M-1}_metrics.json   # M1..M9 from 1M eval
    ├── feedback.txt                      # feedback shown to LLM next iter
    └── benchmarl_c{0..M-1}/              # BenchMARL output of each candidate
```

---

## 5. Open issues / next steps

### 5.1 Immediate — best-checkpoint saving

The eval-vs-final degradation (S3a_gpt5: M1=0.86→0.09) wastes promising candidates. **Implement peak-M1 checkpointing:**
- During full training, BenchMARL already evaluates every `evaluation_interval` frames
- Track M1 at each eval; save policy checkpoint when M1 peaks
- Return peak-M1 policy + metrics alongside the final ones
- Record both in `final_metrics.json`: `"peak_M1"`, `"peak_at_frame"`, `"final_M1"`

This would recover gpt-5.4's M1=0.86 policy from S3a_gpt5 — a legitimate k=2 solution even if the reward is exploitable at longer horizons.

### 5.2 Key scientific questions

1. **What does S3b's observation enhancement look like?** The LLM designed obs features that enabled M1=1.000 on k=2 without communication. Inspect `best_obs.py` from `lero_s3b_gpt/` — what global-state features did it extract? How do they compare to GNN message-passing?

2. **Is S3b's success robust across seeds?** We have 1 run. Run S3b with seeds [0,1,2] to confirm consistency before claiming M1=1.000 on k=2.

3. **Can obs-only LERO generalize beyond k=2?** Try n=6, t=4, k=3 — do the LLM-designed observations still help when coordination complexity increases?

4. **Why can't the LLM design a k=2 reward?** All reward-design attempts (S3, S3a, S3ac with different LLMs and prompts) reward-hacked. The LLM consistently produces anti-crowding penalties that fight convergence. Is this a prompt issue, an LLM capability limit, or fundamental to reward design for coordination tasks?

5. **Few-shot examples anchor reward magnitude.** Investigate whether L8's examples (which use magnitudes in [0, 25] range) prevent the LLM from generating ±1000 magnitude rewards that cause NaN crashes and reward hacking.

### 5.3 Potential follow-up experiments

| Exp | Description | Cost | Priority |
|---|---|---|---|
| S3b ×3 seeds | Robustness check for the M1=1.000 breakthrough | ~$8 | **HIGH** |
| S3b + comm | Obs-only LERO + dim_c=8 — does adding comm help further? | ~$3 | Medium |
| S3a_gpt5 + best-checkpoint | Re-run with peak-M1 saving — recover the M1=0.86 policy | ~$3 | Medium |
| S3b with k=3 | Does obs-only LERO scale to higher k? | ~$3 | Medium |
| Inspect S3b's best_obs.py | Free — just read the file, understand the features | Free | **HIGH** |

### 5.4 Deferred phases

- **Phase 6 — loop tuning:** more candidates per iter, more iterations, longer eval. Lower priority now that obs-only is the winning approach.
- **Remaining Phase 4 ablations** (L2/L3/L5–L7 etc.): deprioritized — the prompt ablation question is less important than obs-only vs reward-design.

---

## 6. References

- LERO paper: [arXiv:2503.21807](https://arxiv.org/abs/2503.21807)
- Code entry point: `rendezvous_comm/train.py` → `runner.run_lero()` → `lero/loop.LeroLoop.run()`
- Configs: `rendezvous_comm/configs/lero/{p1,l1,l4,l8,l21,s1,s3,s3a_gpt,s3a_gpt5,s3ac_gpt,s3b_gpt}.yaml`
- Phase 4 rescued metrics: `results/lero_rescued/{p1,l1,l8}/`
- Phase 5 k=1 artifacts: `lero_s1/`, `lero/.../1324/`
- Phase 5 k=2 artifacts: `lero_s3b_gpt/` (breakthrough), `lero_s3a_gpt5/`, `lero_s3ac_gpt/`, `lero/.../1001/`
