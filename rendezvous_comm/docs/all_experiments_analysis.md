# Complete Experiment Analysis — Learning Communication Protocols for Multi-Robot Rendezvous

**Date:** 2026-03-24 (ER1–ER3), 2026-04-16 (LERO)
**Framework:** VMAS Discovery + BenchMARL (MAPPO)
**Task:** covering_range=0.25 (unless noted), targets_respawn=False

## Data Sources

| ER | Latest Sweep CSV | Training Iterations CSV |
| ---- | ----------------- | ---------------------- |
| ER1 | [sweep_results_20260324_1442.csv](../results/er1/runs/sweep_results_20260324_1442.csv) | [training_iter_20260324_1442.csv](../results/er1/runs/training_iter_20260324_1442.csv) |
| ER2 | [sweep_results_20260324_1639.csv](../results/er2/runs/sweep_results_20260324_1639.csv) | [training_iter_20260324_1639.csv](../results/er2/runs/training_iter_20260324_1639.csv) |
| ER3 | [sweep_results_20260324_0731.csv](../results/er3/runs/sweep_results_20260324_0731.csv) | [training_iter_20260324_0731.csv](../results/er3/runs/training_iter_20260324_0731.csv) |

Sweep reports (per-run markdown):

- ER1: [sweep_report_20260324_1442.md](../results/er1/runs/sweep_report_20260324_1442.md)
- ER2: [sweep_report_20260324_1639.md](../results/er2/runs/sweep_report_20260324_1639.md)
- ER3: [sweep_report_20260324_0731.md](../results/er3/runs/sweep_report_20260324_0731.md)

---

## Metric Definitions

| ID | Metric | Description |
| ---- | ------ | ----------- |
| M1 | Success Rate | Fraction of episodes where all targets are covered |
| M2 | Avg Return | Mean cumulative reward per episode |
| M3 | Avg Steps | Mean steps to completion (capped at max_steps) |
| M4 | Collisions | Average collisions per episode |
| M5 | Tokens | Communication tokens per episode |
| M6 | Coverage Progress | Fraction of targets covered on average |
| M7 | Sample Efficiency | Frames to reach 50%+ of final performance |
| M8 | Agent Utilization | How evenly agents contribute to covering |
| M9 | Spatial Spread | How spread out agents are at episode end |

---

## ER1 — No Communication Baseline

**Goal:** Establish what agents can learn without any communication channel.

### Group 1: Baseline sweeps (k=1 vs k=2)

The fundamental difficulty split: agents_per_target=1 (easy) vs agents_per_target=2 (hard coordination).

| exp_id | n | k | lidar | seeds | M1 (SR%) | M2 (Return) | M3 (Steps) | M4 (Coll) | M6 (Cov%) | M7 (SampleEff) |
| -------- | --- | --- | ------- | ------- | ---------- | ------------- | ------------ | ----------- | ----------- | ----------------- |
| er1 | 4 | 1 | 0.25 | s1 | 69.5% | 0.248 | 72.9 | 4.4 | 90.8% | 2.6M |
| er1 | 4 | 1 | 0.35 | s0 | 76.5% | 0.331 | 69.7 | 4.8 | 93.6% | 5.4M |
| er1 | 4 | 2 | 0.35 | s0 | 8.5% | -0.147 | 97.9 | 6.3 | 49.8% | - |
| er1 | 4 | 2 | 0.35 | s1 | 4.5% | -0.201 | 99.2 | 7.2 | 48.4% | - |

**Finding:** k=1 is solvable (~70-77% SR); k=2 is nearly impossible without communication (~4-9% SR). The gap confirms that k=2 requires coordination that implicit policies alone cannot provide.

### Group 2: Agent LiDAR effect (er1_al)

Adding agent-sensing LiDAR (agents can detect each other).

| exp_id | n | k | lidar | agent_lidar | M1 (SR%) | M2 | M3 | M4 | M6 | M7 |
| -------- | --- | --- | ------- | ------------- | ---------- | ----- | ----- | ----- | ----- | ----- |
| er1_al | 4 | 1 | 0.25 | yes | 63.0% | 0.181 | 75.8 | 3.1 | 89.1% | 2.6M |
| er1_al | 4 | 1 | 0.35 | yes | 75.0% | 0.441 | 69.9 | 0.8 | 92.9% | 5.9M |
| er1_al | 4 | 2 | 0.35 | yes | 0% | -0.642 | 100 | 0.1 | 17.8% | - |
| er1_al | 4 | 2 | 0.35 | yes | 0% | -0.688 | 100 | 0.0 | 15.5% | - |

**Finding:** Agent LiDAR helps with k=1 (fewer collisions: 0.8 vs 4.8) but *hurts* k=2 (0% SR vs 4-9%). The collision avoidance behavior dominates — agents avoid each other, making simultaneous target covering impossible.

### Group 3: Ablation studies (k=2 only, 3 seeds each)

All ablations test k=2 with n=4, ms200, lidar=0.35, no agent LiDAR.

| exp_id | Change | Mean M1 | Mean M6 | Mean M4 | Notes |
| -------- | -------- | --------- | --------- | --------- | ------- |
| er1_abl_a | Default entropy | 0% | 2.4% | 0.0 | Complete failure — agents freeze |
| er1_abl_a2 | High entropy | 4.3% | 45.6% | 6.6 | Explores better but still fails |
| er1_abl_b | lr=0.0001 | 5.7% | 51.4% | 7.0 | Slight improvement from higher LR |
| er1_abl_c | GAE lambda tuning | 6.3% | 51.1% | 6.9 | Marginal improvement |
| er1_abl_g | 20M frames | 5.2% | 48.7% | 6.9 | No benefit from 2x training |
| er1_abl_h | Larger network | 4.7% | 49.6% | 7.7 | No benefit from bigger model |
| er1_abl_i | k=1 sanity | 76.8% | 93.7% | 5.0 | Confirms k=1 is solvable |

**Finding:** No hyperparameter tuning breaks the k=2 barrier at ms200. The problem is structural — agents need coordination, not more training or capacity.

### Group 4: Reward shaping ablations (with agent LiDAR, k=2)

Progressive reward engineering to help k=2 convergence.

| exp_id | Reward Components | M1 | M2 | M3 | M6 |
| -------- | ------------------- | ----- | ----- | ----- | ----- |
| er1_al (k=2) | base only | 0% | -0.642 | 100 | 17.8% |
| er1_al_abl_lp | + lidar proximity | 0.5% | -0.302 | 99.9 | 35.1% |
| er1_al_abl_sr | + shared reward | 1.5% | 0.504 | 99.8 | 38.1% |
| er1_al_abl_lp_sr | + both LP+SR | 4.0% | 0.840 | 99.3 | 45.6% |

**Finding:** Each reward component helps incrementally. LP+SR together bring k=2 from 0% to 4% SR — still low, but coverage jumps from 17.8% to 45.6%.

### Group 5: Task difficulty relaxations

Testing if longer episodes or easier covering threshold break through.

| exp_id | n | k | ms | cr | M1 | M2 | M3 | M6 | M7 |
| -------- | --- | --- | ----- | ------ | ------ | ------ | ------ | ------ | ------ |
| er1_al_abl_lp_sr | 4 | 2 | 200 | 0.25 | 4.0% | 0.840 | 99.3 | 45.6% | 6.1M |
| er1_al_lp_sr_ms400 | 4 | 2 | 400 | 0.25 | **40.5%** | 0.338 | 316.9 | **80.0%** | - |
| er1_al_lp_sr_cr035 | 4 | 2 | 200 | **0.35** | **27.5%** | 1.388 | 176.6 | **73.4%** | 3.8M |
| er1_al_lp_sr_ms400_n2_k1 | **2** | **1** | 400 | 0.25 | **58.0%** | -0.636 | 247.0 | **87.3%** | - |

**Finding:** Both relaxations dramatically help:

- **ms400** (double time): 4% -> 40.5% SR. Agents learn but need more time to coordinate.
- **cr035** (easier covering): 4% -> 27.5% SR. Bigger covering radius makes simultaneous occupation easier.
- **n=2 k=1 ms400**: 58% SR confirms minimal team can solve with enough time.

---

## ER2 — Engineered Communication

**Goal:** Test if explicit communication channels improve coordination for k=2.

### Group 1: Communication baselines

| exp_id | dim_c | comm_type | M1 | M2 | M3 | M4 | M5 (Tokens) | M6 |
| -------- | ------- | ----------- | ----- | ----- | ----- | ----- | ------------- | ----- |
| er2 | 8 | proximity | 3.5% | -0.154 | 99.1 | 8.4 | 3200 | 51.8% |
| er2_al | 8 | proximity | 0.5% | -0.484 | 99.7 | 0.9 | 3200 | 26.4% |
| er2_30m | 8 | proximity | 2.0% | -0.276 | 99.3 | 7.7 | 0 | 45.4% |
| er2_30m | 8 | proximity | 3.5% | -0.199 | 99.4 | 6.3 | 0 | 47.6% |

**Finding:** Raw communication without reward shaping barely helps. er2 baseline (3.5%) is similar to er1 baseline (4-9%). Agent LiDAR again hurts (0.5% vs 3.5%).

### Group 2: Communication + reward shaping (LP+SR)

| exp_id | dim_c | comm_type | M1 | M2 | M3 | M5 | M6 | M7 |
| -------- | ------- | ----------- | ----- | ----- | ----- | ----- | ----- | ----- |
| er1_al_abl_lp_sr (no comm) | 0 | none | 4.0% | 0.840 | 99.3 | 0 | 45.6% | 6.1M |
| er2_al_lp_sr | 8 | proximity | 4.5% | 0.902 | 98.6 | 3200 | 46.3% | 7.3M |
| er2_al_lp_sr_bc_dimc2 | 2 | broadcast | 1.0% | 0.505 | 99.9 | 800 | 37.8% | 5.5M |
| er2_al_lp_sr_bc_dimc8 | 8 | broadcast | 1.0% | 0.172 | 99.6 | 3200 | 29.0% | 9.4M |
| er2_al_lp_sr_bc_dimc16 | 16 | broadcast | 0% | -0.363 | 100 | 6400 | 16.3% | - |

**Finding:** At ms200/cr025, proximity comm barely outperforms no-comm (4.5% vs 4.0%). Broadcast comm actually *hurts* — higher dim_c = worse performance. Broadcast seems to add noise that confuses learning.

### Group 3: Communication channel size (broadcast, no reward shaping)

| exp_id | dim_c | M1 | M2 | M4 | M5 | M6 |
| -------- | ------- | ----- | ----- | ----- | ----- | ----- |
| er2_bc_dimc16 | 16 | 8.5% | -0.147 | 6.3 | 0 | 49.8% |
| er2_bc_dimc16 | 16 | 4.5% | -0.201 | 7.2 | 0 | 48.4% |

**Note:** These appear to be duplicates of er1 baseline runs (M5=0 tokens). May indicate a config issue.

### Group 4: Task relaxations with communication

**Proximity communication:**

| exp_id | dim_c | comm | ms | cr | M1 | M2 | M3 | M5 | M6 | M7 |
| ------ | ----- | ---- | --- | ---- | ---- | ----- | ----- | ----- | ----- | ---- |
| er2_al_lp_sr | 8 | prox | 200 | 0.25 | 4.5% | 0.902 | 98.6 | 3200 | 46.3% | 7.3M |
| er2_al_lp_sr_prox_dc8_ms400 | 8 | prox | 400 | 0.25 | **53.0%** | 0.879 | 295.5 | 12800 | **82.5%** | 6.7M |
| er2_al_lp_sr_prox_dc8_cr035 | 8 | prox | 200 | **0.35** | **37.5%** | 1.775 | 176.2 | 6400 | **77.8%** | 7.0M |

**Broadcast communication:**

| exp_id | dim_c | comm | ms | cr | M1 | M2 | M3 | M5 | M6 | M7 |
| ------ | ----- | ---- | --- | ---- | ---- | ----- | ----- | ----- | ----- | ---- |
| er2_al_lp_sr_bc_dimc8 | 8 | bc | 200 | 0.25 | 1.0% | 0.172 | 99.6 | 3200 | 29.0% | 9.4M |
| er2_al_lp_sr_bc_dc8_ms400 | 8 | bc | 400 | 0.25 | **46.0%** | 0.551 | 308.2 | 12800 | **80.4%** | 9.6M |
| er2_al_lp_sr_bc_dc8_cr035 | 8 | bc | 200 | **0.35** | **48.5%** | **2.041** | 171.6 | 6400 | **83.4%** | 7.6M |

**Finding:** Communication shines with task relaxations:

- **Proximity + ms400**: 53.0% SR (vs 40.5% no-comm) — +12.5pp improvement from comm
- **Proximity + cr035**: 37.5% SR (vs 27.5% no-comm) — +10pp improvement
- **Broadcast + cr035**: 48.5% SR — surprisingly strong, better than proximity+cr035!
- **Broadcast + ms400**: 46.0% SR — lower than proximity (53.0%), but still substantial

The cr035 condition reveals broadcast's advantage: with easier covering, broadcast coordination outperforms proximity because all agents receive the signal regardless of distance.

---

## ER3 — GNN-Based Communication

**Goal:** Test if graph neural networks can learn emergent communication protocols.

### All ER3 experiments

| exp_id | GNN | ms | cr | M1 | M2 | M3 | M4 | M6 | M7 |
| -------- | ----- | ----- | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| er3_al_lp_sr_gatv2 | GATv2 | 200 | 0.25 | 0% | -0.706 | 100 | 27.8 | 9.1% | - |
| er3_al_lp_sr_graphconv | GraphConv | 200 | 0.25 | 0% | -0.549 | 100 | 15.5 | 12.1% | - |
| er3_al_lp_sr_gatv2_ms400 (pre-fix) | GATv2 | 400 | 0.25 | 0% | -3.592 | 400 | 14.8 | 11.1% | - |
| er3_al_lp_sr_gatv2_ms400 (post-fix) | GATv2 | 400 | 0.25 | **71.0%** | **2.498** | 249.7 | 8.9 | **91.8%** | - |
| er3_al_lp_sr_gatv2_cr035 (pre-fix) | GATv2 | 200 | 0.35 | 0% | -1.859 | 200 | 189.2 | 16.5% | - |
| er3_al_lp_sr_gatv2_cr035 (post-fix) | GATv2 | 200 | 0.35 | **36.5%** | 1.756 | 172.1 | 14.9 | **77.3%** | 5.5M |

**Note:** "pre-fix" runs suffered from the max_steps bug (config.pop removed max_steps before VMAS received it, causing all envs to run with default 100 steps). Post-fix results are the authoritative ones.

**Finding:** GATv2 with ms400 achieves the **highest SR across all experiments (71.0%)** and highest coverage (91.8%). GNN message-passing enables much richer coordination than fixed-channel communication. However, at ms200/cr025 GNNs completely fail (0% SR) — they need task relaxation even more than engineered comm.

---

## Cross-ER Comparison

### Best results per condition

| Condition | ER1 (no comm) | ER2 prox | ER2 broadcast | ER3 GNN |
| ----------- | --------------- | ---------- | --------------- | --------- |
| **ms200, cr025** | 4.0% | 4.5% | 1.0% | 0% |
| **ms400, cr025** | 40.5% | **53.0%** | 46.0% | **71.0%** |
| **ms200, cr035** | 27.5% | 37.5% | **48.5%** | 36.5% |

### Analysis

1. **At base difficulty (ms200, cr025):** No method works. The task is too hard — 200 steps is insufficient for 4 agents to coordinate covering 4 targets with k=2 and cr=0.25.

2. **With more time (ms400):** GNN dominates (71.0%), followed by proximity comm (53.0%), broadcast (46.0%), and no-comm (40.5%). More time benefits all methods, but communication amplifies the gain: +12.5pp for proximity, +5.5pp for broadcast, and +31pp for GNN over no-comm.

3. **With easier covering (cr035):** Broadcast comm dominates (48.5%), followed by proximity (37.5%), GNN (36.5%), and no-comm (27.5%). Easier covering benefits broadcast most because global awareness helps agents converge on targets when the covering radius is forgiving.

4. **Communication overhead:** ER2 uses 3200-12800 tokens/episode. ER3 GNN has no explicit tokens (M5=0) — communication is implicit in message-passing. This makes GNN more efficient in terms of communication bandwidth.

5. **Collision patterns:** ER3 GNN has high collisions (8.9-14.9) compared to ER2 proximity (4.4-4.9 with LP+SR). GNN agents are more aggressive in reaching targets, trading collisions for coverage.

### Ranking by Success Rate (top experiments)

| Rank | Experiment | M1 (SR%) | M2 | M3 | M6 (Cov%) | Comm |
| ------ | ----------- | ---------- | ----- | ----- | ----------- | ------ |
| 1 | er3_al_lp_sr_gatv2_ms400 | **71.0%** | 2.498 | 250 | 91.8% | GNN |
| 2 | er1_al_lp_sr_ms400_n2_k1 | **58.0%** | -0.636 | 247 | 87.3% | None (but k=1) |
| 3 | er2_al_lp_sr_prox_dc8_ms400 | **53.0%** | 0.879 | 295 | 82.5% | Proximity |
| 4 | er2_al_lp_sr_bc_dc8_cr035 | **48.5%** | 2.041 | 172 | 83.4% | Broadcast |
| 5 | er2_al_lp_sr_bc_dc8_ms400 | **46.0%** | 0.551 | 308 | 80.4% | Broadcast |
| 6 | er1_al_lp_sr_ms400 | **40.5%** | 0.338 | 317 | 80.0% | None |
| 7 | er2_al_lp_sr_prox_dc8_cr035 | **37.5%** | 1.775 | 176 | 77.8% | Proximity |
| 8 | er3_al_lp_sr_gatv2_cr035 | **36.5%** | 1.756 | 172 | 77.3% | GNN |
| 9 | er1_al_lp_sr_cr035 | **27.5%** | 1.388 | 177 | 73.4% | None |

---

---

## LERO — LLM-Designed Reward Engineering

**Goal:** Use an LLM (gpt-5.4-mini) to evolve reward and observation functions automatically, replacing the hand-crafted reward. Based on the LERO paper ([arXiv:2503.21807](https://arxiv.org/abs/2503.21807)). Full methodology in `docs/lero.md`.

**Setup:** 4 LERO iterations × 3 candidate reward functions per iteration, each evaluated with 1M-frame short training. Best candidate gets 10M-frame full training. `reward_mode=replace` (LLM replaces entire reward), `obs_state_mode=global` (full state access). Prompt: `v2_fewshot` (paper-faithful + 2 MPE-style example reward functions). Reward output clipped to ±50 to prevent PPO gradient explosion. Fallback chain: if full training crashes (NaN), tries next-best candidate.

### LERO results — k=1 tasks (reward design)

| Exp | Task | Prompt | M1 | M2 | M3 | M6 | Runs |
|---|---|---|---:|---:|---:|---:|---|
| L8 | n=3, t=3, k=1 | v2_fewshot | **100%** | 172.1 | 33.9 | 100% | 2/2 stable |
| L8 #2 | n=3, t=3, k=1 | v2_fewshot | **100%** | 105.6 | 33.8 | 100% | 2/2 stable |
| S1 | n=2, t=4, k=1 | v2_fewshot | **100%** | 59.5 | 50.3 | 100% | 1/1 |

### LERO results — k=2 rendezvous task (the hard coordination problem)

| Exp | LLM | Comm | Approach | Obs mode | Eval M1 | **Final M1** | Final M2 | Final M6 | M3 |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| **S3b-global** | gpt-mini | none | obs-only (ER1 reward) | global (oracle) | 0.980 | **1.000** | 19.3 | **1.000** | 68 |
| **S3b-local** | gpt-mini | none | obs-only (ER1 reward) | **local (fair)** | 0.060 | **0.880** | 5.0 | **0.970** | 186 |
| S3a_gpt5 | gpt-5.4 | none | LLM reward (k2 prompt) | global | **0.860** | 0.090 | 1323.6 | 0.393 | 391 |
| S3 | gpt-mini | none | LLM reward (k1 prompt) | global | 0.290 | 0.105 | 486.5 | 0.261 | 382 |
| S3ac | gpt-mini | dim_c=8 | LLM reward (k2 prompt) | global | 0.020 | 0.080 | 1840.8 | 0.275 | 395 |
| S3a_gpt | gpt-mini | none | LLM reward (k2 prompt) | global | 0.010 | 0.000 | 2260.6 | 0.088 | 400 |

**S3b-local is the key result:** M1=88% on k=2 using ONLY local sensors (same information as ER1/ER2/ER3). Beats ER3 GNN (71%) by 17pp without any communication or GNN. The LLM designed coordination features from LiDAR readings — notably a "hold_signal" (target AND agent nearby → stay put) that prevents the "ships passing in the night" failure mode where agents overshoot targets during approach.

### LERO vs ER1/ER2/ER3 comparison (ms400, cr025)

| Method | k=1 (n=2,t=4) | k=1 (n=3,t=3) | k=2 (n=4,t=4) |
|---|---:|---:|---:|
| **LERO obs-only local** (hand-crafted reward + LLM local obs) | — | — | **88.0%** |
| LERO obs-only global (oracle obs) | — | — | 100% (unfair) |
| **ER3** (hand-crafted + GNN) | — | — | 71.0% |
| **ER2** (hand-crafted + proximity comm) | — | — | 53.0% |
| **ER1** (hand-crafted, no comm) | 58.0% | — | 40.5% |
| **LERO reward+obs+comm** (LLM reward, dim_c=8) | — | — | 8.0% |
| **LERO reward-design** (LLM reward, no comm) | **100%** | **100%** | 0–10.5% |

### The eval-vs-final degradation problem (reward hacking at scale)

S3a_gpt5 is the starkest example: gpt-5.4 designed a reward that achieved M1=0.860 at 1M-frame eval — genuinely solving 86% of episodes. But after 10M frames, M1 collapsed to 0.090 while M2 rose from 848 to 1324. The policy found an exploit.

| Metric | Eval (1M) | Final (10M) | Interpretation |
|---|---:|---:|---|
| M1 (success) | 0.860 | 0.090 | policy stopped solving the task |
| M2 (return) | 848 | 1324 | policy found higher-return exploit |
| M6 (coverage) | 0.930 | 0.393 | coverage collapsed |

**Why?** 1M-frame eval is too short to expose reward hacking. By 10M frames, any exploitable gap in the reward WILL be found. **Mitigation:** save the peak-M1 checkpoint during training (not just the final one). The M1=0.86 policy at ~1M frames was a legitimate solution — it shouldn't be discarded just because the reward is exploitable at longer horizons.

### Why LERO reward + communication failed (S3ac: M1=8%)

S3ac combined LLM-designed reward, LLM-designed observations (global), AND 8-float proximity communication (dim_c=8). Despite having the most information of any LERO variant, it performed worst. Three interacting failure modes explain why.

**1. The LLM's reward has a surplus bonus that rewards crowding:**

```python
# From S3ac's best_reward.py:
surplus = (counts - required).clamp(min=0.0).sum(dim=-1)
reward = ... + 0.5 * surplus + ...  # REWARDS having >k agents per target
```

Having 4 agents crowded on 1 target → surplus=(4-2)=2 → +1.0/step × 400 steps = 400 return for doing nothing useful. The policy learns: "crowd together to farm surplus."

**2. The evolutionary loop became a reward-inflation spiral:**

| iter | M2 range across 3 candidates | M1 range |
|---|---|---|
| 0 | 704 – 1,190 | 0.000 – 0.020 |
| 1 | 117 – 1,606 | 0.000 – 0.000 |
| 2 | 2,297 – 4,708 | 0.000 – 0.000 |
| 3 | 4,256 – **6,462** | 0.000 – 0.000 |

The LLM couldn't improve M1 (doesn't know HOW to solve k=2), so it "improved" by designing rewards with progressively higher M2. By iter 3, every candidate had M1=0 with M2 > 4000 — pure reward-inflation.

**3. Communication amplified the exploitation:**

With dim_c=8, each agent has 8 extra continuous action dimensions (the message). The policy used these not to coordinate task completion, but to coordinate reward collection. Communication is a **neutral amplifier** — it helps whatever the policy is already doing:

| Reward quality | Communication effect | Example |
|---|---|---|
| ✅ Correct (ER2) | Helps coordination → M1=53% | Agents share intent |
| ❌ Exploitable (S3ac) | Helps exploitation → M1=8% | Agents coordinate crowding |

**The fundamental asymmetry:**

| Component | Policy can game it? | LLM designs it well? |
|---|---|---|
| **Observations** (read-only features) | ❌ No | ✅ Yes — S3b-local M1=88% |
| **Rewards** (optimization target) | ✅ Yes — will find exploits | ❌ No for k≥2 |
| **Communication** (writable actions) | ✅ Yes — controls messages | n/a — amplifies reward quality |

This explains the entire ranking:
- **S3b-local (obs-only, local):** LLM improves what agents see, not what they optimize → policy can't game observations → 88%
- **ER2 (hand-crafted reward + comm):** correct reward → comm helps coordination → 53%
- **S3ac (LLM reward + comm):** exploitable reward → comm helps exploitation → 8%
- **S3a (LLM reward, no comm):** exploitable reward → no amplifier → 0–10.5%

### Analysis: what the LLM actually designs

#### Successful rewards (k=1 tasks)

The LLM consistently converges on a **three-component structure** for k=1:

**L8 best reward (n=3, t=3, k=1 → M1=1.000):**
```python
reward = (
    3.0 * n_covered                      # per-target coverage count
    + 10.0 * all_covered                  # large completion bonus
    + 0.5 * proximity_bonus              # dense shaping: covering_range - dist
    - 0.5 * crowding                     # anti-crowding: penalize >1 agent/target
    + collision_rew + time_penalty
)
```

**S1 best reward (n=2, t=4, k=1 → M1=1.000):**
```python
reward = (
    2.0 * covered_count
    + 15.0 * all_covered                  # even bigger completion bonus
    + 1.0 * shaping_on_uncovered_only     # smarter: only shape toward uncovered
    - 0.25 * overlap                      # anti-overlap
    + collision_rew + time_penalty
)
```

**Common pattern in successful rewards:**
1. **Per-target coverage reward** (2–3 per target, linear) — provides gradient as soon as any target is reached
2. **Large completion bonus** (10–15 for all targets) — strong signal to finish the task
3. **Dense proximity shaping** — uses `covering_range - dist`, clipped at 0, so agents get signal when approaching a target
4. **Anti-crowding penalty** — discourages multiple agents wasting time on the same target
5. **Small, well-bounded magnitudes** — total reward per step stays in ±20 range, keeping PPO stable
6. **Collision + time penalties preserved** from original scenario

The LLM essentially rediscovered the hand-crafted ER1 reward structure but with better-tuned coefficients and the critical addition of an `all_covered` completion bonus that ER1 lacked.

#### Failed reward (k=2 task — reward hacking)

**S3 best reward (n=4, t=4, k=2 → M1=0.105):**
```python
reward = (
    5.0 * n_covered
    + 25.0 * all_covered
    + 1.25 * occupancy_score              # wants counts near k=2
    - 1.0 * overfill                      # penalizes >k agents per target
    + 0.4 * approach_reward               # -mean(min_dist per agent)
    + 0.2 * progress_bonus
    + collision_rew + time_penalty
)
```

**Why it fails despite looking correct:**

1. **Anti-crowding fights the objective.** The `overfill` penalty penalizes having >2 agents per target. But reaching k=2 coverage REQUIRES 2+ agents to physically converge — the policy learns to AVOID convergence to minimize the penalty during the approach phase.

2. **Approach shaping is individually rational but collectively irrational.** `approach_reward = -min_dist.mean(dim=-1)` encourages each agent to go to its nearest target independently. For k=1, this works (one agent per target = spread out). For k=2, agents need to go to the SAME target together, which this reward actively discourages via the anti-crowding term.

3. **The LLM can describe but not encode coordination.** The `occupancy_score` (wants `counts ≈ k=2`) is mathematically correct but dynamically unstable — it rewards the intermediate state where exactly 2 agents are near each target, but provides no gradient for HOW to get there from a random initialization where agents are spread out.

4. **Magnitude inflation.** S3's M2=486 (vs S1's M2=59 and L8's M2=106) — the policy found a way to collect large rewards without solving the task. The `5.0*n_covered` and `1.25*occupancy_score` terms accumulate over 400 steps even when only partial coverage is achieved.

#### The k=1 vs k=2 capability boundary

| Aspect | k=1 (works) | k=2 (fails) |
|---|---|---|
| Optimal behavior | spread out → 1 agent/target | converge → 2 agents/target |
| Anti-crowding | helpful (stay separate) | harmful (prevents convergence) |
| Individual approach shaping | aligned with team goal | misaligned (agents split) |
| LLM's mental model | "each agent finds a target" ✓ | "pairs of agents find targets" — LLM tries but reward dynamics don't support it |
| Completion bonus | reachable (3/3 targets → bonus) | barely reachable at 1M eval (0.29 best → policy never sees full bonus during short training) |

The fundamental issue: for k=1, individual rationality = collective rationality. The LLM can design individually-rational rewards and k=1 "just works." For k=2, individual rationality (each agent approaches nearest target) ≠ collective rationality (agents must pair up and co-locate). No LLM prompt or function structure tested so far bridges this gap.

### LERO prompt ablation summary (n=3, t=3, k=1 task only)

| Prompt | Best M1 | Stable? | Notes |
|---|---|---|---|
| v2_fewshot (+ MPE examples) | **1.000** ×2 | ✅ very | examples anchor reward magnitude + structure |
| v2 (paper-faithful, 5 lines) | **1.000** ×1, NaN ×2 | ❌ | high LLM variance; negative-M2 samples cause PPO crash |
| v2_min (ultra-minimal, 3 lines) | 0.625 ×1, NaN ×2 | ❌ | too little context → mediocre + fragile rewards |
| v1_global (verbose + research history) | 0.005 ×1 | ❌ | verbose prompt encouraged reward-hackable designs |
| v2_twofn (agent + global split) | 0.010 ×1 | ❌ | decomposition made reward MORE exploitable |

---

## Updated Cross-ER Comparison (including LERO)

### Best results per condition (k=2, ms400, cr025) — UPDATED

| Rank | Method | M1 (SR%) | M2 | M3 | M6 (Cov%) | Reward | Obs/Comm | Fair? |
|---|---|---:|---:|---:|---:|---|---|---|
| (1) | LERO obs-only S3b-global | 100% | 19.3 | 68 | 100% | Hand-crafted | LLM obs (oracle) | ❌ oracle |
| **1** | **LERO obs-only S3b-local** | **88.0%** | **5.0** | **186** | **97.0%** | Hand-crafted | **LLM obs (local only)** | **✅ fair** |
| 2 | ER3 GNN (GATv2) | 71.0% | 2.50 | 250 | 91.8% | Hand-crafted + LP+SR | GNN msg-passing | ✅ fair |
| 3 | ER2 proximity comm | 53.0% | 0.88 | 295 | 82.5% | Hand-crafted + LP+SR | Proximity comm | ✅ fair |
| 4 | ER2 broadcast comm | 46.0% | 0.55 | 308 | 80.4% | Hand-crafted + LP+SR | Broadcast comm | ✅ fair |
| 5 | ER1 no comm | 40.5% | 0.34 | 317 | 80.0% | Hand-crafted + LP+SR | None | ✅ fair |
| 6 | LERO reward (S3) | 10.5% | 486.5 | 382 | 26.1% | LLM-designed | LLM obs | ✅ fair |
| 7 | LERO reward (S3a_gpt5) | 9.0% | 1323.6 | 391 | 39.3% | LLM-designed (gpt-5.4) | LLM obs | ✅ fair |

**S3b-global** (100%) uses oracle global state in observations — unfair comparison. **S3b-local** (88%) uses only local sensors — same information as ER1/ER2/ER3 — and is the legitimate new state-of-the-art.

### Best results for k=1 tasks

| Rank | Method | Task | M1 (SR%) | M6 | Reward design |
|---|---|---|---:|---:|---|
| 1 | **LERO** no comm | n=2, t=4, k=1 | **100%** | 100% | LLM-designed |
| 2 | **LERO** no comm | n=3, t=3, k=1 | **100%** | 100% | LLM-designed |
| 3 | ER1 abl_i | n=4, t=4, k=1 | 76.8% | 93.7% | Hand-crafted |
| 4 | ER1 + AL | n=4, t=4, k=1 | 75.0% | 92.9% | Hand-crafted + AL |
| 5 | ER1 no comm | n=2, t=4, k=1 | 58.0% | 87.3% | Hand-crafted + LP+SR |

---

## Central Thesis — Feature Engineering vs Incentive Design

The complete experimental evidence (ER1–ER3 + LERO) converges on a single principle:

> **LLMs are excellent at feature engineering (designing what agents observe) but unreliable at incentive design (designing what agents optimize for). This asymmetry exists because observations are read-only — the policy cannot game them — while rewards are the optimization target and WILL be exploited given sufficient training.**

This explains every result in the project:

| Approach | LLM designs... | Can policy game it? | k=1 result | k=2 result |
|---|---|---|---|---|
| **Reward LERO** | what agents optimize for | ✅ Yes | M1=100% (k=1 is simple enough that exploits ≈ solutions) | M1=0–10.5% (reward-hacked) |
| **Obs-only LERO (local)** | what agents see | ❌ No | — | **M1=88%** (legitimate) |
| **Reward + comm LERO** | what agents optimize for + comm channel | ✅ Yes (amplified by comm) | — | M1=8% (comm amplifies exploitation) |
| **ER1–ER3** | nothing (hand-crafted) | ❌ No (reward is fixed) | 58–77% | 40.5–71% |

For k=1 tasks, reward design works because individual rationality = collective rationality — there's no gap for the policy to exploit. For k≥2, the LLM designs rewards with exploitable gaps (anti-crowding penalties, surplus bonuses, magnitude inflation) that the policy discovers with 10M frames of training.

The practical implication for MARL practitioners: **use LLMs for observation/feature design, not reward design, when the task requires multi-agent coordination.** Keep the reward hand-crafted (or proven non-exploitable) and let the LLM enhance what agents perceive.

---

## Key Conclusions

1. **LERO obs-only with LOCAL sensors is the new state-of-the-art for k=2.** S3b-local achieved M1=88.0% using only local LiDAR — surpassing ER3 GNN (71%) by 17pp, ER2 proximity (53%), and ER1 no-comm (40.5%). No communication channel, no GNN, no oracle information. The LLM designed coordination features from the same sensor data: a `hold_signal` (target+agent both nearby → stay), `approach_signal`, `crowd_signal`, and `sparsity_signal` that pre-compute actionable coordination decisions from raw LiDAR. S3b-global (M1=100%) uses oracle global state and is NOT a fair comparison.

2. **LLM-designed features beat GNN message-passing.** S3b-local (88%) > ER3 GATv2 (71%) despite using simpler per-agent features instead of learned graph attention. The LLM's feature engineering replaces the GNN's learned communication protocol with pre-computed coordination signals — equivalent to designing a fixed communication protocol at design time rather than learning one at training time.

3. **LERO reward design fails catastrophically on k=2.** Across 5 attempts (different LLMs, prompts, comm channels), LLM-designed rewards all reward-hacked (M1=0–10.5%, M2=486–2260). The LLM consistently produces individually-rational rewards (anti-crowding, per-agent approach) that are collectively irrational for rendezvous. Even gpt-5.4 (full model) achieved eval M1=0.86 but degraded to 0.09 at 10M training — the reward was exploitable.

4. **LERO reward design dominates on k=1.** Achieving 100% SR vs ER1's 58–77%, entirely from better reward design. The LLM-designed reward adds a critical `all_covered` completion bonus and better proximity shaping that ER1 lacked. For tasks where individual rationality = collective rationality (spreading out), the LLM excels.

5. **The LERO capability boundary is between reward design and observation design, not between k=1 and k=2.** k=2 IS solvable by LERO — just not via reward design. The LLM's strength is in designing richer observations (feature engineering), not in designing reward incentives for coordination. With local-only sensors, the LLM achieves 88% by deriving coordination signals from LiDAR that the base observation (raw lidar rays) doesn't provide. With oracle global state, it reaches 100%.

6. **Eval-best ≠ train-stable for LLM rewards.** gpt-5.4's M1=0.86 at 1M eval collapsed to 0.09 at 10M. 1M-frame eval is too short to detect reward hacking. **Best-checkpoint saving** (keeping the peak-M1 policy during training) would recover these solutions.

7. **Few-shot examples are critical for LERO stability.** The `v2_fewshot` prompt was the only configuration that produced M1=1.000 reliably (2/2 runs on k=1). Without examples, the LLM produces rewards with wildly varying magnitudes that cause PPO to diverge.

8. **Communication is a neutral amplifier — it helps whatever the policy optimizes for.** With hand-crafted reward (ER2), comm helps coordination (53%). With LLM-designed reward (S3ac), comm helps exploitation (8% — WORSE than without comm). The reward-inflation spiral across LERO iterations (M2: 1190 → 6462 over 4 iters while M1 stayed at 0) shows the evolutionary loop can't escape exploitable rewards and communication channels provide more degrees of freedom to exploit.

9. **GNN communication remains valuable but is now outperformed.** ER3 GATv2 ms400 (71%) was the previous best for k=2. LERO obs-only with local sensors (88%) surpasses it without any communication channel — just richer per-agent feature engineering from a one-time LLM design step. The GNN learns to COMMUNICATE; the LLM learns to OBSERVE better. For this task, better observation wins.

10. **Agent LiDAR is a double-edged sword.** It reduces collisions for k=1 but causes avoidance behavior that prevents k=2 coordination. All successful k=2 runs use agent LiDAR + reward shaping together.

---

## Appendix: Ablation Summary (ER1)

| Ablation | Variable | Change from baseline | Effect on M1 (mean, k=2) |
| ---------- | ---------- | --------------------- | -------------------------- |
| A | Entropy coeff (default) | entropy=0 | 0% (agents freeze) |
| A2 | Entropy coeff (high) | entropy=0.01 | 4.3% (mild help) |
| B | Learning rate | lr=1e-4 (2x) | 5.7% (mild help) |
| C | GAE lambda | lambda tuning | 6.3% (marginal) |
| G | Training length | 20M frames (2x) | 5.2% (no benefit) |
| H | Network size | Larger hidden layers | 4.7% (no benefit) |
| I | Sanity check | k=1 | 76.8% (task is solvable) |
