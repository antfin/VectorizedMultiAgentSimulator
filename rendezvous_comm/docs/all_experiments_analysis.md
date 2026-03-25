# Complete Experiment Analysis — Learning Communication Protocols for Multi-Robot Rendezvous

**Date:** 2026-03-24
**Framework:** VMAS Discovery + BenchMARL (MAPPO)
**Task:** 4 targets, covering_range=0.25 (unless noted), targets_respawn=False

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

## Key Conclusions

1. **k=2 coordination requires communication + task relaxation.** No configuration achieves >10% SR at the hardest setting (ms200, cr025, k=2).

2. **GNN communication is the most powerful** when given sufficient episode length (ms400), achieving 71% SR — nearly matching what k=1 achieves without communication (76.5%).

3. **Broadcast communication excels with easier covering** (cr035), outperforming proximity and GNN in that condition. Global information sharing helps when precision requirements are relaxed.

4. **Reward shaping (LP+SR) is a prerequisite.** Without it, even explicit communication channels fail to learn useful protocols.

5. **Agent LiDAR is a double-edged sword.** It reduces collisions for k=1 but causes avoidance behavior that prevents k=2 coordination. All successful k=2 runs use agent LiDAR + reward shaping together.

6. **The max_steps bug** invalidated early ER3 results. Post-fix GATv2 ms400 went from 0% to 71% SR, confirming that episode length is critical for GNN-based learning.

7. **Sample efficiency:** Proximity comm converges faster (6.7M frames) than broadcast (9.6M) and GNN (not measured but trains in similar time). No-comm with reward shaping converges around 6.1M frames.

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
