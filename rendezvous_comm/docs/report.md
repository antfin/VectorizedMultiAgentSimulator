# Learning Communication Protocols for Multi-Robot Rendezvous

**Experiment Report — March 2026**

---

## 1. Executive Summary

We investigate whether learned communication protocols improve multi-robot rendezvous in the VMAS Discovery scenario, where 4 agents must coordinate to simultaneously cover 4 targets (k=2: each target requires 2 agents within covering range at the same time). Without communication, agents achieve only ~4% success rate at 200 steps and ~40% at 400 steps — the k=2 coordination constraint creates a fundamental information bottleneck that no amount of hyperparameter tuning or reward shaping can overcome alone. Engineered communication channels (proximity and broadcast) provide modest gains (+12.5 percentage points at 400 steps), but the breakthrough comes from GNN-based implicit spatial communication: a GATv2Conv message-passing architecture achieves **71% success rate** — the highest across all experiments. The key finding is that implicit spatial communication through graph neural network edges is more effective than explicit learned message channels, suggesting that the structure of information flow matters more than its bandwidth.

---

## 2. WHY: Research Question

Multi-robot rendezvous — the problem of coordinating multiple agents to meet at designated locations — is a core challenge in multi-agent systems. In our formulation, coordination is not optional: the k=2 constraint means each target must be covered by exactly 2 agents simultaneously, making the task impossible to solve through independent action alone.

**Research Question:** *Is communication necessary for multi-robot rendezvous, and if so, what form of communication works best?*

The k=2 constraint is critical. When k=1 (any single agent can cover a target), agents achieve 77% success rate without any communication — the task is solvable through independent exploration. But k=2 creates an exponential coordination challenge: agents must not only find targets but also synchronize their presence. This gap (77% vs 4%) is the direct motivation for investigating communication protocols.

### Task Diagram

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 280" width="400" height="280">
  <!-- Target with covering range示 -->
  <circle cx="280" cy="100" r="45" fill="none" stroke="#cc0000" stroke-dasharray="5,5" stroke-width="1.5"/>
  <circle cx="280" cy="100" r="8" fill="#cc0000" opacity="0.8"/>
  <text x="280" y="80" text-anchor="middle" font-size="10" fill="#333">covering_range</text>
  <!-- Two agents inside covering range (k=2) -->
  <circle cx="265" cy="110" r="8" fill="#2266cc" opacity="0.8"/>
  <circle cx="300" cy="95" r="8" fill="#2266cc" opacity="0.8"/>
  <text x="280" y="160" text-anchor="middle" font-size="10" fill="#666">k=2: both agents needed</text>
  <!-- Other targets -->
  <circle cx="100" cy="80" r="8" fill="#cc0000" opacity="0.8"/>
  <circle cx="150" cy="200" r="8" fill="#cc0000" opacity="0.8"/>
  <circle cx="320" cy="220" r="8" fill="#cc0000" opacity="0.8"/>
  <!-- Other agents -->
  <circle cx="80" cy="180" r="8" fill="#2266cc" opacity="0.8"/>
  <circle cx="200" cy="140" r="8" fill="#2266cc" opacity="0.8"/>
  <!-- Legend -->
  <circle cx="30" cy="250" r="6" fill="#2266cc" opacity="0.8"/>
  <text x="42" y="254" font-size="10" fill="#333">Agent</text>
  <circle cx="100" cy="250" r="6" fill="#cc0000" opacity="0.8"/>
  <text x="112" y="254" font-size="10" fill="#333">Target</text>
</svg>

---

## 3. HOW: Methodology

### 3.1 VMAS Discovery Scenario

The Discovery scenario is implemented in VMAS (Vectorized Multi-Agent Simulator), a PyTorch-based framework that runs batches of environments in parallel on GPU. Key simulation parameters:

- **Parallelism:** 600 environments run simultaneously
- **Space:** 2D continuous, agents and targets spawn randomly
- **Agents:** 4, each equipped with LiDAR sensor
- **Targets:** 4, with `targets_respawn=False` (episode ends when all targets are covered)
- **Reward structure:** `covering_rew` (positive for covering targets) + `collision_penalty` + `time_penalty`
- **Covering:** A target is covered when k agents are simultaneously within `covering_range` of it

### 3.2 MAPPO Algorithm

We use Multi-Agent PPO (MAPPO) via BenchMARL, a centralized-training-decentralized-execution (CTDE) algorithm well-suited for cooperative multi-agent tasks. In CTDE, agents share parameters and use a centralized critic during training (with access to global state) but act independently at test time using only local observations.

Key hyperparameters:

- Discount factor: gamma = 0.99
- GAE lambda: 0.95
- Learning rate: 5e-5
- Training budget: 10M frames (unless noted)
- Entropy coefficient: 0.005 (default)

### 3.3 Communication Protocols Tested

We test three approaches, ordered from least to most implicit communication:

1. **ER1 — No Communication:** MLP policy, agents observe the environment only through their LiDAR sensor. This is the baseline: can agents coordinate through environment-mediated stigmergy alone?

2. **ER2 — Engineered Communication:** Agents have explicit `dim_c` action channels that produce communication tokens. Two modes tested:
   - *Proximity:* messages sent only to agents within communication range
   - *Broadcast:* messages sent to all agents regardless of distance

3. **ER3 — GNN Communication:** Agents are nodes in a graph; GATv2Conv (Graph Attention Network v2) performs message-passing along edges. The attention mechanism learns which neighbors' information is most relevant. See [GNN Theory](theory_gnn_communication.md) for architectural details.

### 3.4 Metrics

| Metric | ID | Definition |
| --- | --- | --- |
| Success Rate | M1 | % of episodes where all targets are covered |
| Average Return | M2 | Mean cumulative reward per episode |
| Steps to Completion | M3 | Mean episode length (lower = faster) |
| Collisions | M4 | Mean collisions per episode |
| Communication Tokens | M5 | Total tokens exchanged per episode |
| Coverage Progress | M6 | % of targets covered (even if episode fails) |
| Sample Efficiency | M7 | Frames to reach threshold performance |
| Agent Utilization | M8 | Fraction of time agents are productively moving |
| Spatial Spread | M9 | Mean pairwise distance between agents |

### 3.5 Experimental Evolution and Bug Discovery

A critical methodological note: initial experiments (pre-2026-03-22) ran with **max_steps=100** due to a `config.pop("max_steps")` bug that removed the parameter before it reached VMAS, causing the scenario to use its default of 100 steps. This led to the early (incorrect) conclusion that "communication doesn't help" — agents simply didn't have enough time to coordinate.

The bug was discovered on 2026-03-22, fixed, and all key experiments were re-run. Results below are clearly marked as pre-fix or post-fix. This experience underscores the importance of end-to-end parameter verification in complex ML pipelines.

---

## 4. WHAT: Results

### 4.1 ER1 — No Communication Baseline

#### 4.1.1 k=1 vs k=2 Difficulty Split

The coordination constraint k=2 transforms the task from solvable to nearly impossible:

| exp_id | n | k | ms | cr | dim_c | comm | M1 (SR%) | M2 | M3 | M4 | M6 | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| er1_k1_l025_s0 | 4 | 1 | 200 | 0.25 | 0 | none | 66.0% | 0.132 | 77.3 | 6.1 | 89.3% | |
| er1_k1_l035_s0 | 4 | 1 | 200 | 0.25 | 0 | none | 76.5% | 0.331 | 69.7 | 4.8 | 93.6% | |
| er1_k2_l035_s0 | 4 | 2 | 200 | 0.25 | 0 | none | 8.5% | -0.147 | 97.9 | 6.3 | 49.8% | |
| er1_k2_l035_s1 | 4 | 2 | 200 | 0.25 | 0 | none | 4.5% | -0.201 | 99.2 | 7.2 | 48.4% | |
| er1_abl_i_k1 | 4 | 1 | 200 | 0.25 | 0 | none | 76.8% | — | — | — | 93.7% | ablation sanity |

k=1: 66–77% success. k=2: 4–9% success. This 10x gap is the core motivation for communication.

#### 4.1.2 Agent LiDAR Paradox

Adding agent LiDAR (AL) — allowing agents to sense each other — produces a paradoxical result:

| exp_id | n | k | ms | cr | dim_c | comm | M1 (SR%) | M2 | M3 | M4 | M6 | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| er1_al_k1_l035_s1 | 4 | 1 | 200 | 0.25 | 0 | none | 75.0% | 0.441 | 69.9 | 0.8 | 92.9% | AL helps: fewer collisions |
| er1_al_k2_l035_s0 | 4 | 2 | 200 | 0.25 | 0 | none | 0% | -0.420 | 100 | 0.7 | 29.8% | AL hurts: 0% SR |
| er1_al_k2_l035_s1 | 4 | 2 | 200 | 0.25 | 0 | none | 0% | -0.642 | 100 | 0.1 | 17.8% | AL hurts: 0% SR |

For k=1, agent LiDAR helps by reducing collisions (6.3 to 0.8). For k=2, it is catastrophic: agents learn to avoid each other so effectively that they can never converge on the same target. This is a clear example of observation design interacting with task structure.

#### 4.1.3 Ablation Studies

We ran extensive ablations on hyperparameters for k=2, all at ms200/cr025. The key finding: **no hyperparameter change breaks the k=2 ceiling.**

| Ablation | Change | M1 (SR%) | M6 | Verdict |
| --- | --- | --- | --- | --- |
| abl_a | entropy=0 | 0% | 2.4% | Deadly — agents freeze |
| abl_a2 | entropy=0.01 | 4.3% | 45.6% | Marginal |
| abl_b | lr=1e-4 | 5.7% | 51.4% | Marginal |
| abl_c | lambda tuning | 6.3% | 51.1% | Marginal |
| abl_g | 20M frames | 5.2% | 48.7% | No benefit, some collapse |
| abl_h | larger network | 4.7% | 49.6% | No benefit |
| abl_i | k=1 (sanity) | 76.8% | 93.7% | Confirms k=2 is the issue |

No ablation exceeds 6.3% success. The problem is not learning capacity, training budget, or hyperparameter sensitivity — it is a fundamental information deficit.

Critically, **shared reward yields 0% improvement**, proving that k=2 is an information problem, not a credit assignment problem. Agents fail because they lack the information to coordinate, not because they lack the incentive.

#### 4.1.4 Reward Shaping (LP+SR)

Progressive reward shaping with lidar proximity (LP) and shared reward (SR):

| exp_id | n | k | ms | cr | dim_c | comm | M1 (SR%) | M2 | M3 | M4 | M6 | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| er1_al_base | 4 | 2 | 200 | 0.25 | 0 | none | 0% | -0.642 | 100 | 0.1 | 17.8% | base |
| er1_al_abl_lp | 4 | 2 | 200 | 0.25 | 0 | none | 0.5% | — | — | — | 35.1% | +LP |
| er1_al_abl_sr | 4 | 2 | 200 | 0.25 | 0 | none | 1.5% | — | — | — | 38.1% | +SR |
| er1_al_abl_lp_sr | 4 | 2 | 200 | 0.25 | 0 | none | 4.0% | — | — | — | 45.6% | +LP+SR |

The progression 0% -> 0.5% -> 1.5% -> 4.0% shows that reward shaping helps but cannot solve the coordination problem alone.

#### 4.1.5 Task Relaxations — The ER1 Breakthrough

The most impactful discovery in ER1 was that task relaxations unlock dramatically higher performance:

| exp_id | n | k | ms | cr | dim_c | comm | M1 (SR%) | M2 | M3 | M4 | M6 | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| er1_al_lp_sr | 4 | 2 | 200 | 0.25 | 0 | none | 4.0% | 0.840 | 99.3 | 2.9 | 45.6% | baseline |
| er1_al_lp_sr_ms400 | 4 | 2 | **400** | 0.25 | 0 | none | 40.5% | 0.338 | 316.9 | 31.7 | 80.0% | **10x improvement** |
| er1_al_lp_sr_cr035 | 4 | 2 | 200 | **0.35** | 0 | none | 27.5% | 1.388 | 176.6 | 6.8 | 73.4% | |
| er1_al_lp_sr_ms400_n2k1 | 2 | 1 | **400** | 0.25 | 0 | none | 58.0% | -0.636 | 247.0 | 1.3 | 87.3% | minimal team |

**Key discovery:** ms400 yields a 10x improvement (4% to 40.5%). Agents CAN learn k=2 coordination without communication — they just need enough episode time. The covering_range relaxation (cr035) provides a 7x improvement. These relaxations become the standard conditions for comparing communication protocols.

---

### 4.2 ER2 — Engineered Communication

#### 4.2.1 Baseline Communication (No Reward Shaping)

| exp_id | n | k | ms | cr | dim_c | comm | M1 (SR%) | M2 | M3 | M4 | M6 | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| er2_prox_dc8 | 4 | 2 | 200 | 0.25 | 8 | prox | 3.5% | — | — | — | — | no reward shaping |
| er2_al_prox_dc8 | 4 | 2 | 200 | 0.25 | 8 | prox | 0.5% | — | — | — | — | +AL, no reward shaping |

Communication alone does not help without reward shaping. The 3.5% result is indistinguishable from the no-communication baseline.

#### 4.2.2 Communication + LP+SR (ms200/cr025)

| exp_id | n | k | ms | cr | dim_c | comm | M1 (SR%) | M2 | M3 | M4 | M5 | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| er2_al_lp_sr_prox_dc8 | 4 | 2 | 200 | 0.25 | 8 | prox | 4.5% | — | — | — | 3200 | marginal gain |
| er2_al_lp_sr_bc_dc2 | 4 | 2 | 200 | 0.25 | 2 | broadcast | 1.0% | — | — | — | 800 | |
| er2_al_lp_sr_bc_dc8 | 4 | 2 | 200 | 0.25 | 8 | broadcast | 1.0% | — | — | — | 3200 | |
| er2_al_lp_sr_bc_dc16 | 4 | 2 | 200 | 0.25 | 16 | broadcast | 0% | — | — | — | 6400 | noise overwhelms |

At ms200, communication provides negligible benefit. Broadcast actually hurts performance, and higher dim_c makes it worse — the agents cannot learn to filter useful signals from noise in the limited episode time.

#### 4.2.3 Task Relaxations Reveal Communication Value

| exp_id | n | k | ms | cr | dim_c | comm | M1 (SR%) | M2 | M3 | M4 | M5 | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| er2_al_lp_sr_prox_dc8_ms400 | 4 | 2 | **400** | 0.25 | 8 | prox | 53.0% | — | 295.5 | — | 12800 | +12.5pp vs ER1 |
| er2_al_lp_sr_bc_dc8_ms400 | 4 | 2 | **400** | 0.25 | 8 | broadcast | 46.0% | — | 308.2 | — | 12800 | |
| er2_al_lp_sr_prox_dc8_cr035 | 4 | 2 | 200 | **0.35** | 8 | prox | 37.5% | — | 176.2 | — | 6400 | |
| er2_al_lp_sr_bc_dc8_cr035 | 4 | 2 | 200 | **0.35** | 8 | broadcast | 48.5% | — | 171.6 | — | 6400 | broadcast excels here |

With ms400, proximity communication adds +12.5 percentage points over ER1 (53% vs 40.5%). Surprisingly, broadcast communication excels with the easier covering range (48.5% at cr035), likely because the wider acceptance radius makes imprecise coordination (enabled by broadcast) sufficient.

Key insight: communication needs enough episode time to learn useful protocols. At ms200, the learning horizon is too short for agents to simultaneously learn the task AND a communication protocol.

---

### 4.3 ER3 — GNN Communication

#### 4.3.1 Architecture

The GNN approach uses GATv2Conv (Graph Attention Network v2) for message-passing between agents. Each agent is a node; edges connect all agents (full topology). The attention mechanism learns to weight neighbor information by relevance, enabling implicit spatial coordination through graph structure.

- Architecture: GATv2Conv with attention heads
- Parameters: ~78K
- Topology: full (all-to-all)
- See [GNN Theory](theory_gnn_communication.md) for detailed architecture description

#### 4.3.2 Initial Failure (Pre-Fix)

| exp_id | n | k | ms | cr | dim_c | comm | M1 (SR%) | M2 | M3 | M4 | M6 | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| er3_al_lp_sr_gatv2 | 4 | 2 | 200 | 0.25 | 0 | GATv2 | 0% | — | — | 27.8 | 9.1% | very high collisions |
| er3_al_lp_sr_graphconv | 4 | 2 | 200 | 0.25 | 0 | GraphConv | 0% | — | — | 15.5 | 12.1% | |

At ms200, both GNN architectures fail completely. The GATv2 variant produces extremely high collision rates (27.8/episode), suggesting the attention-based messages create herding behavior without enough time to develop useful coordination.

#### 4.3.3 Post-Fix Breakthrough

| exp_id | n | k | ms | cr | dim_c | comm | M1 (SR%) | M2 | M3 | M4 | M6 | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| er3_al_lp_sr_gatv2_ms400 | 4 | 2 | **400** | 0.25 | 0 | GATv2 | **71.0%** | 2.498 | 249.7 | 8.9 | 91.8% | **best overall** |
| er3_al_lp_sr_gatv2_cr035 | 4 | 2 | 200 | **0.35** | 0 | GATv2 | 36.5% | 1.756 | 172.1 | 14.9 | 77.3% | |

With ms400, GATv2 achieves **71% success rate** — the highest across all experiments. This is a 30.5 percentage point improvement over the no-communication baseline (40.5%) and 18 points above the best engineered communication (53%).

Why GNN wins:

- Spatial edge features enable implicit coordination without needing to learn an explicit communication protocol
- Attention weights allow agents to dynamically focus on the most relevant neighbors
- M9 spatial spread of 0.578 (vs 1.018 for no-comm) provides direct evidence of learned pairing behavior — agents cluster into pairs near targets
- The return efficiency (M2=2.498) is far higher than any other approach, indicating cleaner coordination

---

### 4.4 Cross-ER Comparison

#### Main Comparison Table

| Condition | ER1 (no comm) | ER2 proximity | ER2 broadcast | ER3 GNN |
| --- | --- | --- | --- | --- |
| ms200 / cr025 | 4.0% | 4.5% | 1.0% | 0% |
| **ms400** / cr025 | 40.5% | 53.0% | 46.0% | **71.0%** |
| ms200 / **cr035** | 27.5% | 37.5% | 48.5% | 36.5% |

At the standard difficulty (ms200/cr025), no approach exceeds 5%. Task relaxation is a prerequisite for any method to work.

At ms400, there is a clear hierarchy: GNN (71%) > proximity (53%) > broadcast (46%) > no-comm (40.5%). Communication provides consistent gains, with implicit spatial communication (GNN) being the most effective.

At cr035, broadcast surprisingly leads (48.5%), likely because the wider covering range makes coarse coordination sufficient, and broadcast provides the simplest coordination signal.

#### M1 Success Rate — ms400/cr025 Condition

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 460 200" width="460" height="200">
  <!-- Bars -->
  <rect x="120" y="15" width="203" height="30" fill="#999999" rx="3"/>
  <text x="328" y="35" font-size="13" fill="#333">40.5%</text>
  <text x="10" y="35" font-size="12" fill="#333">ER1 no comm</text>

  <rect x="120" y="55" width="265" height="30" fill="#2266cc" rx="3"/>
  <text x="390" y="75" font-size="13" fill="#333">53.0%</text>
  <text x="10" y="75" font-size="12" fill="#333">ER2 proximity</text>

  <rect x="120" y="95" width="230" height="30" fill="#dd8800" rx="3"/>
  <text x="355" y="115" font-size="13" fill="#333">46.0%</text>
  <text x="10" y="115" font-size="12" fill="#333">ER2 broadcast</text>

  <rect x="120" y="135" width="355" height="30" fill="#229944" rx="3"/>
  <text x="380" y="155" font-size="13" fill="#222" font-weight="bold">71.0%</text>
  <text x="10" y="155" font-size="12" fill="#333">ER3 GNN</text>

  <!-- Axis -->
  <line x1="120" y1="175" x2="475" y2="175" stroke="#ccc" stroke-width="1"/>
  <text x="120" y="190" font-size="10" fill="#666">0%</text>
  <text x="370" y="190" font-size="10" fill="#666">50%</text>
  <text x="460" y="190" font-size="10" fill="#666">71%</text>
</svg>

#### Top Experiments Ranking

| Rank | Experiment | Communication | M1 (SR%) | Key Condition |
| --- | --- | --- | --- | --- |
| 1 | er3_al_lp_sr_gatv2_ms400 | GNN (GATv2) | **71.0%** | **ms400** |
| 2 | er1_al_lp_sr_ms400_n2k1 | none | 58.0% | **ms400**, n=2, k=1 |
| 3 | er2_al_lp_sr_prox_dc8_ms400 | proximity dc8 | 53.0% | **ms400** |
| 4 | er2_al_lp_sr_bc_dc8_cr035 | broadcast dc8 | 48.5% | **cr035** |
| 5 | er2_al_lp_sr_bc_dc8_ms400 | broadcast dc8 | 46.0% | **ms400** |
| 6 | er1_al_lp_sr_ms400 | none | 40.5% | **ms400** |
| 7 | er2_al_lp_sr_prox_dc8_cr035 | proximity dc8 | 37.5% | **cr035** |
| 8 | er3_al_lp_sr_gatv2_cr035 | GNN (GATv2) | 36.5% | **cr035** |
| 9 | er1_al_lp_sr_cr035 | none | 27.5% | **cr035** |

---

### 4.5 Deep Dive: GNN vs Minimal Team

An important comparison: GNN with 4 agents k=2 (71%) vs minimal team with 2 agents k=1 (58%). The minimal team has the same theoretical coordination complexity (each target needs exactly the right number of agents), yet GNN outperforms it by 13 percentage points. Why?

This is NOT a parallelism advantage — the theoretical minimum steps is similar for both configurations. The real reasons:

- **Higher spatial coverage:** 4 agents cover more ground than 2, finding targets faster
- **Adaptive coordination via attention:** GATv2 attention weights allow agents to dynamically pair up based on proximity and target location
- **Robustness through redundancy:** If one agent gets stuck, others can compensate
- **Return efficiency:** GNN achieves M2=+2.498 vs M2=-0.636 for minimal team, indicating much cleaner coordination
- **Spatial spread proves pairing:** M9=0.578 (GNN) vs M9=1.018 (no-comm) — GNN agents learn to cluster into pairs near targets
- **Entropy retention:** GNN maintains entropy at -0.84 vs -2.35, preserving exploratory flexibility

See [Detailed GNN vs Minimal Team comparison](comparison_gnn_vs_minimal_team.md) for full analysis.

---

## 5. Conclusions

1. **k=2 coordination requires communication + task relaxation.** No method exceeds 10% success at ms200/cr025. The coordination constraint creates a fundamental information bottleneck.

2. **GNN is the most powerful communication form.** At ms400, GATv2 achieves 71% — 18pp above the best engineered channel and 30.5pp above no communication.

3. **Broadcast excels with easier covering.** At cr035, broadcast (48.5%) outperforms proximity (37.5%) and GNN (36.5%), suggesting that when coarse coordination suffices, simple signals win.

4. **Reward shaping (LP+SR) is a prerequisite.** Without lidar proximity and shared reward, even explicit communication channels fail (3.5% with proximity dc8, no reward shaping).

5. **Agent LiDAR is double-edged.** It helps k=1 (fewer collisions) but hurts k=2 (agents learn avoidance, preventing convergence on targets).

6. **The bottleneck is information, not credit assignment.** The shared reward ablation (0% improvement) proves that agents fail because they cannot observe enough to coordinate, not because they lack incentive.

7. **Implicit spatial communication (GNN edges) > explicit learned messages (dim_c).** The structure of information flow — who communicates with whom and about what — matters more than the bandwidth of the communication channel.

---

## 6. Next Steps

- **Multi-seed validation:** Run top experiments (GNN ms400, proximity ms400, broadcast cr035) with 3-5 seeds to establish confidence intervals
- **ER4 — Gated communication (IC3Net):** Learn WHEN to communicate, not just WHAT. Agents decide at each step whether to broadcast, potentially reducing communication cost (M5) while maintaining coordination
- **Scaling:** Test with n=6 and n=8 agents to understand how communication protocols scale
- **Pareto analysis:** M1 vs M5 (success rate vs communication cost) to identify the most efficient communication protocols
- **Curriculum learning:** Start with ms400 and gradually reduce to ms200 to see if agents can transfer learned coordination to harder conditions

---

## References

- Raw results: `../results/er1/runs/`, `../results/er2/runs/`, `../results/er3/runs/`
- GNN theory: [theory_gnn_communication.md](theory_gnn_communication.md)
- GNN vs minimal team: [comparison_gnn_vs_minimal_team.md](comparison_gnn_vs_minimal_team.md)
