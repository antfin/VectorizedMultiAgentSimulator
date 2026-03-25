# Experiment Comparison: ER1 vs ER2 vs ER3

**Date**: 2026-03-23
**Task**: Discovery (multi-agent rendezvous) — 4 agents must cooperatively cover 4 targets

## Experimental Variables

| Variable | ER1 | ER2 | ER3 |
|---|---|---|---|
| **Model** | MLP [256, 256] | MLP [256, 256] | GATv2Conv GNN (from_pos) |
| **Communication** | None (dim_c=0) | Proximity, 8-dim (dim_c=8) | None (dim_c=0) — GNN message passing is implicit |
| **Policy params** | 75,013 | 85,269 | 78,444 |

All other parameters are identical (see Shared Configuration below).

## Results: ms400 (max_steps=400)

| Metric | ER1 (MLP) | ER2 (MLP+Comm) | ER3 (GNN) | Best |
|---|---|---|---|---|
| **M1 Success Rate** | 0.405 | 0.530 | **0.710** | ER3 |
| **M2 Avg Return** | 0.338 | 0.879 | **2.498** | ER3 |
| **M3 Avg Steps** | 316.9 | 295.5 | **249.7** | ER3 |
| **M4 Avg Collisions** | 31.7 | 26.1 | **8.9** | ER3 |
| **M5 Avg Tokens** | 0 | 12,800 | 0 | — |
| **M6 Coverage Progress** | 0.800 | 0.825 | **0.918** | ER3 |
| **M9 Spatial Spread** | 0.784 | 0.767 | 0.578 | — |
| **Final Entropy** | -0.737 | -1.979 | -0.838 | — |
| **Training Time** | 1h 34m | 1h 44m | ~8.5h | ER1 |

### Key Findings (ms400)

1. **GNN (ER3) dominates all performance metrics** — 71% success vs 53% (comm) vs 40.5% (MLP)
2. **GNN achieves lowest collisions** (8.9 vs 26-32) — spatial edge features enable better avoidance
3. **GNN fastest to complete** (249.7 steps vs 295-317) — agents coordinate more efficiently
4. **GNN has no explicit communication** but outperforms explicit comm channel (ER2)
5. **ER2 proximity comm helps over baseline** — 53% vs 40.5% (+30% relative improvement)
6. **GNN is much slower to train** — ~8.5h vs ~1.5h (from_pos topology + attention computation)
7. **M9 spatial spread lower for GNN** — agents stay closer together (better coordination)

## Results: cr035 (covering_range=0.35, max_steps=200)

| Metric | ER1 cr035 | ER2 cr035 | ER3 cr035 |
|---|---|---|---|
| **M1 Success Rate** | 0.275 | 0.375 | pending |
| **M2 Avg Return** | 1.388 | 1.775 | pending |
| **M3 Avg Steps** | 176.6 | 176.2 | pending |
| **M4 Avg Collisions** | 6.8 | 4.9 | pending |
| **M5 Avg Tokens** | 0 | 6,400 | pending |
| **M6 Coverage Progress** | 0.734 | 0.778 | pending |
| **M9 Spatial Spread** | 0.843 | 0.804 | pending |
| **Training Time** | 1h 28m | 1h 45m | ~8.5h (est.) |

### Key Findings (cr035)

1. **Larger covering range helps** — easier for agents to "cover" targets
2. **ER2 comm still outperforms ER1** — 37.5% vs 27.5%
3. **Lower collisions than ms400** — fewer steps = fewer collision opportunities
4. **ER3 cr035 pending** — will show if GNN benefits from easier covering

## Pending Experiments

| Experiment | Config | Status | Purpose |
|---|---|---|---|
| **ER3 cr035** | er3_al_lp_sr_gatv2_cr035.yaml | To run | GNN + larger covering range |
| **ER2 broadcast ms400** | To create | To run | Broadcast comm (dim_c=8, proximity=False) with ms400 |

### ER2 Broadcast (comm_proximity=False)

Previous broadcast run (bugged, actual max_steps=100): M1=0.01 (essentially zero).
Needs re-run with proper ms400 to fairly compare proximity vs broadcast communication.

## Bugged Runs (max_steps=100, for reference)

All runs before 2026-03-22 used actual max_steps=100 due to `config.pop("max_steps")` bug.

| Experiment | M1 (bugged) | M1 (fixed ms400) | Improvement |
|---|---|---|---|
| ER1 AL+LP+SR | 0.040 | 0.405 | 10x |
| ER2 AL+LP+SR prox dc8 | 0.045 | 0.530 | 12x |
| ER3 GATv2 (264 params) | 0.000 | — | — |
| ER3 GATv2 (78K params) | — | 0.710 | (new architecture) |
| ER2 broadcast dc8 | 0.010 | pending | — |

## GNN Architecture Details (ER3)

### Old (broken): 264 parameters
```
obs(31) → GATv2Conv(31→2) → actions(2)
```
Single bare GNN layer, no MLP pre/post. Policy never converged (entropy stayed high).

### New (fixed): 78,444 parameters
```
obs(31) → GATv2Conv(from_pos, hidden=148) → MLP(148→148) → MLP(148→actions)
```
- **Topology**: from_pos (distance-based edges, edge_radius=lidar_range=0.35)
- **Dict observations**: scenario returns {observation, pos, vel} enabling spatial edge features
- **Architecture**: SequenceModelConfig(GNN → MLP → MLP)
- **GNN hidden size**: 148 (chosen to match MLP's ~75K param count)

### Why GNN with from_pos works

1. **Spatial edge features**: GATv2 attention uses relative positions between agents to weight messages
2. **Sufficient capacity**: 78K params (vs 264) can learn complex coordination policies
3. **Proximity-based graph**: only nearby agents exchange messages — matches physical communication constraints
4. **Weight sharing**: same GNN weights applied to all agents — permutation equivariant

## Shared Configuration (all experiments)

### Task Parameters
| Parameter | Value |
|---|---|
| n_agents | 4 |
| n_targets | 4 |
| agents_per_target | 2 |
| lidar_range | 0.35 |
| covering_range | 0.25 (ms400) / 0.35 (cr035) |
| max_steps | 400 (ms400) / 200 (cr035) |
| use_agent_lidar | True |
| n_lidar_rays_entities | 15 |
| n_lidar_rays_agents | 12 |
| shared_reward | True |
| agent_collision_penalty | -0.01 |
| covering_rew_coeff | 1.0 |
| time_penalty | -0.01 |
| targets_respawn | False |

### Training Hyperparameters
| Parameter | Value |
|---|---|
| algorithm | MAPPO |
| max_n_frames | 10,000,000 |
| gamma | 0.99 |
| lambda (GAE) | 0.95 |
| learning_rate | 5e-5 |
| frames_per_batch | 60,000 |
| n_envs_per_worker | 600 |
| n_minibatch_iters | 45 |
| minibatch_size | 4,096 |
| share_policy_params | True |
| evaluation_interval | 120,000 frames |
| evaluation_episodes | 200 |
| seed | 0 |

### Observation Space (per agent)
| Component | Dims | Description |
|---|---|---|
| Position | 2 | Agent x, y |
| Velocity | 2 | Agent vx, vy |
| Target lidar | 15 | 15 rays, range to targets |
| Agent lidar | 12 | 12 rays, range to other agents |
| **Total** | **31** | Flat tensor (MLP) or dict with pos/vel keys (GNN) |

### Action Space
| Component | Dims | Description |
|---|---|---|
| Force | 2 | Continuous [fx, fy], range [-1, 1] |

## Run Inventory

### ms400 experiments
| Run | Family | Folder | Duration |
|---|---|---|---|
| ER1 ms400 | er1 | 20260322_2033__er1_al_lp_sr_ms400_mappo_n4_t4_k2_l035_s0 | 1h 34m |
| ER2 ms400 | er2 | 20260322_2209__er2_al_lp_sr_prox_dc8_ms400_mappo_n4_t4_k2_l035_s0 | 1h 44m |
| ER3 ms400 | er3 | 20260323_0949__er3_al_lp_sr_gatv2_ms400_mappo_n4_t4_k2_l035_s0 | ~8.5h |

### cr035 experiments
| Run | Family | Folder | Duration |
|---|---|---|---|
| ER1 cr035 | er1 | 20260323_0139__er1_al_lp_sr_cr035_mappo_n4_t4_k2_l035_s0 | 1h 28m |
| ER2 cr035 | er2 | 20260323_0307__er2_al_lp_sr_prox_dc8_cr035_mappo_n4_t4_k2_l035_s0 | 1h 45m |
| ER3 cr035 | er3 | pending | ~8.5h (est.) |

### Bugged ms100 runs (for reference only)
| Run | Family | Folder | M1 |
|---|---|---|---|
| ER1 ms100_bugged | er1 | 20260322_1304__er1_al_lp_sr_ms100_bugged_mappo_n4_t4_k2_l035_s0 | 0.040 |
| ER2 ms100_bugged | er2 | 20260322_1431__er2_al_lp_sr_prox_dc8_ms100_bugged_mappo_n4_t4_k2_l035_s0 | 0.045 |
| ER3 ms100_bugged | er3 | 20260322_1616__er3_al_lp_sr_gatv2_ms100_bugged_mappo_n4_t4_k2_l035_s0 | 0.000 |
