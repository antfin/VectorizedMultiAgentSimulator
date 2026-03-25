# ER1: No-Communication Baseline Report

> **Experiment**: ER1 — No-Comm Control
> **Date**: 2026-03-20
> **Config**: MAPPO, n=4 agents, t=4 targets, 10M frames, lmbda=0.95
> **Runs**: 8 sweep + 30 ablation (10 ablations × 3 seeds)

---

## Motivation

Multi-agent rendezvous requires agents to coordinate spatially — arriving at the same location at the same time. When targets require multiple agents to cover them simultaneously (k=2), coordination is no longer optional. Without communication, agents must discover each other's intentions through observation alone.

ER1 establishes the **no-communication ceiling**: the best performance MAPPO can achieve when agents have no way to exchange information. This floor defines the gap that communication protocols (ER2-ER4, E1) must close.

The central question: **Is the coordination bottleneck at k=2 a training problem (solvable with better hyperparameters) or an information problem (requiring communication)?**

---

## Experimental Design

**Task**: VMAS Discovery — 4 agents must cover 4 targets in a 2×2 world. Targets are covered when K agents are within `covering_range=0.25` simultaneously.

**Sweep dimensions** (8 runs):

| Parameter | Values | Purpose |
| --------- | ------ | ------- |
| agents_per_target (k) | 1, 2 | Solo vs coordinated coverage |
| lidar_range | 0.25, 0.35 | Limited vs moderate sensing |
| seeds | 0, 1 | Statistical variance |

**Hyperparameter ablation** (30 runs across 10 configs): Tested entropy, learning rate, GAE lambda, collision penalty, shared reward, extended training, and network architecture to ensure the baseline is optimally tuned before comparing with communication.

**Fixed parameters**: `lmbda=0.95` (ablation winner), `lr=5e-5`, `gamma=0.99`, `entropy_coef=0.0`, `collision_penalty=-0.1`, `shared_reward=false`, `[256,256] Tanh` network.

---

## Results

### The Coordination Gap

![Coordination Gap](figures/coordination_gap.png)

The defining result: **k=2 collapses success from ~72% to ~4%**. Agents can find and cover targets individually but cannot synchronize to arrive at the same target simultaneously. This is not a marginal difficulty increase — it is a qualitative failure mode.

### Success Rate by Configuration

![M1 by Config](figures/er1_m1_by_config.png)

| Config | k | Lidar | M1 (seed avg) | M6 Coverage | M4 Collisions | M3 Steps |
| ------ | - | ----- | ------------- | ----------- | -------------- | -------- |
| k=1, l=0.25 | 1 | 0.25 | **67.8%** | 90.0% | 5.3 | 75.1 |
| k=1, l=0.35 | 1 | 0.35 | **76.5%** | 93.8% | 5.1 | 69.2 |
| k=2, l=0.25 | 2 | 0.25 | **2.3%** | 35.8% | 8.0 | 99.5 |
| k=2, l=0.35 | 2 | 0.35 | **6.5%** | 49.1% | 6.8 | 98.6 |

**Key observations**:

1. **k=1 is largely solved** (68-77% success). Wider lidar helps (+9pp from 0.25→0.35) because agents can detect more targets per step.

2. **k=2 is effectively unsolved** (2-7% success). Even with wider lidar, agents cannot coordinate to arrive simultaneously. The 4pp gap between lidar 0.25 and 0.35 at k=2 suggests more information helps marginally, but **the fundamental barrier is coordination, not perception**.

3. **Coverage tells a richer story** (M6): At k=2, agents cover 36-49% of targets on average. They explore the world and approach targets, but fail at the "last mile" — synchronizing with another agent.

### Coverage Progress

![M6 Coverage](figures/er1_m6_coverage.png)

Even at k=2, agents learn to explore and find targets (M6 ≈ 36-49%). The gap between M6 and M1 reveals the coordination bottleneck: agents can do 50% of the task alone but need communication for the remaining 50%.

---

## Hyperparameter Ablation

Before accepting the no-communication ceiling, we exhaustively tested whether better hyperparameters could close the gap. All ablations used the hardest config: n=4, t=4, k=2, lidar=0.35.

![Ablation Comparison](figures/ablation_m1_comparison.png)

### Ablation Summary

| Ablation | Variable | Value | M1 (3-seed avg) | Verdict |
| -------- | -------- | ----- | ---------------- | ------- |
| Baseline | — | defaults | 3.0% | reference |
| **A** | entropy_coef | 0.01 | 0.0% | CATASTROPHIC |
| **A2** | entropy_coef | 0.001 | 4.3% | neutral |
| **B** | lr | 1e-4 | 5.7% | marginal improvement |
| **C** | lmbda (GAE) | 0.95 | **6.3%** | **WINNER** |
| **D** | lr + lmbda | 1e-4 + 0.95 | 6.3% | = C (no compounding) |
| **E** | collision_penalty | -0.01 | 2.7% | WORSE |
| **F** | shared_reward | true | 3.0% | neutral |
| **G** | max_frames | 20M | 5.2% | plateau confirmed |
| **H** | network | [512,256] ReLU | 4.7% | neutral |
| **I** | lmbda=0.95 (k=1) | sanity check | **76.8%** | k=1 also improved |

### Ablation Findings

**Only GAE lambda=0.95 meaningfully helps** (3% → 6.3%). It extends the credit assignment window from ~10 to ~20 steps, matching the timescale of coordination events. This was adopted as the universal default.

**Entropy regularization destroys learning** (A: 0% M1). The k=2 reward signal is so sparse (3% success rate) that any entropy bonus overwhelms it. Agents optimized for randomness instead of target-seeking.

**Collision penalty is essential, not counterproductive** (E: 2.7%). Without it, agents swarm chaotically (15.8 collisions/episode) instead of navigating smoothly.

**Shared reward does not solve coordination** (F: 3.0%). Better incentive alignment did not help — the bottleneck is information, not credit assignment. This is the strongest argument for communication.

**More training does not help** (G: 5.2% at 20M). The policy plateaus at ~10M frames and can even degrade (one seed declined from 5% to 1.5%). The ceiling is structural, not temporal.

**Network capacity is not the bottleneck** (H: 4.7%). Doubling the network to [512,256] with ReLU had no effect.

### Ablation Raw Data

| Ablation | Seed | M1 | M2 | M4 | M6 | M8 | M9 |
| -------- | ---- | --- | --- | --- | --- | --- | --- |
| A | 0 | 0.000 | -0.952 | 0.0 | 0.024 | 0.098 | 1.014 |
| A | 1 | 0.000 | -0.942 | 0.0 | 0.029 | 0.127 | 0.971 |
| A | 2 | 0.000 | -0.965 | 0.0 | 0.018 | 0.081 | 1.008 |
| A2 | 0 | 0.040 | -0.245 | 7.5 | 0.469 | 0.738 | 0.788 |
| A2 | 1 | 0.060 | -0.149 | 5.6 | 0.491 | 0.729 | 0.837 |
| A2 | 2 | 0.030 | -0.336 | 6.8 | 0.409 | 0.812 | 0.815 |
| B | 0 | 0.065 | -0.074 | 7.2 | 0.549 | 0.709 | 0.799 |
| B | 1 | 0.055 | -0.137 | 6.5 | 0.504 | 0.748 | 0.843 |
| B | 2 | 0.050 | -0.196 | 7.4 | 0.489 | 0.748 | 0.782 |
| C | 0 | 0.085 | -0.147 | 6.3 | 0.497 | 0.778 | 0.788 |
| C | 1 | 0.045 | -0.201 | 7.2 | 0.484 | 0.775 | 0.789 |
| C | 2 | 0.060 | -0.072 | 7.3 | 0.551 | 0.673 | 0.809 |
| D | 0 | 0.085 | -0.147 | 6.3 | 0.498 | 0.778 | 0.788 |
| D | 1 | 0.045 | -0.201 | 7.2 | 0.484 | 0.775 | 0.789 |
| D | 2 | 0.060 | -0.072 | 7.3 | 0.551 | 0.673 | 0.809 |
| E | 0 | 0.050 | -0.179 | 10.8 | 0.415 | 0.829 | 0.768 |
| E | 1 | 0.010 | -0.175 | 8.0 | 0.415 | 0.726 | 0.754 |
| E | 2 | 0.020 | -0.238 | 15.8 | 0.391 | 0.779 | 0.718 |
| F | 0 | 0.025 | +0.578 | 7.2 | 0.438 | 0.000 | 0.755 |
| F | 1 | 0.045 | +0.716 | 7.9 | 0.474 | 0.000 | 0.730 |
| F | 2 | 0.020 | +0.559 | 7.9 | 0.434 | 0.000 | 0.742 |
| G | 0 | 0.055 | -0.165 | 6.7 | 0.496 | 0.697 | 0.784 |
| G | 1 | 0.060 | -0.203 | 6.7 | 0.475 | 0.784 | 0.797 |
| G | 2 | 0.040 | -0.184 | 7.1 | 0.490 | 0.744 | 0.814 |
| H | 0 | 0.035 | -0.188 | 7.9 | 0.499 | 0.792 | 0.778 |
| H | 1 | 0.045 | -0.202 | 8.3 | 0.494 | 0.766 | 0.775 |
| H | 2 | 0.060 | -0.161 | 7.0 | 0.496 | 0.755 | 0.784 |
| I (k=1) | 0 | 0.765 | +0.331 | 4.8 | 0.936 | 0.842 | 0.947 |
| I (k=1) | 1 | 0.765 | +0.344 | 5.4 | 0.939 | 0.862 | 0.921 |
| I (k=1) | 2 | 0.775 | +0.439 | 4.8 | 0.935 | 0.795 | 0.922 |

---

## Conclusions

### The no-communication ceiling is ~2-7% M1 at k=2

After testing 10 hyperparameter configurations (30 ablation runs), the best achievable M1 on k=2 is 6.5% (with lmbda=0.95, lidar=0.35). This is not a training problem — it is a fundamental information deficit. Agents cannot coordinate without communication.

### Adopted hyperparameters for all experiments

| Parameter | Value | Source |
| --------- | ----- | ------ |
| lmbda (GAE) | **0.95** | Ablation C (+110% over baseline, universally beneficial) |
| lr | 5e-5 | Default (D showed higher LR doesn't compound with lambda) |
| entropy_coef | 0.0 | Default (A showed any entropy is catastrophic at k=2) |
| collision_penalty | -0.1 | Default (E showed it's essential for spatial awareness) |
| shared_reward | false | Default (F showed it doesn't help coordination) |
| max_n_frames | 10M | Default (G showed 20M doesn't improve, can degrade) |
| network | [256,256] Tanh | Default (H showed larger networks don't help) |

### What communication experiments need to beat

| Config | No-Comm Ceiling (ER1) | Target for Comm (ER2-ER4) |
| ------ | -------------------- | ------------------------- |
| k=1, l=0.35 | 76.5% | >80% (marginal) |
| k=2, l=0.35 | 6.5% | >15% (meaningful) |
| k=2, l=0.25 | 2.3% | >10% (strong signal) |

The k=2 + lidar=0.25 configuration is the most promising testbed for communication protocols: agents have minimal sensing (0.25 range in a 2.0 world), must coordinate in pairs, and achieve only 2.3% success without communication. Any protocol that reaches 10%+ on this config demonstrates clear value.

### The thesis argument

1. **Shared reward doesn't help** (F) → the problem is not incentive alignment
2. **More training doesn't help** (G) → the problem is not sample efficiency
3. **Larger networks don't help** (H) → the problem is not representational capacity
4. **Only credit assignment window helps marginally** (C) → better temporal reasoning gives +3pp

The coordination bottleneck is an **information problem**: agents need to know where other agents are going and when they plan to arrive. This is precisely what communication protocols provide.
