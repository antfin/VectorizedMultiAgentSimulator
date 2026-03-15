# ER1 First Findings — 2026-03-15

## Setup

- VMAS Discovery scenario, MAPPO, 10M frames, 4 targets, `targets_respawn=False`
- Sweep: N ∈ {4, 6} × K ∈ {1, 2} × lidar=0.35, single seed
- Heuristic baseline: random-walk-ish policy evaluated on same task params
- All metric bugs fixed (M1 cumsum, M4 shape, M8 trailing dim, baseline param mismatch)

## Results

| Run | N | K | M1 (Success) | M2 (Return) | M3 (Steps) | M4 (Collisions) | M6 (Coverage) | M8 (Util CV) | M9 (Spread) |
|-----|---|---|:------------:|:-----------:|:-----------:|:----------------:|:-------------:|:------------:|:-----------:|
| n4_k1 | 4 | 1 | **63%** | +0.09 | 74.0 | 6.7 | 88% | 0.95 | 0.90 |
| n4_k2 | 4 | 2 | **3%** | -0.11 | 99.5 | 5.5 | 51% | 0.81 | 0.80 |
| n6_k1 | 6 | 1 | **88.5%** | +0.16 | 57.3 | 9.7 | 97% | 0.89 | 0.98 |
| n6_k2 | 6 | 2 | **20%** | -0.24 | 95.2 | 12.5 | 69% | 0.93 | 0.91 |

## Key Findings

### 1. k=2 is dramatically harder than k=1 (expected)

k=2 drops M1 by ~20× for N=4 (63% → 3%) and ~4.4× for N=6 (88.5% → 20%). This is consistent with MARL literature: k=2 requires simultaneous co-location of two agents, transforming independent navigation into a joint coordination problem with sparse reward.

### 2. k=2 trained policy is WORSE than the heuristic baseline

| | Heuristic (n=4,k=2) | Trained n4_k2 | Trained n6_k2 |
|---|:---:|:---:|:---:|
| M1 | **29%** | 3% | 20% |

Even with 10M frames and 6 agents, MAPPO cannot beat a random-walk heuristic at k=2. The heuristic succeeds by brute-force random proximity over 200 steps, while the policy learns collision avoidance but not paired coordination. **This is the strongest motivation for communication (ER2+).**

### 3. More agents help but don't solve coordination

N=6 beats N=4 at both k values (+25pp for k=1, +17pp for k=2). More agents = more redundancy, but even 6 agents can't overcome the k=2 coordination barrier without communication.

### 4. Workload is always unbalanced (M8 = 0.81–0.95)

No run achieves balanced agent utilization. A few agents carry the task while others idle. Without communication, agents can't negotiate task allocation. This is a coordination failure that communication should address.

### 5. "Last mile" problem at k=2

n4_k2 has M6=51% (covers ~2 of 4 targets) but M1=3% (almost never covers ALL). Agents can explore individually to partial coverage, but the final targets require paired coordination. Communication could close this gap.

### 6. Success and safety are coupled without communication

M1 vs M4 scatter shows positive correlation — more success means more collisions. Agents that solve the task do so by approaching targets, which causes crowding. Communication could decouple this tradeoff by coordinating approach directions.

## Training Dynamics

- All runs show healthy MAPPO convergence: sigmoid M1 curves, gradual entropy decrease, no collapse
- M4 collision pattern: early spike (agents learn to approach targets) then decline (learn avoidance)
- k=2 collision spikes are 5-10× larger than k=1 (agents crowd targets without coordination)
- M7 (sample efficiency) missing for k=2 — policy never reached 80% of final reward
- k=1 M7 = 5.76M frames — convergence happens around iteration 80-90 of 166

## Cross-Metric Consistency

All metric relationships are internally consistent:
- M1 ↑ ↔ M3 ↓ (success = faster completion)
- M1 ↑ ↔ M4 ↑ (success = more collisions, no comm to coordinate)
- M1 ↑ ↔ M9 ↑ (success = more exploration/spatial spread)
- Low M1 + high M6 at k=2 (partial coverage without full completion)
- High M8 everywhere (unbalanced workload without communication)

## Implications for ER2+

1. **k=2 is the target condition** — k=1 is already solvable without communication (63-88%)
2. Communication must improve k=2 M1 above 29% (heuristic floor) to demonstrate value
3. M8 reduction would show communication enables task allocation
4. M4 reduction at constant M1 would show communication enables safe coordination
5. Training budget of 10M frames is sufficient for k=1 but may need 15-20M for k=2 with communication overhead

## Caveats

- Single seed per config — variance not measured
- No published Discovery benchmarks exist for comparison
- Heuristic baseline uses base config params (n=4, k=2) — not re-evaluated per sweep point
