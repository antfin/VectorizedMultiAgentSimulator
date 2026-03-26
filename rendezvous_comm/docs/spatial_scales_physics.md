# Spatial Scales and Physics Reference

This document details the physics simulation parameters and spatial scale relationships in the VMAS Discovery scenario, explaining why certain experimental conditions (ms200 vs ms400, cr025 vs cr035) produce dramatically different results.

## 1. World Geometry

| Parameter | Value | Description |
| --- | --- | --- |
| x_semidim | 1.0 | Half-width of world |
| y_semidim | 1.0 | Half-height of world |
| World bounds | [-1, 1] x [-1, 1] | Full 2D space |
| World size | 2.0 x 2.0 units | Total area = 4.0 sq units |
| World diagonal | ~2.83 units | Corner-to-corner distance |

## 2. Agent and Target Geometry

| Parameter | Value | Source |
| --- | --- | --- |
| Agent radius | 0.05 | `core.py` default |
| Target radius | 0.05 | `discovery.py` |
| min_dist_between_entities | 0.2 | Minimum spawn separation |
| covering_range (default) | 0.25 | Distance to "cover" a target |
| covering_range (relaxed) | 0.35 | Used in cr035 experiments |

## 3. Sensing Ranges

| Parameter | Value | Ratio to world diagonal |
| --- | --- | --- |
| lidar_range (entities) | 0.35 | 12.4% (sees ~1/8 of diagonal) |
| comms_range | 0.35 | Same as lidar_range |
| n_lidar_rays_entities | 15 | Angular resolution for targets |
| n_lidar_rays_agents | 12 | Angular resolution for other agents (when AL=true) |

An agent's LiDAR cone covers roughly 1/8 of the world diagonal. This means agents must actively explore to discover targets — they cannot see the full environment from any single position.

## 4. Physics Integration

VMAS uses Euler integration with drag-based velocity damping:

| Parameter | Value | Source |
| --- | --- | --- |
| dt | 0.1s | Main timestep |
| substeps | 2 | Physics substeps per dt |
| sub_dt | 0.05s | Integration interval |
| drag | 0.25 | Velocity damping per substep |
| collision_force | 500 | Contact resolution force |
| max_speed | None | No velocity cap (Discovery default) |
| max_force | None | No force cap |

**Dynamics:** Holonomic (direct force control in x,y). Action u in [-1, 1] maps to force.

**Per substep integration:**

1. Drag: `vel *= (1 - drag)` = `vel *= 0.75`
2. Acceleration: `accel = force / mass` (mass = 1.0)
3. Velocity: `vel += accel * sub_dt`
4. Position: `pos += vel * sub_dt`

**Steady-state maximum velocity** (sustained max force u=1.0):

At equilibrium, drag removes as much velocity as force adds:
`v_steady = force * sub_dt / drag = 1.0 * 0.05 / 0.25 = 0.2 units/step`

In practice, agents accelerate from rest and rarely sustain max force in one direction, so typical movement is much less — roughly **0.01 units/step** in the first few steps from standstill, ramping up to ~0.1 after sustained acceleration.

## 5. Spatial Scale Relationships

| Quantity | Value | Steps to traverse | % of world width |
| --- | --- | --- | --- |
| Agent diameter | 0.10 | ~1 step | 5% |
| covering_range (cr025) | 0.25 | ~25 steps | 12.5% |
| covering_range (cr035) | 0.35 | ~35 steps | 17.5% |
| LiDAR range | 0.35 | ~35 steps | 17.5% |
| World width | 2.0 | ~200 steps | 100% |
| World diagonal | 2.83 | ~280 steps | 141% |

"Steps to traverse" assumes typical acceleration from rest (~0.01 units/step initially).

## 6. Why ms200 is Tight for k=2

At ~200 steps to cross the world width, the episode budget barely allows an agent to traverse the environment once. For k=2 coordination:

1. **Target discovery:** An agent must explore until its LiDAR (range 0.35, covering ~12% of the world) detects a target. With 4 targets randomly placed in a 2x2 world, an agent needs ~50-100 steps of exploration to find most targets.

2. **Convergence:** Once a target is found, a second agent must also arrive within covering_range (0.25). If the second agent is on the other side of the world (~2.0 units away), it needs ~200 steps to arrive — consuming the entire episode.

3. **Simultaneity:** Both agents must be within 0.25 of the target at the SAME time. If agent A arrives first and moves on, agent B arriving later doesn't count. This synchronization is the core coordination challenge.

4. **Sequential covering:** With 4 targets, the team must repeat this process 4 times. At ms200, there is almost no margin for error.

**ms400 doubles the budget**, giving agents time to:

- Explore more of the environment
- Wait for partners to arrive at targets
- Recover from suboptimal initial movements
- Cover all 4 targets sequentially

This explains the dramatic jump: ER1 goes from 4% (ms200) to 40.5% (ms400), a 10x improvement from doubling episode length.

## 7. Why cr035 Helps

| Metric | cr025 | cr035 | Change |
| --- | --- | --- | --- |
| Covering area per target | 0.196 sq units | 0.385 sq units | **+96%** (nearly 2x) |
| % of world area | 4.9% | 9.6% | +4.7pp |
| Diameter of covering zone | 0.50 | 0.70 | +40% |

With cr035, the covering zone area nearly doubles. This means:

- Two agents have a **40% wider spatial window** to overlap — they don't need to be as precisely synchronized
- The probability of accidental simultaneous coverage increases significantly
- Agents can approach from different angles and still both be "within range"

This explains why cr035 experiments consistently outperform cr025 across all communication types:

| Condition | cr025 | cr035 | Improvement |
| --- | --- | --- | --- |
| ER1 (no comm, ms200) | 4.0% | 27.5% | +23.5pp |
| ER2 proximity (ms200) | 4.5% | 37.5% | +33.0pp |
| ER2 broadcast (ms200) | 1.0% | 48.5% | +47.5pp |
| ER3 GNN (ms200) | 0% | 36.5% | +36.5pp |

## 8. Scale Diagram

```
World: 2.0 x 2.0 units
+--------------------------------------------------+
|                                                  |  <- world boundary
|         (T)  target                              |
|          .    .                                  |
|         .  cr .   <- covering_range = 0.25       |
|        . 025  .                                  |
|         .    .                                   |
|          ....                                    |
|                                                  |
|     [A]---0.35----> LiDAR range                  |
|     agent                                        |
|     (0.05 radius)                                |
|                                                  |
|     Per step from rest: ~0.01 units              |
|     Steady-state max:   ~0.10 units/step         |
|                                                  |
+--------------------------------------------------+
  |<-------------- 2.0 units ---------------->|
  ~200 steps to cross at typical speed
```
