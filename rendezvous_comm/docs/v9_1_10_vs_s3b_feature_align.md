# Feature alignment: v9.1.10 best vs S3b-local best

**Date:** 2026-05-04
**v9.1.10 best:** outer 4, iter 1, cand 0 — M1=0.010, M6=0.244, fitness=+0.108
**S3b-local best:** seed s1, iter 0, cand 0 — M1=0.070, M6=0.435 (1M-eval)

Each row = same conceptual signal. Empty cell = the feature is unique to one side.

| # | concept | S3b-local (17 feats) | v9.1.10 (18 feats) | match? |
|---|---|---|---|---|
| 1 | nearest target distance | `min_t_dist = lidar_t.min(dim=-1).values` | `d_t = lidar_t.min(dim=-1).values` | **identical** |
| 2 | nearest agent distance | `min_a_dist = lidar_a.min(dim=-1).values` | `d_a = lidar_a.min(dim=-1).values` | **identical** |
| 3 | count of close targets | `n_close_targets = (lidar_t < 0.25).sum(-1)` | `t_close = (lidar_t < cover_r).sum(-1)` | **identical (cover_r=0.25)** |
| 4 | count of close agents | `n_close_agents = (lidar_a < 0.25).sum(-1)` | `a_close = (lidar_a < cover_r).sum(-1)` | **identical** |
| 5 | self-speed | `speed = torch.linalg.norm(agent_vel, -1)` | `speed = agent_vel.norm(dim=-1)` | **identical** |
| 6 | soft proximity to target | `t_prox = torch.exp(-3.0 * min_t_dist)` | `soft_t = torch.exp(-3.0 * d_t)` | **identical** |
| 7 | direction-x to nearest target | `t_dir_x = cos(min_t_angle)` | (only via `target_dir_align`, see #11) | only S3b |
| 8 | direction-y to nearest target | `t_dir_y = sin(min_t_angle)` | — | only S3b |
| 9 | direction-x to nearest agent | `a_dir_x = cos(min_a_angle)` | (only via `target_dir_align`) | only S3b |
| 10 | direction-y to nearest agent | `a_dir_y = sin(min_a_angle)` | — | only S3b |
| 11 | target/agent angular alignment | (computable from #7-#10 implicitly) | `target_dir_align = cos(t_ang - a_ang)` | only v9.1.10 (DERIVED form) |
| 12 | crowd / agent-density signal | `crowd = clamp(n_close_agents / 3, 0, 1)` (saturated) | `density_gap = t_close - a_close` (signed) | DIFFERENT — saturated vs signed |
| 13 | velocity direction | `vel_x = cos(vel_dir)`, `vel_y = sin(vel_dir)` | — | only S3b (2 features) |
| 14 | rendezvous opportunity score | — | `rendezvous_score = soft_t * (1 + relu(density_gap))` | only v9.1.10 |
| 15 | "free target" / overlap-avoid signal | — | `avoid_overlap = sigmoid(2.5*(d_a - d_t))` | only v9.1.10 |
| 16 | safe commit gate | — | `safe_commit = sigmoid(3*(0.3-d_t)) * sigmoid(3*wall)` | only v9.1.10 |
| 17 | density × proximity interaction | — | `soft_t * (1 + a_close)` | only v9.1.10 |
| 18 | wall × speed interaction | — | `wall * (1 + speed)` | only v9.1.10 |
| 19 | boundary distance | — | `wall = 1 - pos.abs().max(-1)` | only v9.1.10 |
| 20 | role one-hot (n_agents=4 dims) | `one_hot = torch.zeros(B,n_agents); one_hot[:, agent_idx]=1` | `role = torch.zeros(B,n_agents); role[:, agent_idx]=1` | **identical** |

## Aggregate

| | S3b-local | v9.1.10 |
|---|---|---|
| Total features | 17 | 18 |
| Shared / equivalent | 6 (rows 1-6, 20) → 7 of 17 | 6 (rows 1-6, 20) → 7 of 18 |
| Only S3b-local | rows 7-10 (target/agent dir cos/sin), 13 (vel direction) | — |
| Only v9.1.10 | rows 11, 14-19 (composite + boundary) | — |

## Signal-class summary

| signal class | S3b-local | v9.1.10 |
|---|---|---|
| **proximity** (raw + soft) | min_t, min_a, t_prox | d_t, d_a, soft_t |
| **density** (count + reduction) | n_close_t, n_close_a, crowd (saturated) | t_close, a_close, density_gap (signed) |
| **direction** (raw cos/sin) | 4 dims (t_dir_x, t_dir_y, a_dir_x, a_dir_y) | 0 raw — 1 derived (target_dir_align) |
| **velocity direction** | 2 dims (vel_x, vel_y) | 0 |
| **velocity magnitude** | 1 (speed) | 1 (speed) |
| **boundary** | 0 | 1 (wall) |
| **composite gates** | 0 | 4 (rendezvous_score, avoid_overlap, safe_commit, soft_t*a_close, wall*speed) |
| **role identity** | 4 (one_hot) | 4 (role) |

## Verdict

- **6 features are bit-for-bit identical** (proximity raw/soft for both lidars, count both lidars, speed, role one-hot).
- **3 features are conceptually equivalent but in different forms** (crowd: saturated vs signed; direction: 4 raw dims vs 1 alignment; vel direction: 2 dims vs 0).
- **6 features are unique to v9.1.10** (composite gates encoding the strategy: rendezvous_score, avoid_overlap, safe_commit, plus 2 interactions, plus boundary distance).

The structural "missing" features in v9.1.10 vs S3b-local are: **target direction (cos/sin)**, **agent direction (cos/sin)**, **velocity direction (cos/sin)**. Six raw directional dims that S3b-local has and v9.1.10 lacks.

The v9.1.10 candidate substitutes these with **engineered composites** (rendezvous_score, avoid_overlap, safe_commit) that the LLM derived per the strategy intent. These composites encode strategy-specific decisions but lose raw directional information that PPO might prefer to combine on its own.

**Hypothesis for the M1 gap (0.010 vs 0.070):**

PPO at 1M frames may benefit more from raw directional features (which the policy network can combine in many ways) than from pre-engineered composite gates (which lock in a single decision rule). v9.1.10's composites encode helpful prior knowledge but reduce the representational space the policy can search. S3b-local's flatter, lower-level features give PPO more freedom.

**Cheap experiment to test this:** add target_dir_x/y, agent_dir_x/y, vel_x/y back to the v9.1.10 mandatory_features as 6 raw directional dims (or to the inferable_concepts as recommended). Re-run 1 outer × 3 cands at 1M and check M1.
