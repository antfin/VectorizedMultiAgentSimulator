# v9.1.10 lineage — prompts, observations, evolution, S3b-local comparison

**Date:** 2026-05-04
**v9.1.10 run:** `results/lero_v9/lero_v9_rendezvous_k2_2x3/20260504_1151_s0`
**S3b-local reference prompt:** `src/lero/prompts/v2_fewshot_k2_local/`
**S3b-local results:** `results/s3b_local_replicate/lero_s3b_local_{s0,s1,s2}/`

This doc walks the FULL prompt + observation evolution of v9.1.10:
- **Outer-level evolution**: how each strategy's slot text was authored, refined, or switched
- **Inner-level evolution**: how candidates evolved across the 3 iters inside each outer
- **Side-by-side with S3b-local**: at each step, what would S3b-local have done

---

## Part 1 — S3b-local reference (the hand-crafted prompt)

S3b-local uses ONE static prompt (no meta-LLM, no slot edits, no
strategy switches). Same prompt drives all 4 iters × 3 cands × 3 seeds.

### S3b-local system message

```
You are a reward engineer designing observation enhancement functions for multi-agent reinforcement learning (MARL) tasks. Your objective is to create an observation enhancement function that helps agents coordinate using ONLY their LOCAL sensor data.

CRITICAL: This is a RENDEZVOUS task where k=$agents_per_target agents must simultaneously occupy each target. Agents can only see via LiDAR — they do NOT know other agents' positions, target positions, or which targets are covered. Design features that help agents infer coordination state from sensor readings alone.

All operations must be vectorized using PyTorch (batch dimension first). Only use torch, torch.nn.functional, math.
```

### S3b-local user prompt — key features

**S3b-local 'What you CAN infer':**

### What you CAN infer from LiDAR

- **Direction to nearest target:** the ray with smallest value points toward the closest target
- **Distance to nearest target:** min of lidar_targets rays
- **Number of nearby targets:** count of rays below a threshold (e.g. < covering_range)
- **Agent density nearby:** from lidar_agents, count of rays below threshold → how crowded the area is
- **Angular distribution:** which directions have targets vs agents → helps decide where to go
- **Role differentiation:** agent_idx as one-hot so the shared policy can assign different roles

**S3b-local worked example (gives the LLM a complete template):**

### Example — local observation enhancement

```python
def enhance_observation(scenario_state: dict) -> torch.Tensor:
    lidar_t = scenario_state["lidar_targets"]       # [B, 15]
    agent_idx = scenario_state["agent_idx"]
    n_agents = int(scenario_state["n_agents"])
    B = lidar_t.shape[0]
    device = lidar_t.device

    # Direction to nearest target (which ray is closest)
    min_dist, min_ray = lidar_t.min(dim=-1)         # [B], [B]
    # Encode ray index as angle: ray_i / n_rays * 2*pi
    angle = min_ray.float() / lidar_t.shape[-1] * 2 * 3.14159
    dir_x = torch.cos(angle)                         # [B]
    dir_y = torch.sin(angle)                         # [B]

    # Count of nearby targets (within covering range)
    n_close = (lidar_t < 0.25).float().sum(dim=-1)   # [B]

    # One-hot agent index for role specialization
    one_hot = torch.zeros(B, n_agents, device=device)
    one_hot[:, agent_idx] = 1.0

    return torch.cat([
        min_dist.unsqueeze(-1),   # [B,1] — distance to nearest target
        dir_x.unsqueeze(-1),      # [B,1] — direction x
        dir_y.unsqueeze(-1),      # [B,1] — direction y
        n_close.unsqueeze(-1),    # [B,1] — nearby target count
        one_hot,                  # [B,4] — agent role
    ], dim=-1)  # [B, 8]
```

### S3b-local observation profile (n=36 candidates across 3 seeds × 4 iters × 3 cands)

- avg n_features: 15.8
- avg n_cross_source ops: 1.1
- role_one_hot rate: 97%
- soft_prox (`exp(-α·d)`) rate: 39%
- speed feature rate: 89%

---

## Part 2 — v9.1.10 outer-by-outer prompt + strategy lineage

### Outer 0: `role_split_with_pairing`

**How we got here:** cold-start, chosen by bundle enumeration LLM
(score=8.5, picked over 4 alternatives).

**Strategy CoT:**

- **why_it_works:** The reward is driven by completing exact 2-agent coverage per target, so symmetry breaking is essential; otherwise all agents chase the same nearest target. Role identity plus a target-agent joint signal lets the policy represent 'I should go here' versus 'someone else is already here' using only local observations.
- **what_is_needed:** ['role one-hot so shared-policy MAPPO can specialize agents without explicit communication', 'at least one cross-source scalar combining target-derived and agent-derived summaries', 'a smooth distance feature so PPO can learn gradients before entering the cover zone']
- **failure_modes:** ['Without role features, all agents can collapse to the same attractor and never form stable pairs', 'If the cross-source signal is absent, the network cannot distinguish under-served from already-crowded targets', 'If features are too binary or sparse, PPO may not discover the commit/search boundary within 1M frames']

**Predicted vs actual:** expected M1=0.08, M6_min=0.42 → actual M1=0.000, M6=0.187

**Inner-loop evolution (3 iters × 3 cands):**

| iter | cand | M1 | M6 | fit | n_feat | role | soft_prox | n_cs | rejected? |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.0 | 0.187 | +0.079 | 16 | Y | Y | 4 | no |
| 0 | 1 | — | — | — | — | — | — | — | YES: feature_budget exceeded: AST counted 22 features (hard_cap=2 |
| 0 | 2 | 0.0 | 0.0 | +0.000 | 18 | Y | Y | 4 | no |
| 1 | 0 | 0.0 | 0.163 | +0.073 | 19 | Y | Y | 5 | no |
| 1 | 1 | 0.0 | 0.134 | +0.051 | 19 | Y | Y | 5 | no |
| 1 | 2 | — | — | — | — | — | — | — | YES: feature_budget exceeded: AST counted 22 features (hard_cap=2 |
| 2 | 0 | 0.0 | 0.151 | +0.057 | 17 | Y | Y | 5 | no |
| 2 | 1 | 0.01 | 0.17 | +0.047 | 15 | Y | Y | 5 | no |
| 2 | 2 | 0.01 | 0.247 | -0.006 | 15 | Y | Y | 5 | no |

**End-of-outer reflection CoT:**

- **label:** `partial`
- **action:** `refine_current`

**memory_recall:** "outer 0: predicted M1=0.08, actual M1=0.0; predicted M6=0.42, actual M6=0.18714285714285708". This is the first attempt for this strategy, so the falsification rule does not force a switch yet.

**diff_vs_predicted:** M1 missed badly (0.0 vs 0.08), but M6 also fell short of the minimum target (0.1871 vs 0.42). The structural pattern is present: role one-hot exists, cross-source ops exist, and both lidars are touched. So this is not a translation failure; it is a weak-learning outcome.

**what_went_right:** ['The intended structural ingredients are present: role identity, cross-source combination, and dense features over both lidars.', 'The candidate is at least expressively aligned with the symmetry-breaking idea, so the failure is not from missing the core pattern.']

**what_went_wrong:** ['The learned signal is too weak to produce any M1 at all, suggesting the role/crowding signal is not decisive enough.', 'The current formulation likely mixes proximity and crowding too softly, so the policy cannot cleanly commit to a target or infer that another agent should take it.', 'The feature stack is fully dense with no gating, which may blur the role-specific decision boundary instead of sharpening it.']

**rationale:** This is a partial success with the required pattern present, so the correct move is to keep the same strategy and sharpen the slot text. The first attempt is not yet falsified, and the failure looks like weak signal design rather than a missing structural translation. Per the concrete-fix rule, the refinement should specify one code-level change: make the cross-source crowding signal sharper and more decisive, e.g. replace a soft average with an explicit margin or gated contrast between target proximity and nearby-agent density.

---

### Outer 1: `role_split_with_pairing`

**How we got here:** outer 0 was same strategy, label=`partial`, action=`refine_current`. §2.3 validator decisions:
  - applied: []
  - rejected: ['inferable_hints', 'examples']

**Strategy CoT:**

- **why_it_works:** The reward is driven by completing exact 2-agent coverage per target, so symmetry breaking is essential; otherwise all agents chase the same nearest target. Role identity plus a target-agent joint signal lets the policy represent 'I should go here' versus 'someone else is already here' using only local observations.
- **what_is_needed:** ['role one-hot so shared-policy MAPPO can specialize agents without explicit communication', 'at least one cross-source scalar combining target-derived and agent-derived summaries', 'a smooth distance feature so PPO can learn gradients before entering the cover zone']
- **failure_modes:** ['Without role features, all agents can collapse to the same attractor and never form stable pairs', 'If the cross-source signal is absent, the network cannot distinguish under-served from already-crowded targets', 'If features are too binary or sparse, PPO may not discover the commit/search boundary within 1M frames']

**Predicted vs actual:** expected M1=0.08, M6_min=0.42 → actual M1=0.000, M6=0.167

**Inner-loop evolution (3 iters × 3 cands):**

| iter | cand | M1 | M6 | fit | n_feat | role | soft_prox | n_cs | rejected? |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | — | — | — | — | — | — | — | YES: feature_budget exceeded: AST counted 21 features (hard_cap=2 |
| 0 | 1 | 0.0 | 0.0 | +0.000 | 14 | Y | Y | 5 | no |
| 0 | 2 | 0.0 | 0.154 | -0.260 | 16 | Y | Y | 4 | no |
| 1 | 0 | 0.0 | 0.19 | +0.055 | 17 | Y | Y | 6 | no |
| 1 | 1 | 0.0 | 0.179 | +0.054 | 17 | Y | Y | 5 | no |
| 1 | 2 | — | — | — | — | — | — | — | YES: feature_budget exceeded: AST counted 23 features (hard_cap=2 |
| 2 | 0 | — | — | — | — | — | — | — | YES: feature_budget exceeded: AST counted 26 features (hard_cap=2 |
| 2 | 1 | — | — | — | — | — | — | — | YES: feature_budget exceeded: AST counted 28 features (hard_cap=2 |
| 2 | 2 | 0.0 | 0.18 | -0.126 | 16 | Y | Y | 4 | no |

**End-of-outer reflection CoT:**

- **label:** `partial`
- **action:** `switch_to_next`

**memory_recall:** "outer 0: predicted M1=0.08, actual M1=0.0; predicted M6=0.42, actual M6=0.18714285714285708"; "current outer: predicted M1=0.08, actual M1=0.0; predicted M6=0.42, actual M6=0.16714285714285712". The same strategy has now been attempted 2 times, and both actual M1 values are below 0.5× expected_M1 (0.04), so the falsification rule forces switching away from it.

**diff_vs_predicted:** Pattern requirements are satisfied: inner candidate is present, role one-hot is present, and cross-source operations exist (5). However M1 stayed at 0.0 versus expected 0.08, while M6 fell to 0.1671 versus expected 0.42, so this is a weak partial rather than achieved.

**what_went_right:** ['The structural pattern is present: both lidar sources are touched, role identity is included, and there are multiple cross-source ops.', 'The formulation is still aligned with the symmetry-breaking goal, so the failure is not from a missing core concept.']

**what_went_wrong:** ['Despite the right ingredients, the policy never produced any M1, which means the role/crowding signal is not strong enough to create decisive specialization.', 'The feature stack is entirely dense with no gating, so the decision boundary likely stays blurred instead of clearly separating commit versus divert behavior.', 'Two consecutive attempts on the same strategy both underperformed far below the expected M1 threshold, so further refinement of this exact strategy is unlikely to be productive.']

**rationale:** Per the falsification rule, this strategy has now been attempted at least twice and every actual M1 is below half of expected_M1_at_1M. That makes continued refinement non-viable, even though the pattern is present. The correct action is to switch to the next pending strategy in the bundle.

---

### Outer 2: `pair_completion_pressure`

**How we got here:** outer 1 was `role_split_with_pairing`, label=`partial`, action=`switch_to_next`. §2.10 lazy-authored artifacts for this strategy.

**Strategy CoT:**

- **why_it_works:** Exact-k coverage is fundamentally about completing pairs, so a heuristic that detects 'almost ready' targets directly matches the reward structure. Combining target and agent evidence creates a more task-aligned local proxy than raw distance alone.
- **what_is_needed:** ['a cross-source scalar that mixes target closeness and agent crowding', 'a soft distance channel to distinguish near-but-not-yet-covered targets', 'identity features if the policy must choose distinct pair roles under shared parameters']
- **failure_modes:** ['The proxy can misfire when a target is close but already has two agents arriving from different angles', 'If the completion pressure is too coarse, agents may over-focus on one target and starve others', 'Without role differentiation, multiple agents may respond identically to the same pressure signal']

**Predicted vs actual:** expected M1=0.07, M6_min=0.4 → actual M1=0.000, M6=0.219

**Inner-loop evolution (3 iters × 3 cands):**

| iter | cand | M1 | M6 | fit | n_feat | role | soft_prox | n_cs | rejected? |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.0 | 0.219 | +0.083 | 16 | Y | Y | 3 | no |
| 0 | 1 | 0.0 | 0.196 | -0.160 | 17 | Y | Y | 5 | no |
| 0 | 2 | 0.01 | 0.249 | -3.463 | 18 | Y | Y | 5 | no |
| 1 | 0 | — | — | — | — | — | — | — | YES: feature_budget exceeded: AST counted 23 features (hard_cap=2 |
| 1 | 1 | 0.0 | 0.14 | +0.030 | 20 | Y | Y | 3 | no |
| 1 | 2 | 0.0 | 0.0 | +0.000 | 17 | Y | Y | 4 | no |
| 2 | 0 | 0.0 | 0.151 | +0.060 | 20 | Y | Y | 3 | no |
| 2 | 1 | 0.0 | 0.117 | +0.056 | 19 | Y | Y | 3 | no |
| 2 | 2 | 0.0 | 0.137 | +0.048 | 19 | Y | Y | 5 | no |

**End-of-outer reflection CoT:**

- **label:** `partial`
- **action:** `switch_to_next`

**memory_recall:** "outer 0: predicted M1=0.08, actual M1=0.0; outer 1: predicted M1=0.08, actual M1=0.0; this outer: predicted M1=0.07, actual M1=0.0". Two prior attempts on the same strategy both produced actual M1 far below 0.5× expected_M1_at_1M (0.035), and the current inner still has M1=0.0 despite cross-source ops and role one-hot being present.

**diff_vs_predicted:** M1 missed the expected floor again (actual 0.0 vs expected 0.07), while M6=0.2186 is below the minimum expected 0.4. The structural pattern is present: touches both lidars, role one-hot present, and 3 cross-source ops, so this is not a translation failure.

**what_went_right:** ['The candidate preserves the mandatory structural ingredients: role identity, cross-source mixing, and both sensor sources.', 'The heuristic is still conceptually aligned with exact pair formation, so the core task match is not the problem.']

**what_went_wrong:** ['The signal is still too weak to produce any M1, so the policy is not learning a decisive pair-completion boundary.', 'The feature stack is entirely dense with no gating, which likely blurs the commit/divert distinction instead of sharpening it.', 'Three attempts on closely related role-splitting/pairing formulations have already failed to reach even half the expected M1 threshold.']

**rationale:** By the falsification rule, this strategy has now been attempted at least twice and every attempt's actual M1 is below 0.5× expected_M1_at_1M, so I must switch to the next pending strategy. The current inner is structurally valid but empirically too weak, and further refinement would violate the non-negotiable gate.

---

### Outer 3: `soft_commit_and_release`

**How we got here:** outer 2 was `pair_completion_pressure`, label=`partial`, action=`switch_to_next`. §2.10 lazy-authored artifacts for this strategy.

**Strategy CoT:**

- **why_it_works:** The task rewards stable occupancy by exactly two agents, so a soft field can guide agents toward promising targets while a release cue reduces over-concentration. Continuous features are especially helpful because the environment is sparse and the exact cover event is only observed indirectly.
- **what_is_needed:** ['smooth target proximity features from lidar_targets', 'agent-nearby density features from lidar_agents', 'a way to compare target-side and agent-side local evidence']
- **failure_modes:** ['Soft attraction alone can still cause all agents to pile onto the nearest target', 'If release cues are too weak, agents may oscillate near a target and miss coverage windows', 'Without identity, multiple agents may learn the same soft heuristic and fail to diversify']

**Predicted vs actual:** expected M1=0.06, M6_min=0.35 → actual M1=0.010, M6=0.154

**Inner-loop evolution (3 iters × 3 cands):**

| iter | cand | M1 | M6 | fit | n_feat | role | soft_prox | n_cs | rejected? |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.01 | 0.154 | +0.070 | 14 | Y | Y | 4 | no |
| 0 | 1 | 0.0 | 0.139 | +0.052 | 13 | Y | Y | 3 | no |
| 0 | 2 | 0.0 | 0.191 | -0.646 | 14 | Y | Y | 3 | no |
| 1 | 0 | 0.0 | 0.137 | +0.061 | 19 | Y | Y | 5 | no |
| 1 | 1 | 0.01 | 0.174 | +0.016 | 18 | Y | Y | 6 | no |
| 1 | 2 | 0.0 | 0.134 | -0.550 | 17 | Y | Y | 5 | no |
| 2 | 0 | 0.0 | 0.176 | +0.067 | 20 | Y | Y | 4 | no |
| 2 | 1 | 0.0 | 0.164 | +0.066 | 20 | Y | Y | 4 | no |
| 2 | 2 | 0.0 | 0.134 | +0.055 | 20 | Y | Y | 8 | no |

**End-of-outer reflection CoT:**

- **label:** `partial`
- **action:** `switch_to_next`

**memory_recall:** "outer 0: predicted M1=0.08, actual M1=0.0; outer 1: predicted M1=0.08, actual M1=0.0; outer 2: predicted M1=0.07, actual M1=0.0". Across the last three related attempts, the strategy family stayed below half of expected M1 each time, and the current outer again underperformed with M1=0.01 versus expected_M1_at_1M=0.06. Because this is the same role-splitting/pairing lineage and the falsification rule is triggered, the current soft-commit variant should not be refined further.

**diff_vs_predicted:** Predicted performance was modest but nonzero; actual M1=0.01 fell below expected_M1_at_1M=0.06, while M6=0.154 was also well below expected_M6_at_1M_min=0.35. The structural pattern is present: both lidars are touched, role one-hot is present, and there are 4 cross-source ops. The main gap is that the feature stack is fully dense (14 dense, 0 gated), so the commit/release boundary is still too blurred.

**what_went_right:** ['The candidate preserves the required ingredients: role identity, cross-source mixing, and both sensor sources.', 'The inner result is structurally valid rather than a translation failure, so the issue is learning quality rather than missing pattern support.']

**what_went_wrong:** ['The soft attraction plus release idea did not produce enough M1 to matter, indicating the policy still cannot make a decisive commit-versus-divert choice.', 'All features are dense, with no gating, so the learned signal likely remains too smooth to separate under-served targets from saturated ones.', 'This strategy family has now been attempted repeatedly with every actual M1 below half of the expected threshold, so further refinement is unlikely to pay off.']

**rationale:** The falsification rule applies: the same strategy lineage has been attempted at least twice, and every attempt's actual M1 is below 0.5× its expected M1. Therefore I must recommend switching to the next pending strategy instead of refining current. The current inner also shows a translation-like weakness in practice: despite 4 cross-source ops and role one-hot, the all-dense stack did not learn a useful boundary.

---

### Outer 4: `stay_search_switching`

**How we got here:** outer 3 was `soft_commit_and_release`, label=`partial`, action=`switch_to_next`. §2.10 lazy-authored artifacts for this strategy.

**Strategy CoT:**

- **why_it_works:** The hardest part of exact-2 rendezvous is deciding when to remain at a target after arriving; a dedicated switch directly addresses that bottleneck. The combination of local target distance and nearby-agent density provides the minimum information needed for that binary decision.
- **what_is_needed:** ['one-hot agent identity to let different agents occupy complementary roles', 'smooth target distance and nearby-agent count features', 'a compact decision boundary that PPO can learn from sparse reward']
- **failure_modes:** ['A brittle switch can cause chattering around the covering radius', 'If the policy cannot tell partner presence from incidental nearby agents, it may stay too long or leave too early', 'Shared-policy collapse can still occur if identity is missing or unused']

**Predicted vs actual:** expected M1=0.06, M6_min=0.33 → actual M1=0.010, M6=0.244

**Inner-loop evolution (3 iters × 3 cands):**

| iter | cand | M1 | M6 | fit | n_feat | role | soft_prox | n_cs | rejected? |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.0 | 0.176 | +0.053 | 12 | Y | Y | 4 | no |
| 0 | 1 | 0.0 | 0.121 | -0.050 | 15 | Y | Y | 7 | no |
| 0 | 2 | 0.0 | 0.2 | -0.227 | 16 | Y | Y | 4 | no |
| 1 | 0 | 0.01 | 0.244 | +0.108 | 14 | Y | Y | 5 | no |
| 1 | 1 | 0.0 | 0.19 | +0.079 | 15 | Y | Y | 4 | no |
| 1 | 2 | 0.0 | 0.169 | +0.066 | 15 | Y | Y | 4 | no |
| 2 | 0 | 0.0 | 0.146 | +0.067 | 17 | Y | Y | 6 | no |
| 2 | 1 | 0.0 | 0.096 | -0.001 | 13 | Y | Y | 3 | no |
| 2 | 2 | 0.0 | 0.087 | -0.035 | 14 | Y | Y | 6 | no |

**End-of-outer reflection CoT:**

- **label:** `partial`
- **action:** `switch_to_next`

**memory_recall:** "outer 1: predicted M1=0.08, actual M1=0.0; outer 2: predicted M1=0.07, actual M1=0.0; outer 3: predicted M1=0.06, actual M1=0.01". The same strategy family has now been attempted multiple times, and every actual M1 is below half of its expected threshold (0.03 for the current strategy, 0.03–0.04 for prior related ones), so the falsification rule forces moving on.

**diff_vs_predicted:** Predicted M1=0.06 and M6=0.35, but actual M1=0.01 and M6=0.24428571428571416. Structure is present: both lidars are used, role one-hot is present, and there are 4 cross-source ops, so this is not a translation failure. The issue is underperformance versus the expected M1.

**what_went_right:** ['The candidate preserved the mandatory structural ingredients: role identity, both sensor sources, and multiple cross-source combinations.', 'The inner result shows the intended stay-versus-search idea is expressible in the feature stack, so the pattern itself is not missing.']

**what_went_wrong:** ['The policy still produced very low M1, so the switch is too weak to create reliable commit-versus-divert behavior.', 'All 15 features are dense and none are gated, which likely leaves the decision boundary too blurry for sparse-reward learning.', 'Repeated attempts in this strategy family have not come close to the expected M1 threshold, so refinement is unlikely to rescue it.']

**rationale:** The falsification rule applies: this strategy family has been attempted repeatedly, and every actual M1 is below 0.5× expected_M1. Even though the structural pattern is present, the learning signal is too weak, so we must move to the next pending strategy.

---

## Part 3 — Best v9.1.10 candidate vs S3b-local benchmark

**v9.1.10 best:** outer 4, iter 1, cand 0 — M1=0.01, M6=0.244, fitness=+0.108

```python
import torch
import torch.nn.functional as F
import math

def enhance_observation(scenario_state: dict) -> torch.Tensor:
    lidar_t = scenario_state["lidar_targets"]
    lidar_a = scenario_state["lidar_agents"]
    agent_pos = scenario_state["agent_pos"]
    agent_vel = scenario_state["agent_vel"]
    agent_idx = scenario_state["agent_idx"]
    n_agents = int(scenario_state["n_agents"])
    cover_r = float(scenario_state["covering_range"])

    B = lidar_t.shape[0]

    d_t = lidar_t.min(dim=-1).values
    d_a = lidar_a.min(dim=-1).values
    t_close = (lidar_t < cover_r).float().sum(dim=-1)
    a_close = (lidar_a < cover_r).float().sum(dim=-1)
    speed = agent_vel.norm(dim=-1)
    wall = 1.0 - agent_pos.abs().max(dim=-1).values
    soft_t = torch.exp(-3.0 * d_t)

    t_idx = lidar_t.argmin(dim=-1)
    a_idx = lidar_a.argmin(dim=-1)
    n_t = lidar_t.shape[-1]
    n_a = lidar_a.shape[-1]
    t_ang = 2.0 * math.pi * t_idx.float() / float(n_t)
    a_ang = 2.0 * math.pi * a_idx.float() / float(n_a)

    role = torch.zeros(B, n_agents, device=lidar_t.device, dtype=lidar_t.dtype)
    role[:, agent_idx] = 1.0

    # Joint local features for coordination
    target_dir_align = torch.cos(t_ang - a_ang)
    density_gap = t_close - a_close
    rendezvous_score = soft_t * (1.0 + torch.relu(density_gap))
    avoid_overlap = torch.sigmoid(2.5 * (d_a - d_t))
    safe_commit = torch.sigmoid(3.0 * (0.3 - d_t)) * torch.sigmoid(3.0 * wall)

    feats = torch.stack([
        d_t,
        d_a,
        t_close,
        a_close,
        speed,
        wall,
        soft_t,
        target_dir_align,
        density_gap,
        rendezvous_score,
        avoid_overlap,
        safe_commit,
        soft_t * (1.0 + a_close),
        wall * (1.0 + speed),
    ], dim=-1)

    return torch.cat([feats, role], dim=-1)
```

**S3b-local best:** seed s1, iter 0, cand 0 — M1=0.070, M6=0.435 (1M-eval)

```python
import torch
import math

def enhance_observation(scenario_state: dict) -> torch.Tensor:
    """
    Compute extra features from LOCAL sensor data to help agents coordinate in a rendezvous task.

    Uses only:
      - agent_pos, agent_vel, agent_idx
      - lidar_targets, lidar_agents

    Returns:
      [batch_dim, N] tensor, with N in the requested range.
    """
    lidar_t = scenario_state["lidar_targets"]   # [B, 15]
    lidar_a = scenario_state["lidar_agents"]    # [B, 12]
    agent_vel = scenario_state["agent_vel"]     # [B, 2]
    agent_idx = int(scenario_state["agent_idx"])
    n_agents = int(scenario_state["n_agents"])

    B, n_t = lidar_t.shape
    device = lidar_t.device
    dtype = lidar_t.dtype

    # --- Target-related local features ---
    min_t_dist, min_t_ray = lidar_t.min(dim=-1)  # [B], [B]
    min_t_angle = (min_t_ray.to(dtype) / float(n_t)) * (2.0 * math.pi)
    t_dir_x = torch.cos(min_t_angle)
    t_dir_y = torch.sin(min_t_angle)

    # How many target rays are "close" to the agent (local target clutter / possible multi-target proximity)
    close_t_thresh = 0.25
    n_close_targets = (lidar_t < close_t_thresh).to(dtype).sum(dim=-1)  # [B]

    # Soft target proximity: emphasizes whether a target is very near even if not exactly on a ray
    t_prox = torch.exp(-3.0 * min_t_dist)

    # --- Agent-related local features ---
    min_a_dist, min_a_ray = lidar_a.min(dim=-1)  # [B], [B]
    min_a_angle = (min_a_ray.to(dtype) / float(lidar_a.shape[-1])) * (2.0 * math.pi)
    a_dir_x = torch.cos(min_a_angle)
    a_dir_y = torch.sin(min_a_angle)

    # Nearby agent density: useful for inferring whether another agent may already be at the same target
    close_a_thresh = 0.25
    n_close_agents = (lidar_a < close_a_thresh).to(dtype).sum(dim=-1)  # [B]

    # Crowdiness / local coordination pressure
    crowd = torch.clamp(n_close_agents / max(1.0, float(n_agents - 1)), 0.0, 1.0)

    # --- Motion features ---
    speed = torch.linalg.norm(agent_vel, dim=-1)  # [B]
    vel_dir = torch.atan2(agent_vel[:, 1], agent_vel[:, 0] + 1e-8)  # [B]
    vel_x = torch.cos(vel_dir)
    vel_y = torch.sin(vel_dir)

    # --- Role identity for shared policy specialization ---
    one_hot = torch.zeros(B, n_agents, device=device, dtype=dtype)
    one_hot[:, agent_idx] = 1.0

    # Final feature vector:
    # 1) nearest target distance
    # 2-3) nearest target direction
    # 4) number of nearby target rays
    # 5) target proximity score
    # 6) nearest agent distance
    # 7-8) nearest agent direction
    # 9) number of nearby agent rays
    # 10) crowdiness score
    # 11) speed
    # 12-13) velocity direction
    # 14-17) one-hot agent id
    out = torch.cat([
        min_t_dist.unsqueeze(-1),
        t_dir_x.unsqueeze(-1),
        t_dir_y.unsqueeze(-1),
        n_close_targets.unsqueeze(-1),
        t_prox.unsqueeze(-1),
        min_a_dist.unsqueeze(-1),
        a_dir_x.unsqueeze(-1),
        a_dir_y.unsqueeze(-1),
        n_close_agents.unsqueeze(-1),
        crowd.unsqueeze(-1),
        speed.unsqueeze(-1),
        vel_x.unsqueeze(-1),
        vel_y.unsqueeze(-1),
        one_hot,
    ], dim=-1)

    return out
```

## Part 4 — Feature-by-feature diff: v9.1.10 best vs S3b-local best

| metric | S3b-local best | v9.1.10 best | Δ |
|---|---|---|---|
| n_features | 14 | 14 | +0 |
| n_cross_source | 0 | 5 | +5 |
| n_gated | 0 | 0 | +0 |
| role_one_hot | Y | Y | — |
| soft_prox | Y | Y | — |
| speed | Y | Y | — |
| touches_both_lidars | . | Y | — |
| n_lines | 91 | 58 | -33 |

## Part 5 — Outer-level evolution summary

```
v9.1.10 outer-by-outer (5 strategies attempted):

  cold-start bundle enum → 5 strategies (CoT) ───┐
                                                  │
  outer 0: role_split_with_pairing  ───────────────┤  initial slots
    (3 iters × 3 cands, all train, M1=0 best)    │  authored at enum
                                                  │
  outer 1: role_split_with_pairing (refine)     ──┤  §2.3 rejected
    (3 iters × 3 cands, M1=0)                    │  prose-only edits
                                                  │  → kept good slots
                                                  │
  outer 2: pair_completion_pressure  ────────────┐│  §2.10 lazy
    (M1=0, M6=0.219)                             ││  authored 1885B
                                                  ││  hints + 4013B
                                                  ││  examples
                                                  ││
  outer 3: soft_commit_and_release  ─────────────┐│  §2.10 lazy
    (M1=0.010, fitness=+0.070)                   ││  authored
                                                  ││
  outer 4: stay_search_switching  ───────────────┐│  §2.10 lazy
    (M1=0.010, M6=0.244 — best M6)               ││  authored
                                                  ││
  DONE: 5/5 outers, 45/45 cands trained          ──┘
```

Compare to S3b-local: SINGLE prompt, no strategy concept, 4 iters × 3 cands
× 3 seeds = 36 candidates. Same prompt every time. Best M1 = 0.070, best M6 = 0.435.
