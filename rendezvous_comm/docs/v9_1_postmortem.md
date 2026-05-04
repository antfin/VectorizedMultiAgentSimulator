# v9.1 Phase 6 post-mortem — deep analysis of prompts, rewards, observations

**Date:** 2026-05-04
**Run:** `results/lero_v9/lero_v9_rendezvous_k2_2x3/20260503_2057_s0/`
**Wall:** 3h 6min (53% of v9 Phase 6's 6h 42min)

## TL;DR

v9.1 patches all worked as designed. **One newly-exposed bug**: bundle enumeration only authors slot artifacts for the chosen strategy (#0). Strategies #1-#4 have empty `V9Artifacts()`. When `switch_to_next` fires, empty artifacts → empty slot files → inner LLM with no template → produces bloated/role-less candidates → all rejected pre-eval → 0 trained candidates across outers 2-4.

**Net effect:** v9.1 is structurally correct (validators caught everything they should) but only outers 0+1 actually trained. Best M1=0.010 came from outer 1 with §2.3-preserved good slot text. Outers 2-4 produced 0 trained candidates each (pure pre-eval rejections).

The fix is a single new design change: **§2.10 author artifacts for all bundle strategies at enumeration time** (or lazily at switch time).

## 1. Prompt analysis

### 1.1 Slot-file sizes per outer (the smoking gun)

| outer | strategy | inferable_hints lines | examples lines | python blocks |
|---|---|---:|---:|---:|
| 0 | paired_partition (chosen at bundle enum) | 16 | 154 | 3 |
| 1 | paired_partition (refine kept the good slots via §2.3) | 16 | 154 | 3 |
| 2 | crowd_aware_greedy (switch) | **1** | **1** | **0** |
| 3 | identity_staggered_assignment (switch) | **1** | **1** | **0** |
| 4 | leader_follower_ring (switch) | **1** | **1** | **0** |
| 5 | search_then_commit (switch, never executed — last one) | **1** | **1** | **0** |

**Outers 2-5 inferable_hints text:** literally empty (`""`).
**Outers 2-5 examples text:** literally empty (`""`).

### 1.2 Why this happens

`enumerate_bundle_v9` parses the LLM's `chosen_strategy_artifacts` and writes only to `bundle.strategies[chosen_idx].artifacts`. The other 4 strategies in `bundle.strategies` have the default empty `V9Artifacts()` (no inferable_hints_text, no examples_text, no feedback_template).

When `_strategy_to_slots(bundle.current())` runs at switch time:

```python
def _strategy_to_slots(strategy: V9Strategy) -> Dict[str, str]:
    return {
        "inferable_hints": strategy.artifacts.inferable_hints_text,  # "" for non-chosen
        "examples": strategy.artifacts.examples_text,                 # ""
        "feedback": strategy.artifacts.feedback_template,             # ""
    }
```

→ writes empty strings to slot files → inner LLM sees an empty `inferable_hints` section and an empty `examples` section.

### 1.3 What the inner LLM did with empty slots

It still produced reasonable code (it has the `system` and `state_schema` text), but with no S3b-local-class teaching:
- Outer 2 cand 0 (rejected, 41 features): used `torch.exp(-x)` for closeness, summary stats per stream, no role one-hot.
- Outer 4 cand 0 (rejected, 29 features): similar.

These are competent attempts but without role one-hot guidance the LLM doesn't know it's mandatory.

## 2. Reward analysis

**Identical to S3b-local and v9 Phase 6.** Both runs use `evolve_reward: false`, so the reward is the scenario default:
- `covering_rew_coeff = 1.0` (per target newly covered per step)
- `agent_collision_penalty = -0.01` (per collision per step)
- `time_penalty = -0.01` (per step)
- `shared_reward = true` (team-summed)

No LLM-generated reward function in either run. Reward is NOT the differentiator.

## 3. Observation analysis (outer 1 winner — M1=0.010)

This is the only candidate in v9.1 that produced a non-zero M1. Inspect what worked:

```python
def enhance_observation(scenario_state: dict) -> torch.Tensor:
    lidar_t = scenario_state["lidar_targets"]   # [B, 15]
    lidar_a = scenario_state["lidar_agents"]    # [B, 12]
    agent_pos = scenario_state["agent_pos"]     # [B, 2]
    agent_vel = scenario_state["agent_vel"]     # [B, 2]
    agent_idx = scenario_state["agent_idx"]     # Python int
    n_agents = int(scenario_state["n_agents"])
    cover_r = float(scenario_state["covering_range"])

    B, n_target_rays = lidar_t.shape

    # Target LiDAR — proximity, direction, count
    nearest_target_dist = lidar_t.min(dim=-1).values
    nearest_target_idx = lidar_t.argmin(dim=-1)
    nearest_target_angle = 2.0 * math.pi * nearest_target_idx.float() / float(n_target_rays)
    nearest_target_cos = torch.cos(nearest_target_angle)
    nearest_target_sin = torch.sin(nearest_target_angle)
    target_close_count = (lidar_t < cover_r).float().sum(dim=-1)

    # Agent LiDAR — proximity, direction, count (mirrored)
    # ... (same pattern, omitted for brevity)

    # Cross-source: target proximity vs local agent congestion
    crowd_diff = target_close_count - agent_close_count

    # Role identity (mandatory)
    one_hot = torch.zeros(B, n_agents, device=agent_pos.device)
    one_hot[:, agent_idx] = 1.0

    return torch.cat([...all 14 features...], dim=-1)
```

**14 features. Role one-hot present. Cross-source via `crowd_diff = target_close_count - agent_close_count`.** This is exactly the S3b-local-class profile.

Why did it only get M1=0.010 instead of S3b-local's 0.07 max? Two hypotheses:
- **(a) Sample variance:** S3b-local's per-candidate M1 ranges 0.00 to 0.07 across 36 candidates. Single seed, single candidate → variance dominates.
- **(b) Subtle structural diff:** S3b-local's winning candidates often use `exp(-α·d)` (soft proximity) which is missing here. v9.1's outer 1 has `target_close_count` (gated) but not the smooth proximity form.

## 4. Comparison: v9.1 vs v9 vs S3b-local at 1M

| | best M1 | best M6 | role rate | wall |
|---|---|---|---|---|
| v8 Phase 3 | 0.010 | 0.209 | 0% | 2h 42m |
| v9 Phase 6 | 0.010 | 0.169 | 82% | 6h 42m |
| **v9.1 Phase 6** | **0.010** | **0.194** | **100% (only 6 trained cands)** | **3h 6m** |
| S3b-local 1M (3 seeds) | 0.070 | 0.435 | 97% | per-seed ~3h |

v9.1 matches v9 on M1 but **uses only 6 valid trained candidates** (vs v9's 18 valid trained). Effective LLM-learning data is 1/3 v9's, yet matches its result. If outers 2-4 had had non-empty slots, we'd expect 3× the training data and likely a higher max M1 from candidate diversity.

## 5. Improvements (priority order)

### §2.10 author artifacts for ALL bundle strategies (HIGH — fixes the bug above)

**Option A — eager:** in `enumerate_bundle_v9`, when the LLM returns the bundle, also have it author `inferable_hints_text`/`examples_text`/`feedback_template` for every strategy (not just chosen). Adds ~5× artifact-tokens to the bundle response (~5KB extra → ~$0.05 extra). One call.

**Option B — lazy:** at the moment of `switch_to_next`, if the new strategy's artifacts are empty, fire a one-off meta-LLM call to author them. Adds 1 extra LLM call per switch. v9.1 had 4 switches → ~$0.40 extra.

**Option C — fallback:** if artifacts empty at switch, copy the last passing slot text and let `refine_current` adapt it. Cheapest but loses strategy-specific tailoring.

**Recommendation:** Option B (lazy). The LLM cost is negligible compared to RL training, and the strategy-specific artifacts are higher quality than fallback. Implementation: ~30 LOC in `meta_strategist.py` + outer_loop wiring.

### §2.11 add `soft_proximity` to mandatory_features or examples

S3b-local high performers consistently use `exp(-α·d)` smoothing. v9.1's outer 1 winner uses raw counts. Add to `task_domain.yaml`:

```yaml
inferable_concepts:
  ...existing...
  - concept: "Soft proximity score"
    idiom: "torch.exp(-α * lidar_targets.min(dim=-1).values)  # smoother than raw distance, gives PPO gradient everywhere"
```

Predict: M1 mean climbs from 0.003 to 0.020-0.030 range (S3b-local's mean), without changing anything else.

### §2.12 sanity floor: refuse outer iter if slot text is empty (LOW)

Add a guard at outer-loop entry: if the rendered slot text has 0 python blocks AND ≥0 inferable concepts, log and SKIP this outer (move directly to switch). Saves the ~30s of LLM calls per iter that produce structurally-broken candidates.

This becomes unnecessary once §2.10 lands.

### §2.13 explore evolve_reward (MEDIUM, LATER)

Both v8/v9/v9.1/S3b-local use the default scenario reward. None has tried letting the LLM author a `compute_reward`. The hypothesis: a denser pair-formation bonus could break the M1≈0.01 ceiling at 1M frames. Risk: reward shaping pushes PPO toward the bonus and away from true coverage.

Worth a separate ablation experiment AFTER §2.10 lands.

### §2.14 deep-train v9.1 outer 1 winner at 10M (TEST, low cost)

The outer 1 cand 0 candidate has the right structure (role one-hot, cross-source, 14 features). At 10M frames PPO might unlock it (S3b-local goes from 0.07 at 1M to 0.93 at 10M). Cost: ~3h Mac, $0.

If 10M produces M1>0.5 → confirms task hits ceiling at 1M, structural design is correct.
If 10M still M1<0.05 → suggests this candidate's specific features are weak, need richer set.

## 6. What v9.1 already does correctly

- §2.3 slot-edit validator: kept the good outer-0 slot text into outer 1 — enabled the M1=0.010 result
- §2.6 stronger memory wording: LLM picked `switch_to_next` directly at outers 1, 2, 3, 4 (no override needed)
- §2.1 mandatory_features pre-eval: rejected dozens of role-less candidates → saved ~3h training
- §2.2 feature_budget hard cap: rejected dozens of >20-feature candidates → saved another large chunk
- §2.4 task_context dedup: prompt 50% smaller, no quality regression
- Bundle enumeration: 5 distinct strategies authored with rich CoT (still good — only the artifacts side is incomplete)

## 7. Recommended next steps

1. **Implement §2.10 lazy artifact authoring on switch.** Single highest-leverage fix.
2. Run v9.1.1 (with §2.10) — Mac smoke + full RL. Predict: 5/5 outers actually train, M1 mean 0.005-0.015 range.
3. **Implement §2.11** add soft_proximity to inferable_concepts. Cheap.
4. Run v9.1.2 — Mac full RL. Predict: M1 max climbs to 0.03-0.07 range (S3b-local territory).
5. If still M1<0.05 → §2.14 deep-train winner at 10M.
6. Only then consider §2.13 reward evolution.

Total cost to validate: ~$15 LLM + ~6h Mac RL.
