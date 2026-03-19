# Hyperparameter Analysis & Experiment Design Guide

> **Based on**: 10 completed ER1 runs (8 MAPPO demo sweep + 2 IPPO dry runs)
> **Date**: 2026-03-19

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Hyperparameters](#2-current-hyperparameters)
3. [What the Results Tell Us](#3-what-the-results-tell-us)
4. [Parameter-by-Parameter Analysis](#4-parameter-by-parameter-analysis)
5. [Recommendations Before the Full Sweep](#5-recommendations-before-the-full-sweep)
6. [Proposed Experiments](#6-proposed-experiments)
7. [Communication-Specific Concerns](#7-communication-specific-concerns)
8. [Risk Map](#8-risk-map)

---

## 1. Executive Summary

We have 10 ER1 (no-communication baseline) runs. The headline finding is a **stark bifurcation between k=1 and k=2**:

| Condition | Success Rate (M1) | Avg Steps (M3) | Collisions (M4) |
|-----------|-------------------|-----------------|------------------|
| k=1 (easy) | **63-91%** | 47-74 | 6.5-9.7 |
| k=2 (hard) | **3-32%** | 91-100 | 5.1-12.5 |

This tells us the **coordination bottleneck is real** — which is exactly what the communication experiments (ER2-ER4, E1) are designed to address. However, several hyperparameter choices deserve scrutiny before committing GPU hours to the full 54-run ER1 sweep + 75 communication runs.

**Key concerns:**
- Entropy coefficient = 0.0 may cause premature policy collapse, especially for k=2
- Learning rate 5e-5 is on the low side for the batch sizes used
- GAE lambda 0.9 may be too low for long-horizon coordination tasks
- The MLP architecture [256, 256] may be undersized for k=2 coordination
- Only seed=0 so far — no variance estimates

---

## 2. Current Hyperparameters

### 2.1 Task Parameters (VMAS Discovery)

| Parameter | Current Value | Role |
|-----------|--------------|------|
| `n_agents` | 4, 6 (swept) | Team size |
| `n_targets` | 3, 4 (demo); 7 (full sweep) | Number of targets to cover |
| `agents_per_target` (k) | 1, 2 (swept) | Agents needed simultaneously on a target |
| `covering_range` | 0.25 | Radius within which agent "covers" target |
| `lidar_range` | 0.35 (demo); 0.25/0.35/0.45 (full) | Sensor range |
| `n_lidar_rays_entities` | 15 | Angular resolution for target sensing |
| `n_lidar_rays_agents` | 12 | Angular resolution for agent sensing |
| `use_agent_lidar` | false (ER1) | Whether agents can sense each other |
| `targets_respawn` | false | Covered targets stay covered (finite task) |
| `shared_reward` | false | Individual credit assignment |
| `max_steps` | 200 | Episode timeout |
| `covering_rew_coeff` | 1.0 | Reward per newly-covered target |
| `agent_collision_penalty` | -0.1 | Per-collision penalty |
| `time_penalty` | -0.01 | Per-step cost |
| `min_dist_between_entities` | 0.2 | Spawn spacing |

### 2.2 Training Parameters

| Parameter | Current Value | BenchMARL Default |
|-----------|--------------|-------------------|
| `algorithm` | MAPPO / IPPO | — |
| `max_n_frames` | 10,000,000 | 3,000,000 |
| `gamma` | 0.99 | 0.99 |
| `lr` | 5e-5 | 5e-5 |
| `on_policy_collected_frames_per_batch` | 60,000 | 6,000 |
| `on_policy_n_envs_per_worker` | 600 | 10 |
| `on_policy_n_minibatch_iters` | 45 | 45 |
| `on_policy_minibatch_size` | 4,096 | 400 |
| `share_policy_params` | true | true |
| `evaluation_interval` | 120,000 | 120,000 |
| `evaluation_episodes` | 200 | 10 |

### 2.3 Algorithm Parameters (BenchMARL defaults, NOT overridden)

| Parameter | MAPPO | IPPO | Notes |
|-----------|-------|------|-------|
| `clip_epsilon` | 0.2 | 0.2 | PPO clipping range |
| `entropy_coef` | **0.0** | **0.0** | No entropy bonus |
| `critic_coef` | 1.0 | 1.0 | Critic loss weight |
| `loss_critic_type` | l2 | l2 | MSE critic loss |
| `lmbda` (GAE) | 0.9 | 0.9 | Advantage estimation |
| `scale_mapping` | biased_softplus_1.0 | biased_softplus_1.0 | Action std transform |
| `use_tanh_normal` | true | true | Bounded actions |
| `share_param_critic` | true | true | Shared critic params |

### 2.4 Network Architecture

| Parameter | Value |
|-----------|-------|
| Architecture | MLP |
| Hidden layers | [256, 256] |
| Activation | Tanh |
| Normalization | None |
| Separate actor/critic | Same architecture, different weights |

---

## 3. What the Results Tell Us

### 3.1 The k=1 vs k=2 Gap

```
SUCCESS RATE (M1) BY CONFIGURATION

                k=1         k=2
n4, t3         80%         11%     ← 7x drop
n4, t4         63%          3%     ← 21x drop
n6, t3         91%         32%     ← 3x drop
n6, t4         89%         20%     ← 4x drop
```

**Interpretation**: k=2 is not just "harder" — it represents a qualitatively different problem. With k=1, agents can independently search and cover targets. With k=2, two agents must arrive at the same location at the same time, which requires implicit coordination that pure individual reward cannot easily incentivize.

### 3.2 More Agents Help (but not enough for k=2)

Going from n=4 to n=6:
- k=1: +11% (t=3), +26% (t=4) — extra agents reliably help
- k=2: +21% (t=3), +17% (t=4) — helps, but still far from solved

With n=6 and k=2, you have 6 agents trying to form 3-4 pairs on targets. The combinatorial explosion of "who goes where with whom" is the bottleneck.

### 3.3 Training Dynamics

**k=1 runs show healthy convergence:**
- Reward improves monotonically
- Entropy decays smoothly from ~1.35 to ~-3.13 (policy becomes confident)
- Convergence detected around 5.76M frames (58% of budget)
- Gradient norms well-behaved

**k=2 runs show signs of struggle:**
- Catastrophic collision spike early in training (87 collisions at step 16 for n4_t3_k2)
- Entropy decays much slower — agents never become confident
- Reward is volatile in early training
- No clear convergence detected within 10M frames
- Final loss remains elevated (-0.44 vs -0.024 for k=1)

### 3.4 MAPPO vs IPPO (limited data)

Only two IPPO runs (both k=2, on GPU):

| Config | MAPPO M1 | IPPO M1 |
|--------|----------|---------|
| n4_t3_k2 | 11% | 19% |
| n4_t4_k2 | 3% | 9% |

IPPO actually performed better on k=2, which is surprising. Possible explanations:
- IPPO's independent critics may be better at k=2 where agents have genuinely different roles
- GPU training (IPPO) vs CPU (MAPPO) may cause subtle numerical differences
- n=1 seed — could be noise

**This needs more data points before drawing conclusions.**

### 3.5 Coverage Progress (M6) — The Partial Picture

| Config | M1 (Success) | M6 (Coverage) |
|--------|-------------|---------------|
| n4_t3_k1 | 80% | 93% |
| n4_t3_k2 | 11% | 48% |
| n6_t4_k2 | 20% | 69% |

Even when agents don't fully succeed (M1), they make partial progress (M6). For k=2, agents typically cover ~50-70% of targets. This means they've learned to find targets but struggle with the final coordination push.

---

## 4. Parameter-by-Parameter Analysis

### 4.1 Entropy Coefficient (currently 0.0) — HIGH PRIORITY

**Current**: `entropy_coef = 0.0` (no exploration bonus)

**Problem**: With zero entropy regularization, the policy can collapse prematurely to a suboptimal deterministic strategy. The training curves show entropy dropping to -3.13 for k=1 runs — the policy becomes very confident very early. For k=2, where the reward landscape is sparser, this means agents may settle on a "wander around independently" policy without ever discovering that coordination pays off.

**Evidence from our data**: k=2 entropy decays slower (stays at +0.32 at step 50 vs -1.46 for k=1). This seems like agents are confused, not exploring productively. A small entropy bonus would encourage structured exploration.

**Recommendation**: Test `entropy_coef = 0.01` (mild) and `entropy_coef = 0.05` (moderate).

**Expected effect**: Slower initial convergence for k=1 (which is fine — they already converge), potentially much better k=2 performance as agents explore more coordination strategies before committing.

**Risk**: Too high (>0.1) prevents convergence entirely. Entropy bonus fights the policy gradient — you're telling the agent "be uncertain" while also telling it "maximize reward."

**Literature reference**: Most MAPPO papers (Yu et al. 2022, "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games") use entropy_coef in [0.01, 0.05] range.

---

### 4.2 Learning Rate (currently 5e-5) — MEDIUM PRIORITY

**Current**: `lr = 5e-5`

**Context**: This is the BenchMARL default. However, we're using 10x the default batch size (60K vs 6K frames per batch) and 10x the default minibatch size (4096 vs 400). With larger batches, gradient estimates are more stable, which typically permits a higher learning rate.

**The linear scaling rule** (Goyal et al., 2017): When you multiply the batch size by K, multiply LR by K. Our effective batch size is ~10x the default → LR could be ~5e-4.

However, this rule is approximate and MARL training is notoriously sensitive to LR. A more conservative approach:

**Recommendation**: Test `lr = 1e-4` (2x current) and `lr = 3e-4` (6x current).

**Expected effect**: Faster convergence, potentially reaching higher performance within the same 10M frame budget. But too high → training instability (oscillating loss, reward collapse).

**Risk**: Learning rate is the most dangerous hyperparameter to change. If increased too aggressively, the policy can oscillate or diverge. The clip_epsilon=0.2 provides some protection via PPO's trust region, but not infinite.

**Diagnostic**: Monitor `loss_objective` and `grad_norm`. If loss_objective becomes very noisy or grad_norm spikes, LR is too high.

---

### 4.3 GAE Lambda (currently 0.9) — MEDIUM PRIORITY

**Current**: `lmbda = 0.9`

**What it does**: Controls the bias-variance tradeoff in advantage estimation.
- lambda=0 → one-step TD (high bias, low variance) — good for short-horizon tasks
- lambda=1 → Monte Carlo returns (zero bias, high variance) — good for long-horizon tasks

**Our situation**: Episodes last up to 200 steps. With k=2, the "coordination payoff" might not come until step 50-100 when two agents finally converge on a target. A lambda of 0.9 means the effective credit assignment window is about 10 steps (geometric decay). This may be too short to connect "I started moving toward agent B at step 30" with "we covered a target together at step 45."

**Recommendation**: Test `lmbda = 0.95` and `lmbda = 0.99`.

**Expected effect**: Better credit assignment for long-horizon coordination. The advantage estimate will more accurately reflect whether early movement decisions led to eventual coverage.

**Risk**: Higher lambda means higher variance in the advantage estimates, which can make training noisier. This is partially mitigated by our large batch sizes (60K frames provide good variance reduction).

---

### 4.4 Network Architecture (currently [256, 256]) — LOW-MEDIUM PRIORITY

**Current**: Two hidden layers, 256 units each, Tanh activation.

**Observation space**: 19 dims (without agent lidar) or 31 dims (with agent lidar).

**Assessment**: For a 19-dim input, [256, 256] is generous and probably adequate for k=1 tasks. For k=2 coordination, the question is whether the policy network can learn the more complex decision-making required.

**Potential improvements**:
- **Wider network [512, 512]**: More capacity for complex coordination patterns
- **Deeper network [256, 256, 256]**: More abstraction layers for multi-step reasoning
- **Different activation (ReLU)**: Tanh saturates, which can slow learning for values far from zero. ReLU is more common in modern deep RL.
- **Layer normalization**: Can stabilize training, especially with varying input scales (position [-1,1] vs lidar [0, 0.35])

**Recommendation**: Test [256, 256] with ReLU first (minimal change). If k=2 still struggles, try [512, 256] (wider first layer for richer feature extraction).

**Risk**: Larger networks are slower to train and can overfit with insufficient data. But at 10M frames, overfitting is unlikely.

---

### 4.5 Batch Size & Minibatch Configuration — LOW PRIORITY

**Current**:
- `collected_frames_per_batch`: 60,000 (= 100 episodes of 200 steps with 600 envs × 200 steps / 2000 steps per collection... actually 60K / 200 max_steps = 300 episodes per batch)
- `n_envs_per_worker`: 600
- `minibatch_size`: 4,096
- `n_minibatch_iters`: 45

**Assessment**: The batch contains ~300 episodes worth of experience. Each batch is split into 60K/4096 ≈ 14.6 minibatches, iterated 45 times = ~658 gradient steps per batch. This is a lot of optimization per data collection.

**Concern**: 45 minibatch iterations is aggressive. The PPO clip guards against the policy drifting too far, but with 45 passes over the same data, we might be over-optimizing on stale advantages. This can lead to the "PPO plateau" where the policy gets stuck in a local optimum because it's been over-fitted to a particular batch of data.

**Recommendation**: Consider reducing `n_minibatch_iters` to 15-20 if other changes don't help. Alternatively, increase `collected_frames_per_batch` to 120K to get more diverse data per batch.

**Risk**: Fewer iterations = less sample-efficient but more stable. More data per batch = slower iteration time but better gradient estimates.

---

### 4.6 Discount Factor Gamma (currently 0.99) — NO CHANGE NEEDED

**Current**: `gamma = 0.99`

**Assessment**: Standard for tasks with episodes up to 200 steps. The effective horizon is 1/(1-0.99) = 100 steps, which covers most of the episode. Going higher (0.995 or 0.999) would extend the horizon but increase variance.

**Verdict**: Keep at 0.99. This is fine.

---

### 4.7 PPO Clip Epsilon (currently 0.2) — NO CHANGE NEEDED

**Current**: `clip_epsilon = 0.2`

**Assessment**: This is the standard PPO value from the original paper. There's rarely a good reason to change it. It works well across a wide range of tasks.

**Verdict**: Keep at 0.2.

---

### 4.8 Reward Structure — WORTH INVESTIGATING

**Current**: `shared_reward = false`, individual credit assignment.

**The coordination dilemma**: With k=2 and individual reward, when two agents cover a target together, each gets +1.0. But the "going toward another agent who is near a target" behavior is not directly rewarded — only the final joint coverage is. This creates a sparse reward problem for coordination.

**Potential improvement**: Switch to `shared_reward = true` for k=2 experiments. This would give all agents the same covering reward, removing the credit assignment problem entirely. The downside is that free-riding becomes possible (agents learn to let others do the work).

**Alternative**: Keep individual rewards but increase `covering_rew_coeff` for k=2 to make the coordination payoff more salient relative to the time penalty and collision costs.

**Recommendation**: Run a small comparison:
- k=2 with shared_reward=true
- k=2 with covering_rew_coeff=2.0 (double reward for coverage)

---

### 4.9 Collision Penalty (currently -0.1) — WORTH INVESTIGATING

**Current**: `agent_collision_penalty = -0.1`

**Problem for k=2**: To cover a target together, agents must get very close to each other (within covering_range=0.25 of the target). The collision penalty discourages agents from getting close to each other. With collision distance=0.005 and covering_range=0.25, there's room for two agents to be near a target without colliding — but during learning, agents don't know this yet and may avoid each other.

**Evidence**: k=2 early training shows 87 collisions at step 16, suggesting agents are crashing into each other a lot. The -0.1 penalty per collision × 87 collisions = -8.7 penalty, which could dominate the covering reward (+1.0 per target).

**Recommendation**: Test `agent_collision_penalty = -0.01` or `= 0.0` for k=2 runs. The agents need to learn to approach each other before they learn to avoid colliding.

**Risk**: With zero penalty, agents may develop a swarming behavior with constant collisions. But this might be acceptable if it leads to better coverage.

---

### 4.10 Lidar Range (swept: 0.25, 0.35, 0.45) — KEEP SWEEP

**Current**: 0.35 in demo, full sweep planned across [0.25, 0.35, 0.45].

**Assessment**: This is one of the most important parameters for the overall research question. The lidar range directly controls how much local information agents have:
- 0.25 = very local, agents must explore more blindly
- 0.35 = moderate, covers ~18% of the world diagonal
- 0.45 = generous, covers ~23% of the world diagonal

**For the communication experiments**: The lidar range determines the "information gap" that communication must fill. With 0.45 lidar, agents can see far and may not need communication. With 0.25 lidar, communication should be most valuable.

**Verdict**: Keep the sweep. This is a core independent variable.

---

### 4.11 Max Training Frames (currently 10M) — MIGHT NEED MORE FOR k=2

**Current**: `max_n_frames = 10,000,000`

**Evidence**: k=1 converges at ~5.76M frames. k=2 shows no convergence within 10M frames.

**Question**: Is k=2 just slow to converge, or has it hit a wall?

**Recommendation**: Run one k=2 configuration to 20M or 30M frames to check. If the reward curve is still climbing at 10M, extending training could help. If it's plateaued, more frames won't help and the bottleneck is elsewhere (exploration, architecture, reward structure).

---

## 5. Recommendations Before the Full Sweep

### 5.1 Priority-Ordered Changes

| Priority | Change | Why | Cost |
|----------|--------|-----|------|
| **P0** | Add entropy_coef = 0.01 | Prevent premature exploration collapse for k=2 | Low — single parameter |
| **P1** | Test lr = 1e-4 | Better matched to our batch size | Low — single parameter |
| **P1** | Test lmbda = 0.95 | Better credit assignment for coordination | Low — single parameter |
| **P2** | Reduce collision penalty to -0.01 for k=2 | Stop punishing proximity-seeking behavior | Low — single parameter |
| **P2** | Run one k=2 to 20M frames | Check if more training helps | Medium — one extra long run |
| **P3** | Test shared_reward for k=2 | Remove credit assignment bottleneck | Low — single parameter |
| **P3** | Try ReLU activation | May learn faster than Tanh | Low — model config change |

### 5.2 Suggested Pre-Sweep Ablation (8-12 runs)

Before committing to the 54-run ER1 sweep, run a focused ablation on the **most promising changes** using a single challenging configuration: **n=4, t=4, k=2, lidar=0.35, MAPPO, seed=0** (currently achieves only 3% success rate).

| Run | Change from baseline | Purpose |
|-----|---------------------|---------|
| 1 | baseline (current) | Reference (already have this: 3% M1) |
| 2 | entropy_coef = 0.01 | Test exploration |
| 3 | entropy_coef = 0.05 | Test stronger exploration |
| 4 | lr = 1e-4 | Test faster learning |
| 5 | lmbda = 0.95 | Test better credit assignment |
| 6 | entropy=0.01 + lr=1e-4 | Combined best bets |
| 7 | entropy=0.01 + lmbda=0.95 | Combined exploration + credit |
| 8 | entropy=0.01 + lr=1e-4 + lmbda=0.95 | Full package |
| 9 | collision_penalty = 0.0 | Test without collision avoidance |
| 10 | shared_reward = true | Test cooperative reward |
| 11 | max_frames = 20M (baseline) | Test if more training helps |
| 12 | [512, 256] network | Test larger network capacity |

**Estimated cost**: ~12 runs × 2h each = 24h on CPU, or ~6h on GPU.

### 5.3 What to Update in the Full Sweep Config

Based on ablation results, update `sweep_mappo-ippo_n2-6_l025-045.yaml` with the winning hyperparameters before launching the 54-run sweep.

---

## 6. Proposed Experiments

### 6.1 Experiment A: Entropy Sweep (Quick Win)

**Hypothesis**: Adding entropy regularization will improve k=2 success rates by preventing premature policy collapse.

```yaml
# Add to algorithm config (not currently exposed in TrainConfig)
sweep_dimensions:
  entropy_coef: [0.0, 0.01, 0.05]
  # Fix everything else
  n_agents: 4
  n_targets: 4
  agents_per_target: 2
  lidar_range: 0.35
  algorithm: mappo
  seeds: [0, 1, 2]
# Total: 9 runs
```

**Success criterion**: If best entropy_coef > 0.0 improves M1 by >10 percentage points.

### 6.2 Experiment B: Learning Rate × Lambda Interaction

**Hypothesis**: Higher LR + higher lambda together unlock better k=2 performance than either alone.

```yaml
sweep_dimensions:
  lr: [5e-5, 1e-4, 3e-4]
  lmbda: [0.9, 0.95, 0.99]
  # Fix: n=4, t=4, k=2, lidar=0.35, mappo, seed=0
# Total: 9 runs
```

**Success criterion**: Identify an (lr, lambda) pair that gives >20% M1 on n4_t4_k2.

### 6.3 Experiment C: Reward Engineering for k=2

**Hypothesis**: The reward structure is the main bottleneck for k=2 coordination.

```yaml
sweep_dimensions:
  shared_reward: [false, true]
  collision_penalty: [-0.1, -0.01, 0.0]
  covering_rew_coeff: [1.0, 2.0]
  # Fix: n=4, t=4, k=2, lidar=0.35, mappo, seed=0
# Total: 12 runs
```

**Success criterion**: Find a reward configuration where k=2 reaches >30% M1.

### 6.4 Experiment D: Extended Training Budget

**Hypothesis**: k=2 policies haven't converged — they need more training time.

```yaml
sweep_dimensions:
  max_n_frames: [10_000_000, 20_000_000, 30_000_000]
  # Fix: n=6, t=4, k=2, lidar=0.35, mappo, seed=0
# Total: 3 runs (but 1 already exists)
```

**Success criterion**: If M1 at 20M/30M > M1 at 10M by >10pp, training budget is limiting.

---

## 7. Communication-Specific Concerns

These issues don't affect the ER1 baseline but will matter when ER2-ER4 and E1 add communication channels.

### 7.1 Observation Space Expansion

**Current obs** (ER1): 19 dims = 2 (pos) + 2 (vel) + 15 (target lidar)

**With agent lidar** (E1+): 31 dims = 19 + 12 (agent lidar)

**With communication messages** (ER2-ER4): 31+ dims. Each received message adds dimensions. With n=6 agents and a 4-dim message schema (ER2: target_id, ETA, mode, confidence), that's 5 messages × 4 dims = 20 extra dims → **51 total dims**.

**Impact on network architecture**: A [256, 256] MLP with 51 inputs has 256×51 = 13K weights in the first layer vs 256×19 = 4.9K currently. The network has capacity for this, but:
- The first layer becomes a bottleneck if most inputs are noisy (early training messages are random)
- A wider first layer [512, 256] would handle the extra dims more gracefully
- This is why **ablation H** tests [512, 256] — if it works for ER1, it'll transfer better to comm experiments

### 7.2 Entropy and Communication

When agents have communication channels, the action space effectively grows (actions + messages). Zero entropy means the message policy collapses quickly — agents may learn to send the same message every step (a "null" message) because random messages are punished by the critic.

**With entropy_coef > 0**: Agents explore different messages longer, increasing the chance they discover useful communication patterns. This is especially critical for ER3 (symbolic intent) where the message space is discrete and small — entropy bonus directly encourages trying all symbols.

**Recommendation**: Whatever entropy_coef wins the ablation should be used for ALL experiments. If anything, comm experiments may benefit from *slightly higher* entropy to explore the message space.

### 7.3 Credit Assignment for Communication

When an agent sends a useful message, the reward comes indirectly: the message helps another agent cover a target, which generates reward for the sender only if shared_reward=true. With individual rewards, there's no gradient signal connecting "I sent a good message" → "the team performed better."

**GAE lambda matters here**: A higher lambda (0.95-0.99) allows the advantage estimate to trace back further through the episode, potentially connecting message-sending to eventual coverage. With lambda=0.9, the credit assignment window is ~10 steps — too short for multi-step communication protocols.

### 7.4 Hyperparameter Transfer Risk

The ablation uses n=4, t=4, k=2 — a specific configuration. The winning hyperparameters may not transfer perfectly to:
- **n=2** (too few agents → different coordination dynamics)
- **n=8** (too many agents → different scaling)
- **With comms** (messages change the optimization landscape)

**Mitigation**: Ablation I (k=1 sanity check) partially addresses this. For full confidence, we'd need to spot-check the winning params on 1-2 comm configurations. But this comes later — the ablation gives us a reasonable starting point.

### 7.5 The `shared_reward` Decision

This is the most consequential choice for the entire experimental series:

- **shared_reward=false** (current): Clean experimental comparison — comm should help because it enables better credit assignment. But the baseline is weak, and individual reward may not provide enough gradient signal for communication learning.
- **shared_reward=true**: Stronger baseline AND stronger comm experiments. The comparison becomes "does comm help coordination even with perfect credit assignment?" This is a higher bar but arguably more meaningful.

**If ablation F (shared_reward) shows >2x improvement over individual reward**, seriously consider switching all experiments to shared_reward=true. The thesis argument becomes stronger: "even with shared reward, communication provides additional coordination value."

---

## 8. Risk Map

### 8.1 Things That Could Waste GPU Hours

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Entropy too high → no convergence | Medium | Start with 0.01, not 0.1 |
| LR too high → training collapse | Medium | Monitor grad_norm, use 2x not 10x |
| k=2 fundamentally unsolvable without comms | Low-Medium | That's actually the thesis! ER2-ER4 should fix this |
| Over-tuning on seed=0, doesn't generalize | Medium | Run at least 3 seeds for key configs |
| Network too large → slower training, no benefit | Low | Only try if smaller changes fail |

### 8.2 Things That Could Invalidate Results

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Changing hyperparams between ER1 and ER2-ER4 | High if not careful | Lock hyperparams after ablation, use same for all experiments |
| Different hyperparams being optimal for different experiments | Medium | Use ER1 ablation winner as universal default |
| Targets_respawn accidentally set to True | Low | Already caught and fixed; provenance system will detect |

### 8.3 The Big Strategic Question

**Should we optimize k=2 baseline performance before running communication experiments?**

Arguments FOR:
- Better baseline → more meaningful comparison
- Ensures communication experiments inherit good hyperparams

Arguments AGAINST:
- The thesis is that communication helps coordination — a weak baseline makes the gap larger and more publishable
- Over-optimizing the baseline could make communication appear less valuable
- Time spent tuning baseline = time not spent on the actual research contribution

**Recommendation**: Do a **lightweight ablation** (Section 5.2, ~12 runs). Pick the best hyperparams, then move on. Don't spend more than 2 days on this. The communication experiments are the real contribution — the baseline just needs to be *fair*, not *perfect*.

---

## Appendix A: Parameter Reference Card

Quick reference for all tunable parameters and their safe ranges:

```
TASK PARAMETERS
  n_agents:               2-8 (higher = more coordination overhead)
  n_targets:              3-10 (higher = more exploration needed)
  agents_per_target:      1-3 (exponential difficulty increase)
  covering_range:         0.15-0.40 (smaller = precision task)
  lidar_range:            0.20-0.50 (smaller = information-poor)
  use_agent_lidar:        true/false (enables local agent sensing)
  shared_reward:          true/false (cooperative vs individual)
  covering_rew_coeff:     0.5-5.0 (reward magnitude)
  agent_collision_penalty: -0.5 to 0.0 (proximity cost)
  time_penalty:           -0.05 to 0.0 (urgency pressure)

TRAINING PARAMETERS
  lr:                     1e-5 to 1e-3 (higher with larger batches)
  gamma:                  0.95-0.999 (0.99 is standard)
  entropy_coef:           0.0-0.1 (0.01-0.05 recommended for MARL)
  clip_epsilon:           0.1-0.3 (0.2 is standard)
  lmbda (GAE):            0.9-0.99 (higher for long-horizon tasks)
  minibatch_size:         1024-8192
  n_minibatch_iters:      5-45 (fewer = more stable, less efficient)
  max_n_frames:           5M-30M (depending on task difficulty)

NETWORK PARAMETERS
  hidden_layers:          [128,128] to [512,512]
  activation:             Tanh, ReLU, GELU
  normalization:          None, LayerNorm
```

## Appendix B: Raw Results Table

| Run ID | Algo | n | t | k | lidar | M1 | M2 | M3 | M4 | M6 | M8 | M9 |
|--------|------|---|---|---|-------|----|----|----|----|----|----|-----|
| er1_mappo_n4_t3_k1_l035_s0 | MAPPO | 4 | 3 | 1 | 0.35 | 0.80 | 0.034 | 59.9 | 6.5 | 0.93 | 0.93 | 0.89 |
| er1_mappo_n4_t3_k2_l035_s0 | MAPPO | 4 | 3 | 2 | 0.35 | 0.11 | -0.387 | 97.0 | 5.1 | 0.48 | 0.77 | 0.89 |
| er1_mappo_n4_t4_k1_l035_s0 | MAPPO | 4 | 4 | 1 | 0.35 | 0.63 | 0.089 | 74.0 | 6.7 | 0.88 | 0.95 | 0.90 |
| er1_mappo_n4_t4_k2_l035_s0 | MAPPO | 4 | 4 | 2 | 0.35 | 0.03 | -0.112 | 99.5 | 5.5 | 0.51 | 0.81 | 0.80 |
| er1_mappo_n6_t3_k1_l035_s0 | MAPPO | 6 | 3 | 1 | 0.35 | 0.91 | -0.050 | 47.3 | 8.2 | 0.97 | 1.04 | 0.97 |
| er1_mappo_n6_t3_k2_l035_s0 | MAPPO | 6 | 3 | 2 | 0.35 | 0.32 | -0.455 | 91.8 | 11.6 | 0.68 | 1.06 | 0.99 |
| er1_mappo_n6_t4_k1_l035_s0 | MAPPO | 6 | 4 | 1 | 0.35 | 0.89 | 0.159 | 57.3 | 9.7 | 0.97 | 0.89 | 0.98 |
| er1_mappo_n6_t4_k2_l035_s0 | MAPPO | 6 | 4 | 2 | 0.35 | 0.20 | -0.241 | 95.2 | 12.5 | 0.69 | 0.93 | 0.91 |
| er1_ippo_n4_t3_k2_l035_s0 | IPPO | 4 | 3 | 2 | 0.35 | 0.19 | -0.247 | 94.7 | 7.4 | 0.59 | 0.73 | 0.80 |
| er1_ippo_n4_t4_k2_l035_s0 | IPPO | 4 | 4 | 2 | 0.35 | 0.09 | -0.062 | 97.7 | 9.2 | 0.56 | 0.71 | 0.77 |
