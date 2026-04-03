# LERO — Next Steps & Open Issues

Updated after E1 runs (n=2 k=1, gpt-5.4-mini, multiple attempts).

---

## Summary of Results So Far

| Run | Global Best (1M) | Final (10M) | Verdict |
|---|---|---|---|
| E1 n=2 k=1 (run 1, wrong candidate bug) | M1=0.35 | M1=0.11 | Used iter_3 instead of global best |
| E1 n=2 k=1 (run 2, global best fix) | M1=0.40 | M1=0.065 | Reward hacking: M1 collapsed at 10M |
| ER1 n=2 k=1 (baseline) | — | **M1=0.58** | Baseline wins |

**Core problem: Candidates that look good at 1M frames collapse at 10M frames.** The LLM-shaped reward gets "hacked" — agents optimize the shaped reward signal without actually completing the task.

---

## 1. Observation Enhancement Uses Oracle Information (CRITICAL)

### The Issue

The LLM-generated `enhance_observation` receives `scenario_state` containing **global environment state** that agents should NOT access in a decentralized MARL setting. This makes comparison with ER1/ER2/ER3 unfair.

**What agents normally observe (ER1/ER2/ER3):**
- Own position and velocity (4 dims)
- Target LiDAR: distances to nearby targets in angular rays (15 dims) — no target identity, no covered/uncovered info
- Agent LiDAR: distances to nearby agents in angular rays (12 dims) — no agent identity
- Messages from nearby agents (if dim_c > 0)

**What LERO's `enhance_observation` can access (oracle info):**
| State Variable | Global? | Why It's Unfair |
|---|---|---|
| `agents_pos` [batch, n_agents, 2] | YES | Agents only see others via LiDAR rays, not exact positions |
| `targets_pos` [batch, n_targets, 2] | YES | Agents only detect targets via LiDAR, not exact coordinates |
| `agents_targets_dists` [batch, n, t] | YES | Full pairwise distance matrix — agent only knows its own distances |
| `covered_targets` / `all_time_covered` | YES | Agents can't distinguish covered vs uncovered targets via LiDAR |
| `agents_per_target` [batch, t] | YES | Agent doesn't know how many others are at each target |

**Note:** The reward function using oracle info is FINE (CTDE: Centralized Training, Decentralized Execution). Only the observation enhancement is problematic because it changes what agents see at execution time.

### Fix: Two-tier `scenario_state` (Option C)

Pass full state to `compute_reward` but restricted state to `enhance_observation`:

```python
# For reward (centralized training — full state OK)
reward_state = scenario_state  # everything

# For observation (decentralized execution — local sensors only)
obs_state = {
    "agent_pos": agent.state.pos,
    "agent_vel": agent.state.vel,
    "agent_idx": agent_idx,
    "n_agents": n_agents,
    "n_targets": n_targets,
    "covering_range": covering_range,
    "agents_per_target_required": k,
    "lidar_targets": agent.sensors[0].measure(),
    "lidar_agents": agent.sensors[1].measure(),  # if enabled
}
```

Update `scenario_patch.py` to pass different states to each function. Update prompts to explain what's available for each.

**Priority: CRITICAL — current results are not publishable.**

---

## 2. Reward Hacking: 1M→10M Collapse

### The Problem

Candidates scoring M1=0.35-0.40 at 1M frames evaluation drop to M1=0.065-0.11 after full 10M training. The LLM-designed reward creates shortcuts that get exploited during longer training.

Evidence from best candidate (iter_2, M1=0.40 at 1M):
- Reward had 7 components with hand-tuned weights (1.35, 0.55, 0.30, 0.75, 1.25)
- `approach_uncovered = exp(-4·dist)` — dense gradient but exploitable
- At 10M frames, agent learned to oscillate near targets (collecting approach reward) without actually covering them

### Fixes to Investigate

1. **Longer candidate evaluation**: Increase `eval_frames` from 1M to 2-3M. More expensive but catches hacking earlier. The LERO paper used 30K steps — with our 60K batch, that's only 0.5 batches. We may need more.

2. **Multi-checkpoint evaluation**: Evaluate candidates at 0.5M, 1M, and 2M frames. If M1 drops between checkpoints, penalize the candidate.

3. **Constrained reward design**: In the prompt, tell the LLM:
   - "The reward MUST include the original covering reward as a component"
   - "Do NOT create dense approach rewards that can be exploited"
   - "Reward must be bounded between -X and +X"

4. **Reward = original + shaping**: Instead of replacing the reward entirely, have the LLM generate a BONUS added to the original reward: `R = R_original + α·R_llm_bonus`. This guarantees the base task signal is always present.

5. **Fitness function**: Use M1 at multiple training checkpoints, not just final. E.g., `fitness = min(M1_at_500K, M1_at_1M)` — penalizes candidates that peak early and collapse.

**Priority: HIGH — this is why LERO underperforms ER1.**

---

## 3. MLP Input Size Varies Across Candidates

### The Issue

Each `enhance_observation` returns a different N features (7-23 observed). This means:
- Different MLP first layer size per candidate (38-54 vs ER1's 31)
- ~5-8% more params in LERO vs ER1
- Candidates not comparable to each other

### Fix Options

1. **Fixed obs dim in prompt**: Add `obs_n_features: 8` to LeroConfig, enforce in prompt: "return exactly 8 features"
2. **Reward-only mode** (`evolve_observation: false`): Same MLP as ER1, cleanest comparison
3. Both: reward-only as main experiment, obs enhancement as ablation

**Priority: MEDIUM — important for publishable results but less critical than #1 and #2.**

---

## 4. E2: Communication Experiments

### What Changes with dim_c > 0

| Aspect | E1 (no comm) | E2 (comm) |
|---|---|---|
| Agent action | 2D force | 2D force + 8D message = **10D** |
| Base obs (n=4) | 31D | 31 + 3×8 = **55D** |
| MLP output (n=4) | 8 (4×2D force) | 40 (4×10D) |

### LERO for Communication

The reward can incentivize useful communication:
- Penalize silent agents (all-zero messages)
- Reward message diversity across agents
- Bonus when communicated info leads to coverage

The obs enhancement (if used) can compute message statistics from `scenario_state["messages"]`.

### Key Questions
- Does LERO help agents learn **what to communicate**?
- Can LLM-designed reward make communication emerge faster than ER2's default?

### Blocked By
Issues #1 (oracle obs) and #2 (reward hacking) should be fixed first. E2 adds complexity — fix the fundamentals on E1 first.

---

## 5. LLM Code Quality

### Stats from Runs

Of ~24 candidates generated across runs:
- ~16 valid (67%) — compiled and ran
- ~4 failed extraction (17%) — malformed code blocks
- ~4 runtime errors (17%) — shape mismatches, wrong tensor ops

### Fixes

1. **Pre-training validation**: Run candidate on synthetic batch (1s) before committing to 1M-frame training
2. **Error feedback to LLM**: Include tracebacks in feedback prompt
3. **Temperature tuning**: Try 0.5-0.6 for more reliable code

**Priority: LOW — the loop handles errors gracefully, just wastes ~6min per failed candidate.**

---

## 6. Evolution Dynamics Improvements

### Global Best Tracking
**FIXED** — now tracks best across all iterations, not just last.

### Remaining Issues

1. **Multi-objective fitness**: Rank by composite score, not just M1. E.g., `fitness = M1 - 0.001×M4` to prevent collision regression.

2. **Feedback guidance**: When M4 regresses between iterations, add "DO NOT regress on collision avoidance" to feedback.

3. **Sliding window loses history**: Current sliding window keeps only last iteration's code in conversation. The LLM can't see iteration 0's breakthrough. Consider keeping a "hall of fame" summary.

---

## 7. Technical Debt

- [x] Global best tracking across iterations
- [x] LLM connection error handling (skip iteration, don't crash)
- [x] BenchMARL save_folder creation
- [x] Context window awareness (model-specific)
- [x] Sliding window conversation (OVH token limit fix)
- [ ] Two-tier scenario_state (oracle obs fix)
- [ ] Pre-training candidate validation
- [ ] Failed candidate tracebacks in feedback
- [ ] Multi-objective fitness function
- [ ] Fixed obs dimension option
- [ ] Reward = original + LLM bonus mode
- [ ] Multi-checkpoint candidate evaluation
- [ ] Resume support (continue from iteration N)
- [ ] Multiple seeds per LERO run

---

## 8. Execution Plan (Revised)

### Phase 1: Fix Fundamentals
1. Implement two-tier `scenario_state` (issue #1)
2. Add "reward = original + bonus" mode (issue #2)
3. Add pre-training candidate validation (issue #5)

### Phase 2: Reward-Only Ablation (cleanest test)
4. **E1-RO** n=2 k=1 (`evolve_observation: false`) — same MLP as ER1, only reward differs
5. Compare E1-RO vs ER1 — does LLM reward shaping help or hurt?

### Phase 3: Full LERO with Fixes
6. **E1-FULL** n=2 k=1 with fixed obs (restricted state, N=8)
7. **E1-FULL** n=4 k=2
8. Compare against ER1/ER3 baselines

### Phase 4: Communication
9. **E2-RO** n=4 k=2 dim_c=8 (reward only)
10. **E2-FULL** n=4 k=2 dim_c=8
11. Compare against ER2 baseline

### Phase 5: Analysis
12. Compare all variants
13. Analyze LLM-generated code patterns
14. Test reward transferability across seeds
