# LERO — Next Steps & Open Issues

Based on the first E1 run (n=2, k=1, gpt-5.4-mini, 4 evolutionary iterations).

---

## 1. Fair Comparison: MLP Input Size Problem

### The Issue

Each LERO candidate generates a different `enhance_observation` function that returns a different number of extra features. This means the MLP input layer (and total params) varies across candidates and vs the ER1 baseline:

| Experiment | Obs Dims | MLP Input | First Layer Params | Total Params |
|---|---|---|---|---|
| ER1 baseline | 31 | 31 | 7,936 | 75,012 |
| LERO iter_0/c2 | 31 + 7 = 38 | 38 | 9,728 | 76,804 |
| LERO iter_1/c1 (best) | 31 + 17 = 48 | 48 | 12,288 | 79,364 |
| LERO iter_2/c2 | 31 + 19 = 50 | 50 | 12,800 | 79,876 |
| LERO iter_3/c2 | 31 + 23 = 54 | 54 | 13,824 | 80,900 |

**What stays the same across ALL variants:**
- Hidden layers: 2 × 256 neurons (65,536 + 256 params)
- Activation: Tanh
- Output layer: 4 outputs (2 agents × 2D force action)
- Architecture: MLP (no GNN, no attention)
- All other training hyperparameters (lr, gamma, lambda, batch size, etc.)

**What changes:**
- First layer weight matrix: `[256, input_dim]` — grows with more obs features
- Total param difference: ~5-8% more params in LERO vs ER1 (~75K vs ~80K)

### Impact

The 5-8% param increase is small and unlikely to explain the performance difference. However, the **extra information content** in the enhanced observation is significant — features like `same_target`, `dist_to_uncovered`, `other_agent_target_dist` give the agent coordination signals that are impossible to infer from raw LiDAR.

### Proposed Ablations

To properly isolate LERO's contributions, run these variants:

| Ablation | Reward | Obs | MLP Input | Tests |
|---|---|---|---|---|
| **ER1** (existing) | Original | Original (31D) | 31 | Baseline |
| **E1-RO** (reward only) | LLM-evolved | Original (31D) | 31 | Reward shaping effect alone |
| **E1-OO** (obs only) | Original | LLM-evolved | 31 + N | Obs enhancement effect alone |
| **E1-FULL** (both) | LLM-evolved | LLM-evolved | 31 + N | Combined LERO effect |
| **E1-OO-fixed** | Original | LLM-evolved (N=8 fixed) | 39 | Obs with fixed dim for comparability |

To implement:
- `evolve_reward: true, evolve_observation: false` → E1-RO
- `evolve_reward: false, evolve_observation: true` → E1-OO
- Add `obs_n_features: 8` to LeroConfig to enforce fixed output dim in prompt

**Priority: E1-RO (reward only) is the cleanest comparison** — same MLP as ER1, only the reward function differs. Matches the LERO paper's "LR" ablation.

---

## 2. E2: Communication Experiments

### What Changes with dim_c > 0

When communication is enabled (`dim_c=8`):

| Aspect | E1 (no comm) | E2 (comm) |
|---|---|---|
| Agent action | 2D force | 2D force + 8D message = **10D** |
| Base obs | pos(2) + vel(2) + target_lidar(15) + agent_lidar(12) = **31D** | 31D + (n_agents-1) × dim_c messages |
| For n=4, k=2 | 31D | 31 + 3×8 = **55D** |
| MLP output | 4 (2 agents × 2D) | 20 (4 agents × (2D force + 8D comm)) |

### LERO Implications for E2

The LLM-generated functions get extra data:
- `scenario_state["messages"]`: `[batch, n_agents-1, dim_c]` — raw messages from other agents
- The reward can incentivize useful communication (penalize silence, reward informative messages)
- The obs enhancement can compute message statistics (mean, variance, entropy)

### Key Questions for E2
- Does LERO help agents learn **what to communicate**?
- Can the LLM design reward shaping that makes communication emerge faster?
- Is the improvement from better reward, better obs, or the combo?

### E2 Configs Ready

`configs/e2/single_lero_al_lp_sr_prox_dc8_ms400.yaml` — same task params as `er2_al_lp_sr_prox_dc8_ms400` for direct comparison.

---

## 3. LLM Code Quality Issues

### Observed in E1 Run

Of 12 total candidates generated (4 iters × 3 each):
- **8 valid** — code extracted, compiled, and ran successfully
- **2 failed extraction** — code blocks malformed or missing function name
- **2 runtime errors** — shape mismatches, wrong tensor ops (`~` on float tensor, reshape size mismatch)

### Improvements to Consider

1. **Add runtime validation before full training**: Run the compiled functions on a small synthetic batch (100 envs, 1 step) to catch shape/type errors before committing to 1M frames of training. Cost: ~1s per candidate.

2. **Better error feedback to LLM**: When a candidate fails, include the traceback in the feedback prompt so the LLM can fix it in the next iteration.

3. **Fix obs dimension in prompt**: Tell the LLM exactly how many features to return (e.g., "return exactly 8 features") to make candidates comparable.

4. **Temperature tuning**: Current temperature=0.8. Lower (0.5-0.6) may produce more reliable code with fewer runtime errors. Higher (0.9-1.0) may produce more creative reward designs.

---

## 4. Evolution Dynamics

### What Worked

| Iteration | Best M1 | Best M6 | M4 (collisions) | Key Change |
|---|---|---|---|---|
| 0 | 0.10 | 0.61 | 465 | First attempt — agents cluster |
| 1 | **0.35** | 0.75 | **1.1** | Occupancy-based credit + spread bonus |
| 2 | 0.32 | **0.83** | 106 | Coverage improved but collisions regressed |
| 3 | 0.29 | 0.77 | 261 | Further regression |

**Iteration 1 was the breakthrough**: the LLM added `1/(agents_at_target + 1)` credit splitting and a spread bonus, which virtually eliminated collisions (465 → 1.1) while tripling success rate (0.10 → 0.35).

**Iterations 2-3 regressed** on collisions, likely because the feedback focused on pushing M1/M6 higher, and the LLM traded collision avoidance for more aggressive target pursuit.

### Improvements to Consider

1. **Multi-objective fitness**: Instead of ranking by M1 only, use a composite score that penalizes high M4 (collisions). E.g., `fitness = M1 - 0.001 × M4`.

2. **Keep best across all iterations**: Currently best_candidate is from the last iteration with valid results. Should track global best across all iterations.

3. **Feedback emphasis**: Add explicit "DO NOT regress on collision avoidance" guidance when M4 increases between iterations.

---

## 5. Scaling to n=4, k=2

The n=2, k=1 task is relatively easy — each agent just needs to visit 2 targets. The real test is n=4, k=2 where:
- 4 agents must cover 4 targets
- Each target needs **2 agents simultaneously** — requires rendezvous coordination
- ER1 baseline gets only 3-32% success rate on this task
- ER3 (GNN) achieves 71%

### Estimated Time for n=4, k=2

Based on n=2 timings, n=4 will be ~2× slower per batch (more agents, more obs):
- Evolution: 4 iters × 3 candidates × ~12 min = ~2.5h
- Full training: 167 batches × ~40s = ~1.9h
- **Total: ~4.5h**

### Config Ready

`configs/e1/single_lero_al_lp_sr_ms400.yaml` — n=4, k=2, same hyperparameters.

---

## 6. Execution Plan

### Phase 1: Ablations on n=2, k=1 (fast iteration)
1. **E1-RO** (reward only, `evolve_observation: false`) — ~2h
2. Compare E1-RO vs E1-FULL vs ER1 — isolate reward vs obs contributions

### Phase 2: Main experiment n=4, k=2
3. **E1-FULL** n=4, k=2 — ~4.5h
4. Compare against ER1, ER2, ER3 baselines

### Phase 3: Communication (E2)
5. **E2-FULL** n=4, k=2, dim_c=8 — ~5h
6. Compare against ER2 baseline

### Phase 4: Analysis
7. Compare all: ER1 vs E1-RO vs E1-FULL vs ER2 vs E2-FULL vs ER3
8. Analyze LLM-generated code across iterations — what patterns emerge?
9. Check if LERO-designed rewards transfer across seeds/configurations

---

## 7. Technical Debt

- [ ] Global best tracking across iterations (not just last iteration's best)
- [ ] Runtime validation of candidates before training (synthetic batch test)
- [ ] Failed candidate tracebacks in feedback prompt
- [ ] Configurable fitness function (not just M1 primary)
- [ ] Obs dimension enforcement option (`obs_n_features` in config)
- [ ] Resume support (continue from iteration N if interrupted)
- [ ] Multiple seeds per LERO run (currently seed=0 only)
