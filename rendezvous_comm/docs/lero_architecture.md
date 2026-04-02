# LERO Architecture — How It Works

LERO (LLM-driven Evolutionary Reward & Observation) uses an LLM to **design** reward and observation functions for MARL agents. The LLM is an offline code designer — it runs before RL training, never during.

Adapted from [arXiv:2503.21807](https://arxiv.org/abs/2503.21807) for the VMAS Discovery scenario.

---

## Overview

```
┌──────────────────────────────────────────────────────────────────┐
│  LERO Evolutionary Loop                                          │
│                                                                  │
│  For each iteration (1..4):                                      │
│    1. LLM generates 3 candidate (reward, obs) Python functions   │
│    2. Each candidate: patch into Discovery → train 1M frames     │
│    3. Evaluate → M1-M9 metrics                                   │
│    4. Rank candidates, feed best code + metrics back to LLM      │
│  End                                                             │
│                                                                  │
│  Final: Best candidate trained for full 10M frames               │
└──────────────────────────────────────────────────────────────────┘
```

The LLM is a **code designer**, not an agent. It writes Python functions, then standard MAPPO does all the actual learning. Agents never see or call the LLM.

---

## The Two LLM-Generated Functions

### 1. `compute_reward(scenario_state)` — Hybrid Reward Function

**Replaces**: The default Discovery reward (`vmas/scenarios/discovery.py:158-229`).

**Why**: The default reward has poor credit assignment. Agents get reward for covering a target but no signal for *approaching* uncovered targets, *spreading out*, or *coordinating* with teammates. With k=2, agents need to converge on the same target, which the default reward barely incentivizes.

**Input**: The `scenario_state` dict containing all environment state:

```python
scenario_state = {
    "agents_pos":              # [batch, n_agents, 2] — all agent positions
    "targets_pos":             # [batch, n_targets, 2] — all target positions
    "agents_targets_dists":    # [batch, n_agents, n_targets] — pairwise distances
    "covered_targets":         # [batch, n_targets] — covered this step
    "all_time_covered":        # [batch, n_targets] — cumulative coverage
    "agents_per_target":       # [batch, n_targets] — agents near each target
    "agent_pos":               # [batch, 2] — this agent's position
    "agent_vel":               # [batch, 2] — this agent's velocity
    "agent_idx":               # int — which agent (0..n_agents-1)
    "n_agents":                # int
    "n_targets":               # int
    "covering_range":          # float
    "agents_per_target_required": # int (k)
    "collision_penalty":       # float
    "collision_rew":           # [batch] — collision penalty this step
    "time_penalty":            # float
    "messages":                # [batch, n_agents-1, dim_c] (only if dim_c > 0)
}
```

**Output**: `torch.Tensor` of shape `[batch_dim]`.

**Example** of what the LLM might generate:

```python
def compute_reward(scenario_state):
    import torch
    agent_idx = scenario_state["agent_idx"]
    dists = scenario_state["agents_targets_dists"]     # [batch, n_agents, n_targets]
    covered = scenario_state["all_time_covered"]        # [batch, n_targets]

    # R_local: distance to nearest UNCOVERED target (individual credit)
    my_dists = dists[:, agent_idx, :]
    my_dists_uncov = my_dists + covered.float() * 1e6   # mask covered targets
    min_dist = my_dists_uncov.min(dim=-1).values
    approach_reward = -min_dist * 2.0

    # R_global: bonus when any target gets covered (team credit)
    n_covered = covered.float().sum(dim=-1)
    coverage_bonus = n_covered * 5.0

    # Hybrid: α · R_local + (1-α) · R_global + penalties
    return approach_reward + coverage_bonus \
         + scenario_state["collision_rew"] \
         + scenario_state["time_penalty"]
```

The key LERO insight (from the paper): the reward decomposes into `α · R_local + (1-α) · R_global`. The LLM decides how to balance individual credit vs team credit.

### 2. `enhance_observation(scenario_state)` — Observation Enhancement

**What it does**: Computes extra features appended to each agent's base observation.

The base observation is:

| Component | Dims | Description |
|-----------|------|-------------|
| Position | 2 | Agent x, y |
| Velocity | 2 | Agent vx, vy |
| Target LiDAR | 15 | Distance to nearest target per ray |
| Agent LiDAR | 12 | Distance to nearest agent per ray (if enabled) |
| Messages | (n-1)×dim_c | Received comm messages (if dim_c > 0) |

**Why**: LiDAR tells you *something is nearby* but not *which* target, whether it's covered, or how many other agents are near it. The enhancement gives higher-level coordination signals.

**Output**: `torch.Tensor` of shape `[batch_dim, N]` where N is 4-12 extra features.

**Example**:

```python
def enhance_observation(scenario_state):
    import torch
    agent_idx = scenario_state["agent_idx"]
    dists = scenario_state["agents_targets_dists"]
    covered = scenario_state["all_time_covered"]
    k = scenario_state["agents_per_target_required"]
    agents_near = scenario_state["agents_per_target"]

    # Feature 1: distance to nearest uncovered target
    my_dists = dists[:, agent_idx, :]
    masked = my_dists.clone()
    masked[covered.bool()] = 10.0
    nearest_dist = masked.min(dim=-1).values.unsqueeze(-1)       # [batch, 1]

    # Feature 2: task progress (fraction of targets covered)
    progress = covered.float().sum(dim=-1, keepdim=True) \
             / scenario_state["n_targets"]                        # [batch, 1]

    # Feature 3: how many more agents the nearest target needs
    nearest_idx = masked.argmin(dim=-1)
    agents_at = agents_near.gather(1, nearest_idx.unsqueeze(-1))
    need = (k - agents_at).float().clamp(min=0) / k              # [batch, 1]

    # Feature 4: am I the closest agent to my nearest target?
    all_dists_to_nearest = dists.gather(
        2, nearest_idx.unsqueeze(1).unsqueeze(2).expand(-1, dists.size(1), 1)
    ).squeeze(2)
    am_closest = (all_dists_to_nearest.argmin(dim=-1) == agent_idx) \
                 .float().unsqueeze(-1)                           # [batch, 1]

    return torch.cat([nearest_dist, progress, need, am_closest], dim=-1)
```

---

## How Functions Are Injected at Runtime

The patching happens in `src/lero/scenario_patch.py`:

```
build_experiment()
    → creates BenchMARL Experiment with Discovery scenario
    ↓
_get_scenario_from_experiment()
    → navigates BenchMARL wrappers to find the scenario instance
    ↓
patch_scenario(scenario, reward_source, obs_source)
    ├── exec(reward_source)  → compiles compute_reward into a callable
    ├── exec(obs_source)     → compiles enhance_observation into a callable
    ├── scenario.reward = patched_reward
    └── scenario.observation = patched_observation
```

### Patched Reward

The patched reward keeps the original's infrastructure (computing shared state, collision detection, target respawning) but replaces the final reward calculation:

```python
def patched_reward(agent):
    # 1. Compute shared state (original preamble)
    #    → agents_pos, targets_pos, covered_targets, etc.

    # 2. Collision penalties (original logic)

    # 3. Target respawning (original logic, last agent only)

    # 4. LLM-generated reward (replaces original)
    state = _build_scenario_state(scenario, agent, agent_idx)
    return compute_reward(state)  # ← LLM function
```

### Patched Observation

Simpler — calls the original, then appends extra features:

```python
def patched_observation(agent):
    base_obs = original_observation(agent)          # [batch, 31]
    state = _build_scenario_state(scenario, agent, agent_idx)
    extra = enhance_observation(state)              # [batch, 4-12]
    return torch.cat([base_obs, extra], dim=-1)     # [batch, 35-43]
```

---

## The Three Prompts

The LLM conversation uses three prompt templates stored in `src/lero/prompts/v1/`. Variables (`$name`) are substituted dynamically from the YAML config.

### Prompt 1: System (`system.txt`)

Sent once. Sets the LLM's identity and provides research context.

**Sections**:

1. **Research Context** — Explains the PhD project and what previous experiments found:
   - ER1: no comm, k=2 only gets 3-32% success rate
   - ER2: comm channels exist but agents struggle to learn protocols
   - ER3: GNN achieved 71% on k=2

2. **`$experiment_context`** — Dynamic block built from the YAML config by `_build_experiment_context()` in `loop.py`. Describes the current experiment:
   - For E1: "no-communication experiment, agents must coordinate purely through spatial observation"
   - For E2: "communication experiment with 8-dim channels, proximity-gated"
   - Explains the k=2 rendezvous challenge if applicable

3. **Technical Constraints** — Vectorized PyTorch, no loops over batch dim, allowed imports (torch, math only), correct output tensor shapes.

### Prompt 2: Initial User (`initial_user.txt`)

Sent once at the start. The main task description with everything the LLM needs to write code.

**Sections**:

1. **Task Description** — What Discovery is: `$n_agents` agents, `$n_targets` targets, `$agents_per_target` needed, `$covering_range`, `$max_steps`. All from YAML config.

2. **Agent Capabilities** — What agents can see and do. Dynamic blocks:
   - `$agent_lidar_description` → "ENABLED" or "DISABLED"
   - `$comm_description` → "NONE, agents are silent" or "8-dim continuous channel, proximity-gated..."
   - `$reward_description` → current reward components (covering +1.0, shared reward ON/OFF, collision penalty, time penalty)

3. **Current Implementation** — The actual Python source code of Discovery's `reward()` and `observation()` methods, injected via `$scenario_reward_code` and `$scenario_observation_code`. The LLM sees exactly what it's replacing.

4. **Available State** — The full `scenario_state` dict with tensor shapes and comments. `$comm_state_description` adds `"messages": [batch, 3, 8]` for comm experiments.

5. **Function Signatures** — Exact signatures with docstrings explaining what to optimize. For comm experiments, `$comm_obs_guidance` adds hints about message summary statistics.

### Prompt 3: Feedback (`feedback.txt`)

Sent after each evolutionary iteration. Replaces the user message with training results.

**Contains**:

- **Metrics per candidate**, ranked best to worst:
  - M1 (success rate), M2 (return), M4 (collisions), M6 (coverage progress)
  - M5 (tokens) for comm experiments
- **The actual code** of each candidate
- **Analysis guidance** — diagnostic rules:
  - "M1=0 for all → reward gives no learning signal, redesign completely"
  - "M6 high but M1 low → agents cover some targets but fail to coordinate"
  - "M4 high → strengthen collision avoidance"
  - "Constant component values → rescale or remove"
- **Instruction**: "Best was #N. Generate improved versions. Keep what worked, fix what failed."

### Conversation Flow

```
Turn 1: system.txt       (role: system)     — identity + research context
Turn 2: initial_user.txt (role: user)       — full task + code + signatures
Turn 3: LLM response     (role: assistant)  — generated reward + obs code
Turn 4: feedback.txt     (role: user)       — "M1=0.05, M6=0.3 — improve coordination"
Turn 5: LLM response     (role: assistant)  — improved code
Turn 6: feedback.txt     (role: user)       — "M1=0.15, M6=0.6 — getting better..."
Turn 7: LLM response     (role: assistant)  — further improved code
Turn 8: feedback.txt     (role: user)       — iteration 3 results
Turn 9: LLM response     (role: assistant)  — final code
```

The conversation grows each iteration. The LLM sees all its previous attempts and their metrics — this is the "evolutionary memory" that drives improvement.

### Dynamic Context Generation

All `$variable` substitutions are built by `_build_experiment_context()` in `loop.py`. This function reads the YAML config and generates context blocks that differ between experiments:

| Variable | E1 (no comm) | E2 (with comm) |
|----------|-------------|----------------|
| `$experiment_context` | "no-communication experiment..." | "communication experiment with 8-dim channels..." |
| `$agent_lidar_description` | "ENABLED" | "ENABLED" |
| `$comm_description` | "NONE. Agents are silent..." | "8-dim continuous channel, proximity-gated..." |
| `$reward_description` | covering +1.0, SR on, LP -0.01 | same |
| `$comm_state_description` | `# No communication channels` | `"messages": [batch, 3, 8]` |
| `$comm_obs_guidance` | *(empty)* | "message summary statistics, entropy..." |

This means you only write YAML configs — the prompts adapt automatically to any experiment setup.

---

## Evolutionary Loop Details

### Candidate Evaluation (Short Training)

Each candidate is trained for **1M frames** (vs 10M full) — enough to differentiate good vs bad reward designs without spending hours per candidate.

```
Iteration 0:
  LLM generates 3 candidates
  Train each for 1M frames with MAPPO
  Evaluate 100 episodes → M1, M2, M4, M6
  Results: M1=0.00, M1=0.05, M1=0.02
  Best (#2) + metrics → feedback prompt

Iteration 1:
  LLM sees feedback: "#2 got M1=0.05, M6=0.3 — covers some targets
  but fails to coordinate for full coverage"
  Generates 3 improved versions
  Best gets M1=0.15

Iteration 2:
  Further refinement based on what worked/failed
  Best gets M1=0.30

Iteration 3:
  Final refinement
  Best gets M1=0.45

Final: Best candidate trained for full 10M frames → final M1-M9 metrics
```

### Fitness Ranking

Candidates ranked by: **M1 (success rate)** primary, **M2 (avg return)** secondary.

Top-k candidates (default k=2) are included in the feedback prompt so the LLM can see both what worked and what failed.

### Artifact Storage

Every iteration saves:
```
results/e1/lero/20260402_1430/
├── iter_0/
│   ├── candidate_0_reward.py
│   ├── candidate_0_obs.py
│   ├── candidate_0_metrics.json
│   ├── candidate_0_response.txt    # raw LLM response
│   ├── candidate_1_reward.py
│   ├── ...
│   └── feedback.txt                # what was sent back to LLM
├── iter_1/
│   └── ...
├── messages_initial.json           # starting conversation
├── messages_final.json             # full conversation history
├── evolution_history.json          # M1/M2/M6 per iteration
├── best_reward.py                  # winning reward function
├── best_obs.py                     # winning observation function
├── best_policy.pt                  # trained policy weights
└── final_metrics.json              # M1-M9 from full training
```

---

## Experiment Configs

Three configs mirror the ER1/ER2 baselines for direct comparison:

| Config | Baseline | Key Difference |
|--------|----------|---------------|
| `e1/single_lero_al_lp_sr_ms400.yaml` | `er1/single_al_lp_sr_ms400.yaml` | Same task (n=4, k=2, no comm) + LERO |
| `e1/single_lero_al_lp_sr_ms400_n2_k1.yaml` | `er1/single_al_lp_sr_ms400_n2_k1.yaml` | Same task (n=2, k=1, no comm) + LERO |
| `e2/single_lero_al_lp_sr_prox_dc8_ms400.yaml` | `er2/single_al_lp_sr_prox_dc8_ms400.yaml` | Same task (n=4, k=2, dim_c=8) + LERO |

All task params (n_agents, n_targets, k, lidar_range, shared_reward, collision_penalty, max_steps) match exactly between E and ER configs. Tests enforce this.

---

## LLM Provider Configuration

Uses LiteLLM for unified access to any provider. Configured in YAML:

```yaml
# Anthropic Claude
llm:
  model: "claude-sonnet-4-6"

# OpenAI GPT
llm:
  model: "gpt-4o"

# OVH or any OpenAI-compatible endpoint
llm:
  model: "openai/llama-3.1-70b"
  api_base: "https://your-endpoint.ai.cloud.ovh.net/v1"
  api_key: "your-token"
```

API keys from environment variables: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `api_key` in config for custom endpoints.

---

## Code Structure

```
src/lero/
├── __init__.py           # Exports LeroLoop, LeroConfig, LLMConfig
├── config.py             # LeroConfig (evolution) + LLMConfig (provider)
├── llm_client.py         # LiteLLM wrapper with retry logic
├── codegen.py            # Code extraction from LLM responses, AST validation
├── scenario_patch.py     # Monkey-patch Discovery reward/observation
├── loop.py               # Main evolutionary loop + context generation
└── prompts/
    ├── __init__.py
    ├── loader.py          # Template loading (string.Template)
    └── v1/
        ├── system.txt     # System prompt
        ├── initial_user.txt # Task + code + signatures
        ├── feedback.txt   # Metrics + analysis guidance
        └── meta.yaml      # Version metadata
```

---

## Usage

```python
from rendezvous_comm.src import load_experiment, run_lero

# Load E1 config (no comm, LERO-enabled)
spec = load_experiment("rendezvous_comm/configs/e1/single_lero_al_lp_sr_ms400.yaml")

# Run the full evolutionary loop
results = run_lero(spec, algorithm="mappo", seed=0)

# results contains M1-M9 metrics from the best evolved candidate
print(f"M1 success rate: {results['M1_success_rate']:.3f}")
```
