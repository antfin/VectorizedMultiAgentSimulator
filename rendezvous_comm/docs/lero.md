# LERO — Full Documentation

LERO (LLM-driven Evolutionary Reward & Observation) adapted from [arXiv:2503.21807](https://arxiv.org/abs/2503.21807) for the VMAS Discovery scenario.

---

## 1. Architecture

The LLM is an **offline code designer** — it runs before RL training, never during.

```
For each iteration (1..4):
  1. LLM generates 3 candidate (reward, obs) Python functions
  2. Each candidate: patch into Discovery → train → evaluate
  3. Rank candidates by M1/M6 metrics
  4. Feed best code + metrics back to LLM
Final: best candidate trained for full 10M frames
```

### Two LLM-Generated Functions

**`compute_reward(scenario_state)`** — Reward function (full replacement or additive bonus, configurable via `reward_mode`). Has access to FULL global state (CTDE: centralized training).

**`enhance_observation(scenario_state)`** — Observation enhancement. State access depends on `obs_state_mode` ("global" = paper style, "local" = CTDE-correct).

### Scenario Patching

We create a `PatchedDiscoveryScenario` subclass with LLM methods baked in. This subclass is passed to BenchMARL's env factory so TorchRL sees the correct observation spec at init time.

### Code Structure

```
src/lero/
├── __init__.py
├── config.py             # LeroConfig + LLMConfig
├── llm_client.py         # LiteLLM wrapper (Anthropic/OpenAI/OVH)
├── codegen.py            # Code extraction + AST validation
├── scenario_patch.py     # Patched Scenario subclass
├── loop.py               # Evolutionary loop + experiment builder
└── prompts/
    ├── loader.py
    └── v1/               # Prompt templates
```

---

## 2. Results (12 Runs)

### LERO Paper Reference

MPE Simple Spread (n=3, t=3, k=1): MAPPO 24% → 74.7% coverage (+211%).
Full reward replacement, global obs, no normalization, o3-mini LLM.

### Our Results

#### k=1 (n=2, t=4) — ER1 baseline: M1=0.58

| Run | Reward Mode | Obs Mode | M1 | Issue |
|---|---|---|---|---|
| Replace | Oracle global | 0.065 | Reward hacking at 10M |
| Replace + global best | Oracle global | 0.065 | Same hacking |
| Bonus (name bug) | Local | N/A | Reward not extracted |
| **Bonus (broken approach)** | **Local** | **0.600** | **Bonus was zero, original dominated** |
| Bonus (arena scale) | Local | 0.000 | Bonus too large (M2=1028) |
| Bonus + tanh(0.5) | Local | 0.010 | Bonus active but harmful |

#### k=2 (n=4, t=4) — ER1 baseline: M1=0.405

| Run | Reward Mode | Key Change | M1 | Issue |
|---|---|---|---|---|
| Replace (old prompt) | Oracle | 0.045 | Anti-crowding wrong for k=2 |
| Replace (all M1=0) | Oracle | 0.000 | No signal at 1M eval |
| Bonus (target assign) | Local | 0.010 | Approach=0 bug |
| Bonus (manual, arena fix) | Local | 0.000 | M4=19.8 (interact!) but M1=0 |
| Bonus (LLM, arena fix) | Local | 0.020 | Bonus too large (M2=1398) |
| Bonus + tanh(2.0) | Local | 0.015 | Bounded but no coordination |

### Key Finding

**The only run that beat ER1 (M1=0.60) had a non-functional bonus.** The original Discovery reward alone produced M1=0.58. Every attempt with a functioning LLM bonus performed WORSE.

---

## 3. Analysis — Why LERO Fails On Discovery

### What We Changed vs Paper (Shouldn't Have)

| Our Change | Paper Does | Impact |
|---|---|---|
| Additive bonus `R = R_orig + bonus` | Full replacement | Bonus competes with original |
| tanh normalization | No normalization | Kills LLM's magnitude choices |
| Local-only obs (CTDE) | Global obs | Removes coordination features |
| Single function | `agent_reward` + `global_reward` | Different structure |
| covering_range for distances | Direct comparisons | Zero-gradient bug |

### Root Causes

1. **Discovery reward is already good** — MPE's sparse reward needed LERO. Discovery already has shaped covering reward + collision penalty + time penalty.
2. **Bonus mode limits the LLM** — paper gives LLM full control. Our bonus caps at ±0.5 to ±2.0.
3. **Local obs removes key features** — paper's OEF uses exact distances to landmarks and partners (oracle). Our local-only mode gives only LiDAR rays.
4. **k=2 is fundamentally harder** — requires rendezvous (2 agents at same target), which is far beyond MPE's k=1 spreading task.

---

## 4. Planned Experiments — Paper-Faithful Ablations

### Philosophy

Go back to basics. Replicate the paper exactly, then ablate ONE variable at a time.

### Phase 0: Implementation Changes

- Add `reward_mode: "replace" | "bonus"` to LeroConfig
- Add `obs_state_mode: "global" | "local"` to LeroConfig
- Create prompt `v2/` matching paper's framing ("reward engineer", full replacement, no bonus language)
- Keep `bonus_scale` for bonus mode only

### Phase 1: Paper Replication (P1)

Match paper as closely as possible:

```yaml
task: n=3, t=3, k=1
reward_mode: replace          # full replacement like paper
obs_state_mode: global        # global obs like paper
eval_frames: 1_000_000
llm: gpt-5.4-mini
```

**Expected**: Significant improvement over ER1 baseline on n=3 k=1.
**If this fails**: deeper implementation bug, not methodology issue.

### Phase 2: Reward Mode Ablation

All n=3, t=3, k=1:

| Exp | Reward | Obs | Tests |
|---|---|---|---|
| R1 | Replace (paper) | Global (paper) | Full paper |
| R2 | Replace | Local | Global obs needed? |
| R3 | Bonus | Global | Bonus helps or hurts? |
| R4 | Bonus | Local | Our previous approach |

### Phase 3: Observation Ablation

All n=3, t=3, k=1, reward=replace:

| Exp | OEF | Tests |
|---|---|---|
| O1 | Global (paper) | Full paper |
| O2 | Local sensors | CTDE-correct |
| O3 | None | Reward-only effect |

### Phase 4: Prompt & LLM Ablation

All n=3, t=3, k=1, reward=replace, obs=global.

#### 4A: Prompt Complexity (minimal → maximal)

Our current prompt is 41 lines with research history, technical constraints, distance warnings, code examples. The paper's prompt is ~3 lines: role + env code + signature. We test the spectrum:

| Exp | Prompt Style | Content | Tests |
|---|---|---|---|
| L1 | **Minimal (paper-faithful)** | "You are a reward engineer" + env code + signature. Nothing else. | Paper baseline |
| L2 | **+ Task description** | L1 + natural language description of Discovery task | Does explanation help? |
| L3 | **+ Coordination guidance** | L2 + k-specific strategy (spread for k=1, converge for k=2) | Does strategic context help? |
| L4 | **Full (our v1)** | Everything: research history, ER1-3 results, distance warnings, magnitude tips | Does more context help or confuse? |

#### 4B: Prompt Structure (how to organize information)

| Exp | Structure | Tests |
|---|---|---|
| L5 | **Single system message** | Everything in system prompt (current) |
| L6 | **Role/task split** | System = role only. User = env code + task + signature | Cleaner separation |
| L7 | **Chain-of-thought** | Add "First analyze the task, then design the reward step by step" | Does reasoning help code quality? |
| L8 | **Few-shot examples** | Include 1-2 example reward functions from MPE (paper's Listings 3-4) | Does seeing examples help? |

#### 4C: What the LLM sees

| Exp | Code Injection | Tests |
|---|---|---|
| L9 | **Full scenario source** (paper) | Inject reward() + observation() Python code | Paper approach |
| L10 | **State dict only** | Just the scenario_state dict description, no source code | Does code anchor or help? |
| L11 | **No env details** | Only task description in natural language | Can LLM design from description alone? |

#### 4D: Feedback & Evolution

| Exp | Feedback Style | Tests |
|---|---|---|
| L12 | **Code + metrics only** (paper) | Show best code + coverage rate | Paper approach |
| L13 | **+ Analysis guidance** | Add "if M1=0, rewrite entirely" etc. | Does guidance help evolution? |
| L14 | **+ Failed code tracebacks** | Include error messages from crashed candidates | Does error context help? |
| L15 | **Full conversation history** | No sliding window, keep all iterations | Does full memory help? |

#### 4E: LLM Model

| Exp | Model | Tests |
|---|---|---|
| L16 | gpt-5.4-mini (current) | Baseline |
| L17 | gpt-5.4 (full) | More capable, more expensive |
| L18 | claude-sonnet-4-6 | Different reasoning style |
| L19 | OVH gpt-oss-120b (with context_window fix) | Free, local |

#### 4F: Function Structure

Paper asks for TWO functions (`agent_reward` + `global_reward`). We ask for ONE (`compute_reward`). The split forces the LLM to think about individual vs team credit.

| Exp | Structure | Tests |
|---|---|---|
| L20 | **Single function** (current) | `compute_reward(state)` → scalar | Our approach |
| L21 | **Two functions (paper)** | `agent_reward(state)` + `global_reward(state)`, combined as `α·local + (1-α)·global` | Paper's local/global split |
| L22 | **Three functions** | `approach_reward` + `coordination_reward` + `completion_reward`, LLM designs each | Decomposed shaping |

### Phase 5: Task Scaling

Best config from P1-P4:

| Exp | Task | Tests |
|---|---|---|
| S1 | n=2, t=4, k=1 | Our easy task |
| S2 | n=4, t=4, k=1 | More agents |
| S3 | n=4, t=4, k=2 | Hard coordination |
| S4 | n=4, t=4, k=2, dim_c=8 | + communication |

### Phase 6: Loop Parameters

| Exp | Change | Tests |
|---|---|---|
| E1 | 6 candidates/round | More diversity |
| E2 | 6 iterations | More evolution |
| E3 | eval_frames=3M | Longer eval |
| E4 | Full conversation | Context helps? |

### Priority Order

```text
P1 (paper replication) → validates code works at all
 ↓ works?
YES → Phase 2 (R1-R4): reward replace vs bonus
     → Phase 3 (O1-O3): obs global vs local vs none
NO  → debug implementation against paper
 ↓
Phase 4A (L1-L4): prompt complexity (most impactful, cheap to run)
Phase 4F (L20-L22): function structure (single vs local+global split)
 ↓
Phase 4B-4D (L5-L15): structural/feedback ablations (if 4A shows sensitivity)
Phase 4E (L16-L19): LLM model comparison
 ��
Phase 5 (S1-S4): scale to k=2 with best config
 ↓
Phase 6 (E1-E4): loop parameter tuning
```

**Recommended first batch** (can run in parallel on OVH):
P1 + L1 + L4 + L8 + L21 — covers paper replication, minimal vs full prompt, few-shot, and function split. 5 experiments × ~2h each = 10h on 5 GPUs.

### Hypotheses

#### Core methodology

| ID | Hypothesis | Test |
|---|---|---|
| H1 | Reward replacement > bonus for LERO | R1 vs R3 |
| H2 | Global obs important for LERO's gains | O1 vs O2 |
| H3 | LERO's gains come from obs, not reward | O1 vs O3 |
| H4 | Discovery is harder than MPE Simple Spread | P1 vs paper 74.7% |

#### Prompt engineering

| ID | Hypothesis | Test |
|---|---|---|
| H5 | Minimal prompt works better than overloaded prompt | L1 vs L4 |
| H6 | Task description helps but constraints hurt | L2 vs L4 |
| H7 | Chain-of-thought improves code quality | L7 vs L1 |
| H8 | Few-shot examples > no examples | L8 vs L1 |
| H9 | Seeing scenario source code is critical | L9 vs L11 |
| H10 | Analysis guidance helps evolution | L13 vs L12 |
| H11 | Two-function split (local+global) > single function | L21 vs L20 |

#### LLM model

| ID | Hypothesis | Test |
|---|---|---|
| H12 | Stronger LLM = better reward design | L16 vs L17 |
| H13 | Different LLM families generate different reward strategies | L16 vs L18 |

---

## 5. Technical Notes

### OVH Deployment

LLM API keys for OVH jobs: encrypted via Fernet at submit time, decrypted in container.
```python
from dotenv import dotenv_values
submit_training_job("config.yaml", llm_env=dotenv_values(".env"))
```
Streamlit dashboard auto-detects LERO configs and handles encryption.

### Config Options

```yaml
lero:
  n_iterations: 4
  n_candidates: 3
  eval_frames: 1_000_000
  full_frames: 10_000_000
  evolve_reward: true
  evolve_observation: true
  bonus_scale: 0.5          # only for reward_mode=bonus
  reward_mode: "replace"    # "replace" (paper) | "bonus"
  obs_state_mode: "global"  # "global" (paper) | "local" (CTDE)

llm:
  model: "gpt-5.4-mini"
  temperature: 0.8
  max_retries: 3
  prompt_version: "v1"      # v1=current, v2=paper-faithful
```

### Known Issues Fixed

- Global best tracking across iterations (was last-only)
- LLM connection errors skip iteration (was crash)
- BenchMARL save_folder creation
- Context window awareness (model-specific)
- Sliding window conversation (OVH/vLLM token limit)
- Function name auto-rename (compute_reward → compute_reward_bonus)
- Scenario patching via subclass (not class-level, not instance-level)
