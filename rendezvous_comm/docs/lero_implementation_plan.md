# Plan: Implement LERO in E1 Experiment

## Context

The rendezvous_comm project has ER1 (no comm), ER2 (continuous channels), ER3 (GNN) experiments complete. E1 ("Static LLM Infra") is a placeholder with only a config YAML and empty notebook. The LERO paper (arXiv:2503.21807) proposes using an LLM to **generate and evolve** two Python functions — a hybrid reward function (HRF) and an observation enhancement function (OEF) — via an evolutionary loop. No official LERO code exists. Eureka (github.com/eureka-research/Eureka) is the closest mature implementation for single-agent evolutionary reward generation.

**Goal**: Implement a LERO-style evolutionary LLM loop for the Discovery scenario, reusing the existing runner/config/metrics infrastructure.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│  LERO Evolutionary Loop (new: src/lero/)        │
│                                                  │
│  For each iteration (1..N_ITER):                │
│    1. LLM generates K candidate (HRF, OEF) pairs│
│    2. For each candidate:                        │
│       - Patch Discovery scenario                 │
│       - Short MARL training (reduced frames)     │
│       - Evaluate → M1-M9 metrics                 │
│    3. Rank candidates by fitness                 │
│    4. Feed top-k code + metrics back to LLM      │
│  End                                             │
│                                                  │
│  Final: Full training with best (HRF, OEF)      │
└─────────────────────────────────────────────────┘
```

---

## Step-by-Step Implementation

### Step 1: Create `rendezvous_comm/src/lero/` module

New files:

#### `src/lero/__init__.py`
Exports: `LeroLoop`, `LeroConfig`

#### `src/lero/config.py` — LERO-specific configuration
```python
@dataclass
class LeroConfig:
    # Evolutionary loop
    n_iterations: int = 4          # Evolution generations
    n_candidates: int = 3          # Candidates per iteration
    top_k: int = 2                 # Best candidates kept for feedback
    
    # Short training for candidate evaluation
    eval_frames: int = 1_000_000   # Reduced from 10M (exploration budget)
    eval_episodes: int = 100       # Eval episodes per candidate
    
    # Full training after evolution
    full_frames: int = 10_000_000  # Full training budget for winner
    
    # What to evolve
    evolve_reward: bool = True
    evolve_observation: bool = True

@dataclass
class LLMConfig:
    """LLM provider configuration — abstracted for future DSPy migration."""
    provider: str = "anthropic"          # "anthropic" | "openai" | "dspy" (future)
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.8
    max_tokens: int = 4096
    # Retry/robustness settings
    max_retries: int = 3
    retry_delay: float = 2.0
    # Prompt versioning (track which prompts produced which results)
    prompt_version: str = "v1"
    # Future DSPy fields (not implemented now, but reserved in config)
    # dspy_module: Optional[str] = None     # e.g., "lero.dspy_modules.RewardGenerator"
    # dspy_optimizer: Optional[str] = None  # e.g., "BootstrapFewShotWithRandomSearch"
```

Extend the E1 YAML config with `lero:` and `llm:` sections.

#### `src/lero/prompts/` — Prompt templates (separate files, versioned)

Prompts stored as **individual text files** (not Python strings) for easier iteration, diffing, and future DSPy migration. Each prompt file is a Jinja2 template.

```
src/lero/prompts/
├── v1/
│   ├── system.txt          # Role + constraints
│   ├── initial_user.txt    # Task description + scenario code + signatures
│   ├── feedback.txt        # Metrics + analysis guidance template
│   └── meta.yaml           # Prompt version metadata (author, date, description)
└── loader.py               # PromptLoader class
```

**`loader.py`** — Thin abstraction over prompt loading:
```python
class PromptLoader:
    """Loads and renders prompt templates.
    
    Design note: This is intentionally simple (Jinja2 templates + file I/O).
    When migrating to DSPy, this class gets replaced by DSPy Signatures
    that define input/output fields. The template content becomes the
    docstring of the Signature class. The file-based structure maps 1:1
    to DSPy modules.
    """
    def __init__(self, version: str = "v1"):
        self.template_dir = Path(__file__).parent / version
    
    def render(self, template_name: str, **kwargs) -> str:
        """Render a prompt template with variables."""
        ...
```

Three prompt templates:

1. **System prompt** (`system.txt`): "You are a reward engineer for multi-agent RL. You design Python reward and observation functions for a cooperative target-coverage task..."
2. **Initial user prompt** (`initial_user.txt`): Injects Discovery scenario code (reward/observation methods), task description, available state variables, PyTorch/batch constraints, and the required function signatures.
3. **Feedback prompt** (`feedback.txt`): Injects previous candidate code + M1-M9 metrics + analysis guidance (Eureka pattern: "if success rate is 0, rewrite entirely; if a component is constant, rescale or remove").

**DSPy migration path** (documented, not implemented):
```
# Future: prompts/v1/system.txt + initial_user.txt → DSPy Signature
class RewardGenerator(dspy.Signature):
    """Generate a reward function for multi-agent cooperative coverage."""
    scenario_code: str = dspy.InputField(desc="Discovery scenario source")
    task_description: str = dspy.InputField(desc="Task and constraints")
    state_variables: str = dspy.InputField(desc="Available state dict")
    reward_function: str = dspy.OutputField(desc="Python reward function code")
    observation_function: str = dspy.OutputField(desc="Python obs enhancement code")

# Future: feedback.txt → DSPy Module with ChainOfThought
class RewardEvolver(dspy.Module):
    def __init__(self):
        self.generator = dspy.ChainOfThought(RewardGenerator)
    def forward(self, scenario_code, task_description, state_variables, 
                previous_code=None, metrics_feedback=None):
        ...
```

**Function signatures** the LLM must produce:
```python
def compute_reward(self, agent, agent_idx, scenario_state) -> torch.Tensor:
    """Returns [batch_dim] reward tensor."""
    
def enhance_observation(self, agent, agent_idx, base_obs, scenario_state) -> torch.Tensor:
    """Returns [batch_dim, enhanced_dim] tensor appended to base observation."""
```

`scenario_state` is a dict we build from Discovery's internal state:
```python
scenario_state = {
    "agents_pos": self.agents_pos,           # [batch, n_agents, 2]
    "targets_pos": self.targets_pos,         # [batch, n_targets, 2]
    "agents_targets_dists": self.agents_targets_dists,  # [batch, n_agents, n_targets]
    "covered_targets": self.covered_targets, # [batch, n_targets]
    "agents_per_target": self.agents_per_target,  # [batch, n_targets]
    "agent_pos": agent.state.pos,            # [batch, 2]
    "agent_vel": agent.state.vel,            # [batch, 2]
    "n_agents": len(self.world.agents),
    "n_targets": self.n_targets,
    "covering_range": self._covering_range,
    "agents_per_target_required": self._agents_per_target,
    "step": self.world.step_count,           # Current timestep
}
```

#### `src/lero/llm_client.py` — Provider-agnostic LLM interface

Thin abstraction over LLM providers. The key design principle: **separate the "what to ask" (prompts) from "how to ask" (client)**. This makes future DSPy migration a swap of this one file.

```python
class LLMClient:
    """Provider-agnostic LLM interface.
    
    Current: direct Anthropic/OpenAI SDK calls.
    Future DSPy migration: replace this class with a DSPy-backed client
    that uses compiled prompts and optimized few-shot examples.
    """
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = self._init_provider(config.provider, config.model)
    
    def generate(self, messages: List[Dict], n: int = 1) -> List[str]:
        """Generate n completions from the LLM.
        
        Args:
            messages: Conversation history [{role, content}, ...]
            n: Number of independent completions (candidates)
        Returns:
            List of response strings
        """
        # Retry logic with exponential backoff
        # Provider-specific API call (anthropic vs openai)
        # Returns raw text responses
        ...
    
    @staticmethod
    def _init_provider(provider: str, model: str):
        if provider == "anthropic":
            import anthropic
            return anthropic.Anthropic()
        elif provider == "openai":
            import openai
            return openai.OpenAI()
        else:
            raise ValueError(f"Unknown provider: {provider}. "
                           f"For DSPy, use the dspy integration instead.")
```

**Why not DSPy now**: DSPy adds compilation, optimization, and few-shot bootstrapping — powerful but requires labeled examples (good/bad reward functions) that we don't have yet. After the first round of LERO experiments, we'll have training data to optimize prompts with DSPy's `BootstrapFewShotWithRandomSearch`.

#### `src/lero/codegen.py` — Code extraction + validation (prompt-agnostic)

This module handles **parsing LLM output into executable code** — independent of how the LLM was called.

- `extract_candidates(responses: List[str]) -> List[CandidateCode]`
  - Extracts Python code blocks from LLM responses using regex (Eureka pattern)
  - Validates via `ast.parse()` + signature checking
  - Returns list of `CandidateCode(reward_fn_source, obs_fn_source)` namedtuples

- `validate_function(source: str, expected_name: str, expected_args: List[str]) -> bool`
  - AST-parses the function, checks name and argument list match expectations
  - Rejects code with imports outside allowlist (torch, math, numpy)

- `build_feedback(candidates, metrics_list, top_k) -> str`
  - Uses `PromptLoader.render("feedback.txt", ...)` to format feedback
  - Formats top-k candidates' code + their M1/M2/M4/M6 metrics
  - Adds analysis guidance for the LLM

#### `src/lero/scenario_patch.py` — Monkey-patch Discovery scenario

This is the key integration piece. Rather than subclassing (which would require modifying BenchMARL's task loading), we **monkey-patch** the scenario instance after BenchMARL creates it.

- `patch_scenario(experiment, reward_fn_source, obs_fn_source) -> original_fns`
  - Gets the scenario from `experiment.task.env_func` chain
  - `exec()` the LLM-generated code in a controlled namespace with torch, math imports
  - Replaces `scenario.reward` and augments `scenario.observation` with the generated functions
  - Returns originals for restoration
  
- `unpatch_scenario(experiment, original_fns)`
  - Restores original methods

**Safety**: Each `exec()` runs in a restricted namespace (only torch, math, numpy). Wrapped in try/except — if execution fails, candidate gets fitness=-inf.

#### `src/lero/loop.py` — Main evolutionary loop

```python
class LeroLoop:
    def __init__(self, spec: ExperimentSpec, lero_config: LeroConfig):
        ...
    
    def run(self) -> dict:
        """Run full LERO evolutionary loop + final training."""
        messages = self._build_initial_messages()
        best_candidate = None
        
        for iteration in range(self.lero_config.n_iterations):
            # 1. Generate candidates
            candidates = generate_candidates(messages, self.lero_config.n_candidates, ...)
            
            # 2. Evaluate each candidate (short training)
            results = []
            for candidate in candidates:
                metrics = self._evaluate_candidate(candidate, iteration)
                results.append((candidate, metrics))
            
            # 3. Rank by fitness (M1 primary, M2 secondary)
            results.sort(key=lambda x: (x[1]["M1_success_rate"], x[1]["M2_avg_return"]), reverse=True)
            best_candidate = results[0][0]
            
            # 4. Build feedback for LLM
            feedback = build_feedback(
                [r[0] for r in results],
                [r[1] for r in results],
                self.lero_config.top_k,
            )
            messages.append({"role": "assistant", "content": results[0][0].source})
            messages.append({"role": "user", "content": feedback})
            
            # 5. Log iteration results
            self._log_iteration(iteration, results)
        
        # 6. Full training with best candidate
        final_metrics = self._full_training(best_candidate)
        return final_metrics
    
    def _evaluate_candidate(self, candidate, iteration) -> dict:
        """Short training run with patched scenario."""
        # Build experiment with reduced frames
        short_spec = self.spec.with_overrides(max_n_frames=self.lero_config.eval_frames)
        experiment = build_experiment(...)
        patch_scenario(experiment, candidate.reward_fn, candidate.obs_fn)
        experiment.run()
        metrics = evaluate_trained(...)
        return metrics
```

### Step 2: Extend config.py

Add optional `lero` field to the YAML config structure:
- In `config.py`: add `LeroConfig` import and `lero: Optional[LeroConfig]` to `ExperimentSpec`
- `load_experiment()` parses the `lero:` section if present

### Step 3: Extend runner.py

Add a `run_lero()` entry point (alongside existing `run_sweep()`):
```python
def run_lero(spec: ExperimentSpec) -> dict:
    """Run LERO evolutionary loop for E1 experiments."""
    from .lero import LeroLoop
    loop = LeroLoop(spec, spec.lero_config)
    return loop.run()
```

No changes to existing `run_single()` or `build_experiment()` — LERO wraps them.

### Step 4: Create E1 configs — Two variants (no-comm + comm)

The E1 experiment mirrors the ER1/ER2 split: test LERO's impact both **without** and **with** communication channels. This lets us measure:
- **E1 (no comm)**: Does LLM-evolved reward/obs improve coordination without any comm channel? Direct comparison to ER1.
- **E2 (with comm)**: Does LLM-evolved reward/obs improve how agents learn to use communication? Direct comparison to ER2.

**What changes between variants**:
| Aspect | E1 (no comm) | E2 (with comm) |
|--------|--------------|-----------------|
| `dim_c` | 0 | 8 |
| `use_agent_lidar` | true | true |
| Action space | 2D (force only) | 10D (force + comm) |
| Obs space | base + LLM-enhanced | base + messages + LLM-enhanced |
| Reward can reference | positions, distances, coverage | + communication activity (M5) |
| OEF can compute | spatial features, progress | + message summaries, comm patterns |
| Compares against | ER1 baseline | ER2 baseline |

Both variants use `use_agent_lidar=true` — the only difference is `dim_c`.

**Scale variants**: Each config includes a sweep over two scales:
- **Normal**: `n_agents=4, n_targets=4, agents_per_target=2` (standard k=2 coordination challenge)
- **Minimal**: `n_agents=2, n_targets=2, agents_per_target=1` (simpler k=1, faster iteration, sanity check)

**Implications for the LLM prompts**:
- System prompt includes a `{comm_mode}` variable describing whether comm channels exist
- The `scenario_state` dict passed to LLM-generated functions includes `agent.state.c` (messages) when `dim_c > 0`
- Feedback template includes M5 (tokens) only when `dim_c > 0`
- The initial prompt tells the LLM: "agents have {dim_c}-dimensional communication channels" or "agents cannot communicate"

**Implications for scenario_patch.py**:
- When `dim_c > 0`, the OEF receives the raw messages as part of `scenario_state["messages"]` — the LLM can design features that summarize or transform communication
- The HRF can include communication-related rewards (e.g., penalize silent agents, reward informative messages)

**Config files**:

#### `configs/e1/sweep_lero_mappo.yaml` (E1 — no comm)
```yaml
exp_id: e1
name: "LERO No-Comm"
description: >
  LERO framework without communication channels.
  LLM evolves reward + observation for coordination via spatial awareness only.
  Direct comparison to ER1.

task:
  n_agents: 4
  n_targets: 4
  agents_per_target: 2
  lidar_range: 0.35
  covering_range: 0.25
  use_agent_lidar: true
  targets_respawn: false
  shared_reward: false
  agent_collision_penalty: -0.1
  covering_rew_coeff: 1.0
  time_penalty: -0.01
  max_steps: 200
  dim_c: 0

train:
  algorithm: mappo
  max_n_frames: 10_000_000

lero:
  n_iterations: 4
  n_candidates: 3
  top_k: 2
  eval_frames: 1_000_000
  evolve_reward: true
  evolve_observation: true

llm:
  provider: "anthropic"
  model: "claude-sonnet-4-20250514"
  temperature: 0.8
  max_tokens: 4096
  max_retries: 3
  prompt_version: "v1"

sweep:
  seeds: [0, 1, 2]
  n_agents: [4, 2]
  n_targets: [4, 2]
  agents_per_target: [2, 1]    # n4_t4_k2 (normal) + n2_t2_k1 (minimal)
  algorithms: [mappo]
```

#### `configs/e2/sweep_lero_mappo.yaml` (E2 — with comm)
```yaml
exp_id: e2
name: "LERO With-Comm"
description: >
  LERO framework with 8-dim communication channels.
  LLM evolves reward + observation to improve learned communication.
  Direct comparison to ER2.

task:
  n_agents: 4
  n_targets: 4
  agents_per_target: 2
  lidar_range: 0.35
  covering_range: 0.25
  use_agent_lidar: true
  targets_respawn: false
  shared_reward: false
  agent_collision_penalty: -0.1
  covering_rew_coeff: 1.0
  time_penalty: -0.01
  max_steps: 200
  dim_c: 8
  comm_proximity: true

train:
  algorithm: mappo
  max_n_frames: 10_000_000

lero:
  n_iterations: 4
  n_candidates: 3
  top_k: 2
  eval_frames: 1_000_000
  evolve_reward: true
  evolve_observation: true

llm:
  provider: "anthropic"
  model: "claude-sonnet-4-20250514"
  temperature: 0.8
  max_tokens: 4096
  max_retries: 3
  prompt_version: "v1"

sweep:
  seeds: [0, 1, 2]
  n_agents: [4, 2]
  n_targets: [4, 2]
  agents_per_target: [2, 1]    # n4_t4_k2 (normal) + n2_t2_k1 (minimal)
  algorithms: [mappo]
```

**Note on sweep**: The `n_agents/n_targets/agents_per_target` lists are paired (not Cartesian product). The sweep generator needs to handle this — either via explicit `combinations` list or by zipping. This requires a small config.py extension to support paired sweep dimensions.

**Run order**: Start with minimal (n2_t2_k1) as a fast sanity check (~10x faster), then normal (n4_t4_k2).

### Step 5: Create E1 notebook

`notebooks/experiments/E1_lero.ipynb` following the Golden Circle narrative:
- **WHY**: k=2 coordination fails at 3-32% (ER1), manual reward shaping is tedious
- **HOW**: LLM generates + evolves reward/observation functions
- **WHAT**: Run LERO loop, compare against ER1/ER2/ER3

```python
spec = load_experiment("configs/e1/single_lero_mappo_n4_t4.yaml")
results = run_lero(spec)
```

### Step 6: Store LERO artifacts

Extend `RunStorage` to save per-iteration:
- `lero/iter_{i}/candidate_{j}_reward.py` — generated reward source
- `lero/iter_{i}/candidate_{j}_obs.py` — generated observation source  
- `lero/iter_{i}/candidate_{j}_metrics.json` — evaluation metrics
- `lero/iter_{i}/feedback.txt` — LLM feedback prompt
- `lero/messages.json` — full LLM conversation history

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/lero/__init__.py` | Module exports |
| `src/lero/config.py` | `LeroConfig` + `LLMConfig` dataclasses |
| `src/lero/llm_client.py` | Provider-agnostic LLM interface (Anthropic/OpenAI) |
| `src/lero/codegen.py` | Code extraction, AST validation, feedback formatting |
| `src/lero/prompts/loader.py` | `PromptLoader` — Jinja2 template renderer |
| `src/lero/prompts/v1/system.txt` | System prompt template |
| `src/lero/prompts/v1/initial_user.txt` | Initial user prompt template |
| `src/lero/prompts/v1/feedback.txt` | Feedback/evolution prompt template |
| `src/lero/prompts/v1/meta.yaml` | Prompt version metadata |
| `src/lero/scenario_patch.py` | Monkey-patch Discovery reward/observation |
| `src/lero/loop.py` | Main evolutionary loop orchestration |
| `configs/e1/sweep_lero_mappo.yaml` | E1 no-comm config (vs ER1) |
| `configs/e2/sweep_lero_mappo.yaml` | E2 with-comm config (vs ER2) |
| `notebooks/experiments/E1_lero.ipynb` | Experiment notebook |

## Files to Modify

| File | Change |
|------|--------|
| `src/config.py` | Add `LeroConfig`, extend `ExperimentSpec` with `lero` field |
| `src/runner.py` | Add `run_lero()` entry point |
| `src/storage.py` | Add LERO artifact storage methods |

## Files NOT Modified

- `vmas/scenarios/discovery.py` — no changes to VMAS itself (monkey-patch at runtime)
- Existing ER1/ER2/ER3 configs and code — untouched

---

## Key Design Decisions

1. **Monkey-patch over subclass**: BenchMARL loads scenarios by name; subclassing would require modifying VMAS task registry. Patching the scenario instance post-creation is simpler and reversible.

2. **Separate `run_lero()` over modifying `run_sweep()`**: The evolutionary outer loop is fundamentally different from a parameter sweep. Clean separation avoids complicating the existing pipeline.

3. **Short training for candidates (1M frames) vs full (10M)**: Following LERO paper — 30k steps for evaluation was enough to differentiate candidates. 1M frames (~1/10 budget) should be sufficient for ranking.

4. **Claude Sonnet as default LLM**: Good balance of code quality, speed, and cost for generating Python reward functions. Can upgrade to Opus if quality is insufficient. Configurable via `LLMConfig`.

5. **OEF appends to base observation** (not replaces): The LLM-generated observation enhancement adds extra features alongside the standard obs. This preserves backward compatibility and the existing model can still learn from base features.

6. **Three-layer LLM abstraction** (prompts → client → codegen): Clean separation so each layer can be swapped independently:
   - **Prompts** (text files): What to ask. Future: DSPy Signatures.
   - **LLM Client** (API calls): How to ask. Future: DSPy compiled modules.
   - **Codegen** (parsing): How to interpret responses. Stays the same regardless of LLM layer.

7. **Prompt versioning**: Prompts live in `prompts/v1/`, `v2/`, etc. The `prompt_version` in `LLMConfig` selects which set to use. Every LERO run records which prompt version produced its results — critical for reproducibility.

---

## DSPy: Add Now vs Later?

### Pros of adding DSPy NOW
- **Structured output**: `dspy.Signature` with multiple output fields (reward_function, observation_function) in one API call — reduces parsing failures vs regex extraction
- **Built-in retry/refinement**: `dspy.Refine` and `BestOfN` modules handle failed generations natively
- **Single-signature multi-output**: `"context -> reward_function, observation_function"` generates both in one call
- **Claude support is first-class**: `dspy.LM('anthropic/claude-sonnet-4-5-20250929')` — no issues
- **Future-proofs the codebase**: Won't need to rewrite later

### Cons of adding DSPy NOW
- **No labeled training data yet**: `BootstrapFewShotWithRandomSearch` needs 50+ examples — we have zero. The optimizer (the main value of DSPy) is useless until after first LERO runs
- **No reward-design precedent**: No published DSPy examples for RL reward function generation. We'd be pioneering.
- **Dependency bloat**: 18+ dependencies including litellm, pyarrow (via datasets), pydantic — adds friction to OVH deployment and Dockerfile
- **Overkill for v1**: Our v1 needs: call LLM → parse code → validate. That's 30 lines with raw Anthropic SDK. DSPy adds abstractions we won't leverage until we have training data
- **DSPy's own docs say**: "RL is typically worse on cost/quality basis than prompt optimizers like MIPROv2" — but we need RL-style iteration
- **Learning curve + debugging overhead**: DSPy's magic (compilation, optimization) makes debugging harder when things go wrong — bad for a first experiment

### Decision: Add DSPy LATER (after first LERO runs)

**Phase 1 (now)**: Raw Anthropic SDK with DSPy-ready architecture (file-based prompts, `LLMConfig`, clean separation)
**Phase 2 (after 4+ LERO runs)**: Migrate to DSPy with labeled data from Phase 1 runs. The collected (reward_fn_source, M1-M9 metrics) pairs become training examples for `BootstrapFewShotWithRandomSearch`.

---

## DSPy Migration Path (Future — NOT implemented now)

**When to migrate**: After the first successful LERO runs produce labeled training data (good/bad reward functions + their M1-M9 scores).

**What DSPy gives us**:
- **Prompt optimization**: `BootstrapFewShotWithRandomSearch` finds optimal few-shot examples from our labeled data
- **Structured I/O**: `dspy.Signature` enforces output format, reducing parsing failures
- **Model-agnostic**: Switch between Claude/GPT/open-source without prompt rewriting
- **Assertions**: `dspy.Assert` to enforce constraints (e.g., "reward function must be vectorized")

**Migration steps** (3 files change, everything else stays):
1. `prompts/loader.py` → `prompts/dspy_modules.py` (Signatures + Modules)
2. `llm_client.py` → configure `dspy.settings(lm=...)` instead of raw SDK
3. `codegen.py` → `extract_candidates()` reads from DSPy's structured output fields instead of regex

**What we build now to prepare**:
- File-based prompts (map 1:1 to DSPy Signatures)
- `LLMConfig` with `provider` and `prompt_version` fields
- Clean separation between prompt content and API mechanics
- All LERO artifacts saved (becomes DSPy training data)

---

## Verification Plan

1. **Unit test `scenario_patch.py`**: Create a dummy reward/obs function, patch Discovery, run 1 step, verify output shape and values
2. **Unit test `codegen.py`**: Mock LLM response, verify code extraction and AST validation
3. **Integration test**: Run 1 iteration with 1 candidate on 100k frames, verify metrics are collected and artifacts saved
4. **Full run**: Execute E1 config (4 iterations, 3 candidates), compare M1-M9 against ER1 baseline
5. **Ablation**: Run with `evolve_reward=True, evolve_observation=False` and vice versa to measure individual contribution (matching LERO paper's LR/LO ablation)
