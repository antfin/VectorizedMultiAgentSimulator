# LERO — Full Documentation

LERO (LLM-driven Evolutionary Reward & Observation) adapted from [arXiv:2503.21807](https://arxiv.org/abs/2503.21807) for the VMAS Discovery scenario.

**Status (2026-04-16):** Phase 1–4 partially executed on n=3, t=3, k=1. Per-experiment S3 prefixes, M8 metric, scenario-patch closure scoping, OVH-vs-local VMAS attribute drift, reward magnitude clipping, and a full-training NaN fallback chain are all implemented and verified.

---

## 1. Architecture

The LLM is an **offline code designer** — it runs before RL training, never during.

```
For each iteration (1..n_iterations):
  1. LLM generates n_candidates (reward, obs) Python functions
  2. Each candidate: patch into Discovery → train (1M frames) → evaluate
  3. Rank by (M1, M6, M2)
  4. Feed best code + metrics back to LLM as feedback
After loop:
  5. Sort ALL valid candidates across all iterations by (M1, M6, M2)
  6. Try full training (10M frames) on rank 0; if it crashes (e.g. NaN
     actions), fall back to rank 1, then rank 2, etc.
```

### Two LLM-generated functions

**`compute_reward(scenario_state)`** — reward function. Modes:
- `replace` (paper): full reward replacement; LLM owns the entire reward signal
- `bonus`: `R = R_original + bonus_scale * tanh(compute_reward_bonus(state))`

**`enhance_observation(scenario_state)`** — extra observation features appended to base obs. Modes:
- `global` (paper): function gets full state (all positions, coverage, etc.)
- `local` (CTDE-correct): function gets only own sensors (lidar)

### Scenario patching

`make_patched_scenario_class(...)` in `src/lero/scenario_patch.py` returns a `PatchedDiscoveryScenario` subclass of `vmas.scenarios.discovery.Scenario` with the LLM methods baked in. This subclass is passed to BenchMARL's env factory so TorchRL sees the correct observation spec at init time.

The patched class also overrides `info()` to always return per-agent `agent.covering_reward` (instead of upstream's `shared_covering_rew` when `shared_reward=True`), so M8 (agent utilization CV) is computable.

### Reward safeguards

LLM-generated rewards have wide magnitude variance (we observed M2 ranging from -1192 to +896 across candidates). To prevent PPO from diverging:

1. **Sanitization:** `torch.nan_to_num(r, nan=0, posinf=0, neginf=0)` — handles LLM divisions by zero.
2. **Clipping:** `torch.clamp(r, -reward_clip, +reward_clip)` — default ±50. Configurable via `LeroConfig.reward_clip`. Set to `None` to disable.

Both apply to the final reward in `replace` mode and to the bonus in `bonus` mode (original reward unaffected).

### Full-training fallback chain

After the LERO loop, all valid candidates from all iterations are sorted by (M1, M6, M2). Full training tries rank 0; on `Exception`, falls back to rank 1, then rank 2, etc. The chain is logged to `fallback_chain.json`:

```json
[
  {"rank": 0, "iter": 3, "eval_M1": 0.12, "outcome": "crashed",
   "error": "AssertionError: not action.isnan().any()"},
  {"rank": 1, "iter": 1, "eval_M1": 0.10, "outcome": "success"}
]
```

### Code layout

```
src/lero/
├── __init__.py
├── config.py             # LeroConfig + LLMConfig (+ reward_clip)
├── llm_client.py         # LiteLLM wrapper (Anthropic / OpenAI / OVH)
├── codegen.py            # Code extraction + AST validation
├── scenario_patch.py     # Patched Discovery subclass + reward safeguards
├── loop.py               # Evolutionary loop + fallback chain
└── prompts/
    ├── loader.py
    ├── v1/               # original verbose prompt (bonus + local mode — broken for replace+global)
    ├── v1_global/        # v1 verbose body adapted for replace+global mode (L4 ablation)
    ├── v2/               # paper-faithful minimal (~5 lines)
    ├── v2_min/           # ultra-minimal (3 lines, role + signatures only) — L1 ablation
    ├── v2_fewshot/       # v2 + 2 MPE-style example reward fns — L8 ablation
    └── v2_twofn/         # asks for agent_reward + global_reward decomposition — L21 ablation
```

---

## 2. Experiments executed (2026-04-16)

**Task for all runs below:** Discovery, n=3 agents, t=3 targets, k=1 (1 agent within `covering_range=0.25` to cover), `max_steps=400`, MAPPO, 10M-frame full training, 1M-frame eval per LERO candidate, 4 iterations × 3 candidates.

### 2.1 Final-metric summary (all completed runs)

| Run | Prompt | Final M1 | Final M2 | Final M6 | M4 collisions | M8 util | Outcome | Where |
|---|---|---:|---:|---:|---:|---:|---|---|
| **L8 morning** | `v2_fewshot` | **1.000** | 172.11 | **1.000** | 0.00 | 0.23 | ✅ artifact-clean (rescued from logs) | `lero_rescued/l8/` |
| **L8 afternoon** | `v2_fewshot` | **1.000** | 105.61 | **1.000** | 0.00 | 0.25 | ✅ full artifacts | `lero/.../20260416_1324/` |
| **P1 morning** | `v2` | **1.000** | 104.16 | **1.000** | 0.06 | 0.27 | ✅ artifacts lost (S3 collision) — metrics rescued | `lero_rescued/p1/` |
| **L1 morning** | `v2_min` | 0.625 | 1.16 | 0.857 | 0.00 | 0.92 | ✅ artifacts lost — metrics rescued | `lero_rescued/l1/` |
| **L21 morning** | `v2_twofn` | 0.010 | 149.51 | 0.130 | 35.53 | 0.46 | ✅ full artifacts (reward-hacked) | `lero/.../20260416_1001/` |
| **L4 terminal** | `v1_global` | 0.005 | 334.86 | 0.128 | 0.00 | 0.55 | ✅ full artifacts (reward-hacked) | `lero_l4/` |
| **L1 terminal** | `v2_min` | — | — | — | — | — | ❌ NaN crash @ batch 113/167 (68%) | partial in `lero_l1/` S3 |
| **P1 terminal** | `v2` | — | — | — | — | — | ❌ NaN crash @ batch 151/167 (90%) | partial in `lero_p1/` S3 |
| **L4 morning** | `v1` (broken) | — | — | — | — | — | ❌ all 12 candidates failed `'lidar_targets'` KeyError | `lero/.../20260416_0909/` |
| **L21 afternoon** | `v2_twofn` | — | — | — | — | — | ⚠ stopped early to free quota slot | partial in `lero/.../1501/` |

ER1 baseline for context (n=2, t=4, k=1): **M1 = 0.580**.

### 2.2 Per-experiment summary (best of all attempts)

| Exp | Prompt | Best Final M1 | Stable across runs? | Verdict |
|---|---|---:|---|---|
| **L8** | v2_fewshot (paper-min + 2 examples) | **1.000** (×2) | ✅ very stable | **Best configuration** |
| **P1** | v2 (paper-faithful minimal) | **1.000** (×1, NaN ×2) | ❌ huge variance | Works when LLM samples well; unstable otherwise |
| **L1** | v2_min (ultra-minimal) | 0.625 (×1, NaN ×2) | ❌ huge variance | Minimal prompt → mediocre + unstable rewards |
| **L4** | v1_global (verbose w/ research history) | 0.005 (×1) | n/a (1 run) | Reward-hacked (verbose context didn't help) |
| **L21** | v2_twofn (agent_reward + global_reward split) | 0.010 (×1, partial ×1) | n/a (1 useful run) | Two-function split makes reward MORE exploitable |

### 2.3 Per-iteration evolution (representative)

**P1 morning (M1=1.000) — LERO converged immediately:**

| iter | c0 | c1 | c2 |
|---|---|---|---|
| 0 | M1=0.99 / M2=43 | M1=0.24 / M2=3 | **M1=1.00 / M2=80** ← global best |
| 1 | M1=0.03 / M2=-602 | **M1=1.00 / M2=99** | M1=1.00 / M2=25 |
| 2 | M1=0.09 | M1=0.44 | M1=0.37 |
| 3 | M1=0.12 | M1=0.31 | M1=0.44 |

Iters 2–3 produced strictly worse candidates; iter-1 best held.

**L8 morning (M1=1.000) — LERO converged by iter 1, lockup by iter 3:**

| iter | c0 | c1 | c2 |
|---|---|---|---|
| 0 | ❌ FAIL `'int' has no .float` | **M1=0.65 / M2=70** ← global best | M1=0.02 / M2=-121k |
| 1 | M1=0.04 / M2=-711 | **M1=1.00 / M2=160** ← new best | M1=1.00 / M2=145 |
| 2 | M1=1.00 / M2=137 | M1=0.06 / M2=-1197 | M1=0.96 / M2=123 |
| 3 | **M1=1.00 / M2=150** | M1=1.00 / M2=111 | M1=1.00 / M2=45 |

**P1 terminal (NaN crash) — LLM produced consistently weak rewards:**

| iter | c0 | c1 | c2 |
|---|---|---|---|
| 0 | M1=0.09 / M2=-228 | M1=0.08 / M2=2 | M1=0.01 / M2=87 |
| 1 | **M1=0.12 / M2=-289** | M1=0.07 / M2=-1192 | M1=0.03 / M2=-544 |
| 2 | M1=0.04 / M2=-125 | M1=0.03 / M2=-232 | M1=0.04 / M2=-520 |
| 3 | M1=0.07 / M2=-692 | **M1=0.12 / M2=-175 / M6=0.55** ← picked | M1=0.09 / M2=-241 |

Picked candidate had M2=-175. Trained to batch 151/167 (90%) → NaN actions → crash.

**L4 terminal (M1=0.005, reward-hacked) — all candidates had high positive M2:**

| iter | c0 | c1 | c2 |
|---|---|---|---|
| 0 | M1=0.02 / M2=390 | M1=0.04 / M2=387 | **M1=0.16 / M2=324** ← global best |
| 1 | M1=0.02 / M2=465 | M1=0.02 / M2=319 | M1=0.00 / M2=389 |
| 2 | M1=0.03 / M2=441 | M1=0.02 / M2=402 | M1=0.00 / M2=416 |
| 3 | M1=0.02 / **M2=896** | M1=0.00 / M2=791 | M1=0.02 / M2=602 |

LLM never escaped the high-M2/low-M1 attractor across 4 iters. Verbose v1_global prompt encouraged "scoring" rewards that don't actually solve the task.

### 2.4 Crash pattern — when does NaN happen?

| Run | Picked candidate's M2 | Frames before crash |
|---|---:|---:|
| L1 afternoon (lost) | ~-50 | 7M (77%) |
| P1 terminal | -175 | 9M (90%) |
| L1 terminal | -63 | 6.8M (68%) |

All 3 NaN crashes had **negative M2** at the chosen candidate. L4 had M2=+324 (very positive) and didn't crash, but the policy reward-hacked instead. Pattern:

| Picked candidate's M2 magnitude | Outcome |
|---|---|
| Small (±5..±50) | ✅ stable training |
| Large negative (-100..-1000) | ❌ NaN crash mid-training |
| Large positive (+200..+900) | ⚠ reward hacking (high M2, near-zero M1) |

The 1M-frame eval doesn't separate "stable but mediocre" from "promising but unstable." Hence the need for safeguards (clipping + fallback chain).

### 2.5 Hypothesis status

| ID | Hypothesis | Status | Evidence |
|---|---|---|---|
| H1 | LERO works on Discovery | ✅ Proven | L8 ×2 = 1.000; P1 morning = 1.000 (vs ER1 baseline 0.58) |
| H4 | Discovery > MPE Simple Spread in difficulty | ✅ Supported | High LLM reward variance, 3 NaN crashes from same prompt |
| H5 | Less prompt is better | ⚠ Mixed | L1 (more minimal than v2) hurt; L4 (more verbose than v2) also hurt — v2 is the sweet spot |
| H8 | Few-shot examples help | ✅ Strongly supported | L8 = best + most reliable; M2 = 172 (highest of any successful run) |
| H11 | Two-function split (paper) helps | ❌ Refuted | L21 reward-hacked at 0.01 |

---

## 3. Implementation lessons learned (2026-04-16 incident log)

These are the bugs we hit and fixed in this session. Documenting for future-us so we don't re-do the same diagnostics.

### 3.1 Scenario patch closure scoping (`_obs_mode` NameError)

**Symptom:** All candidates with `obs_source` failed with `NameError: name '_obs_mode' is not defined`.

**Root cause:** A class-body-level `_obs_mode = obs_state_mode` is **not** visible inside method bodies (Python scoping rule — class body is not an enclosing scope for nested functions). The `reward()` path worked because it used `_bs` only as a default arg (evaluated at class-body time).

**Fix (`scenario_patch.py:225`):** Capture via method default arg from the enclosing function's parameter:
```python
def observation(self, agent, _mode=obs_state_mode):
    ...
    if _mode == "global": ...
```

### 3.2 VMAS pip-installed scenario lacks newer kwargs

**Symptom:** `'PatchedDiscoveryScenario' object has no attribute 'dim_c'` (and similar for `comm_proximity`, `_comms_range`).

**Root cause:** OVH containers `pip install vmas` from PyPI; that version of `discovery.py` doesn't `kwargs.pop("dim_c", 0)` like our local fork does. The `comm_proximity`, `dict_obs` kwargs aren't consumed → never set as scenario attributes.

**Fix (`scenario_patch.py:80,113`):** Use `getattr(scenario, "dim_c", 0)` everywhere instead of direct attribute access. Same for `comm_proximity`, `_comms_range`, `use_agent_lidar`, `all_time_covered_targets`.

### 3.3 M8 agent utilization always 0

**Symptom:** `M8_agent_utilization: 0.0000` in every LERO run.

**Root cause:** With `shared_reward=True` (which we use for stability), Discovery's `info()` returns `self.shared_covering_rew` — the same scalar for every agent. Per-agent CV is therefore 0.

**Fix (`scenario_patch.py:194`):** Override `info()` in `PatchedDiscoveryScenario` to always return per-agent `agent.covering_reward`:
```python
def info(self, agent):
    base = super().info(agent)
    base["covering_reward"] = agent.covering_reward
    return base
```
Safe because MAPPO reads rewards from `reward()`, not `info`. Verified: M8 now ranges 0.23–0.92.

### 3.4 OVH S3 prefix bug (the worst one)

**Symptom:** Per-experiment S3 prefixes (`lero_p1`, `lero_l1`, …) failed silently — every parallel job synced to bucket root, overwriting each other's results during FINALIZE.

**Root cause:** Our volume string was `bucket@region/exp_id/:mount:rwd` — **trailing slash on the prefix**. The `ovhai` CLI parses `container@alias[/prefix]:mount_path[:permission]`. With a trailing slash on prefix, OVH's parser fails silently and falls back to `prefix=None`. Every job mounted bucket root.

**Fix (`src/ovh.py:222`):** Build the volume string conditionally without trailing slash:
```python
results_volume = (
    f"{bucket_results}@{region}/{exp_id}:{mount_results}:rwd"
    if exp_id
    else f"{bucket_results}@{region}:{mount_results}:rwd"
)
```

**Verification:** A probe job inspected via `ovhai job get <id>` now shows `prefix: 'lero_p1'` (was `None`).

### 3.5 Streamlit module caching

**Symptom:** After fixing `src/ovh.py`, the user submitted via Streamlit and the bug came back.

**Root cause:** Streamlit imports modules once into a long-running session. `submit_training_job` was the OLD buggy version cached in memory. **Restarting Streamlit** OR submitting via `python -c '...'` (fresh process) is required.

**Mitigation:** Submit via terminal, not Streamlit, for code that's just been changed.

### 3.6 v1 prompt incompatibility with `replace+global` mode

**Symptom:** L4 (using v1 prompt) — every candidate's `enhance_observation` failed with `KeyError: 'lidar_targets'`.

**Root cause:** v1's prompt advertises a separate `obs_state` dict (with `lidar_targets`) for `enhance_observation`. But when run with `obs_state_mode=global`, the function actually receives the global state dict (which has no `lidar_targets`). LLM follows prompt spec → references missing key → crash.

**Fix:** New prompt `v1_global/` — keeps v1's verbose research-history content but uses the single global state dict and `compute_reward` (not `compute_reward_bonus`) signature.

### 3.7 PPO NaN actions from large-magnitude rewards

**Symptom:** `AssertionError: not action.isnan().any()` ~70-90% into 10M-frame full training.

**Root cause:** LLM-generated rewards with |M2| > ~100 (especially negative) cause PPO value function and policy gradients to diverge. Eval at 1M frames doesn't catch this.

**Fix (Layer B + Layer C — `scenario_patch.py:140`, `loop.py:730`):**
- B: clamp reward to `[-reward_clip, +reward_clip]` (default ±50) after `nan_to_num`
- C: full-training fallback chain — on crash, try the next-best candidate from cross-iter rankings

**Caveat:** B deviates from the LERO paper's "raw rewards" claim. Justification: paper used MPE Simple Spread with naturally bounded rewards [0, 5]; Discovery LLMs produce magnitudes 100–1000× larger. Document as an explicit deviation.

---

## 4. Infrastructure reference

### 4.1 LeroConfig

```yaml
lero:
  n_iterations: 4
  n_candidates: 3
  top_k: 2
  eval_frames: 1_000_000
  eval_episodes: 100
  full_frames: 10_000_000
  evolve_reward: true
  evolve_observation: true
  reward_mode: "replace"      # "replace" (paper) | "bonus"
  obs_state_mode: "global"    # "global" (paper) | "local" (CTDE)
  bonus_scale: 0.5            # only for reward_mode=bonus
  reward_clip: 50.0           # ±clip after nan_to_num. Set null to disable.
```

### 4.2 LLMConfig

```yaml
llm:
  model: "gpt-5.4-mini"       # LiteLLM identifier
  temperature: 0.8
  max_tokens: null
  api_base: null              # custom endpoint (OVH or compatible)
  api_key: null               # else read from env (OPENAI_API_KEY etc.)
  context_window: null        # auto-detect, set explicit for vLLM/OVH
  max_retries: 3
  retry_delay: 2.0
  prompt_version: "v2"        # v1 | v1_global | v2 | v2_min | v2_fewshot | v2_twofn
```

### 4.3 OVH submission

```python
from dotenv import dotenv_values
from rendezvous_comm.src.ovh import submit_training_job

submit_training_job(
    "rendezvous_comm/configs/lero/<exp>.yaml",
    llm_env=dotenv_values("rendezvous_comm/.env"),
)
```

**Always submit via terminal (fresh Python process)**, never via Streamlit unless you've restarted Streamlit since the last code change.

Verify the parsed prefix on the resulting job:
```bash
ovhai job get <uuid> --output json | python -c "
import json, sys
d = json.load(sys.stdin)
for v in d['spec']['volumes']:
    if v['mountPath'] == '/workspace/results':
        print('prefix:', v['dataStore']['prefix'])"
```
Must print `prefix: 'lero_<exp>'`, not `prefix: None`.

### 4.4 Output files (per LERO run)

```
results/lero_<exp>/lero/runs/lero/<timestamp>/
├── messages_initial.json    # system + initial user prompt
├── messages_final.json      # final conversation (sliding window)
├── evolution_history.json   # per-iter best M1/M2/M6
├── fallback_chain.json      # which candidates were tried in full training
├── final_metrics.json       # M1..M9 from 10M training of chosen candidate
├── best_reward.py           # source of the candidate that completed full training
├── best_obs.py              # source of corresponding enhance_observation
├── best_policy.pt           # trained MAPPO policy weights
├── benchmarl_final/         # full BenchMARL output of final training
└── iter_{0..N-1}/
    ├── candidate_{0..M-1}_response.txt   # raw LLM output
    ├── candidate_{0..M-1}_reward.py      # extracted compute_reward
    ├── candidate_{0..M-1}_obs.py         # extracted enhance_observation
    ├── candidate_{0..M-1}_metrics.json   # M1..M9 from 1M eval
    ├── feedback.txt                      # feedback shown to LLM next iter
    └── benchmarl_c{0..M-1}/              # BenchMARL output of each candidate
```

---

## 5. Open issues / next steps

### 5.1 Quick wins (low effort)

- **Re-run P1 and L1** with the new safeguards (`reward_clip=50` + fallback chain). Should now complete training even with large-magnitude or unlucky LLM reward samples. ~$4, ~2h parallel.
- **Document the deviation from "raw rewards"** in any paper write-up — note the clip + fallback safeguards, with justification.

### 5.2 Open scientific questions

- **Why is L8 (few-shot) so much more reliable than P1 (no examples)?** L8 hit M1=1.000 ×2; P1 hit it ×1, NaN ×2. Same task, same LLM, just different prompt context. Worth investigating whether examples constrain the LLM's reward magnitude (i.e. examples were in [0, 25] range, anchoring the LLM).
- **L4 (verbose v1_global) reward-hacked.** The verbose prompt's "design dense approach signals + reward partial progress" encouraged the LLM to define rewards that *score* but don't *solve*. Compare what L4's `compute_reward.py` actually looks like vs L8's.
- **L21 (two-function split) reward-hacked too.** Decomposing into `agent_reward` + `global_reward` made the reward MORE exploitable, not less. Counter to paper's H11. Worth a careful read of the LERO paper's reward functions to understand if MPE Simple Spread is just trivially safer.

### 5.3 Deferred phases (per original plan)

- **Phase 5 — task scaling:** n=2 t=4 k=1, n=4 t=4 k=1, **n=4 t=4 k=2 (rendezvous, the actually-hard task)**, n=4 t=4 k=2 with `dim_c=8` communication.
- **Phase 6 — loop tuning:** more candidates per iter, more iterations, longer eval, full conversation (no sliding window).

### 5.4 Phase 4 ablations not yet run

The original plan listed L2, L3, L5–L7, L9–L20, L22, plus E1–E4 loop ablations. Of those, only L1, L4, L8, L21 ran. Whether to continue depends on what we learn from re-running the L1+P1 with safeguards — if reward-magnitude is the dominant variable, it may matter more to study **how examples constrain magnitude** (a specific L8 follow-up) than to grind through the full prompt-structure ablation.

---

## 6. References

- LERO paper: [arXiv:2503.21807](https://arxiv.org/abs/2503.21807)
- Code entry point: `rendezvous_comm/train.py` → `runner.run_lero()` → `lero/loop.LeroLoop.run()`
- Configs: `rendezvous_comm/configs/lero/{p1,l1,l4,l8,l21}.yaml`
- Rescued metrics: `results/lero_rescued/{p1,l1,l8}/`
- Full surviving artifacts: `results/lero/runs/lero/{20260416_1001,20260416_1324}/`, `results/lero_l4/`
