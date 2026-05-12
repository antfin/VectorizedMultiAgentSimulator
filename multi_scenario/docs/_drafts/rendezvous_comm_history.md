# rendezvous_comm experiment history — reference doc

> **Status: scaffolding (F10.9).** Distilled from `rendezvous_comm/docs/lero.md`
> (the 674-line canonical doc, dated 2026-04-17). Use this as a citation
> source when writing new-repo reports; **delete after reproducing the
> relevant results in coopvmas** (F10.8 cleanup pass picks up `_drafts/`).
>
> Original lives at `rendezvous_comm/docs/lero.md` if you need the verbose
> form. This doc keeps only the load-bearing facts.

## 1. Headline results

The rendezvous_comm LERO work delivered ONE landmark result and a set of
supporting ablations across two phases:

### 1.1 The breakthrough: S3b-local — **M1 = 0.88 on k=2 with LOCAL sensors only**

| Method | Information available | M1 | M3 (steps to coverage) | M6 |
|---|---|---:|---:|---:|
| **LERO S3b-local** (obs-only, ER1 reward, LOCAL sensors) | LiDAR + LLM-designed features | **0.880** | 186 | 0.970 |
| ER3 GNN (GATv2) | LiDAR + GNN message-passing | 0.710 | 250 | 0.918 |
| ER2 proximity comm | LiDAR + proximity messages | 0.530 | 295 | 0.825 |
| ER1 no comm | LiDAR only | 0.405 | 317 | 0.800 |
| LERO S3b-global (obs-only, ORACLE) | LiDAR + global state | **1.000** | 68 | 1.000 |

Task setup: 4 agents, 4 targets, k=2 agents-per-target rendezvous,
covering_range=0.25, max_steps=400, 10M-frame full training.

**Key claim**: S3b-local beats ER3 GNN by 17 percentage points using only
local sensors — no communication channel, no GNN, no oracle. The
improvement comes entirely from LLM-designed feature engineering of the
same LiDAR data.

### 1.2 Phase 4 prompt ablations (n=3, t=3, k=1 — easier task)

| Run | Prompt | M1 (full) | M2 | M6 | Notes |
|---|---|---:|---:|---:|---|
| L8 (morning + afternoon) | `v2_fewshot` | **1.000** | 105.6–172.1 | 1.000 | Best prompt — examples anchor reward magnitude |
| P1 morning | `v2` | **1.000** | 104.2 | 1.000 | Paper-faithful minimal |
| L1 morning | `v2_min` | 0.625 | 1.2 | 0.857 | 3-line ultra-minimal — partial success |
| L21 | `v2_twofn` | 0.010 | 149.5 | 0.130 | Reward-hacked (decomposed signature) |
| L4 | `v1_global` | 0.005 | 334.9 | 0.128 | Reward-hacked |
| P1 terminal | `v2` | — | — | — | NaN @ 90% — needed reward_clip fallback |
| L1 terminal | `v2_min` | — | — | — | NaN @ 68% |

### 1.3 Phase 5 k=1 task scaling

| Run | Task | M1 | Baseline (ER1) |
|---|---|---:|---:|
| S1 | n=2, t=4, k=1 | **1.000** | 0.580 |
| L8 | n=3, t=3, k=1 | **1.000** | ~0.58 |

LERO dominates all k=1 tasks tested.

### 1.4 Phase 5 k=2 reward-design failures

| Run | LLM | Comm | Approach | Final M1 |
|---|---|---|---|---:|
| S3a_gpt5 | gpt-5.4 | none | LLM reward (k2 prompt) | 0.090 (eval-best 0.86!) |
| S3 | gpt-mini | none | LLM reward (k1 prompt) | 0.105 |
| S3ac | gpt-mini | dim_c=8 | LLM reward + comm | 0.080 |
| S3a_gpt | gpt-mini | none | LLM reward (k2 prompt) | 0.000 |

**Reward-design path failed for k=2 with every LLM and prompt tried.**
Anti-crowding penalties consistently fight convergence. S3a_gpt5 hit
M1=0.86 at one eval but degraded to 0.09 at final training (the
eval-vs-final regression motivated the peak-M1-checkpoint follow-up
still pending).

## 2. The winning S3b-local observation function

Iteration 3 candidate 2, ~28 features from local sensors only:

```text
# Target features (8)
t1, t2          # nearest, 2nd-nearest target distance (from sorted lidar)
tdx, tdy        # direction to nearest target (cos/sin of ray angle)
t_gap = t2 - t1 # gap → "is this target isolated?"
t_count         # number of target rays < covering_range
t_intensity     # sum(1/dist) for close target rays — sharper proximity signal

# Agent features (8) — same shape, but on lidar_agents
a1, a2; adx, ady; a_gap; a_count; a_intensity

# Coordination signals (4) — THE KEY INNOVATION
target_near = (t1 < covering_range)         # "target in range"
agent_near  = (a1 < covering_range)         # "partner in range"
hold_signal       = target_near * agent_near  # "STAY: target + partner both here"
approach_signal   = target_near * (1 - agent_near)  # "need partner"
crowd_signal      = clamp((a_count - 1) / 3)  # "too crowded, move"
sparsity_signal   = clamp(1 - t_count / n_targets)  # "few targets, explore"

# Self features (4)
speed, vx, vy, polar position, one-hot role
```

**Why this works**: `hold_signal` (line 4 of coordination signals)
breaks the "ships passing in the night" failure — without it, two
agents converging on the same target tend to OVERSHOOT (one arrives,
sees lidar reading drop, leaves before the second arrives). With it,
the first arrival holds while the second closes in. This single
feature is responsible for the gap between iter-3-cand-2 (M1=0.88) and
iter-0-cand-1 (M1=0.04) — the LLM discovered it in iteration 3.

### How LERO evolved the observation across 4 iterations

| iter | candidate | inner-loop M1 | feature additions |
|---|---|---:|---|
| 0 | c1 | **0.040** | basic: nearest target/agent distance + direction + count |
| 1 | c0/c1 | 0.030 | sector-based features, polar position |
| 2 | c0 | **0.020** | gaps, intensity, more agent-count features |
| 3 | c2 | **0.060** | **hold/approach/crowd/sparsity signals** ← breakthrough |

Inner-loop M1 at 1M frames was a weak signal (0.06 max) — the
candidate selection bet on coordination-signal-rich code AND the
full-training step (10M frames) lifted it from 0.06 to 0.88. **The
LLM's contribution was the inductive prior baked into the obs features;
MAPPO did the rest with full training time.**

## 3. Implementation lessons (the bug incident log)

These are the bugs hit and fixed during rendezvous_comm's LERO work.
Document them in the new repo's `docs/operations/` so we don't re-do
the diagnostics.

### 3.1 Scenario-patch closure scoping (`NameError: _obs_mode`)

**Symptom**: All candidates with `obs_source` failed with
`NameError: name '_obs_mode' is not defined`.

**Root cause**: A class-body-level `_obs_mode = obs_state_mode` is
**not visible** inside method bodies (Python scoping rule — class body
is not an enclosing scope for nested functions).

**Fix**: Capture via method default arg from the enclosing function's
parameter:

```python
def observation(self, agent, _mode=obs_state_mode):
    ...
    if _mode == "global": ...
```

Same fix pattern applies to `reward_clip`, `bonus_scale`, and any
LLM-config field referenced inside the patched class's methods.

### 3.2 OVH-installed VMAS lacks newer kwargs

**Symptom**: `'PatchedDiscoveryScenario' object has no attribute 'dim_c'`
(and similar for `comm_proximity`, `_comms_range`, `use_agent_lidar`,
`all_time_covered_targets`).

**Root cause**: OVH containers `pip install vmas` from PyPI; that
version's `discovery.py` doesn't `kwargs.pop("dim_c", 0)` like the
local fork does.

**Fix**: Use `getattr(scenario, "dim_c", 0)` everywhere instead of
direct attribute access. Same for the other kwargs above.

### 3.3 M8 agent utilization always 0

**Symptom**: `M8_agent_utilization: 0.0000` in every LERO run.

**Root cause**: With `shared_reward=True`, Discovery's `info()` returns
`self.shared_covering_rew` — same scalar for every agent → per-agent
coefficient-of-variation is 0.

**Fix**: Override `info()` in the patched class to always return
per-agent `agent.covering_reward`:

```python
def info(self, agent):
    base = super().info(agent)
    base["covering_reward"] = agent.covering_reward
    return base
```

Safe because MAPPO reads rewards from `reward()`, not from `info`.
Verified: M8 now ranges 0.23–0.92 across runs.

### 3.4 OVH S3 prefix silent-fallback bug (the worst one)

**Symptom**: Parallel jobs synced to bucket root, overwriting each
other's results during FINALIZE. Per-experiment prefixes
(`lero_p1`, `lero_l1`, …) silently failed.

**Root cause**: Volume string was `bucket@region/prefix/:mount:rwd` —
**trailing slash on prefix**. `ovhai` parses
`container@alias[/prefix]:mount_path[:permission]`. Trailing slash on
prefix silently parses as `prefix=None`. Every job mounted bucket root.

**Fix**: Build the volume string conditionally without trailing slash:

```python
results_volume = (
    f"{bucket}@{region}/{exp_id}:{mount}:rwd"
    if exp_id else f"{bucket}@{region}:{mount}:rwd"
)
```

**Verification**: `ovhai job get <id>` shows `prefix: 'lero_p1'`
(was `None` pre-fix).

### 3.5 Streamlit module caching

**Symptom**: After fixing `src/ovh.py`, submitting via Streamlit hit
the old bug.

**Root cause**: Streamlit imports modules once into a long-running
session. Old buggy version cached.

**Mitigation**: Restart Streamlit after editing imported modules.
Or use the terminal CLI for code that's just been changed.

### 3.6 `v1` prompt incompatibility with `replace+global` mode

**Symptom**: L4 (v1 prompt) — every candidate's `enhance_observation`
failed with `KeyError: 'lidar_targets'`.

**Root cause**: v1's prompt advertises a separate `obs_state` dict
(with `lidar_targets`) for `enhance_observation`. But in
`obs_state_mode=global`, the function receives the GLOBAL state dict
(no `lidar_targets`). LLM follows prompt → references missing key →
crash.

**Fix**: New prompt `v1_global/` — keeps v1's verbose content but uses
the single global state dict and `compute_reward` (not
`compute_reward_bonus`) signature.

### 3.7 PPO NaN actions from large-magnitude rewards

**Symptom**: `AssertionError: not action.isnan().any()` ~70–90% into
10M-frame full training.

**Root cause**: LLM-generated rewards with `|M2| > ~100` (especially
negative) cause PPO value-function and policy gradients to diverge.
Eval at 1M frames doesn't catch this.

**Fix** (two-layer):

- **B**: clamp reward to `[-reward_clip, +reward_clip]` (default ±50)
  after `nan_to_num`.
- **C**: full-training fallback chain — on crash, try the next-best
  candidate.

**Caveat**: B deviates from the LERO paper's "raw rewards" claim. The
paper used MPE Simple Spread with naturally bounded rewards `[0, 5]`;
Discovery LLMs produce magnitudes 100–1000× larger. Document as
explicit deviation when citing.

## 4. Pending experiments rendezvous_comm never ran

Listed for context — useful framing for new-repo follow-ups (Phase 11).

### Immediate (high priority)

- **S3b-local × 3 seeds** — robustness check for the M1=0.88 claim.
  Single-seed result; need n=3 to publish.
- **Inspect `best_obs.py` for S3b-global** — what oracle features did
  the M1=1.000 candidate use? Compare against S3b-local's coordination
  signals.

### Medium

- **S3b + dim_c=8 comm** — does adding communication on top of LERO's
  obs help further? Or is the obs sufficient?
- **S3a_gpt5 with peak-M1-checkpointing** — recover the M1=0.86 policy
  that degraded to 0.09 during full training. Implements
  best-checkpoint policy (motivated by this run).
- **Obs-only LERO at k=3** (e.g. n=6, t=4, k=3) — does the approach
  generalise to higher coordination complexity?

### Deferred / dropped

- **More Phase 4 prompt ablations** (L2/L3/L5–L7) — deprioritised; the
  prompt ablation question was less important than obs-only vs
  reward-design.
- **Phase 6 loop tuning** (more candidates, more iters, longer eval) —
  lower priority now that obs-only is the winning approach.

## 5. Open scientific questions

1. **What makes S3b-local's observation enhancement work?** Section 2
   above documents the features; the *why* is the hold-signal +
   coordination-signal combination. But: is this scenario-specific or
   does the pattern generalise? Test on Navigation / Transport.

2. **Why can't the LLM design a k=2 reward?** Every reward-design
   attempt reward-hacked. Anti-crowding penalties consistently fight
   convergence. Prompt issue, LLM capability limit, or fundamental to
   reward design for coordination tasks?

3. **Few-shot examples anchor reward magnitude.** L8's examples use
   magnitudes in `[0, 25]` range. Does that prevent the LLM from
   generating ±1000-magnitude rewards that cause NaN crashes? Test
   with a stripped-examples variant of v2_fewshot.

4. **Inner-loop M1 was a poor selection signal.** Iter-3-cand-2 had
   M1=0.06 at 1M frames → M1=0.88 at 10M. Same pattern would have
   missed the winner under stricter pruning. Implication: LERO
   candidate ranking should weight feature richness, not just
   1M-frame M1.

## 6. How this informs the new-repo (coopvmas) work

When porting this story into coopvmas reports:

- Cite §1.1 as the **rendezvous_comm headline** (S3b-local M1=0.88, ER1
  baseline 0.405, GNN ER3 0.71).
- Cite §1.4 to motivate why **obs-only is the right approach** (every
  reward-design attempt failed for k=2).
- Cite §2 (the winning code) as the **specific contribution** —
  hold_signal etc. — that coopvmas needs to reproduce to claim a
  successful reproduction.
- Cite §3 as the **operational know-how** (which bugs we already
  paid for). Move these into `docs/operations/lessons.md` once the
  doc structure lands.
- Cite §4 to seed **Phase 11's per-scenario campaign** decisions
  (which experiments to actually fire).

## 7. References

- Source: `rendezvous_comm/docs/lero.md` (674 lines, 2026-04-17).
- LERO paper: <https://arxiv.org/abs/2503.21807>
- Phase 6 coopvmas comparison report: `docs/f8_4_phase6_comparison.md`
  (cites this doc; will move to `docs/reproducibility/lero_s3b_local.md`
  under F10.1's mkdocs structure).
