# Lessons learned — incident log

Bugs hit and fixed during reproducibility work. Distilled here so
future-us doesn't re-do the diagnostics.

Sources: `rendezvous_comm/docs/lero.md` §3 (2026-04-16 incident log) +
F8.4 Phase 1-11 audit work (2026-05).

## Architecture / scoping

### 1. Class-body locals not visible in method bodies

A class-body-level `_obs_mode = obs_state_mode` is **not** visible
inside method bodies (Python scoping rule — class body is not an
enclosing scope for nested functions).

**Fix**: capture via method default arg from the enclosing function's
parameter:

```python
def observation(self, agent, _mode=obs_state_mode):
    ...
    if _mode == "global": ...
```

Same fix for `reward_clip`, `bonus_scale`, any closed-over LLM-config
field.

### 2. VMAS pip version lacks newer kwargs

OVH containers `pip install vmas`; that version of `discovery.py`
doesn't `kwargs.pop("dim_c", 0)` like local fork. Symptom:
`'PatchedDiscoveryScenario' object has no attribute 'dim_c'`.

**Fix**: `getattr(scenario, "dim_c", 0)` everywhere instead of direct
attribute access. Same for `comm_proximity`, `_comms_range`,
`use_agent_lidar`, `all_time_covered_targets`.

### 3. Pickle can't serialise local classes

`make_patched_discovery_class()` returns a local class (defined inside
a function). BenchMARL's `config.pkl` write fails:
`AttributeError: Can't pickle local object 'make_patched_discovery_class.<locals>.PatchedDiscoveryScenario'`.

**Fix**: `ScenarioEnvFunFactory.__getstate__` persists `config` +
`patched_kwargs` (primitive strings); `__setstate__` rebuilds the
class via `make_patched_discovery_class(**kwargs)`.

## Metric semantics

### 4. M8 agent utilization always 0

With `shared_reward=True`, Discovery's `info()` returns
`self.shared_covering_rew` — same scalar for every agent → per-agent CV
is 0.

**Fix**: Override `info()` in patched class to return per-agent
`agent.covering_reward`. Safe — MAPPO reads rewards from `reward()`,
not `info`.

### 5. M6 > 1.0 from broken cumsum on tc

`_extract_targets_covered` applied `cumsum` to VMAS's per-step
`covered_targets.sum(-1)` (a state, not a delta). With
`targets_respawn=False`, this monotonically accumulates to
`count × time_remaining` and overshoots `n_targets`.

**Fix attempt #1 (wrong)**: removed the cumsum entirely → M1 became
mathematically impossible (n_targets simultaneous coverage requires
K×T > N agents).

**Fix attempt #2 (correct, post-2026-05-12 audit)**: restored the
cumsum (matches rendezvous_comm's formula in `src/metrics.py:109`),
ALSO added `.clamp(max=n_targets)` to M6 before dividing (matches
rendezvous_comm's `src/metrics.py:165` clamp). M1 = "did cumulative
coverage events cross `n_targets` at any step"; M6 ∈ [0, 1].

### 6. Cumulative `targets_covered` exceeds `n_targets`

Empirically (Phase 5b eval data) the same target gets covered multiple
times per episode despite VMAS's teleport-on-cover. Cause not fully
understood from static reading — possibly the teleport's far-outside
position lands near arena under some configs.

**Workaround**: the clamp on M6 absorbs this; M1's threshold is still
meaningful (any-step crossing). Documented as open question; doesn't
block reproducibility.

## OVH dispatch

### 7. OVH S3 prefix silent-fallback (trailing slash)

Volume string `bucket@region/prefix/:mount:rwd` — **trailing slash on
prefix**. `ovhai` parses it as `prefix=None`. Every parallel job
mounted bucket root; FINALIZING overwrote each other.

**Fix**: build the volume string conditionally without trailing
slash. **Verify**: `ovhai job get <id>` shows `prefix: 'lero_p1'`,
not `None`.

### 8. Streamlit module caching

Streamlit imports modules once into a long-running session. After
editing `src/ovh.py`, the Submit page still hit the old bug.

**Fix**: restart Streamlit after editing imported modules. Or use the
terminal CLI for code that's just been changed.

### 9. PPO NaN actions from large LERO rewards

`AssertionError: not action.isnan().any()` ~70-90% into 10M-frame full
training. LLM-generated rewards with `|M2| > 100` cause value-function
+ policy gradients to diverge. Eval at 1M frames doesn't catch this.

**Fix**: two-layer.

- **B**: clamp reward to `[-reward_clip, +reward_clip]` (default ±50)
  after `nan_to_num`.
- **C**: full-training fallback chain — on crash, try the next-best
  candidate from the iter-cross rankings.

Documented deviation from the LERO paper's "raw rewards" claim.

## LERO orchestrator

### 10. `train_full()` return value discarded

`LeroOrchestrator._run_full_training` had:

```python
_ = self._full_trainer.train_full(...)  # ← return value discarded
```

Result: `final_summary.json` reported the inner-loop (1M-frame) M1,
not the post-full-train (10M-frame) M1. Phase 5a showed M1=0.035 when
the real number was 0.795.

**Fix**: capture the return, plumb into `LeroRunSummary.best_candidate_full_metrics`.

### 11. Inner-loop checkpoints throwaway

Each LERO candidate's BenchMARL training writes ~200 MiB of
checkpoints during 1M-frame eval. The full-trainer retrains from
scratch — these are never loaded again. Phase 5a shipped 2.5 GiB of
throwaway state to S3 per run.

**Fix**: `_prune_inner_loop_checkpoints(cand_run_dir)` deletes the
`checkpoints/` subtree after metrics extraction. Preserves
`config.pkl` + `scalars/` (small, useful for analysis).

### 12. Full-train kept all checkpoints (`keep_checkpoints_num=1000`)

Phase 5a YAML had `keep_checkpoints_num: 1000` so the full-training
saved 17 × 107 MiB = 1.83 GiB.

**Fix**: `training.delete_intermediate_checkpoints_on_success: true`
in LERO YAMLs. On success, `prune_intermediate_checkpoints_keep_latest`
drops all but the final checkpoint (Streamlit replay still works).

## Other

### 13. Submit page dirty-on-load (LERO YAMLs)

LERO YAMLs partial-fill `lero:` / `llm:` sections; my F9.8 widgets
render the full Pydantic field set; snapshot didn't have the schema
fill the form did → `current_form != snapshot_form` → "Save" shown
on a fresh load.

**Fix**: `submit_workflow._fill_pydantic_defaults` aligns the snapshot
with what the widgets emit. `_schema_from_pydantic` skips
`None`-default fields entirely so they don't add phantom keys.

## Citation source

For numerical results context, see
[`docs/_drafts/rendezvous_comm_history.md`](../_drafts/rendezvous_comm_history.md)
(scaffolding, deleted post-F10.6 extraction once cited from coopvmas).
