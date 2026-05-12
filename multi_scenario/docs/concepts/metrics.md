# Metrics M1–M9

Universal metric IDs used across every scenario, Streamlit page, and CSV output.

| ID | Name | Source | Units |
|---|---|---|---|
| M1 | Success rate | `Scenario.success_predicate(rollout)` per scenario; for Discovery: `(cumsum_targets_covered.max ≥ n_targets).mean()` | fraction in `[0, 1]` |
| M2 | Avg return | `rollout.episode_returns.mean()` | unbounded float |
| M3 | Avg steps to completion | `rollout.episode_lengths.mean()` (or `done_at_step`) | steps |
| M4 | Avg collisions per episode | `rollout.episode_collisions.mean()` | count |
| M5 | Comm tokens per episode | Comm-gated; `None` when `scenario.has_comm() == False` | count |
| M6 | Coverage progress | `Scenario.coverage_progress(rollout)` per scenario; for Discovery: `(cumsum.max.clamp(max=n_targets) / n_targets).mean()` | fraction in `[0, 1]` |
| M7 | Sample efficiency | end-of-run; deferred | TBD |
| M8 | Agent utilization (CV) | per-agent covering count std/mean | unbounded; 0 = balanced |
| M9 | Spatial spread | mean pairwise agent distance | unbounded |

## M1 / M6 formula alignment with rendezvous_comm

After the Phase 11 metric audit (2026-05-12), our M1/M6 match rendezvous_comm's
formula byte-for-byte:

```python
# rendezvous_comm/src/metrics.py:109,135,165
targets_covered_total += info["targets_covered"]    # cumsum across time
task_done = targets_covered_total >= n_targets      # any-step crossing
success_rate = is_done.float().mean()               # M1

coverage = targets_covered_total.clamp(max=n_targets)
m6 = (coverage / n_targets).mean()                  # clamped to [0, 1]
```

The clamp is load-bearing: empirically cumulative `targets_covered`
can exceed `n_targets` within an episode (targets get re-covered
despite VMAS's teleport-on-cover). Without the clamp, M6 > 1.0.

See [Reproducibility → LERO S3b-local](../reproducibility/lero_s3b_local_reproduction.md)
for the empirical numbers.

## M5 — comm tokens

Counted per agent-step where `dim_c > 0 and not agent.silent`. With
`shared_reward=True` and no comm scenarios in the current set, M5
returns `None` for every run today. Becomes relevant for ER2/ER3
ablations under Phase 11.

## M8 — agent utilization

Coefficient of variation of per-agent covering counts. Requires the
patched Discovery scenario's `info()` override to return per-agent
`agent.covering_reward` instead of `shared_covering_rew`; see
[Operations → Lessons learned](../operations/lessons_learned.md).
