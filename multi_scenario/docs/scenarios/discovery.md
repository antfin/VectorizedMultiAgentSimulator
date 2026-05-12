# Discovery

VMAS scenario where **N agents** must cover **T targets**, with **K
agents required simultaneously per target** (rendezvous). Once
covered, the target teleports out of the arena.

## Default knobs

```yaml
scenario:
  type: discovery
  params:
    n_agents: 4
    n_targets: 4
    agents_per_target: 2          # k = 2 → rendezvous task
    covering_range: 0.25          # rendezvous_comm S3b-local; 0.35 = ER1
    max_steps: 400                # rendezvous_comm; 200 = ER1
    targets_respawn: false        # REQUIRED for M1 cumsum semantics
    shared_reward: true
    lidar_range: 0.35
    n_lidar_rays_entities: 15     # target lidar
    n_lidar_rays_agents: 12       # agent lidar (if use_agent_lidar=true)
    use_agent_lidar: true
    covering_rew_coeff: 1.0
    time_penalty: -0.01
    agent_collision_penalty: -0.01
```

## Why `targets_respawn=false` is required

M1 semantics rely on cumulative coverage events crossing `n_targets`.
With `targets_respawn=true`, agents could trivially farm the same
target indefinitely → M1 hits 1 spuriously. See
[Concepts → Metrics](../concepts/metrics.md).

## M1 / M3 / M6 wiring

- **M1 success**: `(cumsum(targets_covered).max(time) >= n_targets).mean()`.
- **M3 steps to coverage**: first step where the cumsum crosses `n_targets`; falls back to `max_steps` for episodes that never succeed.
- **M6 coverage progress**: `(cumsum.max.clamp(max=n_targets) / n_targets).mean()`.

The clamp on M6 matches rendezvous_comm's formula
(`rendezvous_comm/src/metrics.py:165`). See
[Concepts → Metrics](../concepts/metrics.md) for the full derivation.

## Canonical configs

- `experiments/discovery/baseline/configs/baseline.yaml` — ER1 reference (cr=0.35, ms=200, non-LERO).
- `experiments/discovery/lero/configs/lero_s3b_local.yaml` — rendezvous_comm S3b-local port (cr=0.25, ms=400).
- `experiments/discovery/lero/configs/lero_s3b_local_er1params.yaml` — LERO at ER1 params (cr=0.35, ms=200).

## See also

- [Concepts → LERO](../concepts/lero.md) — what LERO evolves on top of Discovery.
- [Reproducibility → LERO S3b-local](../reproducibility/lero_s3b_local_reproduction.md) — the Phase 6 comparison.
