# ER3 Verification Report

**Date**: 2026-03-22
**Runs verified**: 3

## Summary Table

| Variant | GNN Class | M1 (SR) | M2 (Return) | M3 (Steps) | M4 (Coll) | M5 (Tokens) | M6 (Coverage) | M8 (Util) | M9 (Spread) | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| al_lp_sr_gatv2_k2_l035_s0 | GATv2Conv | 0.0 | -0.706 | 100.0 | 27.815 | 0.0 | 0.091 | 0.0 | 0.540 | max_steps_actual=100 (cfg=200, actual=100) |
| al_lp_sr_gatv2_ms100_k2_l035_s0 | GATv2Conv | 0.0 | -0.706 | 100.0 | 27.815 | 0.0 | 0.091 | 0.0 | 0.540 | max_steps_actual=100 (cfg=400, actual=100); DUPLICATE METRICS |
| al_lp_sr_graphconv_k2_l035_s0 | GraphConv | 0.0 | -0.549 | 100.0 | 15.465 | 0.0 | 0.121 | 0.0 | 0.449 | max_steps_actual=100 (cfg=200, actual=100) |

## Detailed Verification

### al_lp_sr_gatv2_k2_l035_s0

- **Config verification**:
  - `model_type`: gnn -- OK
  - `gnn_class`: GATv2Conv -- OK (matches variant tag "gatv2")
  - `gnn_topology`: full -- OK
  - `use_agent_lidar`: true -- OK (variant has "al")
  - `shared_reward`: true -- OK (variant has "sr")
  - `agent_collision_penalty`: -0.01 -- OK (variant has "lp" = low penalty)
  - `max_steps`: 200 (configured), 100 (actual due to bug)
  - `dim_c`: 0, no communication channel -- OK
  - `targets_respawn`: false -- OK
- **Metrics**: M1-M6, M8-M9 present. M7 (sample_efficiency) absent (SR=0, never converged). All values match master_results.csv.
- **Anomalies**:
  - M1=0.0: GATv2Conv completely failed to learn the task
  - M4=27.815: Very high collision count (worst across all ER3 runs)
  - M3=100.0: Capped at actual max_steps (bug), never completed any episode
  - M8=0.0: Zero agent utilization
  - max_steps_actual=100 confirmed (cfg=200, actual=100)
- **Status**: ANOMALY -- zero success rate with very high collisions; max_steps_actual=100 present

### al_lp_sr_gatv2_ms100_k2_l035_s0

- **Config verification**:
  - `model_type`: gnn -- OK
  - `gnn_class`: GATv2Conv -- OK
  - `gnn_topology`: full -- OK
  - `use_agent_lidar`: true -- OK
  - `shared_reward`: true -- OK
  - `agent_collision_penalty`: -0.01 -- OK
  - `max_steps`: 400 (configured), 100 (actual due to bug)
  - `dim_c`: 0 -- OK
  - `targets_respawn`: false -- OK
- **Metrics**: M1-M6, M8-M9 present. All values match master_results.csv.
- **Anomalies**:
  - **CRITICAL**: Metrics are IDENTICAL to al_lp_sr_gatv2_k2_l035_s0 (same M1-M9 values, same final_entropy=-2.1808, same final_eval_reward=-0.7233, same n_iterations=166, same policy_params=264). This strongly suggests the ms100 run reused the non-ms100 evaluation results or the max_steps_actual=100 made both runs behave identically (both ran at actual max_steps=100).
  - max_steps_actual=100 confirmed (cfg=400, actual=100) -- this is the **intended fix target**
  - M1=0.0, M4=27.815: Same failure pattern as non-ms100 variant
- **Status**: ANOMALY -- max_steps_actual=100 present; metrics are duplicated from non-ms100 run (invalidates the ms100 experiment purpose)

### al_lp_sr_graphconv_k2_l035_s0

- **Config verification**:
  - `model_type`: gnn -- OK
  - `gnn_class`: GraphConv -- OK (matches variant tag "graphconv")
  - `gnn_topology`: full -- OK
  - `use_agent_lidar`: true -- OK
  - `shared_reward`: true -- OK
  - `agent_collision_penalty`: -0.01 -- OK
  - `max_steps`: 200 (configured), 100 (actual due to bug)
  - `dim_c`: 0 -- OK
  - `targets_respawn`: false -- OK
- **Metrics**: M1-M6, M8-M9 present. All values match master_results.csv.
- **Anomalies**:
  - M1=0.0: GraphConv also completely failed to learn
  - M4=15.465: High collisions but significantly fewer than GATv2Conv (27.8 vs 15.5)
  - M3=100.0: Capped at actual max_steps
  - M8=0.0: Zero agent utilization
  - max_steps_actual=100 confirmed
- **Status**: ANOMALY -- zero success rate; max_steps_actual=100 present; slightly better than GATv2 on collisions/return

## Cross-Run Sanity Checks

### GATv2 vs GraphConv differences

Yes, the two GNN architectures show different results despite both failing (SR=0):

| Metric | GATv2Conv | GraphConv | Delta |
|---|---|---|---|
| M2 (Return) | -0.706 | -0.549 | GraphConv +0.157 better |
| M4 (Collisions) | 27.815 | 15.465 | GraphConv 44% fewer collisions |
| M6 (Coverage) | 0.091 | 0.121 | GraphConv slightly better |
| M9 (Spread) | 0.540 | 0.449 | GraphConv lower spread (more clustered) |
| Policy params | 264 | 252 | GATv2 slightly larger (attention heads) |
| final_entropy | -2.181 | -0.386 | GraphConv much higher entropy (less collapsed) |
| Throughput | 3426 fps | 4159 fps | GraphConv 21% faster |

GraphConv performed marginally better on all metrics but both architectures fundamentally failed. GATv2Conv's very low entropy (-2.18) suggests its policy collapsed early.

### Is ms100 flagged with max_steps_actual=100?

Yes. The master_results.csv notes column contains `max_steps_actual=100:cfg=400_actual=100` for the ms100 variant. The config shows `max_steps: 400` but the actual runtime used 100 steps due to the known `config.pop("max_steps")` bug.

**Additional concern**: The ms100 run has identical metrics to the non-ms100 GATv2 run, suggesting either (a) the bug made both runs identical since actual max_steps=100 in both cases, or (b) metrics were accidentally copied. Both runs also show identical n_iterations (166), policy_params (264), and final_entropy (-2.1808), which is consistent with explanation (a) -- the bug nullified the ms100 config difference.

### Is topology "full" for all runs?

Yes. All three configs have `gnn_topology: full`. This is correct per the known constraint that `from_pos` topology requires dict observations which are not available in the current training setup.

## Anomalies Found

1. **ALL ER3 RUNS: Zero success rate (M1=0.0)** -- Neither GNN architecture learned the rendezvous task. Both GATv2Conv and GraphConv completely failed. This is a fundamental ER3 result issue, not just a max_steps_actual=100 artifact. For comparison, the MLP baseline (ER1 al_lp_sr_k2_l035_s0) achieved M1=0.04 under the same bug conditions.

2. **ALL ER3 RUNS: max_steps_actual=100 present** -- All three runs were affected by the `config.pop("max_steps")` bug, running at actual max_steps=100 instead of configured 200/400. These runs must be re-executed after the bug fix.

3. **ms100 DUPLICATE METRICS** -- The ms100 GATv2 run produced identical metrics to the non-ms100 GATv2 run (down to floating point precision). This is expected because the max_steps_actual=100 set actual max_steps=100 in both cases, making the configured difference (200 vs 400) irrelevant. The ms100 experiment is therefore invalid and must be re-run after the bug fix.

4. **Very high collision counts** -- GATv2Conv produced M4=27.815 collisions per episode, which is the highest across all experiments in master_results.csv. GraphConv produced M4=15.465, also elevated. The GNN models appear to create collision-prone policies.

5. **Zero agent utilization (M8=0.0)** -- All runs show M8=0.0, same as other shared_reward runs (ER1 sr variants also show M8=0.0). This is expected behavior for shared_reward=true since M8 measures individual agent contribution differentiation.

6. **GATv2Conv policy collapse** -- GATv2Conv's final entropy of -2.18 (vs GraphConv's -0.39) indicates severe policy collapse, suggesting the attention mechanism may be detrimental in this setting or requires different hyperparameters.

7. **Only single seed (s0)** -- All ER3 runs use seed=0 only. No multi-seed verification is available, making statistical conclusions unreliable.
