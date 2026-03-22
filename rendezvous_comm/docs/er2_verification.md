# ER2 Verification Report

**Date**: 2026-03-22
**Runs verified**: 11

## Summary Table

| Variant | dim_c | Comm Mode | AL | LP | SR | M1 (SR) | M2 (Return) | M3 (Steps) | M4 (Coll) | M5 (Tokens) | M6 (Cov) | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prox_dc8_k2_l035_s0 | 8 | proximity | N | N | N | 0.035 | -0.154 | 99.13 | 8.38 | 3200 | 0.518 | max_steps_bug |
| prox_dc8_30m_k2_l035_s0 | 8 | proximity | N | N | N | 0.020 | -0.276 | 99.32 | 7.71 | 0 | 0.454 | max_steps_bug; **M5=0 anomaly** |
| prox_dc8_30m_k2_l035_s1 | 8 | proximity | N | N | N | 0.035 | -0.199 | 99.42 | 6.33 | 0 | 0.476 | max_steps_bug; **M5=0 anomaly** |
| al_prox_dc8_k2_l035_s0 | 8 | proximity | Y | N | N | 0.005 | -0.484 | 99.67 | 0.93 | 3200 | 0.264 | max_steps_bug |
| al_lp_sr_prox_dc8_k2_l035_s0 | 8 | proximity | Y | Y | Y | 0.045 | 0.902 | 98.56 | 4.39 | 3200 | 0.463 | max_steps_bug |
| al_lp_sr_prox_dc8_ms100_k2_l035_s0 | 8 | proximity | Y | Y | Y | 0.045 | 0.902 | 98.56 | 4.39 | 3200 | 0.463 | max_steps_actual=100_cfg_intended=400_actual=100 |
| al_lp_sr_bc_dc2_k2_l035_s0 | 2 | broadcast | Y | Y | Y | 0.010 | 0.505 | 99.88 | 5.83 | 800 | 0.378 | max_steps_bug |
| al_lp_sr_bc_dc8_k2_l035_s0 | 8 | broadcast | Y | Y | Y | 0.010 | 0.172 | 99.58 | 4.31 | 3200 | 0.290 | max_steps_bug |
| al_lp_sr_bc_dc16_k2_l035_s0 | 16 | broadcast | Y | Y | Y | 0.000 | -0.363 | 100.00 | 7.16 | 6400 | 0.163 | max_steps_bug |
| bc_dc16_k2_l035_s0 | 16 | broadcast | N | N | N | 0.085 | -0.147 | 97.94 | 6.31 | 0 | 0.498 | max_steps_bug; **comm_ignored** |
| bc_dc16_k2_l035_s1 | 16 | broadcast | N | N | N | 0.045 | -0.201 | 99.23 | 7.22 | 0 | 0.484 | max_steps_bug; **comm_ignored** |

## Spot-Check Details

### 1. prox_dc8_k2_l035_s0 (baseline proximity comm)

**Config verification**:
- `dim_c=8`: MATCH
- `comm_proximity=true`: MATCH
- `use_agent_lidar=false`: MATCH (no AL in tag)
- `shared_reward=false`: MATCH (no SR in tag)
- `agent_collision_penalty=-0.1`: MATCH (no LP in tag)

**Metrics**: All M1-M9 present and match master CSV. M5=3200 is correct for 4 agents x 8 channels x 100 steps (but see max_steps_bug note -- actual_max_steps=100 not 200).

**Verdict**: PASS

---

### 2. al_lp_sr_prox_dc8_k2_l035_s0 (best ER1 config + proximity comm)

**Config verification**:
- `dim_c=8`: MATCH
- `comm_proximity=true`: MATCH
- `use_agent_lidar=true`: MATCH (AL in tag)
- `shared_reward=true`: MATCH (SR in tag)
- `agent_collision_penalty=-0.01`: MATCH (LP in tag)

**Metrics**: All M1-M9 present and match master CSV. M5=3200, M7=7320000 (convergence detected).

**Verdict**: PASS

---

### 3. al_lp_sr_bc_dc8_k2_l035_s0 (broadcast comm)

**Config verification**:
- `dim_c=8`: MATCH
- `comm_proximity=false`: MATCH (broadcast = not proximity)
- `use_agent_lidar=true`: MATCH (AL in tag)
- `shared_reward=true`: MATCH (SR in tag)
- `agent_collision_penalty=-0.01`: MATCH (LP in tag)

**Metrics**: All M1-M9 present and match master CSV. M5=3200, M7=9360000.

**Verdict**: PASS

---

### 4. bc_dc16_k2_l035_s0 (broadcast without LP+SR -- should be comm_ignored)

**Config verification**:
- `dim_c=16`: MATCH
- `comm_proximity=false`: MATCH (broadcast mode)
- `use_agent_lidar=false`: MATCH (no AL in tag)
- `shared_reward=false`: MATCH (no SR in tag)
- `agent_collision_penalty=-0.1`: MATCH (no LP in tag)

**Metrics**: M5=0.0 (tokens not produced). M1=0.085, which matches ER1 baseline `default_k2_l035_s0` exactly (same M1, M2, M3, M4 values). This confirms communication was ignored -- the run is effectively identical to a no-comm baseline.

**comm_ignored flag**: CORRECTLY APPLIED in master CSV.

**Why comm was ignored**: Without `use_agent_lidar=true` (AL), agents do not have observations of other agents' communication. Without `shared_reward=true` (SR), there is no gradient signal incentivizing communication. The broadcast channel exists in the config but the policy cannot learn to use it.

**Verdict**: PASS (with comm_ignored correctly flagged)

---

### 5. al_lp_sr_prox_dc8_ms100_k2_l035_s0 (ms100 run)

**Config verification**:
- `dim_c=8`: MATCH
- `comm_proximity=true`: MATCH
- `use_agent_lidar=true`: MATCH
- `shared_reward=true`: MATCH
- `agent_collision_penalty=-0.01`: MATCH
- `max_steps=400` in config: MATCH (ms100 tag)

**Metrics**: ALL metrics are byte-identical to `al_lp_sr_prox_dc8_k2_l035_s0` (the ms100 version):
- M1=0.045, M2=0.902, M3=98.555, M4=4.39, M5=3200, M6=0.4625
- This confirms the **max_steps_bug**: despite config saying 400, `config.pop("max_steps")` removed the value before it reached VMAS, so the scenario defaulted to 100 steps.

**max_steps_bug flag**: CORRECTLY APPLIED in master CSV as `max_steps_actual=100_cfg_intended=400_actual=100`.

**Additional note**: Config also contains leftover GNN fields (`gnn_class: GATv2Conv`, `gnn_topology: from_pos`, `model_type: mlp`) -- these are harmless since `model_type=mlp` is used, but they indicate the config template was shared with ER3.

**Verdict**: PASS (with max_steps_bug correctly flagged)

## Cross-Run Sanity Checks

### Proximity vs Broadcast differences

Comparing runs with identical settings except comm mode (all with AL+LP+SR):

| Metric | prox_dc8 | bc_dc8 | bc_dc2 | bc_dc16 |
|---|---|---|---|---|
| M1 (SR) | 0.045 | 0.010 | 0.010 | 0.000 |
| M2 (Return) | 0.902 | 0.172 | 0.505 | -0.363 |
| M6 (Coverage) | 0.463 | 0.290 | 0.378 | 0.163 |

**Finding**: Proximity comm outperforms all broadcast variants. Among broadcast runs, smaller `dim_c` performs better (dc2 > dc8 > dc16), suggesting larger comm channels introduce noise or make optimization harder. This is a plausible and expected pattern.

### Are bc runs without LP+SR flagged as comm_ignored?

- `bc_dc16_k2_l035_s0`: YES, flagged `comm_ignored` in master CSV.
- `bc_dc16_k2_l035_s1`: YES, flagged `comm_ignored` in master CSV.

Both have M5=0 (no tokens counted) and their M1/M2/M3/M4 match ER1 baseline runs exactly, confirming communication had no effect.

### Is ms100 flagged with max_steps_bug?

- `al_lp_sr_prox_dc8_ms100_k2_l035_s0`: YES, flagged `max_steps_actual=100_cfg_intended=400_actual=100`.
- Metrics are identical to the ms100 counterpart, confirming the bug was present.

## Anomalies Found

### ANOMALY 1: prox_dc8_30m runs report M5=0 (tokens)

Runs `prox_dc8_30m_k2_l035_s0` and `prox_dc8_30m_k2_l035_s1` have `dim_c=8` and `comm_proximity=true` but report M5=0.0 tokens. Other proximity runs with identical comm settings (e.g., `prox_dc8_k2_l035_s0`) correctly report M5=3200.

**Likely cause**: These 30M runs were executed on OVH (cuda, torch 2.2.0) while the 10M prox_dc8 run was local (cpu, torch 2.10.0). The M5 token extraction may not have been implemented in the earlier runner version, or the 30M evaluation used a different code path. The M5=0 value should be treated as **missing data, not as evidence of zero communication**.

**Impact**: M5 is unreliable for these two runs. All other metrics appear consistent.

### ANOMALY 2: Identical metrics between ms100 and ms100 proximity runs

As noted above, `al_lp_sr_prox_dc8_ms100_k2_l035_s0` has byte-identical metrics to `al_lp_sr_prox_dc8_k2_l035_s0`. This is a direct consequence of the max_steps_bug. Both runs effectively trained and evaluated with max_steps=100. The ms100 run needs to be re-executed after the bug fix.

### ANOMALY 3: bc_dc16 (comm_ignored) matches ER1 baseline exactly

`bc_dc16_k2_l035_s0` produces M1=0.085, M2=-0.147, M3=97.935, M4=6.305 -- identical to ER1 `default_k2_l035_s0` and `abl_lam095_k2_l035_s0`. This is internally consistent (communication was ignored, so it is an ER1 baseline re-run) but means these runs provide no information about broadcast communication. They should not be used to draw conclusions about broadcast comm effectiveness.

### ANOMALY 4: M8 (Agent Utilization) = 0.0 for all AL+LP+SR runs

All runs with `shared_reward=true` show M8=0.0. This is a known artifact: the M8 metric measures per-agent reward variance, and shared reward eliminates inter-agent reward differences by design. This is expected behavior, not a bug, but M8 is uninformative for SR runs.
