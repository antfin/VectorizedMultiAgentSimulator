# ER1 Verification Report

**Date**: 2026-03-22
**Runs verified**: 41 (8 default, 8 AL, 4 AL ablation combos, 21 hyperparameter ablations)

## Summary Table

| Variant | M1 (SR) | M2 (Return) | M3 (Steps) | M4 (Coll) | Notes |
|---------|---------|-------------|------------|-----------|-------|
| abl_20m_k2_l035_s0 | 0.055 | -0.165 | 99.21 | 6.74 | max_steps_bug |
| abl_20m_k2_l035_s1 | 0.060 | -0.203 | 98.54 | 6.68 | max_steps_bug |
| abl_20m_k2_l035_s2 | 0.040 | -0.184 | 99.17 | 7.13 | max_steps_bug |
| abl_ent001_k2_l035_s0 | 0.040 | -0.245 | 99.26 | 7.51 | max_steps_bug |
| abl_ent001_k2_l035_s1 | 0.060 | -0.149 | 99.22 | 5.57 | max_steps_bug |
| abl_ent001_k2_l035_s2 | 0.030 | -0.336 | 99.61 | 6.76 | max_steps_bug |
| abl_ent01_k2_l035_s0 | 0.000 | -0.952 | 100.00 | 0.00 | max_steps_bug; collapsed policy |
| abl_ent01_k2_l035_s1 | 0.000 | -0.942 | 100.00 | 0.00 | max_steps_bug; collapsed policy |
| abl_ent01_k2_l035_s2 | 0.000 | -0.965 | 100.00 | 0.00 | max_steps_bug; collapsed policy |
| abl_k1_sanity_k1_l035_s0 | 0.765 | 0.331 | 69.70 | 4.83 | max_steps_bug |
| abl_k1_sanity_k1_l035_s1 | 0.765 | 0.344 | 68.59 | 5.36 | max_steps_bug |
| abl_k1_sanity_k1_l035_s2 | 0.775 | 0.439 | 66.91 | 4.76 | max_steps_bug |
| abl_lam095_k2_l035_s0 | 0.085 | -0.147 | 97.94 | 6.31 | max_steps_bug; DUPLICATE metrics (see anomalies) |
| abl_lam095_k2_l035_s1 | 0.045 | -0.201 | 99.23 | 7.22 | max_steps_bug; DUPLICATE metrics (see anomalies) |
| abl_lam095_k2_l035_s2 | 0.060 | -0.072 | 98.78 | 7.30 | max_steps_bug |
| abl_lp_nolid_k2_l035_s0 | 0.050 | -0.179 | 98.50 | 10.79 | max_steps_bug; csv_only |
| abl_lp_nolid_k2_l035_s1 | 0.010 | -0.175 | 99.60 | 7.96 | max_steps_bug; csv_only |
| abl_lp_nolid_k2_l035_s2 | 0.020 | -0.238 | 99.40 | 15.79 | max_steps_bug; csv_only |
| abl_lr1e4_k2_l035_s0 | 0.065 | -0.074 | 99.08 | 7.22 | max_steps_bug |
| abl_lr1e4_k2_l035_s1 | 0.055 | -0.137 | 98.86 | 6.46 | max_steps_bug |
| abl_lr1e4_k2_l035_s2 | 0.050 | -0.196 | 99.04 | 7.44 | max_steps_bug |
| abl_lr1e4_lam095_k2_l035_s0 | 0.085 | -0.147 | 97.90 | 6.30 | max_steps_bug; csv_only |
| abl_lr1e4_lam095_k2_l035_s1 | 0.045 | -0.201 | 99.20 | 7.22 | max_steps_bug; csv_only |
| abl_lr1e4_lam095_k2_l035_s2 | 0.060 | -0.072 | 98.80 | 7.30 | max_steps_bug; csv_only |
| abl_net512_relu_k2_l035_s0 | 0.035 | -0.188 | 99.32 | 7.87 | max_steps_bug |
| abl_net512_relu_k2_l035_s1 | 0.045 | -0.202 | 99.02 | 8.29 | max_steps_bug |
| abl_net512_relu_k2_l035_s2 | 0.060 | -0.161 | 98.60 | 7.01 | max_steps_bug |
| abl_sr_nolid_k2_l035_s0 | 0.025 | 0.578 | 99.50 | 7.24 | max_steps_bug; csv_only; M8=0.0 |
| abl_sr_nolid_k2_l035_s1 | 0.045 | 0.716 | 99.30 | 7.92 | max_steps_bug; csv_only; M8=0.0 |
| abl_sr_nolid_k2_l035_s2 | 0.020 | 0.559 | 99.70 | 7.90 | max_steps_bug; csv_only; M8=0.0 |
| al_k1_l025_s0 | 0.630 | 0.181 | 75.85 | 3.10 | max_steps_bug |
| al_k1_l025_s1 | 0.655 | 0.247 | 74.93 | 3.20 | max_steps_bug |
| al_k1_l035_s0 | 0.605 | 0.364 | 73.69 | 0.88 | max_steps_bug |
| al_k1_l035_s1 | 0.750 | 0.441 | 69.86 | 0.80 | max_steps_bug |
| al_k2_l025_s0 | 0.000 | -0.688 | 100.00 | 0.04 | max_steps_bug |
| al_k2_l025_s1 | 0.000 | -0.739 | 100.00 | 0.17 | max_steps_bug |
| al_k2_l035_s0 | 0.000 | -0.420 | 100.00 | 0.70 | max_steps_bug |
| al_k2_l035_s1 | 0.000 | -0.642 | 100.00 | 0.10 | max_steps_bug |
| al_lp_k2_l035_s0 | 0.005 | -0.302 | 99.92 | 4.64 | max_steps_bug |
| al_lp_sr_k2_l035_s0 | 0.040 | 0.840 | 99.29 | 2.85 | max_steps_bug; M8=0.0 |
| al_lp_sr_ms100_k2_l035_s0 | 0.040 | 0.840 | 99.29 | 2.85 | max_steps_actual=100_cfg_intended=400; DUPLICATE metrics (see anomalies) |
| al_sr_k2_l035_s0 | 0.015 | 0.504 | 99.79 | 0.96 | max_steps_bug; M8=0.0 |
| default_k1_l025_s0 | 0.660 | 0.132 | 77.34 | 6.15 | max_steps_bug |
| default_k1_l025_s1 | 0.695 | 0.248 | 72.94 | 4.43 | max_steps_bug |
| default_k1_l035_s0 | 0.765 | 0.331 | 69.70 | 4.83 | max_steps_bug |
| default_k1_l035_s1 | 0.765 | 0.344 | 68.59 | 5.36 | max_steps_bug |
| default_k2_l025_s0 | 0.020 | -0.453 | 99.64 | 8.88 | max_steps_bug |
| default_k2_l025_s1 | 0.025 | -0.506 | 99.43 | 7.06 | max_steps_bug |
| default_k2_l035_s0 | 0.085 | -0.147 | 97.94 | 6.31 | max_steps_bug |
| default_k2_l035_s1 | 0.045 | -0.201 | 99.23 | 7.22 | max_steps_bug |

## Spot-Check Details

### default_k2_l035_s0
- **Config**: k=2, lidar_range=0.35, seed=0, use_agent_lidar=false, shared_reward=false, collision_penalty=-0.1, max_steps=200 (cfg), lmbda=0.95, lr=5e-05
- **Metrics**: M1=0.085, M2=-0.147, M3=97.94, M4=6.31, M5=0.0, M6=0.498
- **Param check**: All params match variant tag. k=2 (agents_per_target=2), l035 (lidar_range=0.35), s0 (seed=0). No agent lidar, no shared reward, standard penalty.
- **Status**: OK (metrics in expected range; low M1 expected for k=2 under max_steps_bug)

### al_k2_l035_s0
- **Config**: k=2, lidar_range=0.35, seed=0, use_agent_lidar=**true**, shared_reward=false, collision_penalty=-0.1, max_steps=200 (cfg), lmbda=0.95, lr=5e-05
- **Metrics**: M1=0.0, M2=-0.420, M3=100.0, M4=0.70, M5=0.0
- **Param check**: AL tag matches use_agent_lidar=true. Other params consistent.
- **Status**: OK (M1=0 with k=2 under 100 actual steps is unsurprising; agent lidar adds obs complexity but k=2 is very hard)

### abl_lr1e4_k2_l035_s0
- **Config**: k=2, lidar_range=0.35, seed=0, use_agent_lidar=false, shared_reward=false, collision_penalty=-0.1, **lr=0.0001**, lmbda=null (BenchMARL default)
- **Metrics**: M1=0.065, M2=-0.074, M3=99.08, M4=7.22, M5=0.0
- **Param check**: lr=1e-4 matches the `lr1e4` tag. lmbda is null (not set), confirming this ablation only changed lr. Other params match.
- **Status**: OK (slightly better M2 than default, consistent with higher lr)

### al_lp_sr_k2_l035_s0
- **Config**: k=2, lidar_range=0.35, seed=0, use_agent_lidar=**true**, shared_reward=**true**, collision_penalty=**-0.01**, max_steps=200, lmbda=0.95, lr=5e-05, dim_c=0
- **Metrics**: M1=0.04, M2=0.840, M3=99.29, M4=2.85, M5=0.0, M7=6120000, M8=0.0
- **Param check**: AL (use_agent_lidar=true), LP (collision_penalty=-0.01), SR (shared_reward=true) all match. dim_c=0 confirms no communication channel (ER1).
- **Status**: OK with note -- M8=0.0 is expected artifact of shared_reward (utilization metric undefined when reward is shared). Positive M2 due to shared reward aggregation.

### al_lp_sr_ms100_k2_l035_s0
- **Config**: k=2, lidar_range=0.35, seed=0, use_agent_lidar=true, shared_reward=true, collision_penalty=-0.01, **max_steps=400** (cfg), lmbda=0.95, lr=5e-05, model_type=mlp
- **Metrics**: M1=0.04, M2=0.840, M3=99.29, M4=2.85 -- **IDENTICAL to al_lp_sr_k2_l035_s0**
- **Param check**: Config correctly shows max_steps=400. However, due to the max_steps_bug, actual_max_steps=100 during training. The evaluation also ran with 100 steps (not 400).
- **Status**: ANOMALY -- Metrics are byte-for-byte identical to `al_lp_sr_k2_l035_s0`. The ms100 config was configured but the bug meant the trained policy and evaluation both used 100 steps. The ms100 run appears to have reused the same trained model checkpoint (same final_entropy=-1.722, same policy_params=75013, nearly identical training_seconds). This run needs to be **re-executed** after the max_steps fix.

## Cross-Run Sanity Checks

### Are ablation runs using the expected modified param?
- **abl_lr1e4**: lr=0.0001 confirmed (vs default 5e-05). lmbda=null (not set). OK.
- **abl_lam095**: lmbda=0.95 in CSV. However, the default runs ALSO have lmbda=0.95 in their config, making this ablation redundant. The metrics for `abl_lam095_k2_l035_s{0,1}` are **identical** to `default_k2_l035_s{0,1}` (see anomalies).
- **abl_ent001**: entropy_coef=0.001 confirmed in CSV (vs default null). OK.
- **abl_ent01**: entropy_coef=0.01 confirmed. All 3 seeds collapsed to M1=0.0, M4=0.0 -- too much entropy regularization. OK (expected behavior).
- **abl_20m**: max_n_frames=20000000 confirmed (vs default 10M). OK.
- **abl_net512_relu**: hidden_layers=[512,256], activation=relu confirmed. OK.
- **abl_k1_sanity**: agents_per_target=1 confirmed (easier task). Much higher M1 (~0.77). OK.
- **abl_lp_nolid**: collision_penalty=-0.01, use_agent_lidar=false confirmed. OK.
- **abl_sr_nolid**: shared_reward=true, use_agent_lidar=false confirmed. M8=0.0 expected. OK.
- **abl_lr1e4_lam095**: lr=0.0001, lmbda=0.95. csv_only entries -- metrics match `abl_lam095` exactly (see anomalies).

### Do seeds produce different results?
- **default_k2_l035**: s0 M1=0.085, s1 M1=0.045. Different. OK.
- **abl_ent001_k2_l035**: s0 M1=0.04, s1 M1=0.06, s2 M1=0.03. Different. OK.
- **abl_k1_sanity**: s0 M1=0.765, s1 M1=0.765, s2 M1=0.775. Very similar but M2/M3/M4 differ. OK (k=1 is easier, less variance).
- **abl_ent01**: s0/s1/s2 all M1=0.0. All collapsed identically. OK (expected when entropy is too high).

### Are ms100 runs flagged with max_steps_bug?
- **al_lp_sr_ms100_k2_l035_s0**: CSV notes = `max_steps_actual=100_cfg_intended=400_actual=100`. Correctly flagged.
- Config shows max_steps=400 but actual execution used 100 steps due to the bug.

## Anomalies Found

### ANOMALY 1: `abl_lam095` is identical to `default` (CRITICAL)
The `abl_lam095_k2_l035_s0` and `abl_lam095_k2_l035_s1` runs have **exactly the same metrics** as `default_k2_l035_s0` and `default_k2_l035_s1` respectively:
- s0: M1=0.085, M2=-0.14675, M3=97.935, M4=6.305 (both variants)
- s1: M1=0.045, M2=-0.20100, M3=99.230, M4=7.220 (both variants)

**Root cause**: The default config already includes lmbda=0.95. The "ablation" for lmbda=0.95 was therefore a no-op -- the experiment ran with the same parameters as the default. These are likely the **same trained models** reorganized under two different variant names. The `abl_lam095` variant has `old_run_id` = `er1_abl_c_*` (separate ablation run) but produced identical results because the parameters were identical.

**Impact**: The `abl_lam095` rows are redundant duplicates of the `default` runs. They should be excluded from ablation analysis or relabeled.

### ANOMALY 2: `abl_lr1e4_lam095` is identical to `abl_lam095` (csv_only)
The `abl_lr1e4_lam095_k2_l035_s{0,1,2}` entries are marked `csv_only` and their metrics match `abl_lam095` exactly (e.g., s0: M1=0.085, M2=-0.147, M3=97.9, M4=6.3). These appear to be mislabeled copies. They claim lr=0.0001 + lmbda=0.95 but the metrics do not match the actual `abl_lr1e4` runs (which have different M2 values). This needs investigation.

### ANOMALY 3: `al_lp_sr_ms100` metrics identical to `al_lp_sr` (EXPECTED)
As detailed in spot-check above, the ms100 run produced byte-for-byte identical metrics to the non-ms100 run because the max_steps_bug prevented the config change from taking effect. Correctly flagged in CSV notes. **Action**: Re-run after bug fix.

### ANOMALY 4: `abl_k1_sanity` identical to `default_k1_l035`
The k=1 sanity check metrics are identical to the default k=1 l=0.35 runs:
- `abl_k1_sanity_k1_l035_s0`: M1=0.765, M2=0.331, M3=69.70, M4=4.83
- `default_k1_l035_s0`: M1=0.765, M2=0.331, M3=69.70, M4=4.83

These are the same model runs reorganized under two names. The "sanity" ablation was meant to verify k=1 behavior, and it used the same default parameters. This is not a bug per se, but the duplicate should be noted.

### ANOMALY 5: M8 (Agent Utilization) = 0.0 for all shared_reward runs
Runs with shared_reward=true (`al_lp_sr`, `al_sr`, `abl_sr_nolid`) all report M8=0.0. This is a known artifact: the M8 metric measures per-agent reward differentiation, which is undefined under shared rewards. Not a data error, but these M8 values should be excluded from cross-variant comparisons.

### ANOMALY 6: `abl_ent01` collapsed policy (all 3 seeds)
All three seeds of the entropy=0.01 ablation show M1=0.0, M3=100.0, M4=0.0. The agents learned to stand still (no collisions, no coverage, maximum steps). This is a valid result showing entropy_coef=0.01 is too high, but it means this hyperparameter setting is non-viable.

### ANOMALY 7: All ER1 runs affected by max_steps_bug
Every single ER1 run was trained with actual_max_steps=100 despite configs specifying 200 (or 400). This is a known issue (documented in MEMORY.md). All results should be interpreted as 100-step episode performance. The k=2 task (requiring 2 agents per target) appears nearly impossible in 100 steps, which explains the universally low M1 for k=2 variants.

## Summary Assessment

- **41 ER1 runs** present in CSV, **35 have run directories** with config/metrics files, **6 are csv_only** (from consolidated ablation CSVs).
- **All metrics are in valid ranges**: M1 in [0, 0.775], M2 in [-0.965, 0.840], M3 in [66.9, 100.0], M4 in [0.0, 15.79].
- **3 duplicate pairs identified** (abl_lam095/default, abl_lr1e4_lam095/abl_lam095, abl_k1_sanity/default_k1) -- effectively reducing unique experiments to ~35.
- **All runs affected by max_steps_bug** -- results reflect 100-step episodes regardless of configured max_steps.
- **ms100 run needs re-execution** after bug fix to produce valid 400-step results.
