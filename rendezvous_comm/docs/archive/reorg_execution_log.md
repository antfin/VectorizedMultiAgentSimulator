# Results Reorganization — Execution Log

**Started**: 2026-03-22
**Status**: PHASES 1-6 COMPLETE (2026-03-22)

## Pre-flight Findings (Phase 0 ✅)

- **96 total run directories** found across all result dirs
- **86 with metrics.json** (valid completed runs)
- **9 in e1/** with no config/metrics (failed runs → archive)
- **1 incomplete** (er3_al_lp_sr_prox, no metrics → archive)
- **max_steps bug confirmed**: ALL 86 runs used max_steps=100 regardless of config
  - Configs said 200 or 400, VMAS default of 100 was used
  - Bug is now fixed in runner.py
  - All results are comparable (all used 100) but need annotation
- **Duplicates confirmed**: er1_ablation/{a2,b,c,g,h,i} = standalone er1_abl_{a2,b,c,g,h,i}
- **Unique content in er1_ablation/**: er1_abl_a (3 runs), er1_old (13 runs), consolidated CSV

## Execution Checklist

### Phase 1: Create target structure ✅
- [x] mkdir er1/runs, er1/archive
- [x] mkdir er2/runs, er2/archive
- [x] mkdir er3/runs, er3/archive
- [x] mkdir docs/archive

### Phase 2: Copy runs to new locations ✅

#### ER1 — 41 runs total

**ER1 focused sweep (8 runs)** — source: `er1/`
- [x] `default_k1_l025_s0` ← er1/20260320_0841__er1_mappo_n4_t4_k1_l025_s0
- [x] `default_k1_l025_s1` ← er1/20260320_0841__er1_mappo_n4_t4_k1_l025_s1
- [x] `default_k1_l035_s0` ← er1/20260320_0841__er1_mappo_n4_t4_k1_l035_s0
- [x] `default_k1_l035_s1` ← er1/20260320_0841__er1_mappo_n4_t4_k1_l035_s1
- [x] `default_k2_l025_s0` ← er1/20260320_0841__er1_mappo_n4_t4_k2_l025_s0
- [x] `default_k2_l025_s1` ← er1/20260320_0841__er1_mappo_n4_t4_k2_l025_s1
- [x] `default_k2_l035_s0` ← er1/20260320_1337__er1_mappo_n4_t4_k2_l035_s0
- [x] `default_k2_l035_s1` ← er1/20260320_1337__er1_mappo_n4_t4_k2_l035_s1

**ER1 + agent lidar (8 runs)** — source: `er1_al/`
- [x] `al_k1_l025_s0` ← er1_al/20260320_1806__er1_al_mappo_n4_t4_k1_l025_s0
- [x] `al_k1_l025_s1` ← er1_al/20260320_1806__er1_al_mappo_n4_t4_k1_l025_s1
- [x] `al_k1_l035_s0` ← er1_al/20260320_1806__er1_al_mappo_n4_t4_k1_l035_s0
- [x] `al_k1_l035_s1` ← er1_al/20260320_1806__er1_al_mappo_n4_t4_k1_l035_s1
- [x] `al_k2_l025_s0` ← er1_al/20260320_1806__er1_al_mappo_n4_t4_k2_l025_s0
- [x] `al_k2_l025_s1` ← er1_al/20260320_1806__er1_al_mappo_n4_t4_k2_l025_s1
- [x] `al_k2_l035_s0` ← er1_al/20260320_1806__er1_al_mappo_n4_t4_k2_l035_s0
- [x] `al_k2_l035_s1` ← er1_al/20260320_1806__er1_al_mappo_n4_t4_k2_l035_s1

**ER1 AL ablations (3 runs)** — source: `er1_al_abl_*/`
- [x] `al_lp_k2_l035_s0` ← er1_al_abl_lp/...
- [x] `al_sr_k2_l035_s0` ← er1_al_abl_sr/...
- [x] `al_lp_sr_k2_l035_s0` ← er1_al_abl_lp_sr/...

**ER1 AL+LP+SR ms100 (1 run)** — source: `er1_al_lp_sr_ms400/`
- [x] `al_lp_sr_ms100_k2_l035_s0` ← er1_al_lp_sr_ms400/...

**ER1 ablation A: entropy=0.01 (3 runs)** — source: `er1_ablation/er1_abl_a/`
- [x] `abl_ent01_k2_l035_s0` ← er1_ablation/er1_abl_a/...s0
- [x] `abl_ent01_k2_l035_s1` ← er1_ablation/er1_abl_a/...s1
- [x] `abl_ent01_k2_l035_s2` ← er1_ablation/er1_abl_a/...s2

**ER1 ablation A2: entropy=0.001 (3 runs)** — source: `er1_abl_a2/`
- [x] `abl_ent001_k2_l035_s0`
- [x] `abl_ent001_k2_l035_s1`
- [x] `abl_ent001_k2_l035_s2`

**ER1 ablation B: lr=1e-4 (3 runs)** — source: `er1_abl_b/`
- [x] `abl_lr1e4_k2_l035_s0`
- [x] `abl_lr1e4_k2_l035_s1`
- [x] `abl_lr1e4_k2_l035_s2`

**ER1 ablation C: lmbda=0.95 (3 runs)** — source: `er1_abl_c/`
- [x] `abl_lam095_k2_l035_s0`
- [x] `abl_lam095_k2_l035_s1`
- [x] `abl_lam095_k2_l035_s2`

**ER1 ablation G: 20M frames (3 runs)** — source: `er1_abl_g/`
- [x] `abl_20m_k2_l035_s0`
- [x] `abl_20m_k2_l035_s1`
- [x] `abl_20m_k2_l035_s2`

**ER1 ablation H: [512,256]+ReLU (3 runs)** — source: `er1_abl_h/`
- [x] `abl_net512_relu_k2_l035_s0`
- [x] `abl_net512_relu_k2_l035_s1`
- [x] `abl_net512_relu_k2_l035_s2`

**ER1 ablation I: k=1 sanity (3 runs)** — source: `er1_abl_i/`
- [x] `abl_k1_sanity_k1_l035_s0`
- [x] `abl_k1_sanity_k1_l035_s1`
- [x] `abl_k1_sanity_k1_l035_s2`

#### ER2 — 11 runs total

- [x] `prox_dc8_s0` ← er2/...
- [x] `prox_dc8_30m_s0` ← er2_30m/...s0
- [x] `prox_dc8_30m_s1` ← er2_30m/...s1
- [x] `al_prox_dc8_s0` ← er2_al/...
- [x] `al_lp_sr_prox_dc8_s0` ← er2_al_lp_sr/...
- [x] `al_lp_sr_prox_dc8_ms100_s0` ← er2_al_lp_sr_prox_dc8_ms400/...
- [x] `bc_dc16_s0` ← er2_bc_dimc16/...s0
- [x] `bc_dc16_s1` ← er2_bc_dimc16/...s1
- [x] `al_lp_sr_bc_dc2_s0` ← er2_al_lp_sr_bc_dimc2/...
- [x] `al_lp_sr_bc_dc8_s0` ← er2_al_lp_sr_bc_dimc8/...
- [x] `al_lp_sr_bc_dc16_s0` ← er2_al_lp_sr_bc_dimc16/...

#### ER3 — 3 runs (+ 1 incomplete → archive)

- [x] `al_lp_sr_gatv2_s0` ← er3_al_lp_sr_gatv2/...
- [x] `al_lp_sr_gatv2_ms100_s0` ← er3_al_lp_sr_gatv2_ms400/...
- [x] `al_lp_sr_graphconv_s0` ← er3_al_lp_sr_graphconv/...
- [x] ARCHIVE: er3_al_lp_sr_prox/ (incomplete, no metrics)

### Phase 3: Verify copies ✅
- [x] Every copied run: config.yaml matches source
- [x] Every copied run: metrics.json matches source
- [x] File count matches per run

### Phase 4: Build master_results.csv ✅
- [x] Read all metrics.json from new er{N}/runs/
- [x] Read ablation D,E,F from er1_ablation/ablation_consolidated.csv
- [x] Add variant, experiment, family, old_run_id columns
- [x] Add actual_max_steps=100 column for all pre-fix runs
- [x] Add notes column (max_steps_bug, comm_ignored, csv_only)
- [x] Validate row count: 55 runs + 9 csv-only = 64 rows

### Phase 5: Archive ✅
- [x] e1/ → er1/archive/e1/
- [x] er1_ablation/er1_old/ → er1/archive/er1_old/
- [x] er1_ablation/ablation_consolidated.csv → er1/er1_ablation_consolidated.csv
- [x] er3_al_lp_sr_prox/ → er3/archive/incomplete_prox/
- [x] Sweep files (sweep_report, sweep_results, training_iter) → archive/sweep_files/
- [x] docs/er1_first_findings_2026-03-15.md → docs/archive/
- [x] docs/hyperparameter_analysis.md → docs/archive/

### Phase 6: Generate changelog.md ✅
- [x] Auto-generated from all Phase 2-5 actions
- [x] Includes old path → new path for every move

### Phase 7: Write per-experiment verification report ✅
- [x] er1_verification.md — 41 runs, 5 spot-checked, 7 anomalies documented
- [x] er2_verification.md — 11 runs, 5 spot-checked, 4 anomalies documented
- [x] er3_verification.md — 3 runs, all checked, M1=0 across all GNN runs
- [x] Each report: config used, metrics obtained, sanity checks, anomalies

### Phase 8: Final validation ✅ (automated by script)
- [x] er1/runs/ count = 41
- [x] er2/runs/ count = 11
- [x] er3/runs/ count = 3
- [x] master_results.csv row count = 64
- [x] Every non-csv_only row has matching runs/ dir
- [x] No metrics.json lost (count before = count after)

### Phase 9: Delete originals (ONLY after Phase 8 passes)
- [x] Remove old flat directories
- [x] Remove er1_ablation duplicate copies
- [x] Remove empty dirs

## Bugs Found During Reorganization

| Bug | Impact | Status |
|-----|--------|--------|
| max_steps silently ignored | ALL experiments ran with 100 steps, not configured 200 | FIXED in runner.py |
| ms100 experiments (intended ms400) | Ran with 100 steps, not configured 400 | Need re-run with actual ms400 |
| M5 not extracted on OVH | OVH code didn't have M5 fix | Need re-deploy for future runs |

## Resolved: Variant Tag Collisions ✅

All collisions fixed in reorganize.py:

1. **er1_ablation/ duplicate subdirs** — excluded `{a2,b,c,g,h,i}` from collection (confirmed dupes of standalone dirs)
2. **er1_old/ runs** — excluded from main collection, archived separately to `er1/archive/er1_old/`
3. **Ablation identity** — `build_variant_tag()` now checks both `exp_dir` and `run_dir_name` for ablation markers
4. **Double-20m bug** — `abl_g` no longer appends redundant `20m` suffix (was `abl_20m_20m_*`, now `abl_20m_*`)
5. **Safety net** — added `n{N}_t{T}` encoding for non-standard agent/target counts, algorithm encoding for non-MAPPO

## Decisions

1. **DO NOT modify metrics.json** — add actual_max_steps column in CSV instead
2. **Copy first, verify, then delete** — never move directly
3. **er1_abl_c = er1 focused sweep** — ablation C (lmbda=0.95) became the default, so er1 focused sweep IS ablation C. They share the same results on matching seeds.
4. **er2_bc_{8,2,16} without LP+SR** — results identical to ER1, mark as comm_ignored
5. **ms100 runs** (originally configured as ms400) — all used actual max_steps=100, need re-run for real ms400
