# Results Reorganization Plan (v2)

**Date**: 2026-03-21
**Status**: Plan — execute after local broadcast+LP+SR experiments finish

## Why This Matters

Every experiment variant must be instantly comparable. A researcher looking at the
results for the first time should understand in seconds: what was tested, what
changed between runs, and where to find the data. Right now that's not possible —
21 scattered directories, inconsistent naming, duplicated data, and no single CSV
to compare everything.

## Design Principles

1. **One CSV to rule them all**: `master_results.csv` at the results root with every
   run ever completed. Every dimension that varies gets its own column. Filter/group
   in pandas, not in folder names.
2. **Two-level folders**: `er{N}/runs/{variant_tag}/` — experiment family + variant.
3. **Variant tags are short and parseable**: `al_lp_sr_prox_dc8_s0` not
   `er2_al_lp_sr_mappo_n4_t4_k2_l035_s0`.
4. **Rename directories** for consistency (old names tracked in changelog).
5. **Never delete data** — archive what's obsolete, mark what's known-invalid.
6. **Docs updated** to reference new paths and names.

## Master CSV Schema

`results/master_results.csv` — the single source of truth for all experiments.

### Identification columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `variant` | str | **Primary key**. Short, unique tag. | `al_lp_sr_prox_dc8_s0` |
| `experiment` | str | Human label for grouping/display. | `ER2 + AL + LP+SR prox dc=8` |
| `family` | str | Top-level experiment: `er1`, `er2`, `er3`, `er4`. | `er2` |
| `old_run_id` | str | Original run_id (for tracing back to logs). | `er2_al_lp_sr_mappo_n4_t4_k2_l035_s0` |
| `old_dir` | str | Original directory name before reorg. | `er2_al_lp_sr/20260321_1723__er2_al_...` |

### Experimental dimensions (what varies)

| Column | Type | Description | Values |
|--------|------|-------------|--------|
| `agent_lidar` | bool | Agent-to-agent lidar sensor | true/false |
| `dim_c` | int | Communication channel size | 0, 2, 8, 16 |
| `comm_mode` | str | Communication gating | `none`, `proximity`, `broadcast` |
| `shared_reward` | bool | Team reward signal | true/false |
| `collision_penalty` | float | Agent collision penalty | -0.1, -0.01 |
| `agents_per_target` | int | k — agents required to cover | 1, 2 |
| `lidar_range` | float | Lidar/comm range | 0.25, 0.35 |
| `n_agents` | int | Number of agents | 4 |
| `n_targets` | int | Number of targets | 4 |
| `algorithm` | str | RL algorithm | `mappo` |
| `seed` | int | Random seed | 0, 1, 2 |
| `max_frames` | int | Training budget | 10M, 20M, 30M |
| `lr` | float | Learning rate | 5e-5, 1e-4 |
| `lmbda` | float | GAE lambda | 0.9, 0.95 |
| `entropy_coef` | float | Entropy bonus (0=default) | 0, 0.001, 0.01 |
| `hidden_layers` | str | Network architecture | `[256,256]`, `[512,256]` |
| `activation` | str | Activation function | `tanh`, `relu` |

### Metrics

| Column | Type | Description |
|--------|------|-------------|
| `M1_success_rate` | float | Fraction of episodes where all targets covered |
| `M2_avg_return` | float | Mean episodic return |
| `M3_avg_steps` | float | Mean steps to completion |
| `M4_avg_collisions` | float | Mean collisions per episode |
| `M5_avg_tokens` | float | Mean comm tokens per episode |
| `M6_coverage_progress` | float | Fraction of targets covered by episode end |
| `M7_sample_efficiency` | float | Frames to 80% of final reward |
| `M8_agent_utilization` | float | CV of per-agent covering (0=balanced) |
| `M9_spatial_spread` | float | Mean pairwise agent distance |

### Execution metadata

| Column | Type | Description |
|--------|------|-------------|
| `training_seconds` | float | Wall-clock training time |
| `device` | str | `cpu` or `cuda` |
| `source` | str | Where it ran: `local` or `ovh` |
| `date` | str | Run date (YYYY-MM-DD) |
| `notes` | str | Special annotations (e.g. `comm_ignored`) |

### The `notes` column

Critical for flagging known issues:
- `comm_ignored` — results byte-identical to no-comm baseline (policy ignored comm channel)
- `pre_ablation` — run before lmbda=0.95 was adopted as default
- `csv_only` — no run directory exists, metrics from consolidated CSV only (ablations D, E, F)

## Target Folder Structure

```
results/
├── master_results.csv              # ALL experiments, ALL runs
├── changelog.md                    # Log of every rename/move/archive action
├── comparison_er1_er2_k2.md        # Cross-experiment analysis (updated)
│
├── er1/
│   ├── runs/
│   │   ├── default_k1_l025_s0/          # ER1 baseline k=1
│   │   ├── default_k1_l025_s1/
│   │   ├── default_k1_l035_s0/
│   │   ├── default_k1_l035_s1/
│   │   ├── default_k2_l025_s0/          # ER1 baseline k=2
│   │   ├── default_k2_l025_s1/
│   │   ├── default_k2_l035_s0/
│   │   ├── default_k2_l035_s1/
│   │   ├── al_k1_l025_s0/               # ER1 + agent lidar
│   │   ├── al_k1_l025_s1/
│   │   ├── al_k1_l035_s0/
│   │   ├── al_k1_l035_s1/
│   │   ├── al_k2_l025_s0/
│   │   ├── al_k2_l025_s1/
│   │   ├── al_k2_l035_s0/
│   │   ├── al_k2_l035_s1/
│   │   ├── al_lp_k2_l035_s0/            # AL ablations
│   │   ├── al_sr_k2_l035_s0/
│   │   ├── al_lp_sr_k2_l035_s0/
│   │   ├── abl_ent01_k2_l035_s0/        # Hyperparameter ablations
│   │   ├── abl_ent01_k2_l035_s1/
│   │   ├── abl_ent01_k2_l035_s2/
│   │   ├── abl_ent001_k2_l035_s0/
│   │   ├── abl_ent001_k2_l035_s1/
│   │   ├── abl_ent001_k2_l035_s2/
│   │   ├── abl_lr1e4_k2_l035_s0/
│   │   ├── abl_lr1e4_k2_l035_s1/
│   │   ├── abl_lr1e4_k2_l035_s2/
│   │   ├── abl_lam095_k2_l035_s0/
│   │   ├── abl_lam095_k2_l035_s1/
│   │   ├── abl_lam095_k2_l035_s2/
│   │   ├── abl_20m_k2_l035_s0/
│   │   ├── abl_20m_k2_l035_s1/
│   │   ├── abl_20m_k2_l035_s2/
│   │   ├── abl_net512_relu_k2_l035_s0/
│   │   ├── abl_net512_relu_k2_l035_s1/
│   │   ├── abl_net512_relu_k2_l035_s2/
│   │   ├── abl_k1_sanity_k1_l035_s0/
│   │   ├── abl_k1_sanity_k1_l035_s1/
│   │   └── abl_k1_sanity_k1_l035_s2/
│   ├── er1_ablation_consolidated.csv   # Original CSV (only source for D,E,F)
│   └── archive/
│       ├── er1_old/                    # Pre-focused sweep (n_targets=3,4,7)
│       └── e1/                         # Very early test run
│
├── er2/
│   ├── runs/
│   │   ├── prox_dc8_s0/                        # Base ER2
│   │   ├── prox_dc8_30m_s0/                    # Extended training
│   │   ├── prox_dc8_30m_s1/
│   │   ├── al_prox_dc8_s0/                     # + agent lidar
│   │   ├── al_lp_sr_prox_dc8_s0/               # + AL + LP+SR
│   │   ├── bc_dc16_s0/                          # Broadcast (comm ignored)
│   │   ├── bc_dc16_s1/
│   │   ├── al_lp_sr_bc_dc2_s0/                 # Full stack broadcast ablations
│   │   ├── al_lp_sr_bc_dc8_s0/                 # (pending local runs)
│   │   └── al_lp_sr_bc_dc16_s0/
│   └── archive/                                # (empty for now)
│
├── er3/                             # Future
│   ├── runs/
│   └── archive/
│
└── er4/                             # Future
    ├── runs/
    └── archive/
```

### Variant tag format

```
{modifiers}_{k}{k_val}_{l}{lidar}_{s}{seed}
```

Rules:
- `default` = no agent lidar, standard reward (-0.1 penalty, individual)
- `al` = agent lidar on
- `lp` = low penalty (-0.01)
- `sr` = shared reward
- `abl_*` = hyperparameter ablation (prefix identifies what changed)
- `prox` / `bc` = proximity / broadcast (ER2+ only)
- `dc{N}` = dim_c channel size (ER2+ only)
- `30m` = 30M frames (only when different from standard 10M)
- k and lidar included when the sweep varies them; omit when fixed (k2_l035 for all current ER2)

For ER2 where k=2 and l=0.35 are always fixed, simplify:
`al_lp_sr_prox_dc8_s0` (not `al_lp_sr_prox_dc8_k2_l035_s0`)

For ER1 where k and l vary in the focused sweep, include them:
`default_k1_l025_s0`, `default_k2_l035_s1`

## Complete Rename Mapping

### ER1 focused sweep (er1/)

| Old path | New variant | New path |
|----------|-------------|----------|
| `er1/20260320_0841__er1_mappo_n4_t4_k1_l025_s0/` | `default_k1_l025_s0` | `er1/runs/default_k1_l025_s0/` |
| `er1/20260320_0841__er1_mappo_n4_t4_k1_l025_s1/` | `default_k1_l025_s1` | `er1/runs/default_k1_l025_s1/` |
| `er1/20260320_0841__er1_mappo_n4_t4_k1_l035_s0/` | `default_k1_l035_s0` | `er1/runs/default_k1_l035_s0/` |
| `er1/20260320_0841__er1_mappo_n4_t4_k1_l035_s1/` | `default_k1_l035_s1` | `er1/runs/default_k1_l035_s1/` |
| `er1/20260320_0841__er1_mappo_n4_t4_k2_l025_s0/` | `default_k2_l025_s0` | `er1/runs/default_k2_l025_s0/` |
| `er1/20260320_0841__er1_mappo_n4_t4_k2_l025_s1/` | `default_k2_l025_s1` | `er1/runs/default_k2_l025_s1/` |
| `er1/20260320_1337__er1_mappo_n4_t4_k2_l035_s0/` | `default_k2_l035_s0` | `er1/runs/default_k2_l035_s0/` |
| `er1/20260320_1337__er1_mappo_n4_t4_k2_l035_s1/` | `default_k2_l035_s1` | `er1/runs/default_k2_l035_s1/` |

### ER1 + agent lidar (er1_al/)

| Old path | New variant | New path |
|----------|-------------|----------|
| `er1_al/20260320_1806__er1_al_mappo_n4_t4_k{K}_l{L}_s{S}/` | `al_k{K}_l{L}_s{S}` | `er1/runs/al_k{K}_l{L}_s{S}/` |

(8 runs: k∈{1,2}, l∈{025,035}, s∈{0,1})

### ER1 AL ablations (er1_al_abl_*/)

| Old path | New variant | New path |
|----------|-------------|----------|
| `er1_al_abl_lp/20260321_1057__..._s0/` | `al_lp_k2_l035_s0` | `er1/runs/al_lp_k2_l035_s0/` |
| `er1_al_abl_sr/20260321_1222__..._s0/` | `al_sr_k2_l035_s0` | `er1/runs/al_sr_k2_l035_s0/` |
| `er1_al_abl_lp_sr/20260321_1351__..._s0/` | `al_lp_sr_k2_l035_s0` | `er1/runs/al_lp_sr_k2_l035_s0/` |

### ER1 hyperparameter ablations

| Old exp | Old path | New variant | New path |
|---------|----------|-------------|----------|
| er1_ablation/er1_abl_a | `..._s{S}/` | `abl_ent01_k2_l035_s{S}` | `er1/runs/abl_ent01_k2_l035_s{S}/` |
| er1_abl_a2 | `..._s{S}/` | `abl_ent001_k2_l035_s{S}` | `er1/runs/abl_ent001_k2_l035_s{S}/` |
| er1_abl_b | `..._s{S}/` | `abl_lr1e4_k2_l035_s{S}` | `er1/runs/abl_lr1e4_k2_l035_s{S}/` |
| er1_abl_c | `..._s{S}/` | `abl_lam095_k2_l035_s{S}` | `er1/runs/abl_lam095_k2_l035_s{S}/` |
| er1_abl_g | `..._s{S}/` | `abl_20m_k2_l035_s{S}` | `er1/runs/abl_20m_k2_l035_s{S}/` |
| er1_abl_h | `..._s{S}/` | `abl_net512_relu_k2_l035_s{S}` | `er1/runs/abl_net512_relu_k2_l035_s{S}/` |
| er1_abl_i | `..._s{S}/` | `abl_k1_sanity_k1_l035_s{S}` | `er1/runs/abl_k1_sanity_k1_l035_s{S}/` |

(3 seeds each = 21 runs)

### ER2 runs

| Old path | New variant | New path |
|----------|-------------|----------|
| `er2/20260320_1945__er2_..._s0/` | `prox_dc8_s0` | `er2/runs/prox_dc8_s0/` |
| `er2_30m/20260321_1004__..._s0/` | `prox_dc8_30m_s0` | `er2/runs/prox_dc8_30m_s0/` |
| `er2_30m/20260321_1004__..._s1/` | `prox_dc8_30m_s1` | `er2/runs/prox_dc8_30m_s1/` |
| `er2_al/20260321_0905__..._s0/` | `al_prox_dc8_s0` | `er2/runs/al_prox_dc8_s0/` |
| `er2_al_lp_sr/20260321_1723__..._s0/` | `al_lp_sr_prox_dc8_s0` | `er2/runs/al_lp_sr_prox_dc8_s0/` |
| `er2_bc_dimc16/20260321_1640__..._s0/` | `bc_dc16_s0` | `er2/runs/bc_dc16_s0/` |
| `er2_bc_dimc16/20260321_1640__..._s1/` | `bc_dc16_s1` | `er2/runs/bc_dc16_s1/` |
| `er2_al_lp_sr_bc_dimc2/..._s0/` | `al_lp_sr_bc_dc2_s0` | `er2/runs/al_lp_sr_bc_dc2_s0/` |
| `er2_al_lp_sr_bc_dimc8/..._s0/` | `al_lp_sr_bc_dc8_s0` | `er2/runs/al_lp_sr_bc_dc8_s0/` |
| `er2_al_lp_sr_bc_dimc16/..._s0/` | `al_lp_sr_bc_dc16_s0` | `er2/runs/al_lp_sr_bc_dc16_s0/` |

## Ablations D, E, F — CSV-only runs

These have NO run directories. Their metrics exist only in
`er1_ablation/ablation_consolidated.csv`. They will appear in
`master_results.csv` with `notes=csv_only` and no corresponding
`er1/runs/` directory.

| Old CSV row | New variant | experiment label |
|-------------|-------------|-----------------|
| D, seed 0-2 | `abl_lr1e4_lam095_k2_l035_s{S}` | `ER1 abl: lr=1e-4 + lmbda=0.95` |
| E, seed 0-2 | `abl_lp_nolid_k2_l035_s{S}` | `ER1 abl: low penalty (no AL)` |
| F, seed 0-2 | `abl_sr_nolid_k2_l035_s{S}` | `ER1 abl: shared reward (no AL)` |

## Docs to Update

| File | Action | Why |
|------|--------|-----|
| `docs/er1_first_findings_2026-03-15.md` | Move to `docs/archive/` | Pre-ablation, superseded by er1_report.md |
| `docs/hyperparameter_analysis.md` | Move to `docs/archive/` | Predictions superseded by ablation_results.md |
| `docs/ablation_results.md` | Keep as-is | Ground truth for ablation analysis |
| `docs/er1_report.md` | Keep as-is | Official ER1 report, still accurate |
| `docs/communication_design.md` | Keep as-is | Foundational design doc |
| `docs/er3_er4_plan.md` | Keep as-is | Future roadmap |
| `docs/ovh_gpu_setup.md` | Keep as-is | Operational, recently updated |
| `results/comparison_er1_er2_k2.md` | Update paths + add broadcast findings | References old dir names |

## Changelog Format

`results/changelog.md` — tracks every action taken during reorganization.

```markdown
# Results Reorganization Changelog

## 2026-03-21 — Initial reorganization

### Renames
| Action | Old path | New path |
|--------|----------|----------|
| MOVE | er1/20260320_0841__er1_mappo_n4_t4_k1_l025_s0/ | er1/runs/default_k1_l025_s0/ |
| ... | ... | ... |

### Archives
| Action | Old path | New path | Reason |
|--------|----------|----------|--------|
| ARCHIVE | e1/ | er1/archive/e1/ | Early test, no metrics |
| ARCHIVE | er1_ablation/er1_old/ | er1/archive/er1_old/ | Pre-focused sweep |
| ARCHIVE | docs/er1_first_findings_2026-03-15.md | docs/archive/ | Superseded |
| ARCHIVE | docs/hyperparameter_analysis.md | docs/archive/ | Superseded |

### Deletions (duplicates only)
| Action | Path | Reason |
|--------|------|--------|
| DELETE | er1_ablation/er1_abl_a2/ | Duplicate of standalone er1_abl_a2/ |
| DELETE | er1_ablation/er1_abl_b/ | Duplicate of standalone er1_abl_b/ |
| ... | ... | ... |
| DELETE | er1_ablation/ (empty after moves) | All content moved |
| DELETE | er1_al/ (empty after moves) | All runs moved to er1/runs/ |
| DELETE | er3/ er4/ | Empty placeholders |

### CSV created
- master_results.csv: {N} rows from {M} source CSVs + consolidated CSV
```

## Execution Steps (Updated)

### Step 0: Wait for local experiments
Wait for `er2_al_lp_sr_bc_dimc{2,8,16}` to finish.

### Step 1: Verify S3 ↔ local completeness
For each S3 prefix, verify local has same or more data (local may have
additional runs from local experiments not uploaded).

### Step 2: Create target structure
```bash
mkdir -p results/er1/runs results/er1/archive
mkdir -p results/er2/runs results/er2/archive
mkdir -p docs/archive
```

### Step 3: Move & rename all ER1 runs
Script iterates over all ER1 source dirs, extracts run_id, maps to new
variant tag, moves to `er1/runs/{variant}/`. Logs every action to changelog.

### Step 4: Move & rename all ER2 runs
Same as step 3 for ER2.

### Step 5: Archive obsolete data
- `e1/` → `er1/archive/e1/`
- `er1_ablation/er1_old/` → `er1/archive/er1_old/`
- `docs/er1_first_findings*` → `docs/archive/`
- `docs/hyperparameter_analysis*` → `docs/archive/`

### Step 6: Copy preserved files
- `er1_ablation/ablation_consolidated.csv` → `er1/er1_ablation_consolidated.csv`

### Step 7: Delete confirmed duplicates
- `er1_ablation/er1_abl_{a2,b,c,g,h,i}/` (copies of standalone dirs)
- Empty parent dirs after all moves
- Empty `er3/`, `er4/` placeholders

### Step 8: Build master_results.csv
Collect metrics from:
- All `*/output/metrics.json` files in new `runs/` dirs
- Rows D, E, F from `er1_ablation_consolidated.csv`
- Add `variant`, `experiment`, `family`, `old_run_id`, `old_dir`, `source`, `notes` columns

### Step 9: Move sweep-level files to archive
Old `sweep_report_*.md`, `sweep_results_*.csv`, `training_iter_*.csv` from
each original exp dir → move to `er1/archive/sweep_files/` and
`er2/archive/sweep_files/` (kept for provenance, superseded by master CSV).

### Step 10: Update comparison report
Update `results/comparison_er1_er2_k2.md` to reference new variant names
and include broadcast+LP+SR findings.

### Step 11: Clear S3
```bash
ovhai bucket object delete rendezvous-results@GRA --all --yes
```

### Step 12: Upload clean structure
```bash
ovhai bucket object upload rendezvous-results@GRA \
  --remove-prefix "rendezvous_comm/results/" \
  rendezvous_comm/results/er1/ rendezvous_comm/results/er2/ \
  rendezvous_comm/results/master_results.csv
```

## Validation Checks

Before deleting anything:
- [ ] `master_results.csv` row count matches expected:
  - ER1: 8 (focused) + 8 (AL) + 3 (AL ablations) + 3×7 (abl a,a2,b,c,g,h,i) + 3×3 (D,E,F csv-only) = **40 rows**
  - ER2: 1 + 2 + 1 + 1 + 2 + 1 + 1 + 1 = **10 rows** (+ pending broadcast runs)
  - Total: **~50 rows**
- [ ] Every variant in CSV with `notes != csv_only` has a matching `runs/` directory
- [ ] Every `runs/` directory has `output/metrics.json` (except archived/early runs)
- [ ] `er1_ablation_consolidated.csv` preserved and accessible
- [ ] No metrics.json lost: count before = count after
- [ ] `changelog.md` documents every action taken
- [ ] All old sweep CSVs preserved in archive (provenance)
