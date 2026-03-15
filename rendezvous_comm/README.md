# Rendezvous Comm — Experiment Manager

Experiment infrastructure for **"Learning Communication Protocols for Multi-Robot Rendezvous"**.
Trains multi-agent policies using VMAS Discovery + BenchMARL, manages OVH cloud GPU jobs,
and provides a web dashboard for analysis.

---

## Quick Start

### 1. Install dependencies

```bash
# From the repo root (VectorizedMultiAgentSimulator/)
pip install -e .                                      # VMAS
pip install -r rendezvous_comm/requirements.txt       # Core deps
pip install streamlit scipy                           # Web app + stats
```

### 2. Run a local training experiment

```bash
cd rendezvous_comm
python3 train.py configs/er1/demo.yaml
```

This will:
- Load the experiment config (which agents, targets, algorithm, sweep dimensions)
- Auto-detect GPU (falls back to CPU)
- Train all sweep combinations
- Save results to `results/er1/` with metrics, policies, and training logs

### 3. Launch the web dashboard

```bash
cd rendezvous_comm
streamlit run app.py
```

Opens at `http://localhost:8501`. Use the sidebar to navigate between pages.

---

## Project Structure

```
rendezvous_comm/
├── app.py                  # Streamlit home page
├── train.py                # Headless training script (local or OVH)
├── Dockerfile              # Container for OVH AI Training
├── requirements.txt
├── configs/                # Experiment YAML configs
│   ├── er1/                #   ER1: No communication (baseline)
│   ├── er2/                #   ER2: Engineered schema
│   ├── er3/                #   ER3: Symbolic intent
│   ├── er4/                #   ER4: Event-triggered
│   └── e1/                 #   E1: Static LLM
├── pages/                  # Streamlit pages (auto-discovered)
│   ├── 1_Experiments.py    #   Config browser + editor + run status
│   ├── 2_OVH_Jobs.py       #   Launch/monitor/download OVH jobs
│   ├── 3_Training_Curves.py #  Per-run training curves
│   ├── 4_Sweep_Analysis.py  #  Heatmaps, seed variance, scatter plots
│   ├── 5_Cross_Experiment.py # Statistical comparison across experiments
│   └── 6_Run_Detail.py      # Per-run deep dive (KPIs, config, policy)
├── src/                    # Core library
│   ├── config.py           #   YAML loading, ExperimentSpec, sweep iteration
│   ├── storage.py          #   Run/experiment storage, metrics I/O
│   ├── runner.py           #   BenchMARL training loop + evaluation
│   ├── metrics.py          #   M1-M9 metric computation
│   ├── plotting.py         #   All matplotlib plot functions
│   ├── stats.py            #   Mann-Whitney U, bootstrap CI, Pareto frontier
│   ├── ovh.py              #   OVH CLI wrapper (job submit, monitor, download)
│   ├── provenance.py       #   Config/code freshness tracking
│   └── ...
├── notebooks/              # Jupyter experiment notebooks
├── results/                # Training output (auto-created)
├── tests/                  # pytest suite
└── docs/                   # Analysis documents
```

---

## Training Script

The `train.py` script runs experiments headlessly — designed for cloud GPUs but works locally too.

```bash
# Basic usage
python3 train.py configs/er1/demo.yaml

# Force GPU
python3 train.py configs/er1/demo.yaml --device cuda

# Preview without training
python3 train.py configs/er1/demo.yaml --dry-run

# Limit to first N runs (for testing)
python3 train.py configs/er1/demo.yaml --max-runs 2

# Re-run even if results exist
python3 train.py configs/er1/demo.yaml --force-retrain
```

**Output** goes to `results/<exp_id>/YYYYMMDD_HHMM__<run_id>/`:
- `input/config.yaml` — frozen config snapshot
- `logs/run.log` — training log
- `output/metrics.json` — final M1-M9 metrics
- `output/policy.pt` — trained policy weights
- `output/benchmarl/` — BenchMARL CSV scalars and checkpoints

After all runs complete, a `sweep_summary.json` is written to `results/<exp_id>/`.

---

## Experiment Configs

Each YAML file in `configs/<exp_id>/` is self-contained:

```yaml
exp_id: er1
name: "No-Comm Control"
description: >
  Baseline without communication.

task:
  n_agents: 4
  n_targets: 4
  agents_per_target: 2      # k=2 means 2 agents must cover a target simultaneously
  lidar_range: 0.35
  targets_respawn: false     # Required for M1/M3 metrics to work
  max_steps: 200

train:
  algorithm: mappo
  max_n_frames: 10_000_000  # BenchMARL standard budget
  evaluation_interval: 960_000
  train_device: cpu          # Overridden by --device or auto-detect

sweep:
  seeds: [0]
  n_agents: [4, 6]
  n_targets: [3, 4]
  agents_per_target: [1, 2]
  lidar_range: [0.35]
  algorithms: [mappo]
```

The sweep section generates all combinations. The config above produces
`2 x 2 x 2 x 1 x 1 x 1 = 8` runs.

**Naming convention:**
- `demo.yaml` — quick test config
- `single_mappo_n4_l035.yaml` — single run
- `sweep_mappo-ippo_n2-6_l025-045.yaml` — full parameter sweep

---

## Web Dashboard Pages

### 1. Experiments
Browse and edit YAML configs. See run status (DONE/PENDING) and freshness badges
(whether results match the current config).

### 2. OVH Jobs
Four tabs for cloud GPU management:
- **Launch Job** — Select config, GPU model, and buckets, then submit
- **Monitor Jobs** — View status table, logs, stop running jobs
- **Download Results** — Pull results from OVH Object Storage to local
- **Cost Estimator** — Calculate GPU cost before submitting

Requires `ovhai` CLI installed and authenticated (see [OVH Cloud Setup](#ovh-cloud-setup)).

### 3. Training Curves
Select an experiment and one or more runs. Shows:
- M1 (success rate) and M4 (collisions) eval curves
- 6-panel training dashboard (reward, targets covered, collisions, entropy, etc.)
- Multi-run overlay for comparison

### 4. Sweep Analysis
For experiments with parameter sweeps:
- **Heatmap** — metric values across two swept parameters
- **Seed variance** — bar chart with error bars grouped by algorithm or parameter
- **Cross-metric scatter** — 2x2 plot of M1 vs M4, M1 vs M3, M6 vs M8, M1 vs M9
- Raw metrics table

### 5. Cross-Experiment Comparison
Compare across ER1/ER2/ER3/ER4:
- Bar chart comparing any metric across experiments
- **Mann-Whitney U test** — non-parametric significance test with p-value and effect size
- **Bootstrap confidence intervals** (95%)
- Radar chart for multi-metric profiles
- **Pareto frontier** — M1 (success) vs M5 (communication cost)

### 6. Run Detail
Deep dive into a single run:
- KPI cards (M1-M9)
- Provenance check (config freshness)
- Saved config YAML
- Training curves
- Policy file info
- Generated report

---

## OVH Cloud Setup

### Prerequisites

1. **OVH account** with Public Cloud project (AI Training enabled)
2. **ovhai CLI** — install from [OVH docs](https://docs.ovh.com/gb/en/ai-training/install-client/)

```bash
# Install and authenticate
ovhai login
```

3. **Two S3 buckets** (create via OVH console or CLI):
   - `rendezvous-code` — for uploading your code
   - `rendezvous-results` — for training output

### Running on OVH

**Option A: From the web dashboard**

1. `streamlit run app.py`
2. Go to **OVH Jobs** page
3. Click **Upload Code to OVH** (uploads the repo to your code bucket)
4. Select config, GPU model (L4/L40S/H100), and click **Submit Job**
5. Monitor progress in the **Monitor Jobs** tab
6. When done, **Download Results** to your local machine

**Option B: From the command line**

```bash
# Upload code
ovhai bucket object upload rendezvous-code@GRA .

# Submit a job
ovhai job run pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime \
  --name rendezvous_er1 \
  --gpu 1 --gpu-model L4 \
  --volume rendezvous-code@GRA/:/workspace/code:ro \
  --volume rendezvous-results@GRA/:/workspace/results:rwd \
  --env RESULTS_DIR=/workspace/results \
  -- bash -c "pip install -e /workspace/code && cd /workspace/code && python3 rendezvous_comm/train.py rendezvous_comm/configs/er1/demo.yaml --device cuda"

# Check status
ovhai job list

# Download results
ovhai bucket object download rendezvous-results@GRA --output-dir ./results
```

### GPU Models and Pricing

| GPU    | VRAM  | EUR/hr | Best for                    |
|--------|-------|--------|-----------------------------|
| L4     | 24 GB | ~0.75  | Quick tests, small sweeps   |
| L40S   | 48 GB | ~1.40  | Full sweeps                 |
| H100   | 80 GB | ~3.10  | Large batch / fast training |

Use the **Cost Estimator** tab in the dashboard to estimate total cost before submitting.

---

## Metrics Reference

| ID | Name              | What it measures                                       |
|----|-------------------|--------------------------------------------------------|
| M1 | Success Rate      | Fraction of episodes where all targets are covered     |
| M2 | Avg Return        | Mean episode reward                                    |
| M3 | Avg Steps         | Mean steps to complete (when successful)               |
| M4 | Avg Collisions    | Mean agent-agent collisions per episode                |
| M5 | Avg Tokens        | Communication tokens per episode (0 for ER1)           |
| M6 | Coverage Progress | Partial credit: fraction of targets covered            |
| M7 | Sample Efficiency | Frames needed to reach threshold (computed from CSVs)  |
| M8 | Agent Utilization | CV of per-agent covering contributions (0 = balanced)  |
| M9 | Spatial Spread    | Mean pairwise agent distance                           |

---

## Running Tests

```bash
cd rendezvous_comm
python3 -m pytest tests/ -q
```

---

## Environment Variables

| Variable      | Default                          | Description                          |
|---------------|----------------------------------|--------------------------------------|
| `RESULTS_DIR` | `rendezvous_comm/results`        | Where training output is stored      |
| `DISPLAY`     | (system)                         | If unset, matplotlib uses Agg backend|
