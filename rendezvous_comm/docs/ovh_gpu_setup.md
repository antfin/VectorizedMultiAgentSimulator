# OVH Public Cloud — GPU Training Setup

User has an OVH Public Cloud account. Goal: accelerate BenchMARL/TorchRL/VMAS training with GPU.

## GPU Compute Options

### 1. AI Notebooks (managed Jupyter, recommended for interactive work)
- Pre-installed PyTorch, per-minute billing, zero setup
- GPU tiers: L4 (24GB) ~0.75 EUR/hr, L40S (48GB) ~1.40 EUR/hr, H100 (80GB) ~3.10 EUR/hr
- L4 is sufficient for VMAS+BenchMARL workloads
- `/workspace` persists across stop/start (synced to Object Storage)
- Anything outside `/workspace` is ephemeral
- Launch: `ovhai notebook run pytorch jupyterlab --gpu 1 --volume results@GRA/:/workspace/results:rw`

### 2. AI Training (managed batch jobs, recommended for sweeps)
- Submit job, auto-stops when done, no idle cost
- Same GPU tiers and pricing as AI Notebooks
- Lifecycle: INITIALIZING (sync from bucket) → RUNNING → FINALIZING (sync back) → DONE
- Can run multiple sweep configs in parallel on separate GPUs
- Needs script, not notebook (or use papermill)
- Launch: `ovhai job run pytorch/pytorch:2.x-cuda12.x --gpu 1 --volume code@GRA/:/workspace/code:ro --volume results@GRA/:/workspace/results:rw -- bash -c "..."`

### 3. GPU Instance (full VM, best for persistent environment)
- Full SSH access, install anything, keep running
- Same GPU tiers: L4 ~0.75 EUR/hr, L40S ~1.40 EUR/hr
- Pay while instance is up (even idle) — remember to stop/delete
- Supports VS Code Remote-SSH (local IDE, remote compute)
- Also supports SSH tunnel for remote Jupyter kernel

## Storage

### Object Storage (S3-compatible)
- Standard S3 (1-AZ): ~0.007 EUR/GB/month
- Standard S3 (3-AZ): 0.014 EUR/GB/month
- High Performance S3: ~0.018 EUR/GB/month
- Cold Archive: 0.0013 EUR/GB/month (6-month minimum)
- **Egress: FREE on Standard** (as of Jan 2026)
- API calls: included, no per-request charges
- Ingress (upload): always free
- Internal traffic (between OVH services): free

### Storage for training results
- All file types supported: .pt, .mp4, .png, .json, .yaml, .csv
- A single run's results: ~20-150 MB. 8-run sweep: ~0.2-1.2 GB
- Cost for 50 sweeps (~50 GB): ~0.35 EUR/month — negligible
- GPU compute time is where the money goes

### Volume mount permissions
- `ro` — read-only input, not synced back
- `rw` — read-write, synced back on job end, deletions NOT propagated
- `rwd` — read-write-delete, full sync including deletions

## CLI Commands

```bash
# Create buckets
ovhai bucket create rendezvous-code@GRA
ovhai bucket create rendezvous-results@GRA

# Upload code
ovhai bucket object upload rendezvous-code@GRA ./rendezvous_comm/

# Download results
ovhai bucket object download rendezvous-results@GRA
ovhai bucket object download rendezvous-results@GRA --prefix er1_mappo_n4/
ovhai bucket object download rendezvous-results@GRA --workers 8

# Monitor jobs
ovhai job logs <job-id>
ovhai job get <job-id>
```

### Alternative download tools
- rclone: `rclone sync ovh-s3:rendezvous-results ./results/ --progress`
- aws s3: `aws s3 sync s3://rendezvous-results ./results/` (set endpoint to `https://s3.gra.io.cloud.ovh.us`)

## Config change for GPU
```yaml
train:
  train_device: cuda
  sampling_device: cuda   # or keep cpu if GPU memory tight
```

## Code change needed
`config.py` has `RESULTS_DIR = Path(__file__).parent.parent / "results"` — needs to be configurable via env var to point to `/workspace/results` on OVH.
