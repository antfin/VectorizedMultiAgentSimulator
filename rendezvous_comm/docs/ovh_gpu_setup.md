# OVH Public Cloud — GPU Training Setup

OVH Startup Program account. Only Public Cloud services available.
Available GPUs: **V100S** (32GB VRAM) via AI Training flavors.

## Quick Start

```bash
# 1. Install CLI
curl -sSf https://cli.bhs.ai.cloud.ovh.net/install.sh | bash

# 2. Authenticate
ovhai login

# 3. Create buckets
ovhai bucket create GRA rendezvous-code
ovhai bucket create GRA rendezvous-results

# 4. Upload code (strip local path prefix)
ovhai bucket object upload rendezvous-code@GRA \
  --remove-prefix "/path/to/VectorizedMultiAgentSimulator/" \
  /path/to/VectorizedMultiAgentSimulator/rendezvous_comm

# 5. Submit job (or use Streamlit OVH Jobs page)
ovhai job run pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime \
  --name rendezvous_dry_run \
  --flavor ai1-1-gpu --gpu 1 \
  --volume "rendezvous-code@GRA/:/workspace/code:ro" \
  --volume "rendezvous-results@GRA/er1/:/workspace/results:rwd" \
  --env "RESULTS_DIR=/workspace/results" \
  --env "CHECKPOINTS_DIR=/workspace/results/checkpoints" \
  -- bash -c "export HOME=/tmp && \
    pip install vmas benchmarl tensordict torchrl \
    pyyaml pandas scipy imageio matplotlib && \
    cd /workspace/code/rendezvous_comm && \
    python train.py /workspace/code/rendezvous_comm/configs/er1/dry_run.yaml \
    --device cuda"
```

## GPU Compute Options

### AI Training (batch jobs — recommended for sweeps)
- Submit job, auto-stops when done, no idle cost
- Lifecycle: INITIALIZING → PENDING → RUNNING → FINALIZING → DONE/FAILED
- Can run multiple sweep configs in parallel on separate GPUs (use per-experiment bucket prefix)
- Launch via Streamlit OVH Jobs page or CLI

### AI Notebooks (managed Jupyter — interactive work)
- Pre-installed PyTorch, per-minute billing
- `/workspace` persists across stop/start (synced to Object Storage)
- Anything outside `/workspace` is ephemeral

### Available Flavors
```
ai1-1-gpu    NVIDIA Tesla V100S   13 vCPU, 40GB RAM, 32GB VRAM
ai1-1-cpu    Intel CPU vCores     1 vCPU, 4GB RAM
```
Run `ovhai capabilities flavor list` to check current availability.

## Known Issues & Workarounds

### Parallel jobs overwrite each other's results
OVH AI Training does NOT use a live S3 mount. The lifecycle is:
1. **INITIALIZING**: bucket contents are copied to a local filesystem
2. **RUNNING**: job reads/writes the local filesystem only
3. **FINALIZING**: entire local filesystem is synced back to the bucket prefix

When multiple parallel jobs mount the same bucket root with `rwd`, the last
job to finalize **overwrites all previous results** — only its local files
survive. This caused data loss for broadcast/dimc2 experiments (2026-03-21).

**Fix**: Each job now mounts a per-experiment prefix:
```
# Before (all jobs share one mount — DANGEROUS in parallel):
--volume "rendezvous-results@GRA/:/workspace/results:rwd"

# After (each job isolated by exp_id):
--volume "rendezvous-results@GRA/er2_broadcast/:/workspace/results:rwd"
```
`submit_training_job()` extracts `exp_id` from the config YAML and uses it
as the bucket prefix automatically. Parallel jobs with different exp_ids
are now safe.

**Note**: `sleep` or `sync` at the end of the training command is unnecessary —
OVH handles the sync during FINALIZING, not during RUNNING.

### Permission denied for pip install
The container runs as non-root with HOME=/workspace (a mounted volume).
Pip cannot write to `/workspace/.cache` or `/workspace/.local`.
**Fix:** Set `export HOME=/tmp` before pip install.

### Read-only code volume
Code bucket is mounted `:ro`. Any directory creation inside `/workspace/code/`
fails with `OSError: Read-only file system`.
**Fix:** Use env vars to redirect writable paths to the results volume:
- `RESULTS_DIR=/workspace/results` (already set)
- `CHECKPOINTS_DIR=/workspace/results/checkpoints`

### Upload preserves local absolute paths
`ovhai bucket object upload` stores files with full local paths by default.
**Fix:** Use `--remove-prefix "/local/path/"` (trailing slash required).

### No `--gpu-model` flag
The `ovhai job run` command uses `--flavor`, not `--gpu-model`.
Use `--flavor ai1-1-gpu` for V100S.

### `--tail` syntax for logs
Use `--tail=N` (equals sign), not `--tail N` (space).

### Video generation skipped on OVH
`generate_run_videos` requires pyglet/OpenGL which is unavailable on
headless OVH containers. Videos are skipped (try/except) during training.
Rebuild locally with `python train.py --rebuild-videos <config.yaml>`
or via the Streamlit "Rebuild Videos" button.

## Storage

### Object Storage (S3-compatible)
- Standard S3 (1-AZ): ~0.007 EUR/GB/month
- Egress: FREE on Standard
- A single run: ~20-150 MB. 8-run sweep: ~0.2-1.2 GB
- Cost for 50 sweeps (~50 GB): ~0.35 EUR/month — negligible

### Volume Mount Permissions
- `ro` — read-only input, not synced back
- `rw` — read-write, synced back on job end, deletions NOT propagated
- `rwd` — read-write-delete, full sync including deletions

## CLI Reference

```bash
# List flavors
ovhai capabilities flavor list

# Create buckets (data_store first, then container name)
ovhai bucket create GRA rendezvous-code

# Upload with path stripping
ovhai bucket object upload rendezvous-code@GRA \
  --remove-prefix "/local/prefix/" ./rendezvous_comm

# List bucket contents
ovhai bucket object list rendezvous-code@GRA

# Clear bucket
ovhai bucket object delete rendezvous-code@GRA --all --yes

# Download results
ovhai bucket object download rendezvous-results@GRA \
  --output-dir ./results --workers 8

# Job management
ovhai job list
ovhai job get <job-id>
ovhai job logs <job-id> --tail=50
ovhai job stop <job-id>
```

## Config for GPU Training
```yaml
train:
  train_device: cuda
  sampling_device: cuda   # or keep cpu if GPU memory tight
```

## Architecture on OVH
```
rendezvous-code (bucket, read-only mount)
  └── rendezvous_comm/
      ├── train.py
      ├── src/
      └── configs/

rendezvous-results (bucket, read-write-delete mount)
  ├── er1/
  │   └── YYYYMMDD_HHMM__<run_id>/
  │       ├── input/
  │       ├── logs/
  │       └── output/
  └── checkpoints/
```
