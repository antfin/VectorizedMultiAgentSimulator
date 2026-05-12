# `multi-scenario regenerate-videos`

Replay a saved policy and write before/after-training mp4 videos.

```bash
multi-scenario regenerate-videos <run_dir>
```

## When to use

- OVH containers are headless (no X11) — videos can't be recorded during training; this regenerates them locally from the saved checkpoint.
- The Streamlit Submit page calls this automatically post-pullback when `<run_dir>/videos/` is empty AND a checkpoint exists (Phase 8 auto-poll lifecycle).

## What it produces

```text
<run_dir>/videos/before_training.mp4   # untrained policy rollout
<run_dir>/videos/after_training.mp4    # trained policy rollout
```

Both ~30-60s clips at default frame rate. Skips silently when the run
has no BenchMARL checkpoint (smoke runs intentionally disable
checkpointing).
