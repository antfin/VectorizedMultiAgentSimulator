# `multi-scenario eval`

Re-evaluate a trained policy without retraining. LERO-aware.

```bash
multi-scenario eval <run_dir> [--episodes N] [--name TAG]
```

## What it does

- Loads `<run_dir>/input/config.json` → `ExperimentConfig`.
- Locates the latest BenchMARL checkpoint under `<run_dir>/output/benchmarl/`.
- Reloads the experiment via `Experiment.reload_from_file(...)` with overrides:
    - `save_folder` → the local exp dir (the pickled cfg's path is the container-side `/workspace/results`).
    - `sampling_device` / `train_device` / `buffer_device` → `cfg.training.device` (typically `cpu` for replay).
    - `restore_map_location` → same — CUDA-saved tensors get materialised on CPU.
- Runs `cfg.evaluation.episodes` (or `--episodes` override) eval rollouts.
- Writes `<run_dir>/output/eval_runs/<tag>.json` with the resulting `EvalRunRecord`.

## LERO awareness

For LERO runs (`cfg.lero is not None`), the CLI also:

- Loads the winning candidate's `obs_source` / `reward_source` from `output/lero/evolution_history.json`.
- Rebuilds the patched Discovery class via `make_patched_discovery_class(...)`.
- Monkey-patches `ScenarioEnvFunFactory.__setstate__` so the pickled task's factory restores the patched class on unpickle (covers legacy Phase 5a artefacts whose pickle stored a dummy state).

Without LERO awareness, the eval would use the bare Discovery scenario
and either crash on tensor-shape mismatch (LERO-enhanced obs) or
silently report meaningless metrics.

## Use cases

- Re-run with `--episodes 200` for tighter CIs after a small training-time eval.
- Compute the post-full-train M1 for a pre-Phase-1 LERO run whose `final_summary.json` lacks `best_candidate_full_metrics`.
- Verify reproducibility (paired with [`tests/reproducibility/test_same_seed_byte_equal.py`](../reproducibility/determinism_test.md)).
