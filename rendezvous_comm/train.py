#!/usr/bin/env python3
"""Standalone training script for OVH AI Training (or any headless env).

Usage:
    python -m rendezvous_comm.train configs/er1/demo.yaml
    python -m rendezvous_comm.train configs/er1/demo.yaml --device cuda
    python -m rendezvous_comm.train configs/er1/demo.yaml --dry-run
    python -m rendezvous_comm.train configs/er1/demo.yaml --rebuild-csv

Environment variables:
    RESULTS_DIR  — override results output path (default: rendezvous_comm/results)
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Disable matplotlib GUI in CLI
os.environ.setdefault("MPLBACKEND", "Agg")


def main():
    parser = argparse.ArgumentParser(
        description="Run experiment sweep (headless, for OVH or CI).",
    )
    parser.add_argument(
        "config", type=str,
        help="Path to experiment YAML config file",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Training device: cuda, cpu (default: auto-detect)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build experiments without training",
    )
    parser.add_argument(
        "--max-runs", type=int, default=None,
        help="Cap number of runs (for testing)",
    )
    parser.add_argument(
        "--skip-complete", action="store_true", default=True,
        help="Skip runs that already have results (default: True)",
    )
    parser.add_argument(
        "--force-retrain", action="store_true",
        help="Re-run even if results exist",
    )
    parser.add_argument(
        "--rebuild-csv", action="store_true",
        help="Rebuild consolidated CSVs without training",
    )
    parser.add_argument(
        "--rebuild-videos", action="store_true",
        help="Regenerate videos for all completed runs",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("train")

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        log.error(f"Config file not found: {config_path}")
        sys.exit(1)

    from src.config import load_experiment, RESULTS_DIR

    spec = load_experiment(config_path)
    spec.ensure_dirs()

    log.info(f"Experiment: {spec.exp_id} — {spec.name}")
    log.info(f"Config:     {config_path}")
    log.info(f"Results:    {RESULTS_DIR}")

    # Rebuild CSVs only (no training)
    if args.rebuild_csv:
        from src.consolidate import consolidate_csvs
        paths = consolidate_csvs(spec.exp_id)
        for kind, path in paths.items():
            log.info(f"  {kind}: {path}")
        if not paths:
            log.warning("No data to consolidate.")
            sys.exit(1)
        sys.exit(0)

    # Rebuild videos for all completed runs
    if args.rebuild_videos:
        from src.storage import ExperimentStorage
        from src.runner import generate_run_videos
        storage = ExperimentStorage(spec.exp_id)
        run_dirs = storage.list_run_dirs()
        if not run_dirs:
            log.warning("No completed runs found.")
            sys.exit(1)
        from src.storage import _extract_run_id, RunStorage
        n_ok, n_fail = 0, 0
        for rd in run_dirs:
            rid = _extract_run_id(rd.name)
            rs = RunStorage(rd, rid)
            if not rs.has_policy():
                log.info(f"  SKIP {rid} — no policy")
                continue
            log.info(f"  Generating videos for {rid}...")
            try:
                # Parse task_overrides from saved config
                import yaml
                cfg_path = rs.input_dir / "config.yaml"
                overrides = None
                if cfg_path.exists():
                    with open(cfg_path) as f:
                        cfg = yaml.safe_load(f) or {}
                    overrides = cfg.get("task_overrides")
                generate_run_videos(rs, spec.task, overrides, log)
                n_ok += 1
            except Exception as exc:
                log.warning(f"  FAIL {rid}: {exc}")
                n_fail += 1
        log.info(f"Videos: {n_ok} generated, {n_fail} failed")
        sys.exit(0)

    # Override device if specified
    if args.device:
        spec.train.train_device = args.device
        spec.train.sampling_device = args.device
        log.info(f"Device:     {args.device}")
    else:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        spec.train.train_device = device
        spec.train.sampling_device = device
        log.info(f"Device:     {device} (auto-detected)")

    # Count sweep runs
    all_runs = list(spec.iter_runs())
    log.info(f"Sweep:      {len(all_runs)} runs")

    # Run sweep
    skip = not args.force_retrain
    t0 = time.monotonic()

    from src.runner import run_sweep

    results = run_sweep(
        spec,
        skip_complete=skip,
        dry_run=args.dry_run,
        max_runs=args.max_runs,
    )
    elapsed = time.monotonic() - t0

    # Summary
    m, s = divmod(int(elapsed), 60)
    h, m = divmod(m, 60)
    log.info(f"Sweep complete: {len(results)} runs in {h}h {m}m {s}s")

    # Write summary JSON
    summary_path = spec.results_dir / "sweep_summary.json"
    summary = {
        "exp_id": spec.exp_id,
        "config": str(config_path),
        "n_runs": len(results),
        "elapsed_seconds": int(elapsed),
        "device": spec.train.train_device,
        "runs": {},
    }
    for run_id, metrics in results.items():
        summary["runs"][run_id] = {
            k: (v if isinstance(v, (int, float, str)) else str(v))
            for k, v in metrics.items()
        }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Summary:    {summary_path}")

    csv_matches = sorted(
        spec.results_dir.glob("sweep_results_*.csv")
    )
    if csv_matches:
        log.info(f"CSV:        {csv_matches[-1]}")

    # Exit code
    if not results and not args.dry_run:
        log.warning("No runs completed.")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
