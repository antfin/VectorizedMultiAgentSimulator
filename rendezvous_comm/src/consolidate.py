"""CSV consolidation: merge per-run scalars into experiment-level CSVs.

Produces three timestamped CSVs in results/<exp_id>/:
  - sweep_results_{ts}.csv  — 1 row per run, all config + metrics
  - training_iter_{ts}.csv  — 1 row per (run_id, step), iteration-freq
  - training_eval_{ts}.csv  — 1 row per (run_id, step), eval-freq

Usage:
    from rendezvous_comm.src.consolidate import consolidate_csvs
    consolidate_csvs("er1")

CLI:
    python -m rendezvous_comm.src.consolidate er1
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

_log = logging.getLogger("rendezvous")

# Row-count threshold: CSVs with <= this many rows are eval-frequency
_EVAL_THRESHOLD = 20

# Columns to exclude from training CSVs (redundant with step)
_EXCLUDE_COLS = {"counters_iter"}


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


def _load_run_config(run_dir: Path) -> dict:
    """Load saved config.yaml from a run's input/ dir."""
    config_path = run_dir / "input" / "config.yaml"
    if not config_path.exists():
        return {}
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _build_sweep_row(run_dir: Path, run_id: str) -> Optional[dict]:
    """Build a single sweep_results row from a completed run."""
    from .storage import _parse_run_id

    metrics_path = run_dir / "output" / "metrics.json"
    if not metrics_path.exists():
        return None

    with open(metrics_path) as f:
        metrics = json.load(f)

    row = {"run_id": run_id}
    row.update(metrics)

    # Backfill from config.yaml if metrics are missing config fields.
    # task_overrides take precedence over base task config.
    config = _load_run_config(run_dir)
    if config:
        task = config.get("task", {})
        overrides = config.get("task_overrides", {})
        # Merge: base task + overrides (overrides win)
        effective_task = {**task, **overrides}
        train = config.get("train", {})
        for k, v in effective_task.items():
            if k not in row:
                row[k] = v
        for k, v in train.items():
            if k not in row:
                row[k] = v
        if "algorithm" not in row and "algorithm" in config:
            row["algorithm"] = config["algorithm"]
        if "seed" not in row and "seed" in config:
            row["seed"] = config["seed"]
        if "exp_id" not in row and "exp_id" in config:
            row["exp_id"] = config["exp_id"]

    # Backfill from run_id parsing as last resort
    parsed = _parse_run_id(run_id)
    for k, v in parsed.items():
        if k not in row:
            row[k] = v

    return row


def _load_scalars(run_dir: Path) -> dict:
    """Load all BenchMARL CSV scalars as {name: [(step, value), ...]}."""
    import csv as csv_mod
    scalars = {}
    benchmarl_dir = run_dir / "output" / "benchmarl"
    if not benchmarl_dir.exists():
        return scalars
    for sd in benchmarl_dir.glob("**/scalars"):
        for csv_file in sorted(sd.glob("*.csv")):
            name = csv_file.stem
            if name in _EXCLUDE_COLS:
                continue
            rows = []
            with open(csv_file) as f:
                for row in csv_mod.reader(f):
                    if len(row) >= 2:
                        try:
                            rows.append(
                                (int(row[0]), float(row[1]))
                            )
                        except ValueError:
                            continue
            if rows:
                scalars[name] = rows
    return scalars


def _classify_scalars(scalars: dict):
    """Split scalars into iter-frequency and eval-frequency groups.

    Also aligns custom eval metrics (eval_M1, eval_M4) whose step
    indices are off-by-one vs native BenchMARL eval scalars, because
    on_evaluation_end fires after on_batch_collected increments iter.
    """
    iter_scalars = {}
    eval_scalars = {}
    for name, data in scalars.items():
        if len(data) <= _EVAL_THRESHOLD:
            eval_scalars[name] = data
        else:
            iter_scalars[name] = data

    # Align off-by-one custom eval metrics to native eval steps
    custom_keys = {"eval_M1_success_rate", "eval_M4_avg_collisions"}
    native_keys = set(eval_scalars.keys()) - custom_keys
    if custom_keys & set(eval_scalars.keys()) and native_keys:
        # Get native step set for alignment
        native_ref = next(
            eval_scalars[k] for k in native_keys
            if k in eval_scalars
        )
        native_steps = {s for s, _ in native_ref}
        for ck in custom_keys & set(eval_scalars.keys()):
            data = eval_scalars[ck]
            aligned = []
            for step, val in data:
                # Shift step back by 1 to match native eval
                adj = step - 1
                if adj in native_steps:
                    aligned.append((adj, val))
                else:
                    aligned.append((step, val))
            eval_scalars[ck] = aligned

    return iter_scalars, eval_scalars


def _scalars_to_wide_df(run_id: str, scalars: dict) -> pd.DataFrame:
    """Convert {name: [(step, val)]} to wide DataFrame with run_id, step.

    Only keeps rows where ALL scalar columns have values (no NaN).
    Different scalars may log at slightly different steps; we use
    the intersection of step sets to guarantee no NaN.
    """
    if not scalars:
        return pd.DataFrame()

    # Find common steps across all scalars (intersection)
    step_sets = [set(s for s, _ in data) for data in scalars.values()]
    common_steps = step_sets[0]
    for ss in step_sets[1:]:
        common_steps = common_steps & ss

    if not common_steps:
        # Fallback: use all steps but drop rows with NaN after
        all_steps = set()
        for data in scalars.values():
            for step, _ in data:
                all_steps.add(step)
        target_steps = sorted(all_steps)
    else:
        target_steps = sorted(common_steps)

    rows = []
    for step in target_steps:
        row = {"run_id": run_id, "step": step}
        has_all = True
        for name, data in scalars.items():
            step_map = dict(data)
            if step in step_map:
                row[name] = step_map[step]
            else:
                has_all = False
        if has_all:
            rows.append(row)

    if not rows:
        # If no common steps, keep all but accept some NaN
        for step in target_steps:
            row = {"run_id": run_id, "step": step}
            for name, data in scalars.items():
                step_map = dict(data)
                if step in step_map:
                    row[name] = step_map[step]
            rows.append(row)

    return pd.DataFrame(rows)


def consolidate_csvs(
    exp_id: str,
    results_root: Optional[Path] = None,
) -> Dict[str, Path]:
    """Consolidate all run data into three experiment-level CSVs.

    Args:
        exp_id: experiment identifier (e.g. "er1")
        results_root: override for RESULTS_DIR

    Returns:
        {"sweep": Path, "iter": Path, "eval": Path}
    """
    from .storage import ExperimentStorage, _extract_run_id

    storage = ExperimentStorage(exp_id, results_root)
    run_dirs = storage.list_run_dirs()

    if not run_dirs:
        _log.warning(f"No completed runs for {exp_id}")
        return {}

    ts = _timestamp()
    results_dir = storage.results_dir

    # ── Sweep results ────────────────────────────────────────────────
    sweep_rows = []
    iter_frames = []
    eval_frames = []

    for run_dir in run_dirs:
        run_id = _extract_run_id(run_dir.name)

        # Sweep row
        sweep_row = _build_sweep_row(run_dir, run_id)
        if sweep_row:
            sweep_rows.append(sweep_row)

        # Training scalars
        scalars = _load_scalars(run_dir)
        if scalars:
            iter_sc, eval_sc = _classify_scalars(scalars)

            iter_df = _scalars_to_wide_df(run_id, iter_sc)
            if not iter_df.empty:
                iter_frames.append(iter_df)

            eval_df = _scalars_to_wide_df(run_id, eval_sc)
            if not eval_df.empty:
                eval_frames.append(eval_df)

    paths = {}

    # Write sweep CSV
    if sweep_rows:
        sweep_df = pd.DataFrame(sweep_rows)
        # Stable column order: run_id first, then sorted
        cols = ["run_id"] + sorted(
            [c for c in sweep_df.columns if c != "run_id"]
        )
        sweep_df = sweep_df[cols]
        sweep_path = results_dir / f"sweep_results_{ts}.csv"
        sweep_df.to_csv(sweep_path, index=False)
        paths["sweep"] = sweep_path
        _log.info(
            f"sweep_results: {len(sweep_df)} runs, "
            f"{len(sweep_df.columns)} cols → {sweep_path.name}"
        )

    # Write training_iter CSV
    if iter_frames:
        iter_all = pd.concat(iter_frames, ignore_index=True)
        cols = ["run_id", "step"] + sorted(
            [c for c in iter_all.columns if c not in ("run_id", "step")]
        )
        iter_all = iter_all[cols]
        iter_path = results_dir / f"training_iter_{ts}.csv"
        iter_all.to_csv(iter_path, index=False)
        paths["iter"] = iter_path
        _log.info(
            f"training_iter: {len(iter_all)} rows, "
            f"{len(iter_all.columns)} cols → {iter_path.name}"
        )

    # Write training_eval CSV
    if eval_frames:
        eval_all = pd.concat(eval_frames, ignore_index=True)
        cols = ["run_id", "step"] + sorted(
            [c for c in eval_all.columns if c not in ("run_id", "step")]
        )
        eval_all = eval_all[cols]
        eval_path = results_dir / f"training_eval_{ts}.csv"
        eval_all.to_csv(eval_path, index=False)
        paths["eval"] = eval_path
        _log.info(
            f"training_eval: {len(eval_all)} rows, "
            f"{len(eval_all.columns)} cols → {eval_path.name}"
        )

    return paths


def load_latest_csv(
    results_dir: Path,
    prefix: str,
) -> Optional[pd.DataFrame]:
    """Load the most recent timestamped CSV matching prefix.

    Args:
        results_dir: e.g. results/er1/
        prefix: one of "sweep_results", "training_iter", "training_eval"

    Returns:
        DataFrame or None if no matching CSV exists.
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return None

    matches = sorted(results_dir.glob(f"{prefix}_*.csv"))
    if not matches:
        return None

    latest = matches[-1]  # sorted by timestamp in filename
    return pd.read_csv(latest)


def list_experiments_with_data(
    results_root: Optional[Path] = None,
) -> list:
    """Return exp_ids that have at least one completed run or CSV."""
    if results_root is None:
        from .config import RESULTS_DIR
        results_root = RESULTS_DIR

    exp_ids = []
    if not results_root.exists():
        return exp_ids
    for d in sorted(results_root.iterdir()):
        if not d.is_dir():
            continue
        # Has a consolidated CSV?
        if list(d.glob("sweep_results_*.csv")):
            exp_ids.append(d.name)
            continue
        # Has any completed run (metrics.json)?
        for sub in d.iterdir():
            if sub.is_dir() and (sub / "output" / "metrics.json").exists():
                exp_ids.append(d.name)
                break
    return exp_ids


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python -m rendezvous_comm.src.consolidate <exp_id>")
        sys.exit(1)
    result = consolidate_csvs(sys.argv[1])
    for kind, path in result.items():
        print(f"  {kind}: {path}")
