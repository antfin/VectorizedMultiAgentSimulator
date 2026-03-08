"""Results storage and retrieval for experiment runs.

Directory layout per run:
  results/<exp_id>/<run_id>/
    input/
      config.yaml           # Frozen config snapshot
    logs/
      run.log               # Python logger output
      training_log.csv      # Per-iteration training metrics
    output/
      benchmarl/            # Raw BenchMARL artifacts (CSV, checkpoints)
      metrics.json          # Final aggregate metrics
      eval_episodes.json    # Per-episode eval data (optional)
    report.txt              # Human-readable run summary
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _json_serializable(obj):
    """Convert non-serializable types for JSON."""
    if hasattr(obj, "item"):  # torch/numpy scalar
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not serializable: {type(obj)}")


class RunStorage:
    """Manages data for a single experiment run.

    Creates a structured directory with input/, logs/, and output/ subdirs.
    """

    def __init__(self, results_dir: Path, run_id: str):
        self.run_dir = results_dir / run_id
        self.run_id = run_id

        # Create structured subdirectories
        self.input_dir = self.run_dir / "input"
        self.logs_dir = self.run_dir / "logs"
        self.output_dir = self.run_dir / "output"
        self.benchmarl_dir = self.output_dir / "benchmarl"

        for d in (self.input_dir, self.logs_dir, self.output_dir, self.benchmarl_dir):
            d.mkdir(parents=True, exist_ok=True)

    def save_config(self, config: Dict[str, Any]):
        """Save the full config snapshot to input/config.yaml."""
        config["_saved_at"] = datetime.now().isoformat()
        with open(self.input_dir / "config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    def save_metrics(self, metrics: Dict[str, float]):
        """Save final aggregate metrics to output/metrics.json."""
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=_json_serializable)

    def load_metrics(self) -> Optional[Dict[str, float]]:
        path = self.output_dir / "metrics.json"
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def append_training_log(self, row: Dict[str, Any]):
        """Append a row to logs/training_log.csv."""
        import csv

        path = self.logs_dir / "training_log.csv"
        file_exists = path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def save_eval_episodes(self, episodes: List[Dict[str, Any]]):
        """Save per-episode evaluation data to output/eval_episodes.json."""
        with open(self.output_dir / "eval_episodes.json", "w") as f:
            json.dump(episodes, f, indent=2, default=_json_serializable)

    def is_complete(self) -> bool:
        return (self.output_dir / "metrics.json").exists()


class ExperimentStorage:
    """Manages all runs for an experiment."""

    def __init__(self, exp_id: str, results_root: Optional[Path] = None):
        if results_root is None:
            results_root = Path(__file__).parent.parent / "results"
        self.results_dir = results_root / exp_id.lower()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.exp_id = exp_id

    def get_run(self, run_id: str) -> RunStorage:
        return RunStorage(self.results_dir, run_id)

    def list_runs(self) -> List[str]:
        """List all completed run IDs."""
        runs = []
        if self.results_dir.exists():
            for d in sorted(self.results_dir.iterdir()):
                if d.is_dir() and (d / "output" / "metrics.json").exists():
                    runs.append(d.name)
        return runs

    def load_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Load metrics from all completed runs."""
        result = {}
        for run_id in self.list_runs():
            metrics = self.get_run(run_id).load_metrics()
            if metrics is not None:
                result[run_id] = metrics
        return result

    def to_dataframe(self):
        """Load all metrics into a pandas DataFrame."""
        import pandas as pd

        all_metrics = self.load_all_metrics()
        if not all_metrics:
            return pd.DataFrame()
        rows = []
        for run_id, metrics in all_metrics.items():
            row = {"run_id": run_id}
            row.update(metrics)
            row.update(_parse_run_id(run_id))
            rows.append(row)
        return pd.DataFrame(rows)


def _parse_run_id(run_id: str) -> Dict[str, Any]:
    """Extract structured params from run_id string.

    Example: 'er1_mappo_n_agents4_n_targets7_agents_per_target2_lidar_range0.35_s0'
    """
    parsed = {}
    # Extract seed
    if "_s" in run_id:
        parts = run_id.rsplit("_s", 1)
        try:
            parsed["seed"] = int(parts[1])
        except ValueError:
            pass
    # Extract algorithm
    for algo in ["mappo", "ippo", "qmix", "maddpg"]:
        if f"_{algo}_" in run_id:
            parsed["algorithm"] = algo
            break
    # Extract exp_id
    for prefix in ["er1", "er2", "er3", "er4", "e1"]:
        if run_id.startswith(prefix):
            parsed["exp_id"] = prefix
            break
    return parsed


def load_cross_experiment(
    exp_ids: List[str],
    results_root: Optional[Path] = None,
):
    """Load metrics across multiple experiments for comparison.

    Returns a pandas DataFrame with all runs from all specified experiments.
    """
    import pandas as pd

    frames = []
    for exp_id in exp_ids:
        store = ExperimentStorage(exp_id, results_root)
        df = store.to_dataframe()
        if not df.empty:
            df["experiment"] = exp_id
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
