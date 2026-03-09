"""Results storage and retrieval for experiment runs.

Directory layout per run:
  results/<exp_id>/YYYYMMDD_HHMM__<run_id>/
    input/
      config.yaml           # Frozen config snapshot
    logs/
      run.log               # Python logger output
    output/
      benchmarl/            # Raw BenchMARL artifacts (CSV, checkpoints)
      metrics.json          # Final aggregate metrics
      eval_episodes.json    # Per-episode eval data (optional)
      policy.pt             # Trained policy state dict (for export/import)
    report.txt              # Human-readable run summary
"""
import json
import os
import re
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


def _make_folder_name(run_id: str) -> str:
    """Create a timestamped folder name: YYYYMMDD_HHMM__<run_id>."""
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    return f"{ts}__{run_id}"


def _extract_run_id(folder_name: str) -> str:
    """Extract the parametric run_id from a timestamped folder name.

    '20260309_1430__er1_mappo_n4_t7_k2_l035_s0' → 'er1_mappo_n4_t7_k2_l035_s0'
    Also handles plain run_id without timestamp prefix.
    """
    m = re.match(r"\d{8}_\d{4}__(.*)", folder_name)
    if m:
        return m.group(1)
    return folder_name


class RunStorage:
    """Manages data for a single experiment run.

    Creates a structured directory with input/, logs/, and output/ subdirs.
    """

    def __init__(self, run_dir: Path, run_id: str):
        self.run_dir = run_dir
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

    def save_policy(self, policy):
        """Save trained policy state dict to output/policy.pt."""
        import torch
        torch.save(policy.state_dict(), self.output_dir / "policy.pt")

    def load_policy_state_dict(self) -> Optional[dict]:
        """Load policy state dict from output/policy.pt.

        Returns None if no saved policy exists.
        To use: rebuild the experiment with same config, then call
        experiment.policy.load_state_dict(state_dict).
        """
        path = self.output_dir / "policy.pt"
        if not path.exists():
            return None
        import torch
        return torch.load(path, map_location="cpu", weights_only=True)

    def load_benchmarl_scalars(self) -> dict:
        """Load all BenchMARL CSV scalars as {name: [(step, value), ...]}."""
        import csv as csv_mod
        scalars = {}
        for sd in self.benchmarl_dir.glob("**/scalars"):
            for csv_file in sorted(sd.glob("*.csv")):
                rows = []
                with open(csv_file) as f:
                    for row in csv_mod.reader(f):
                        if len(row) >= 2:
                            try:
                                rows.append((int(row[0]), float(row[1])))
                            except ValueError:
                                continue
                if rows:
                    scalars[csv_file.stem] = rows
        return scalars

    def is_complete(self) -> bool:
        return (self.output_dir / "metrics.json").exists()

    def has_policy(self) -> bool:
        return (self.output_dir / "policy.pt").exists()


class ExperimentStorage:
    """Manages all runs for an experiment.

    Run folders are timestamped: YYYYMMDD_HHMM__<run_id>/
    The parametric run_id (e.g., er1_mappo_n4_t7_k2_l035_s0) is used
    for matching/skip_complete; the timestamp makes each run unique.
    """

    def __init__(self, exp_id: str, results_root: Optional[Path] = None):
        if results_root is None:
            results_root = Path(__file__).parent.parent / "results"
        self.results_dir = results_root / exp_id.lower()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.exp_id = exp_id

    def get_run(self, run_id: str) -> RunStorage:
        """Get or create a RunStorage for a given run_id.

        If a completed run with this run_id already exists, returns it.
        Otherwise creates a new timestamped folder.
        """
        # Check for existing run with this parametric id
        existing = self._find_run_dir(run_id)
        if existing is not None:
            return RunStorage(existing, run_id)

        # Create new timestamped folder
        folder_name = _make_folder_name(run_id)
        run_dir = self.results_dir / folder_name
        return RunStorage(run_dir, run_id)

    def _find_run_dir(self, run_id: str) -> Optional[Path]:
        """Find an existing run folder matching this parametric run_id."""
        if not self.results_dir.exists():
            return None
        for d in self.results_dir.iterdir():
            if d.is_dir() and _extract_run_id(d.name) == run_id:
                return d
        return None

    def list_runs(self) -> List[str]:
        """List all completed parametric run IDs."""
        runs = []
        if self.results_dir.exists():
            for d in sorted(self.results_dir.iterdir()):
                if d.is_dir() and (d / "output" / "metrics.json").exists():
                    runs.append(_extract_run_id(d.name))
        return runs

    def list_run_dirs(self) -> List[Path]:
        """List all completed run directories."""
        dirs = []
        if self.results_dir.exists():
            for d in sorted(self.results_dir.iterdir()):
                if d.is_dir() and (d / "output" / "metrics.json").exists():
                    dirs.append(d)
        return dirs

    def load_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Load metrics from all completed runs.

        Keys are parametric run_ids (without timestamp prefix).
        """
        result = {}
        for d in self.list_run_dirs():
            run_id = _extract_run_id(d.name)
            rs = RunStorage(d, run_id)
            metrics = rs.load_metrics()
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

    Handles short format: er1_mappo_n4_t7_k2_l035_s0
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

    # Extract short-format params: n4, t7, k2, l035
    m = re.search(r"_n(\d+)", run_id)
    if m:
        parsed["n_agents"] = int(m.group(1))
    m = re.search(r"_t(\d+)", run_id)
    if m:
        parsed["n_targets"] = int(m.group(1))
    m = re.search(r"_k(\d+)", run_id)
    if m:
        parsed["agents_per_target"] = int(m.group(1))
    m = re.search(r"_l(\d+)", run_id)
    if m:
        # Convert back: "035" → 0.35
        raw = m.group(1)
        if len(raw) >= 2:
            parsed["lidar_range"] = float(f"0.{raw.lstrip('0') or '0'}")

    return parsed


def load_cross_experiment(
    exp_ids: List[str],
    results_root: Optional[Path] = None,
):
    """Load metrics across multiple experiments for comparison."""
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
