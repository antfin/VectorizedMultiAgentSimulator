"""Provenance tracking for experiment runs.

Stores config + code hashes to detect stale results.

Config freshness compares the non-sweep parameters (task + train)
stored in each run's input/config.yaml against the current spec.
This means two different YAML files that produce the same run
parameters are treated as equivalent — no false "config changed".
"""
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import yaml


# Source files whose changes invalidate results
_CODE_FILES = ["config.py", "runner.py", "metrics.py"]
_SRC_DIR = Path(__file__).parent


class Freshness(Enum):
    VALID = "valid"
    CONFIG_CHANGED = "config_changed"
    CODE_CHANGED = "code_changed"
    BOTH_CHANGED = "both_changed"
    NO_PROVENANCE = "no_provenance"


@dataclass
class Provenance:
    config_hash: str
    code_hash: str
    git_commit: str
    git_dirty: bool
    created_at: str
    hashed_source_files: List[str]


def compute_config_hash(task_dict: Dict, train_dict: Dict) -> str:
    """SHA-256 of non-sweep task + train parameters.

    Accepts the task and train dicts (as stored in config.yaml),
    normalizes them via sorted YAML dump, and hashes the result.
    This is independent of which YAML file was used.
    """
    combined = {"task": task_dict, "train": train_dict}
    canonical = yaml.dump(combined, default_flow_style=False, sort_keys=True)
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()[:16]


def compute_code_hash() -> str:
    """SHA-256 of concatenated source files that affect results."""
    h = hashlib.sha256()
    for fname in sorted(_CODE_FILES):
        path = _SRC_DIR / fname
        if path.exists():
            h.update(path.read_bytes())
    return "sha256:" + h.hexdigest()[:16]


def _git_info() -> tuple:
    """Return (commit_hash, is_dirty). Defaults to ('unknown', False)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=_SRC_DIR,
        )
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD"],
            capture_output=True, cwd=_SRC_DIR,
        )
        return (
            commit.stdout.strip() if commit.returncode == 0 else "unknown",
            dirty.returncode != 0,
        )
    except FileNotFoundError:
        return ("unknown", False)


def save_provenance(
    run_dir: Path,
    task_dict: Dict,
    train_dict: Dict,
):
    """Save provenance.json to the run's input/ directory.

    Args:
        run_dir: path to the run directory
        task_dict: task config dict (from spec.task.to_dict())
        train_dict: train config dict (from spec.train.__dict__)
    """
    git_commit, git_dirty = _git_info()
    prov = Provenance(
        config_hash=compute_config_hash(task_dict, train_dict),
        code_hash=compute_code_hash(),
        git_commit=git_commit,
        git_dirty=git_dirty,
        created_at=datetime.now().isoformat(),
        hashed_source_files=sorted(_CODE_FILES),
    )
    prov_path = run_dir / "input" / "provenance.json"
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prov_path, "w") as f:
        json.dump(asdict(prov), f, indent=2)


def load_provenance(run_dir: Path) -> Optional[Provenance]:
    """Load provenance.json from a run directory."""
    prov_path = run_dir / "input" / "provenance.json"
    if not prov_path.exists():
        return None
    with open(prov_path) as f:
        data = json.load(f)
    # Handle old format that had source_config_path field
    data.pop("source_config_path", None)
    return Provenance(**data)


def _load_run_config(run_dir: Path) -> Optional[Dict]:
    """Load the saved config.yaml from a run's input/ directory."""
    cfg_path = run_dir / "input" / "config.yaml"
    if not cfg_path.exists():
        return None
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def check_freshness(run_dir: Path, spec) -> Freshness:
    """Check if a run's results are still valid against current spec.

    Compares the non-sweep parameters (task + train) from the
    current spec against what was stored when the run was created.

    Args:
        run_dir: path to the run directory
        spec: ExperimentSpec with current task/train config
    """
    prov = load_provenance(run_dir)
    if prov is None:
        return Freshness.NO_PROVENANCE

    # Compare config: hash current spec's task+train params
    current_config_hash = compute_config_hash(
        spec.task.to_dict(),
        {k: v for k, v in spec.train.__dict__.items()},
    )
    config_ok = prov.config_hash == current_config_hash
    code_ok = prov.code_hash == compute_code_hash()

    if config_ok and code_ok:
        return Freshness.VALID
    elif not config_ok and not code_ok:
        return Freshness.BOTH_CHANGED
    elif not config_ok:
        return Freshness.CONFIG_CHANGED
    else:
        return Freshness.CODE_CHANGED
