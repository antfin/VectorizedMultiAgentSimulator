"""Shared fixtures for the F7.7.C2 pytest-bdd Submit-page features."""

# pylint: disable=missing-function-docstring,redefined-outer-name,import-outside-toplevel

from pathlib import Path
from typing import Any

import pytest
import yaml


_SUBMIT_PAGE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "multi_scenario"
    / "frontend"
    / "pages"
    / "submit.py"
)


def _minimal_cfg() -> dict[str, Any]:
    """Smallest valid ExperimentConfig — same shape used by the AppTest tests."""
    return {
        "experiment": {"id": "demo", "seed": 0},
        "scenario": {
            "type": "discovery",
            "params": {"n_agents": 2, "n_targets": 2, "max_steps": 5},
        },
        "algorithm": {"type": "mappo", "params": {}},
        "training": {
            "max_iters": 1,
            "num_envs": 1,
            "device": "cpu",
            "frames_per_batch": 50,
            "minibatch_size": 25,
            "n_minibatch_iters": 1,
        },
        "evaluation": {"interval_iters": 1, "episodes": 1},
        "runtime": {
            "runner": {"type": "local", "params": {"record_video": False}},
            "storage": {"type": "fs", "path": "PLACEHOLDER", "params": {}},
        },
    }


@pytest.fixture
def submit_page_path() -> Path:
    """Absolute path of the Submit page (input to ``AppTest.from_file``)."""
    return _SUBMIT_PAGE_PATH


@pytest.fixture
def experiments_root(tmp_path: Path) -> Path:
    """Per-scenario tmp experiments root with one valid discovery YAML."""
    cfg_dir = tmp_path / "discovery" / "baseline" / "configs"
    cfg_dir.mkdir(parents=True)
    storage_dir = tmp_path / "results"
    storage_dir.mkdir()
    cfg = _minimal_cfg()
    cfg["runtime"]["storage"]["path"] = str(storage_dir)
    (cfg_dir / "seed0.yaml").write_text(
        yaml.dump(cfg, sort_keys=False), encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def context() -> dict[str, Any]:
    """Per-scenario shared bag — step funcs read/write the AppTest here."""
    return {}
