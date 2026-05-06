"""Shared pytest fixtures for multi_scenario tests."""

from pathlib import Path

import pytest


@pytest.fixture
def tmp_results_dir(tmp_path: Path) -> Path:
    """Per-test isolated results directory matching the §3.5.2 run layout root."""
    d = tmp_path / "results"
    d.mkdir()
    return d


@pytest.fixture
def repo_root() -> Path:
    """Path to the multi_scenario package root (contains docs/, src/, tests/)."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def fake_config_builder():
    """Factory: builds minimal valid ExperimentConfig dicts for tests.

    Sections mirror the v5 schema (§3.5 in implementation_plan.md). Pass
    keyword overrides to replace whole sections.
    """

    def _build(**overrides) -> dict:
        """Return a minimal valid config dict; ``overrides`` replace whole sections."""
        cfg = {
            "experiment": {"id": "test", "seed": 0},
            "scenario": {"type": "discovery", "params": {}},
            "algorithm": {"type": "mappo", "params": {}},
            "training": {"max_iters": 1, "num_envs": 1, "device": "cpu"},
            "evaluation": {"interval_iters": 1, "episodes": 1},
        }
        cfg.update(overrides)
        return cfg

    return _build
