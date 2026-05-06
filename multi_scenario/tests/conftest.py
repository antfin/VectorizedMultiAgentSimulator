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
def fake_config_builder():
    """Factory: builds minimal experiment config dicts for tests.

    Sections mirror the F1.1 ExperimentConfig schema; update when that lands.
    Pass keyword overrides to replace whole sections.
    """

    def _build(**overrides) -> dict:
        cfg = {
            "experiment": {"id": "test", "seed": 0},
            "scenario": {"name": "discovery"},
            "algorithm": {"name": "mappo"},
            "training": {"max_iters": 1},
            "runner": {"name": "local"},
            "metrics": {"set": "common"},
            "storage": {"name": "local"},
        }
        cfg.update(overrides)
        return cfg

    return _build
