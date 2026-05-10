"""F8.2.G — BenchMARL `keep_checkpoints_num` propagation.

ER1 dry-run lost the iter-125 peak because BenchMARL's default retention
of 3 silently overwrote earlier snapshots. Pin the contract that non-smoke
runs propagate ``cfg.training.keep_checkpoints_num`` (default 1000 = "all")
into the BenchMARL ExperimentConfig.
"""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

from pathlib import Path

import pytest

from multi_scenario.adapters.algorithms.mappo import MappoAdapter
from multi_scenario.domain.models import ExperimentConfig


def _cfg(tmp_path: Path, *, exp_id: str, keep: int | None = None) -> ExperimentConfig:
    training = {
        "max_iters": 2,
        "num_envs": 1,
        "device": "cpu",
        "frames_per_batch": 100,
        "minibatch_size": 50,
        "n_minibatch_iters": 1,
    }
    if keep is not None:
        training["keep_checkpoints_num"] = keep
    return ExperimentConfig.model_validate(
        {
            "experiment": {"id": exp_id, "seed": 0},
            "scenario": {
                "type": "discovery",
                "params": {"n_agents": 2, "n_targets": 2, "max_steps": 5},
            },
            "algorithm": {"type": "mappo", "params": {}},
            "training": training,
            "evaluation": {"interval_iters": 1, "episodes": 1},
            "runtime": {
                "runner": {"type": "local", "params": {}},
                "storage": {"type": "fs", "path": str(tmp_path), "params": {}},
            },
        }
    )


def test_non_smoke_default_keeps_all_checkpoints(tmp_path: Path):
    """Default ``keep_checkpoints_num=1000`` propagates into BenchMARL config.

    Pre-fix: BenchMARL kept the rolling default of 3, so a 167-iter ER1 run
    only retained the last 3 snapshots (≈ iters 150/160/167) — the iter-125
    peak that F8.5.D's best-checkpoint policy needs was overwritten. With
    the override, all 16+ snapshots survive.
    """
    cfg = _cfg(tmp_path, exp_id="er1_cr035")
    bm = MappoAdapter()._experiment_config(cfg, save_folder=str(tmp_path))
    assert bm.keep_checkpoints_num == 1000, (
        f"non-smoke run should keep all checkpoints (1000); got "
        f"{bm.keep_checkpoints_num} — F8.2.G regression"
    )
    assert bm.checkpoint_at_end is True
    assert bm.checkpoint_interval > 0


def test_non_smoke_custom_keep_value_round_trips(tmp_path: Path):
    """User can override ``keep_checkpoints_num`` per-experiment if disk-bound."""
    cfg = _cfg(tmp_path, exp_id="er1_cr035", keep=5)
    bm = MappoAdapter()._experiment_config(cfg, save_folder=str(tmp_path))
    assert bm.keep_checkpoints_num == 5


def test_smoke_runs_skip_checkpoints_entirely(tmp_path: Path):
    """Smoke runs disable checkpoints altogether; retention knob is unused.

    Smoke runs are 1-2 iter sanity checks — there's nothing to retain.
    The adapter sets ``checkpoint_interval=0`` so no snapshots are written;
    we don't even bother setting ``keep_checkpoints_num``.
    """
    cfg = _cfg(tmp_path, exp_id="mappo_smoke")
    bm = MappoAdapter()._experiment_config(cfg, save_folder=str(tmp_path))
    assert bm.checkpoint_interval == 0
    assert bm.checkpoint_at_end is False


@pytest.mark.parametrize("invalid", [0, -1])
def test_keep_checkpoints_num_must_be_positive(tmp_path: Path, invalid: int):
    """Pydantic enforces ``gt=0`` on the field."""
    with pytest.raises(ValueError, match="greater than 0"):
        _cfg(tmp_path, exp_id="x", keep=invalid)
