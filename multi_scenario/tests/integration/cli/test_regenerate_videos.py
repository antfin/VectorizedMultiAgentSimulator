"""F6.6 tests: ``multi-scenario regenerate-videos <run_dir>``."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from multi_scenario.adapters.logging.file_logger import FileLogger
from multi_scenario.adapters.runners.local import LocalRunner
from multi_scenario.adapters.storage.local import LocalStorageAdapter
from multi_scenario.cli import app
from multi_scenario.domain.models import ExperimentConfig, RunId, RunReport


def _non_smoke_cfg_dict(storage_path: Path) -> dict:
    """A non-smoke config that triggers checkpoint writing (per F5.7's heuristic)."""
    return {
        # Note: id does NOT end in "_smoke" so checkpoints + videos are enabled.
        "experiment": {"id": "regen_videos_test", "seed": 0},
        "scenario": {
            "type": "discovery",
            "params": {
                "n_agents": 2,
                "n_targets": 2,
                "agents_per_target": 2,
                "targets_respawn": False,
                "shared_reward": True,
                "max_steps": 10,
            },
        },
        "algorithm": {"type": "mappo", "params": {}},
        "training": {
            "max_iters": 1,
            "num_envs": 1,
            "device": "cpu",
            "frames_per_batch": 50,
            "minibatch_size": 25,
            "n_minibatch_iters": 1,
            "checkpoint_interval_iters": 1,
        },
        "evaluation": {"interval_iters": 1, "episodes": 1},
        "runtime": {
            # Disable record_video: the inline F2.11 path would record during
            # training; we want the regenerate-videos CLI to be the one
            # producing the MP4s, so the test starts with no videos on disk.
            "runner": {"type": "local", "params": {"record_video": False}},
            "storage": {"type": "fs", "path": str(storage_path), "params": {}},
        },
    }


def test_regenerate_videos_refuses_when_config_missing(tmp_path: Path) -> None:
    """No input/config.json → exit 2."""
    run_dir = tmp_path / "no_config"
    run_dir.mkdir()
    result = CliRunner().invoke(app, ["regenerate-videos", str(run_dir)])
    assert result.exit_code == 2
    assert "config.json" in result.output


def test_regenerate_videos_refuses_when_no_checkpoint(tmp_path: Path) -> None:
    """Cfg present but no BenchMARL checkpoint → exit 2."""
    run_dir = tmp_path / "no_ckpt"
    run_dir.mkdir()
    storage = LocalStorageAdapter()
    storage.save_config(run_dir, ExperimentConfig.model_validate(_non_smoke_cfg_dict(tmp_path)))
    result = CliRunner().invoke(app, ["regenerate-videos", str(run_dir)])
    assert result.exit_code == 2
    assert "checkpoint" in result.output.lower()


@pytest.mark.slow
def test_regenerate_videos_end_to_end(tmp_path: Path) -> None:
    """Train (no video) → delete videos dir → regenerate → both MP4s + report links resolve."""
    cfg = ExperimentConfig.model_validate(_non_smoke_cfg_dict(tmp_path))
    rid = RunId(exp_id=cfg.experiment.id, seed=cfg.experiment.seed)
    rd = tmp_path / f"{rid}__test"
    rd.mkdir(parents=True)

    # Phase 1: train (record_video=false → no inline videos produced).
    LocalRunner(logger=FileLogger(rd / "logs" / "run.log")).run(cfg, run_dir=rd)
    assert not (rd / "output" / "videos").exists()

    # Phase 2: regenerate.
    result = CliRunner().invoke(app, ["regenerate-videos", str(rd)])
    assert result.exit_code == 0, result.output

    # Both MP4s exist + are non-empty.
    before = rd / "output" / "videos" / "before_training.mp4"
    after = rd / "output" / "videos" / "after_training.mp4"
    assert before.is_file() and before.stat().st_size > 0
    assert after.is_file() and after.stat().st_size > 0

    # Report.json refreshed: video links populated and resolve.
    report = RunReport.model_validate_json(
        (rd / "output" / "report.json").read_text(encoding="utf-8")
    )
    assert report.links.videos.before_training is not None
    assert report.links.videos.after_training is not None
    assert (rd / report.links.videos.before_training).is_file()
    assert (rd / report.links.videos.after_training).is_file()
