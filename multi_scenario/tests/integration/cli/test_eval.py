"""F5.8 integration tests: ``multi-scenario eval <run_dir>``."""

import json
from pathlib import Path

import pytest

from multi_scenario.adapters.logging.file_logger import FileLogger
from multi_scenario.adapters.runners.local import LocalRunner
from multi_scenario.adapters.storage.local import LocalStorageAdapter
from multi_scenario.cli import app
from multi_scenario.domain.models import EvalRunRecord, ExperimentConfig, RunId
from typer.testing import CliRunner


def _smoke_cfg_dict(storage_path: Path) -> dict:
    return {
        "experiment": {"id": "eval_only_test", "seed": 0},
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
            "max_iters": 2,
            "num_envs": 1,
            "device": "cpu",
            "frames_per_batch": 50,
            "minibatch_size": 25,
            "n_minibatch_iters": 1,
            "checkpoint_interval_iters": 1,
        },
        "evaluation": {"interval_iters": 1, "episodes": 1},
        "runtime": {
            "runner": {"type": "local", "params": {"record_video": False}},
            "storage": {"type": "fs", "path": str(storage_path), "params": {}},
        },
    }


def test_eval_refuses_when_config_missing(tmp_path: Path) -> None:
    """No input/config.json → exit 2 with helpful message."""
    run_dir = tmp_path / "no_config"
    run_dir.mkdir()
    result = CliRunner().invoke(app, ["eval", str(run_dir)])
    assert result.exit_code == 2
    assert "config.json" in result.output


def test_eval_refuses_when_no_checkpoint(tmp_path: Path) -> None:
    """Cfg present but no BenchMARL checkpoint → exit 2 with helpful message."""
    run_dir = tmp_path / "no_ckpt"
    run_dir.mkdir()
    storage = LocalStorageAdapter()
    storage.save_config(
        run_dir, ExperimentConfig.model_validate(_smoke_cfg_dict(tmp_path))
    )
    result = CliRunner().invoke(app, ["eval", str(run_dir)])
    assert result.exit_code == 2
    assert "checkpoint" in result.output.lower()


@pytest.mark.slow
def test_eval_end_to_end(tmp_path: Path) -> None:
    """Train → eval CLI with --episodes override → record produced + parses cleanly."""
    cfg = ExperimentConfig.model_validate(_smoke_cfg_dict(tmp_path))
    rid = RunId(exp_id=cfg.experiment.id, seed=cfg.experiment.seed)
    rd = tmp_path / f"{rid}__test"
    rd.mkdir(parents=True)

    # Phase 1: train to produce a checkpoint.
    LocalRunner(logger=FileLogger(rd / "logs" / "run.log")).run(cfg, run_dir=rd)

    # Phase 2: eval-only with override.
    result = CliRunner().invoke(
        app,
        ["eval", str(rd), "--episodes", "2", "--name", "post_hoc"],
    )
    assert result.exit_code == 0, result.output

    out = rd / "output" / "eval_runs" / "post_hoc.json"
    assert out.is_file()
    record = EvalRunRecord.model_validate_json(out.read_text(encoding="utf-8"))
    assert record.eval_id == "post_hoc"
    assert record.run_id == "eval_only_test_s0"
    assert record.n_eval_episodes == 2
    # M1/M2/M3 should be real floats (or None for not-applicable scenarios).
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert "M1_success_rate" in on_disk["metrics"]
    assert "M2_avg_return" in on_disk["metrics"]
    # Policy checkpoint path resolves under run_dir.
    assert (rd / record.policy_checkpoint).is_file()


@pytest.mark.slow
def test_eval_default_name_is_timestamped(tmp_path: Path) -> None:
    """Without --name, the eval_id is ``eval_<timestamp>`` and the file lands."""
    cfg = ExperimentConfig.model_validate(_smoke_cfg_dict(tmp_path))
    rid = RunId(exp_id=cfg.experiment.id, seed=cfg.experiment.seed)
    rd = tmp_path / f"{rid}__test"
    rd.mkdir(parents=True)
    LocalRunner(logger=FileLogger(rd / "logs" / "run.log")).run(cfg, run_dir=rd)

    result = CliRunner().invoke(app, ["eval", str(rd)])
    assert result.exit_code == 0, result.output
    eval_runs = list((rd / "output" / "eval_runs").glob("eval_*.json"))
    assert len(eval_runs) == 1
    assert eval_runs[0].name.startswith("eval_")


@pytest.mark.slow
def test_eval_multiple_runs_coexist(tmp_path: Path) -> None:
    """Multiple ``eval`` invocations produce separate files."""
    cfg = ExperimentConfig.model_validate(_smoke_cfg_dict(tmp_path))
    rid = RunId(exp_id=cfg.experiment.id, seed=cfg.experiment.seed)
    rd = tmp_path / f"{rid}__test"
    rd.mkdir(parents=True)
    LocalRunner(logger=FileLogger(rd / "logs" / "run.log")).run(cfg, run_dir=rd)

    CliRunner().invoke(app, ["eval", str(rd), "--name", "first"])
    CliRunner().invoke(app, ["eval", str(rd), "--name", "second"])

    eval_runs_dir = rd / "output" / "eval_runs"
    assert (eval_runs_dir / "first.json").is_file()
    assert (eval_runs_dir / "second.json").is_file()
