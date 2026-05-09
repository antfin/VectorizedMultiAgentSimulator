"""F5.7 integration tests: ``multi-scenario resume <run_dir>`` (local-only)."""

# Helpers seed minimal §3.5.2 layouts with whatever state the test demands.
# pylint: disable=too-many-arguments,too-many-positional-arguments

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from multi_scenario.adapters.logging.file_logger import FileLogger
from multi_scenario.adapters.runners.local import LocalRunner
from multi_scenario.adapters.storage.local import LocalStorageAdapter
from multi_scenario.cli import app
from multi_scenario.domain.models import (
    ExperimentConfig,
    RunId,
    RunState,
    RunStateRecord,
    RunStateTransition,
)
from typer.testing import CliRunner


def _smoke_cfg_dict(storage_path: Path, runner_type: str = "local") -> dict:
    return {
        "experiment": {"id": "resume_smoke", "seed": 0},
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
        },
        "evaluation": {"interval_iters": 1, "episodes": 1},
        "runtime": {
            "runner": {"type": runner_type, "params": {}},
            "storage": {"type": "fs", "path": str(storage_path), "params": {}},
        },
    }


def _ts(minute: int) -> datetime:
    return datetime(2026, 5, 7, 12, minute, tzinfo=timezone.utc)


def _seed_run(
    run_dir: Path,
    state: RunState,
    runner_type: str = "local",
    seed_checkpoint: bool = True,
) -> None:
    """Lay down a minimal run folder at ``state``, optionally with a fake checkpoint."""
    run_dir.mkdir(parents=True)
    storage = LocalStorageAdapter()
    cfg = ExperimentConfig.model_validate(
        _smoke_cfg_dict(run_dir.parent, runner_type=runner_type)
    )
    storage.save_config(run_dir, cfg)
    transitions = [RunStateTransition(state=RunState.INITIALIZING, ts=_ts(0))]
    if state in (RunState.RUNNING, RunState.DONE, RunState.CRASHED):
        transitions.append(RunStateTransition(state=RunState.RUNNING, ts=_ts(1)))
    if state == RunState.DONE:
        transitions.append(RunStateTransition(state=RunState.DONE, ts=_ts(5)))
    if state == RunState.CRASHED:
        transitions.append(RunStateTransition(state=RunState.CRASHED, ts=_ts(5)))
    storage.save_run_state(
        run_dir, RunStateRecord(state=state, transitions=transitions)
    )
    if seed_checkpoint:
        ckpt_dir = (
            run_dir / "output" / "benchmarl" / "fake_run" / "fake_run" / "checkpoints"
        )
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "checkpoint_0.pt").write_text("fake-policy", encoding="utf-8")


def test_resume_refuses_when_state_is_done(tmp_path: Path) -> None:
    """A DONE run has nothing to resume — exit 2 with helpful message."""
    run_dir = tmp_path / "done"
    _seed_run(run_dir, RunState.DONE)

    result = CliRunner().invoke(app, ["resume", str(run_dir)])
    assert result.exit_code == 2
    assert "DONE" in result.output


def test_resume_refuses_when_runner_unsupported(tmp_path: Path) -> None:
    """OVH (or any non-local) runner type → exit 2 with capability message."""
    run_dir = tmp_path / "ovh_run"
    _seed_run(run_dir, RunState.RUNNING, runner_type="ovh")

    result = CliRunner().invoke(app, ["resume", str(run_dir)])
    assert result.exit_code == 2
    assert "ovh" in result.output.lower()
    assert "local" in result.output.lower()


def test_resume_refuses_when_no_checkpoint(tmp_path: Path) -> None:
    """No BenchMARL checkpoint on disk → exit 2 with helpful message."""
    run_dir = tmp_path / "no_ckpt"
    _seed_run(run_dir, RunState.RUNNING, seed_checkpoint=False)

    result = CliRunner().invoke(app, ["resume", str(run_dir)])
    assert result.exit_code == 2
    assert "checkpoint" in result.output.lower()


def test_local_runner_advertises_resume_support() -> None:
    """LocalRunner exposes the F5.7 capability flag as True."""
    assert LocalRunner.supports_resume is True


@pytest.mark.slow
def test_resume_end_to_end(tmp_path: Path) -> None:
    """Full happy path: train completes, simulated crash, resume CLI succeeds.

    Confirms:
    - BenchMARL writes checkpoints when ``checkpoint_interval_iters`` is set.
    - ``Experiment.reload_from_file`` reconstructs from our run-folder layout.
    - State machine transitions ``RUNNING → CRASHED → RESUMED → RUNNING → DONE``.
    """
    cfg_dict = {
        "experiment": {"id": "resume_e2e", "seed": 0},
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
            "storage": {"type": "fs", "path": str(tmp_path), "params": {}},
        },
    }
    cfg = ExperimentConfig.model_validate(cfg_dict)
    rid = RunId(exp_id=cfg.experiment.id, seed=cfg.experiment.seed)
    rd = tmp_path / f"{rid}__test"
    rd.mkdir(parents=True)

    # Phase 1: full run.
    runner = LocalRunner(logger=FileLogger(rd / "logs" / "run.log"))
    runner.run(cfg, run_dir=rd)
    assert LocalStorageAdapter().load_run_state(rd).state == RunState.DONE

    # Phase 2: simulate a mid-run crash by rewriting state to RUNNING.
    state_path = rd / "run_state.json"
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    raw["state"] = "RUNNING"
    raw["transitions"] = raw["transitions"][:-1]  # drop the DONE transition
    state_path.write_text(json.dumps(raw), encoding="utf-8")

    # Phase 3: resume.
    result = CliRunner().invoke(app, ["resume", str(rd)])
    assert result.exit_code == 0, result.output

    final = LocalStorageAdapter().load_run_state(rd)
    assert final.state == RunState.DONE
    transition_states = [t.state.value for t in final.transitions]
    # The state-machine log records the synthetic crash + resume transitions.
    assert "CRASHED" in transition_states
    assert "RESUMED" in transition_states
    assert transition_states[-1] == "DONE"
