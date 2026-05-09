"""F2.6 integration tests: LocalRunner — Protocol satisfaction + end-to-end smoke."""

# The fake logger is intentionally a stub; pylint's noisy here.
# pylint: disable=missing-function-docstring,unused-argument,too-few-public-methods

from datetime import datetime, timezone
from pathlib import Path

import pytest

from multi_scenario.adapters.runners.local import LocalRunner
from multi_scenario.domain.models import (
    ExperimentConfig,
    ExperimentResult,
    LibraryVersions,
    Provenance,
    RunState,
)
from multi_scenario.domain.ports import Runner


class _NoopLogger:
    """Drops all log lines; sufficient for smoke."""

    def info(self, msg: str) -> None:
        ...

    def debug(self, msg: str) -> None:
        ...

    def warning(self, msg: str) -> None:
        ...

    def error(self, msg: str) -> None:
        ...


def _stub_provenance(_cfg: ExperimentConfig) -> Provenance:
    return Provenance(
        config_hash="sha256:test",
        code_hash="sha256:test",
        hashed_source_files=[],
        git_sha="0",
        git_dirty=False,
        created_at=datetime(2026, 5, 6, 14, 23, tzinfo=timezone.utc),
        library_versions=LibraryVersions(
            python="3.11",
            torch="2.4",
            vmas="1.4",
            benchmarl="1.3",
            multi_scenario="0.0.1",
        ),
    )


def test_local_runner_implements_runner_protocol():
    """LocalRunner satisfies the Runner port."""
    runner = LocalRunner(logger=_NoopLogger(), provenance_factory=_stub_provenance)
    assert isinstance(runner, Runner)


def _smoke_config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "experiment": {"id": "local_runner_smoke", "seed": 0},
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
                "runner": {"type": "local", "params": {}},
                "storage": {"type": "fs", "path": str(tmp_path), "params": {}},
            },
        }
    )


@pytest.mark.slow
def test_run_routes_through_factories_and_service(tmp_path: Path):
    """LocalRunner.run wires real adapters via factories and runs end-to-end."""
    runner = LocalRunner(logger=_NoopLogger(), provenance_factory=_stub_provenance)
    cfg = _smoke_config(tmp_path)

    result = runner.run(cfg, run_dir=tmp_path)

    # Returned result has the expected identity.
    assert isinstance(result, ExperimentResult)
    assert result.run_id == "local_runner_smoke_s0"
    assert result.scenario == "discovery"
    assert result.algorithm == "mappo"

    # Storage produced the §3.5.2 layout files on disk.
    assert (tmp_path / "input" / "config.json").is_file()
    assert (tmp_path / "input" / "provenance.json").is_file()
    assert (tmp_path / "output" / "metrics.json").is_file()
    assert (tmp_path / "run_state.json").is_file()

    # Final state on disk is DONE.
    state_text = (tmp_path / "run_state.json").read_text(encoding="utf-8")
    assert RunState.DONE.value in state_text
