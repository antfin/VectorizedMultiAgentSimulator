"""F5.2 integration tests: ``multi-scenario consolidate <exp_type_dir>``."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from multi_scenario.adapters.storage.local import LocalStorageAdapter
from multi_scenario.cli import app
from multi_scenario.domain.models import (
    ExperimentResult,
    RunState,
    RunStateRecord,
    RunStateTransition,
)


def _seed_done_run(parent: Path, run_id: str) -> None:
    run_dir = parent / f"{run_id}__20260507_1400"
    run_dir.mkdir(parents=True)
    storage = LocalStorageAdapter()
    storage.save_result(
        run_dir,
        ExperimentResult(
            run_id=run_id,
            exp_id=run_id.rsplit("_s", 1)[0],
            scenario="discovery",
            algorithm="mappo",
            seed=0,
            run_timestamp="20260507_1400",
            metrics={
                "M1_success_rate": 0.5,
                "M2_avg_return": 1.0,
                "M3_steps": 10.0,
                "M4_collisions": 0.0,
                "M5_tokens": None,
                "M6_coverage_progress": None,
                "M7_sample_efficiency": None,
                "M8_agent_utilization": None,
                "M9_spatial_spread": None,
            },
            config_snapshot={"n_agents": 2},
            n_envs=1,
            n_eval_episodes=1,
        ),
    )
    ts0 = datetime(2026, 5, 7, 14, 0, 0, tzinfo=timezone.utc)
    ts1 = datetime(2026, 5, 7, 14, 5, 0, tzinfo=timezone.utc)
    storage.save_run_state(
        run_dir,
        RunStateRecord(
            state=RunState.DONE,
            transitions=[
                RunStateTransition(state=RunState.INITIALIZING, ts=ts0),
                RunStateTransition(state=RunState.DONE, ts=ts1),
            ],
        ),
    )


def test_consolidate_command_writes_runs_csv(tmp_path: Path) -> None:
    """End-to-end: CLI consolidates a populated exp_type folder."""
    _seed_done_run(tmp_path, "smoke_s0")
    _seed_done_run(tmp_path, "smoke_s1")

    result = CliRunner().invoke(app, ["consolidate", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "runs.csv" in result.output

    df = pd.read_csv(tmp_path / "runs.csv")
    assert len(df) == 2


def test_consolidate_command_with_missing_dir_returns_nonzero() -> None:
    """Pointing at a non-existent dir exits non-zero."""
    result = CliRunner().invoke(app, ["consolidate", "/tmp/does_not_exist_xyz"])
    assert result.exit_code != 0
