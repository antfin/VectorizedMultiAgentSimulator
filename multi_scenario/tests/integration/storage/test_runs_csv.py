"""F5.2 integration tests: RunsCsvWriter — cross-run leaderboard CSV."""

# The fixture helpers below take a small handful of orthogonal kwargs (run_id,
# scenario, algo, state, m1) — bundling them into a dataclass would add noise.
# pylint: disable=too-many-arguments,too-many-positional-arguments

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from multi_scenario.adapters.storage.local import LocalStorageAdapter
from multi_scenario.adapters.storage.runs_csv import RunsCsvWriter
from multi_scenario.domain.models import (
    ExperimentResult,
    RunState,
    RunStateRecord,
    RunStateTransition,
)


def _ts(minute: int = 0) -> datetime:
    return datetime(2026, 5, 7, 14, minute, 0, tzinfo=timezone.utc)


def _result(
    run_id: str, scenario: str, algo: str, seed: int = 0, m1: float | None = 0.5
) -> ExperimentResult:
    return ExperimentResult(
        run_id=run_id,
        exp_id=run_id.rsplit("_s", 1)[0],
        scenario=scenario,
        algorithm=algo,
        seed=seed,
        run_timestamp="20260507_1400",
        metrics={
            "M1_success_rate": m1,
            "M2_avg_return": 12.34,
            "M3_steps": 25.0,
            "M4_collisions": 0.0,
            "M5_tokens": None,
            "M6_coverage_progress": None,
            "M7_sample_efficiency": None,
            "M8_agent_utilization": None,
            "M9_spatial_spread": None,
        },
        config_snapshot={"n_agents": 2, "max_steps": 10},
        n_envs=1,
        n_eval_episodes=1,
    )


def _done_state() -> RunStateRecord:
    return RunStateRecord(
        state=RunState.DONE,
        transitions=[
            RunStateTransition(state=RunState.INITIALIZING, ts=_ts(0)),
            RunStateTransition(state=RunState.RUNNING, ts=_ts(1)),
            RunStateTransition(state=RunState.DONE, ts=_ts(5)),
        ],
    )


def _running_state() -> RunStateRecord:
    return RunStateRecord(
        state=RunState.RUNNING,
        transitions=[
            RunStateTransition(state=RunState.INITIALIZING, ts=_ts(0)),
            RunStateTransition(state=RunState.RUNNING, ts=_ts(1)),
        ],
    )


def _seed_run(
    parent: Path,
    run_id: str,
    scenario: str,
    algo: str,
    state: RunStateRecord,
    m1: float | None = 0.5,
) -> Path:
    """Lay down a minimal run folder under ``parent`` with the §3.5.2 layout."""
    run_dir = parent / f"{run_id}__20260507_1400"
    run_dir.mkdir(parents=True)
    storage = LocalStorageAdapter()
    storage.save_result(run_dir, _result(run_id, scenario, algo, m1=m1))
    storage.save_run_state(run_dir, state)
    return run_dir


def test_consolidate_one_row_per_done_run(tmp_path: Path) -> None:
    """Three completed runs → CSV has 3 rows (record_type=final)."""
    _seed_run(tmp_path, "smoke_mappo_s0", "discovery", "mappo", _done_state())
    _seed_run(tmp_path, "smoke_ippo_s0", "discovery", "ippo", _done_state())
    _seed_run(tmp_path, "smoke_mappo_s1", "discovery", "mappo", _done_state(), m1=0.75)

    out = RunsCsvWriter().consolidate(tmp_path)

    assert out == tmp_path / "runs.csv"
    df = pd.read_csv(out)
    assert len(df) == 3
    assert (df["record_type"] == "final").all()
    assert set(df["run_id"]) == {"smoke_mappo_s0", "smoke_ippo_s0", "smoke_mappo_s1"}


def test_consolidate_skips_non_done_runs(tmp_path: Path) -> None:
    """Runs in RUNNING / CRASHED / etc. don't appear in the leaderboard."""
    _seed_run(tmp_path, "done_s0", "discovery", "mappo", _done_state())
    _seed_run(tmp_path, "still_running_s0", "discovery", "mappo", _running_state())

    out = RunsCsvWriter().consolidate(tmp_path)
    df = pd.read_csv(out)
    assert len(df) == 1
    assert df.iloc[0]["run_id"] == "done_s0"


def test_consolidate_skips_dirs_without_metrics_json(tmp_path: Path) -> None:
    """Folders without output/metrics.json are silently skipped."""
    (
        tmp_path / "configs"
    ).mkdir()  # sibling configs/ folder shouldn't break consolidation
    _seed_run(tmp_path, "ok_s0", "discovery", "mappo", _done_state())

    df = pd.read_csv(RunsCsvWriter().consolidate(tmp_path))
    assert len(df) == 1


def test_consolidate_renders_none_metrics_as_na_string(tmp_path: Path) -> None:
    """JSON null metric values render as 'N/A' in the CSV (not empty / 'None')."""
    _seed_run(tmp_path, "stub_s0", "flocking", "mappo", _done_state(), m1=None)

    out = RunsCsvWriter().consolidate(tmp_path)
    raw = out.read_text(encoding="utf-8")
    # M1 column for the flocking row is None → "N/A" via pandas na_rep.
    assert "N/A" in raw
    df = pd.read_csv(out, keep_default_na=False)
    assert df.iloc[0]["M1_success_rate"] == "N/A"


def test_consolidate_includes_metric_and_config_columns(tmp_path: Path) -> None:
    """Schema includes M1-M9 columns and flattened config_snapshot keys."""
    _seed_run(tmp_path, "smoke_s0", "discovery", "mappo", _done_state())

    df = pd.read_csv(RunsCsvWriter().consolidate(tmp_path))
    cols = set(df.columns)
    assert {"record_type", "run_id", "exp_id", "scenario", "algorithm", "seed"} <= cols
    assert {"M1_success_rate", "M2_avg_return", "M9_spatial_spread"} <= cols
    # Flattened config_snapshot keys.
    assert "n_agents" in cols
    assert "max_steps" in cols
    # Duration computed from run_state transitions.
    assert "duration_seconds" in cols


def test_consolidate_atomic_overwrite_creates_previous(tmp_path: Path) -> None:
    """Re-consolidating moves the old runs.csv to runs.previous.csv (one-step rollback)."""
    _seed_run(tmp_path, "first_s0", "discovery", "mappo", _done_state())
    RunsCsvWriter().consolidate(tmp_path)
    first_content = (tmp_path / "runs.csv").read_text(encoding="utf-8")

    # Add another run and re-consolidate.
    _seed_run(tmp_path, "second_s0", "discovery", "ippo", _done_state())
    RunsCsvWriter().consolidate(tmp_path)

    assert (tmp_path / "runs.csv").is_file()
    assert (tmp_path / "runs.previous.csv").is_file()
    # The .previous.csv is the *prior* runs.csv content.
    assert (tmp_path / "runs.previous.csv").read_text(encoding="utf-8") == first_content
    # The new runs.csv has both runs.
    new_df = pd.read_csv(tmp_path / "runs.csv")
    assert len(new_df) == 2


def test_consolidate_returns_path_to_runs_csv(tmp_path: Path) -> None:
    """The writer returns the path to the produced runs.csv."""
    _seed_run(tmp_path, "smoke_s0", "discovery", "mappo", _done_state())
    out = RunsCsvWriter().consolidate(tmp_path)
    assert out == tmp_path / "runs.csv"
    assert out.is_file()


def test_consolidate_empty_dir_writes_empty_csv(tmp_path: Path) -> None:
    """No runs at all → CSV with header only."""
    out = RunsCsvWriter().consolidate(tmp_path)
    assert out.is_file()
    df = pd.read_csv(out)
    assert len(df) == 0
