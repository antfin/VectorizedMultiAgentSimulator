"""F7.1 tests: runs_loader walks an experiments tree and pivots metrics into columns."""

# Pytest fixtures intentionally compose synthetic run dirs; test names doc them.
# pylint: disable=missing-function-docstring

from datetime import datetime, timezone
from pathlib import Path

from multi_scenario.adapters.storage.local import LocalStorageAdapter
from multi_scenario.domain.models import (
    ExperimentResult,
    MetricRecord,
    RunState,
    RunStateRecord,
)
from multi_scenario.frontend.runs_loader import load_runs


def _seed_run(run_dir: Path, run_id: str, m1: float, state: RunState | None = None) -> None:
    """Write a minimal output/metrics.json (+ optional run_state.json) under run_dir."""
    storage = LocalStorageAdapter()
    result = ExperimentResult(
        run_id=run_id,
        exp_id=run_id.rsplit("_s", 1)[0],
        scenario="discovery",
        algorithm="mappo",
        seed=int(run_id.rsplit("_s", 1)[1]),
        run_timestamp="20260507_1500",
        metrics=[
            MetricRecord(name="M1_success_rate", value=m1),
            MetricRecord(name="M2_avg_return", value=1.5),
            MetricRecord(name="M3_steps", value=20.0),
            MetricRecord(name="M4_collisions", value=0.5),
            MetricRecord(name="M5_tokens", value=None),
            MetricRecord(name="M6_coverage_progress", value=0.8),
            MetricRecord(name="M7_sample_efficiency", value=None),
            MetricRecord(name="M8_agent_utilization", value=None),
            MetricRecord(name="M9_spatial_spread", value=None),
        ],
        config_snapshot={"n_agents": 2},
        n_envs=1,
        n_eval_episodes=10,
    )
    storage.save_result(run_dir, result)
    if state is not None:
        ts = datetime(2026, 5, 7, 15, 0, tzinfo=timezone.utc)
        # Walk the legal transition graph: INITIALIZING → RUNNING → {DONE,CRASHED}.
        record = RunStateRecord.initial(ts).transition_to(RunState.RUNNING, ts)
        if state is not RunState.RUNNING:
            record = record.transition_to(state, ts)
        storage.save_run_state(run_dir, record)


def test_load_runs_returns_empty_for_missing_dir(tmp_path: Path) -> None:
    df = load_runs(tmp_path / "does_not_exist")
    assert df.empty


def test_load_runs_returns_empty_when_no_metrics_files(tmp_path: Path) -> None:
    (tmp_path / "some_unrelated_file.txt").write_text("noise", encoding="utf-8")
    df = load_runs(tmp_path)
    assert df.empty


def test_load_runs_pivots_metrics_to_columns(tmp_path: Path) -> None:
    _seed_run(tmp_path / "demo_s0__t1", "demo_s0", m1=0.7, state=RunState.DONE)
    df = load_runs(tmp_path)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["run_id"] == "demo_s0"
    assert row["scenario"] == "discovery"
    assert row["algorithm"] == "mappo"
    assert row["M1_success_rate"] == 0.7
    assert row["M6_coverage_progress"] == 0.8
    assert row["state"] == "DONE"
    assert "run_dir" in row.index


def test_load_runs_recursively_finds_nested_run_dirs(tmp_path: Path) -> None:
    """Nested ``<exp_type>/<run_dir>/`` layouts are walked recursively."""
    _seed_run(tmp_path / "discovery" / "baseline" / "demo_s0__t1", "demo_s0", m1=0.4)
    _seed_run(tmp_path / "discovery" / "baseline" / "demo_s1__t1", "demo_s1", m1=0.9)
    df = load_runs(tmp_path)
    assert len(df) == 2
    assert set(df["run_id"]) == {"demo_s0", "demo_s1"}


def test_load_runs_skips_corrupt_metrics_json(tmp_path: Path) -> None:
    """A bad metrics.json is skipped, not raised — best-effort walk."""
    good = tmp_path / "good_s0__t1"
    _seed_run(good, "good_s0", m1=0.5, state=RunState.DONE)
    bad = tmp_path / "bad_s0__t1"
    (bad / "output").mkdir(parents=True)
    (bad / "output" / "metrics.json").write_text("{not valid json", encoding="utf-8")
    df = load_runs(tmp_path)
    assert len(df) == 1
    assert df.iloc[0]["run_id"] == "good_s0"


def test_load_runs_marks_state_unknown_when_run_state_missing(tmp_path: Path) -> None:
    """Runs without run_state.json (older / partial) get ``UNKNOWN``."""
    _seed_run(tmp_path / "no_state_s0__t1", "no_state_s0", m1=0.3, state=None)
    df = load_runs(tmp_path)
    assert len(df) == 1
    assert df.iloc[0]["state"] == "UNKNOWN"
