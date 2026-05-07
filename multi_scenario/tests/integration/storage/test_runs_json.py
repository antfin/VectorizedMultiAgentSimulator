"""F5.3 integration tests: RunsJsonWriter — slim cross-run manifest."""

# Helpers fan a small handful of orthogonal fixture kwargs around — bundling
# them into a dataclass would add noise.
# pylint: disable=too-many-arguments,too-many-positional-arguments

from datetime import datetime, timezone
from pathlib import Path

from multi_scenario.adapters.storage.local import LocalStorageAdapter
from multi_scenario.adapters.storage.runs_json import RunsJsonWriter
from multi_scenario.domain.models import (
    ExperimentResult,
    ReportLinks,
    RunReport,
    RunsManifest,
    RunState,
    RunStateRecord,
    RunStateTransition,
)


def _ts(minute: int = 0) -> datetime:
    return datetime(2026, 5, 7, 14, minute, 0, tzinfo=timezone.utc)


def _result(run_id: str, exp_id: str, algo: str, m1: float | None, m2: float) -> ExperimentResult:
    return ExperimentResult(
        run_id=run_id,
        exp_id=exp_id,
        scenario="discovery",
        algorithm=algo,
        seed=int(run_id.rsplit("_s", 1)[-1]),
        run_timestamp="20260507_1400",
        metrics={
            "M1_success_rate": m1,
            "M2_avg_return": m2,
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
    )


def _done_state() -> RunStateRecord:
    return RunStateRecord(
        state=RunState.DONE,
        transitions=[
            RunStateTransition(state=RunState.INITIALIZING, ts=_ts(0)),
            RunStateTransition(state=RunState.DONE, ts=_ts(5)),
        ],
    )


def _running_state() -> RunStateRecord:
    return RunStateRecord(
        state=RunState.RUNNING,
        transitions=[RunStateTransition(state=RunState.INITIALIZING, ts=_ts(0))],
    )


def _stub_report() -> RunReport:
    return RunReport(
        status="DONE",
        started_at=_ts(0),
        finished_at=_ts(5),
        duration_seconds=300.0,
        summary={"M1_success_rate": 0.5, "M2_avg_return": 1.0},
        links=ReportLinks(
            config="input/config.json",
            provenance="input/provenance.json",
            log="logs/run.log",
            metrics="output/metrics.json",
        ),
    )


def _seed_run(
    parent: Path,
    run_id: str,
    exp_id: str,
    algo: str,
    state: RunStateRecord,
    m1: float | None = 0.5,
    m2: float = 1.0,
    write_report: bool = True,
) -> Path:
    """Lay down a minimal §3.5.2 run folder under ``parent``."""
    run_dir = parent / f"{run_id}__20260507_1400"
    run_dir.mkdir(parents=True)
    storage = LocalStorageAdapter()
    storage.save_result(run_dir, _result(run_id, exp_id, algo, m1, m2))
    storage.save_run_state(run_dir, state)
    if write_report:
        storage.save_report(run_dir, _stub_report())
    return run_dir


def test_consolidate_writes_runs_json(tmp_path: Path) -> None:
    """Three completed runs → manifest with 3 runs entries."""
    _seed_run(tmp_path, "smoke_a_s0", "smoke_a", "mappo", _done_state(), m1=0.3, m2=2.0)
    _seed_run(tmp_path, "smoke_a_s1", "smoke_a", "mappo", _done_state(), m1=0.7, m2=1.0)
    _seed_run(tmp_path, "smoke_b_s0", "smoke_b", "ippo", _done_state(), m1=0.5, m2=3.0)

    out = RunsJsonWriter().consolidate(tmp_path)
    assert out == tmp_path / "runs.json"

    manifest = RunsManifest.model_validate_json(out.read_text(encoding="utf-8"))
    assert manifest.scope.n_runs == 3
    assert manifest.csv == "runs.csv"
    assert {r.run_id for r in manifest.runs} == {"smoke_a_s0", "smoke_a_s1", "smoke_b_s0"}


def test_scope_aggregates_unique_exp_ids_seeds_algorithms(tmp_path: Path) -> None:
    """Scope deduplicates exp_ids / seeds / algorithms across runs."""
    _seed_run(tmp_path, "smoke_a_s0", "smoke_a", "mappo", _done_state())
    _seed_run(tmp_path, "smoke_a_s1", "smoke_a", "mappo", _done_state())
    _seed_run(tmp_path, "smoke_b_s0", "smoke_b", "ippo", _done_state())

    manifest = RunsManifest.model_validate_json(
        RunsJsonWriter().consolidate(tmp_path).read_text(encoding="utf-8")
    )
    assert sorted(manifest.scope.exp_ids) == ["smoke_a", "smoke_b"]
    assert sorted(manifest.scope.seeds) == [0, 1]
    assert sorted(manifest.scope.algorithms) == ["ippo", "mappo"]


def test_rankings_sorted_descending_per_metric(tmp_path: Path) -> None:
    """Each metric's ranking list is sorted descending by value."""
    _seed_run(tmp_path, "low_s0", "smoke", "mappo", _done_state(), m1=0.1, m2=1.0)
    _seed_run(tmp_path, "high_s0", "smoke", "ippo", _done_state(), m1=0.9, m2=3.0)
    _seed_run(tmp_path, "mid_s0", "smoke", "maddpg", _done_state(), m1=0.5, m2=2.0)

    manifest = RunsManifest.model_validate_json(
        RunsJsonWriter().consolidate(tmp_path).read_text(encoding="utf-8")
    )
    m1 = manifest.rankings["M1_success_rate"]
    assert [e.run_id for e in m1] == ["high_s0", "mid_s0", "low_s0"]
    assert [e.value for e in m1] == [0.9, 0.5, 0.1]


def test_rankings_exclude_none_values(tmp_path: Path) -> None:
    """Metrics that are None for a run are skipped from that metric's ranking."""
    _seed_run(tmp_path, "ok_s0", "smoke", "mappo", _done_state(), m1=0.5)
    _seed_run(tmp_path, "stub_s0", "smoke", "ippo", _done_state(), m1=None)

    manifest = RunsManifest.model_validate_json(
        RunsJsonWriter().consolidate(tmp_path).read_text(encoding="utf-8")
    )
    # M1 ranking contains only the run where M1 is non-None.
    assert [e.run_id for e in manifest.rankings["M1_success_rate"]] == ["ok_s0"]
    # Metrics that are None for ALL runs are absent from rankings entirely.
    assert "M5_tokens" not in manifest.rankings
    assert "M9_spatial_spread" not in manifest.rankings


def test_runs_entries_link_to_per_run_report(tmp_path: Path) -> None:
    """Each runs[] entry's report path resolves to that run's output/report.json."""
    _seed_run(tmp_path, "smoke_s0", "smoke", "mappo", _done_state())

    manifest = RunsManifest.model_validate_json(
        RunsJsonWriter().consolidate(tmp_path).read_text(encoding="utf-8")
    )
    assert len(manifest.runs) == 1
    entry = manifest.runs[0]
    assert entry.report is not None
    assert (tmp_path / entry.report).is_file()


def test_runs_entry_report_none_when_missing(tmp_path: Path) -> None:
    """If a run has no report.json yet, its runs[] entry has report=None."""
    _seed_run(tmp_path, "no_report_s0", "smoke", "mappo", _done_state(), write_report=False)

    manifest = RunsManifest.model_validate_json(
        RunsJsonWriter().consolidate(tmp_path).read_text(encoding="utf-8")
    )
    assert manifest.runs[0].report is None


def test_skips_non_done_runs(tmp_path: Path) -> None:
    """RUNNING / CRASHED runs don't appear in the manifest."""
    _seed_run(tmp_path, "done_s0", "smoke", "mappo", _done_state())
    _seed_run(tmp_path, "running_s0", "smoke", "mappo", _running_state())

    manifest = RunsManifest.model_validate_json(
        RunsJsonWriter().consolidate(tmp_path).read_text(encoding="utf-8")
    )
    assert manifest.scope.n_runs == 1
    assert [e.run_id for e in manifest.runs] == ["done_s0"]


def test_atomic_overwrite_creates_previous(tmp_path: Path) -> None:
    """Re-consolidating moves the old runs.json to runs.previous.json."""
    _seed_run(tmp_path, "first_s0", "smoke", "mappo", _done_state())
    RunsJsonWriter().consolidate(tmp_path)
    first_content = (tmp_path / "runs.json").read_text(encoding="utf-8")

    _seed_run(tmp_path, "second_s0", "smoke", "ippo", _done_state())
    RunsJsonWriter().consolidate(tmp_path)

    assert (tmp_path / "runs.json").is_file()
    assert (tmp_path / "runs.previous.json").is_file()
    assert (tmp_path / "runs.previous.json").read_text(encoding="utf-8") == first_content


def test_empty_dir_writes_empty_manifest(tmp_path: Path) -> None:
    """No runs at all → scope.n_runs=0, empty rankings/runs."""
    out = RunsJsonWriter().consolidate(tmp_path)
    manifest = RunsManifest.model_validate_json(out.read_text(encoding="utf-8"))
    assert manifest.scope.n_runs == 0
    assert manifest.runs == []
    assert manifest.rankings == {}
