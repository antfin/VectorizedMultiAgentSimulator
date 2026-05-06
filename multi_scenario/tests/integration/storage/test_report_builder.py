"""F2.10 tests: ReportBuilder — assembles ``output/report.json`` from run-dir state."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from multi_scenario.adapters.storage.local import LocalStorageAdapter
from multi_scenario.adapters.storage.report_builder import ReportBuilder
from multi_scenario.domain.models import (
    ExperimentResult,
    RunReport,
    RunState,
    RunStateRecord,
    RunStateTransition,
)


def _ts(minute: int = 0) -> datetime:
    return datetime(2026, 5, 6, 14, minute, 0, tzinfo=timezone.utc)


def _result() -> ExperimentResult:
    """A populated result with all four headline metrics."""
    return ExperimentResult(
        run_id="report_smoke_s0",
        exp_id="report_smoke",
        scenario="discovery",
        algorithm="mappo",
        seed=0,
        run_timestamp="20260506_1400",
        metrics={
            "M1_success_rate": 0.5,
            "M2_avg_return": 12.34,
            "M3_steps": 25.0,
            "M4_collisions": 3.0,
            "M5_tokens": None,
            "M6_coverage_progress": 0.42,
            "M7_sample_efficiency": None,
            "M8_agent_utilization": None,
            "M9_spatial_spread": None,
        },
        config_snapshot={"n_agents": 2},
        n_envs=1,
        n_eval_episodes=2,
    )


def _done_state() -> RunStateRecord:
    """A run-state record covering INITIALIZING → RUNNING → DONE over 10 minutes."""
    return RunStateRecord(
        state=RunState.DONE,
        transitions=[
            RunStateTransition(state=RunState.INITIALIZING, ts=_ts(0)),
            RunStateTransition(state=RunState.RUNNING, ts=_ts(1)),
            RunStateTransition(state=RunState.DONE, ts=_ts(10)),
        ],
    )


def _seed_required_files(run_dir: Path) -> None:
    """Lay down the four mandatory artefacts (config / provenance / log / metrics)."""
    (run_dir / "input").mkdir(parents=True)
    (run_dir / "input" / "config.json").write_text("{}", encoding="utf-8")
    (run_dir / "input" / "provenance.json").write_text("{}", encoding="utf-8")
    (run_dir / "logs").mkdir()
    (run_dir / "logs" / "run.log").write_text("ok\n", encoding="utf-8")
    (run_dir / "output").mkdir()
    (run_dir / "output" / "metrics.json").write_text("{}", encoding="utf-8")


def _seed_benchmarl(run_dir: Path, bm_run: str = "mappo_xyz_20260506_1400") -> Path:
    """Lay down a fake BenchMARL output dir with a checkpoints policy + scalars."""
    bm_dir = run_dir / "output" / "benchmarl" / bm_run
    (bm_dir / "checkpoints").mkdir(parents=True)
    (bm_dir / "checkpoints" / "checkpoint_0.pt").write_text("policy", encoding="utf-8")
    (bm_dir / "scalars").mkdir()
    (bm_dir / "scalars" / "train.csv").write_text("", encoding="utf-8")
    return bm_dir


def test_build_with_fully_populated_layout(tmp_path: Path) -> None:
    """Every link resolves to an existing file when the run folder is fully populated."""
    _seed_required_files(tmp_path)
    bm_dir = _seed_benchmarl(tmp_path)

    report = ReportBuilder().build(tmp_path, _result(), _done_state())

    assert isinstance(report, RunReport)
    # Required artefacts.
    for rel in (
        report.links.config,
        report.links.provenance,
        report.links.log,
        report.links.metrics,
    ):
        assert rel is not None
        assert (tmp_path / rel).is_file()
    # BenchMARL artefacts.
    assert report.links.benchmarl_dir is not None
    assert (tmp_path / report.links.benchmarl_dir).is_dir()
    assert (tmp_path / report.links.benchmarl_dir).resolve() == bm_dir.resolve()
    assert report.links.benchmarl_scalars is not None
    assert (tmp_path / report.links.benchmarl_scalars).is_dir()
    assert report.links.policy is not None
    assert (tmp_path / report.links.policy).is_file()
    # eval_episodes / videos default-None until F2.10.1 / F2.11.
    assert report.links.eval_episodes is None
    assert report.links.videos.before_training is None
    assert report.links.videos.after_training is None


def test_build_with_missing_optional_artefacts(tmp_path: Path) -> None:
    """No benchmarl / no videos → optional links are None; required ones still set."""
    _seed_required_files(tmp_path)

    report = ReportBuilder().build(tmp_path, _result(), _done_state())

    assert report.links.benchmarl_dir is None
    assert report.links.benchmarl_scalars is None
    assert report.links.policy is None
    assert report.links.eval_episodes is None
    assert report.links.videos.before_training is None
    assert report.links.videos.after_training is None
    # Required artefacts still resolve.
    assert (tmp_path / report.links.config).is_file()


def test_status_and_duration_from_run_state(tmp_path: Path) -> None:
    """status mirrors the final state; duration_seconds is finished − started."""
    _seed_required_files(tmp_path)

    report = ReportBuilder().build(tmp_path, _result(), _done_state())

    assert report.status == "DONE"
    assert report.started_at == _ts(0)
    assert report.finished_at == _ts(10)
    assert report.duration_seconds == timedelta(minutes=10).total_seconds()


def test_summary_extracts_headline_metrics(tmp_path: Path) -> None:
    """summary lifts the M1–M4 headline subset from the result; passes through None."""
    _seed_required_files(tmp_path)

    report = ReportBuilder().build(tmp_path, _result(), _done_state())

    assert report.summary == {
        "M1_success_rate": 0.5,
        "M2_avg_return": 12.34,
        "M3_steps": 25.0,
        "M4_collisions": 3.0,
    }


def test_save_report_round_trips_through_local_storage(tmp_path: Path) -> None:
    """LocalStorageAdapter.save_report writes JSON that parses back as RunReport."""
    _seed_required_files(tmp_path)
    report = ReportBuilder().build(tmp_path, _result(), _done_state())

    LocalStorageAdapter().save_report(tmp_path, report)
    on_disk = tmp_path / "output" / "report.json"
    assert on_disk.is_file()
    parsed = RunReport.model_validate_json(on_disk.read_text(encoding="utf-8"))
    assert parsed == report
