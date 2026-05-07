"""F7.3 tests: run_detail_loader probes a run folder for every artefact."""

# pylint: disable=missing-function-docstring,redefined-outer-name

from datetime import datetime, timezone
from pathlib import Path

import pytest

from multi_scenario.adapters.storage.local import LocalStorageAdapter
from multi_scenario.domain.models import (
    ExperimentConfig,
    ExperimentResult,
    MetricRecord,
    RunState,
    RunStateRecord,
)
from multi_scenario.frontend.run_detail_loader import RunDetail, load_run_detail


def _seed_required(run_dir: Path) -> None:
    """Write a minimal config.json + metrics.json (the two required artefacts)."""
    storage = LocalStorageAdapter()
    cfg = ExperimentConfig.model_validate(
        {
            "experiment": {"id": "demo", "seed": 0},
            "scenario": {"type": "discovery", "params": {}},
            "algorithm": {"type": "mappo", "params": {}},
            "training": {"max_iters": 1},
            "evaluation": {"interval_iters": 1, "episodes": 1},
        }
    )
    storage.save_config(run_dir, cfg)
    storage.save_result(
        run_dir,
        ExperimentResult(
            run_id="demo_s0",
            exp_id="demo",
            scenario="discovery",
            algorithm="mappo",
            seed=0,
            run_timestamp="20260507_1500",
            metrics=[MetricRecord(name="M1_success_rate", value=0.5)],
            config_snapshot={"n_agents": 2},
            n_envs=1,
            n_eval_episodes=10,
        ),
    )


def _seed_run_state(run_dir: Path, final: RunState = RunState.DONE) -> None:
    storage = LocalStorageAdapter()
    ts = datetime(2026, 5, 7, 15, 0, tzinfo=timezone.utc)
    rec = RunStateRecord.initial(ts).transition_to(RunState.RUNNING, ts)
    if final is not RunState.RUNNING:
        rec = rec.transition_to(final, ts)
    storage.save_run_state(run_dir, rec)


@pytest.fixture
def happy_run_dir(tmp_path: Path) -> Path:
    """A run dir with config + result + state + benchmarl scalars + videos + log."""
    run_dir = tmp_path / "demo_s0__t1"
    run_dir.mkdir()
    _seed_required(run_dir)
    _seed_run_state(run_dir)
    # Inner BenchMARL folder — note BenchMARL nests the experiment name TWICE
    # (``benchmarl/<exp>/<exp>/scalars/*.csv``); the loader walks recursively.
    inner = run_dir / "output" / "benchmarl" / "demo_inner" / "demo_inner"
    (inner / "scalars").mkdir(parents=True)
    (inner / "scalars" / "reward.csv").write_text("step,value\n0,0.0\n1,1.0\n", encoding="utf-8")
    (inner / "scalars" / "loss.csv").write_text("step,value\n0,0.5\n", encoding="utf-8")
    # Videos
    videos = run_dir / "output" / "videos"
    videos.mkdir()
    (videos / "before_training.mp4").write_bytes(b"\x00")
    (videos / "after_training.mp4").write_bytes(b"\x00")
    # Log — F2.7 writes to ``<run_dir>/logs/run.log`` (top-level, not under output/).
    logs = run_dir / "logs"
    logs.mkdir()
    (logs / "run.log").write_text("started\n", encoding="utf-8")
    # Eval-episodes flag
    (run_dir / "output" / "eval_episodes.json").write_text("{}", encoding="utf-8")
    return run_dir


def test_load_run_detail_returns_none_when_dir_missing(tmp_path: Path) -> None:
    assert load_run_detail(tmp_path / "nope") is None


def test_load_run_detail_returns_none_when_required_artefacts_missing(tmp_path: Path) -> None:
    """Missing config or metrics → loader bails early with None."""
    run_dir = tmp_path / "empty"
    run_dir.mkdir()
    assert load_run_detail(run_dir) is None


def test_load_run_detail_happy_path_populates_all_fields(happy_run_dir: Path) -> None:
    rd = load_run_detail(happy_run_dir)
    assert isinstance(rd, RunDetail)
    assert rd.run_dir == happy_run_dir
    assert rd.cfg.experiment.id == "demo"
    assert rd.result.run_id == "demo_s0"
    assert rd.run_state is not None
    assert rd.run_state.state.value == "DONE"
    assert rd.benchmarl_dir is not None
    assert rd.benchmarl_dir.name == "demo_inner"
    assert {p.name for p in rd.scalar_csvs} == {"reward.csv", "loss.csv"}
    assert set(rd.videos) == {"before", "after"}
    assert rd.log_path is not None
    assert rd.log_path.name == "run.log"
    assert rd.has_eval_episodes is True


def test_load_run_detail_skips_optional_artefacts_when_absent(tmp_path: Path) -> None:
    run_dir = tmp_path / "minimal"
    run_dir.mkdir()
    _seed_required(run_dir)
    rd = load_run_detail(run_dir)
    assert rd is not None
    assert rd.run_state is None
    assert rd.benchmarl_dir is None
    assert rd.scalar_csvs == []
    assert rd.videos == {}
    assert rd.log_path is None
    assert rd.has_eval_episodes is False


def test_load_run_detail_picks_most_recent_benchmarl_inner(tmp_path: Path) -> None:
    """Multiple BenchMARL inner folders → pick the one with newest mtime."""
    run_dir = tmp_path / "multi"
    run_dir.mkdir()
    _seed_required(run_dir)
    older = run_dir / "output" / "benchmarl" / "older"
    newer = run_dir / "output" / "benchmarl" / "newer"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    # Touch newer to bump its mtime above older's.
    import os
    import time

    time.sleep(0.01)
    os.utime(newer, None)
    rd = load_run_detail(run_dir)
    assert rd is not None
    assert rd.benchmarl_dir is not None
    assert rd.benchmarl_dir.name == "newer"


def test_load_run_detail_falls_back_to_alt_log_path(tmp_path: Path) -> None:
    """Older runs may use ``output/log.log`` instead of ``output/logs/run.log``."""
    run_dir = tmp_path / "alt_log"
    run_dir.mkdir()
    _seed_required(run_dir)
    (run_dir / "output").mkdir(exist_ok=True)
    alt = run_dir / "output" / "log.log"
    alt.write_text("legacy\n", encoding="utf-8")
    rd = load_run_detail(run_dir)
    assert rd is not None
    assert rd.log_path == alt


def test_load_run_detail_partial_videos(tmp_path: Path) -> None:
    """Only ``before_training.mp4`` present → ``videos`` has just that key."""
    run_dir = tmp_path / "before_only"
    run_dir.mkdir()
    _seed_required(run_dir)
    videos = run_dir / "output" / "videos"
    videos.mkdir(parents=True)
    (videos / "before_training.mp4").write_bytes(b"\x00")
    rd = load_run_detail(run_dir)
    assert rd is not None
    assert set(rd.videos) == {"before"}
