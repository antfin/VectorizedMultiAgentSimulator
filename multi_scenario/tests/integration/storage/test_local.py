"""F2.5 integration tests: LocalStorageAdapter — Storage port + on-disk round-trips."""

from datetime import datetime, timezone
from pathlib import Path

from multi_scenario.adapters.storage.local import LocalStorageAdapter
from multi_scenario.domain.models import (
    ExperimentConfig,
    ExperimentResult,
    LibraryVersions,
    MetricRecord,
    Provenance,
    RunState,
    RunStateRecord,
)
from multi_scenario.domain.ports import Storage


def _ts() -> datetime:
    return datetime(2026, 5, 6, 14, 23, 0, tzinfo=timezone.utc)


def _provenance() -> Provenance:
    return Provenance(
        config_hash="sha256:abc",
        code_hash="sha256:def",
        hashed_source_files=["src/foo.py"],
        git_sha="1a2b3c4d",
        git_dirty=False,
        created_at=_ts(),
        library_versions=LibraryVersions(
            python="3.11.4",
            torch="2.4.0",
            vmas="1.4.0",
            benchmarl="1.3.0",
            multi_scenario="0.0.1",
        ),
    )


def _result() -> ExperimentResult:
    return ExperimentResult(
        run_id="disc_baseline_smoke_mappo_s0",
        exp_id="disc_baseline_smoke_mappo",
        scenario="discovery",
        algorithm="mappo",
        seed=0,
        run_timestamp="20260506_1423",
        metrics=[
            MetricRecord(name="M1_success_rate", value=0.42),
            MetricRecord(name="M5_tokens", value=None),
        ],
        config_snapshot={"n_agents": 2, "lr": 0.0003},
        n_envs=1,
        n_eval_episodes=10,
    )


def test_implements_storage_protocol():
    """LocalStorageAdapter satisfies the Storage port."""
    assert isinstance(LocalStorageAdapter(), Storage)


def test_save_load_config_roundtrip(repo_root: Path, tmp_path: Path):
    """ExperimentConfig parsed from YAML round-trips through disk."""
    storage = LocalStorageAdapter()
    cfg = ExperimentConfig.from_yaml(repo_root / "docs" / "example_config.yaml")
    storage.save_config(tmp_path, cfg)
    assert (tmp_path / "input" / "config.json").exists()
    assert storage.load_config(tmp_path) == cfg


def test_save_load_provenance_roundtrip(tmp_path: Path):
    """Provenance round-trips with the datetime field preserved."""
    storage = LocalStorageAdapter()
    p = _provenance()
    storage.save_provenance(tmp_path, p)
    assert (tmp_path / "input" / "provenance.json").exists()
    assert storage.load_provenance(tmp_path) == p


def test_save_load_result_roundtrip(tmp_path: Path):
    """ExperimentResult with metrics list + config_snapshot round-trips."""
    storage = LocalStorageAdapter()
    r = _result()
    storage.save_result(tmp_path, r)
    assert (tmp_path / "output" / "metrics.json").exists()
    assert storage.load_result(tmp_path) == r


def test_save_load_run_state_roundtrip(tmp_path: Path):
    """RunStateRecord with multi-transition history round-trips."""
    storage = LocalStorageAdapter()
    record = (
        RunStateRecord.initial(_ts())
        .transition_to(RunState.RUNNING, _ts())
        .transition_to(RunState.DONE, _ts())
    )
    storage.save_run_state(tmp_path, record)
    assert (tmp_path / "run_state.json").exists()
    loaded = storage.load_run_state(tmp_path)
    assert loaded == record
    assert [t.state for t in loaded.transitions] == [
        RunState.INITIALIZING,
        RunState.RUNNING,
        RunState.DONE,
    ]


def test_save_creates_directories(tmp_path: Path):
    """save_* on a fresh run_dir creates the layout subdirectories on demand."""
    storage = LocalStorageAdapter()
    run_dir = tmp_path / "fresh_run"
    assert not run_dir.exists()
    storage.save_config(
        run_dir,
        ExperimentConfig.model_validate(
            {
                "experiment": {"id": "test", "seed": 0},
                "scenario": {"type": "discovery", "params": {}},
                "algorithm": {"type": "mappo", "params": {}},
                "training": {"max_iters": 1},
                "evaluation": {"interval_iters": 1, "episodes": 1},
            }
        ),
    )
    assert (run_dir / "input").is_dir()
    assert (run_dir / "input" / "config.json").is_file()
