"""F6.3 tests: S3StorageAdapter — Storage port + sync helpers (moto-mocked S3)."""

# Pytest fixtures intentionally share a name between definition + injection;
# pylint flags it as shadowing — that's a fixture-pattern false positive.
# pylint: disable=redefined-outer-name

from datetime import datetime, timezone
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from multi_scenario.adapters.storage.s3 import S3StorageAdapter
from multi_scenario.domain.models import (
    ExperimentConfig,
    ExperimentResult,
    LibraryVersions,
    Provenance,
    RunState,
    RunStateRecord,
    RunStateTransition,
    S3StorageConfig,
)
from multi_scenario.domain.ports import Storage

_BUCKET = "ms-test-bucket"
_PREFIX = "experiments/discovery/baseline"


def _config() -> S3StorageConfig:
    return S3StorageConfig(bucket=_BUCKET, prefix=_PREFIX, region="us-east-1")


def _exp_cfg() -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "experiment": {"id": "s3_demo", "seed": 0},
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
        }
    )


def _provenance() -> Provenance:
    return Provenance(
        config_hash="sha256:c",
        code_hash="sha256:d",
        hashed_source_files=[],
        git_sha="abc",
        git_dirty=False,
        created_at=datetime(2026, 5, 7, 12, 0, tzinfo=timezone.utc),
        library_versions=LibraryVersions(
            python="3.11",
            torch="2.4",
            vmas="1.4",
            benchmarl="1.3",
            multi_scenario="0.0.1",
        ),
    )


def _result() -> ExperimentResult:
    return ExperimentResult(
        run_id="s3_demo_s0",
        exp_id="s3_demo",
        scenario="discovery",
        algorithm="mappo",
        seed=0,
        run_timestamp="20260507_1200",
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
    )


def _run_state() -> RunStateRecord:
    return RunStateRecord(
        state=RunState.DONE,
        transitions=[
            RunStateTransition(
                state=RunState.INITIALIZING,
                ts=datetime(2026, 5, 7, 12, 0, tzinfo=timezone.utc),
            ),
            RunStateTransition(
                state=RunState.DONE, ts=datetime(2026, 5, 7, 12, 5, tzinfo=timezone.utc)
            ),
        ],
    )


@pytest.fixture
def mocked_s3():
    """Spin up a moto S3 with the test bucket pre-created."""
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=_BUCKET)
        yield client


def test_implements_storage_protocol(mocked_s3) -> None:
    """``S3StorageAdapter`` satisfies the ``Storage`` Protocol."""
    adapter = S3StorageAdapter(_config(), client=mocked_s3)
    assert isinstance(adapter, Storage)
    assert adapter.name == "s3"


def test_save_load_config_round_trip(mocked_s3) -> None:
    """save_config + load_config reproduce the original ``ExperimentConfig``."""
    adapter = S3StorageAdapter(_config(), client=mocked_s3)
    run_dir = Path("s3_demo_s0__20260507_1200")
    adapter.save_config(run_dir, _exp_cfg())
    out = adapter.load_config(run_dir)
    assert out.experiment.id == "s3_demo"
    assert out.experiment.seed == 0


def test_save_provenance_round_trip(mocked_s3) -> None:
    """Provenance save/load via S3."""
    adapter = S3StorageAdapter(_config(), client=mocked_s3)
    run_dir = Path("s3_demo_s0__20260507_1200")
    prov = _provenance()
    adapter.save_provenance(run_dir, prov)
    assert adapter.load_provenance(run_dir).git_sha == "abc"


def test_save_result_round_trip(mocked_s3) -> None:
    """ExperimentResult save/load via S3."""
    adapter = S3StorageAdapter(_config(), client=mocked_s3)
    run_dir = Path("s3_demo_s0__20260507_1200")
    adapter.save_result(run_dir, _result())
    out = adapter.load_result(run_dir)
    assert out.run_id == "s3_demo_s0"
    assert {m.name for m in out.metrics} >= {"M1_success_rate", "M2_avg_return"}


def test_save_run_state_round_trip(mocked_s3) -> None:
    """RunStateRecord save/load via S3."""
    adapter = S3StorageAdapter(_config(), client=mocked_s3)
    run_dir = Path("s3_demo_s0__20260507_1200")
    adapter.save_run_state(run_dir, _run_state())
    out = adapter.load_run_state(run_dir)
    assert out.state == RunState.DONE


def test_keys_include_run_dir_name_under_prefix(mocked_s3) -> None:
    """Keys are ``<prefix>/<run_dir.name>/<rel>`` — preserves §3.5.2 layout."""
    adapter = S3StorageAdapter(_config(), client=mocked_s3)
    run_dir = Path("s3_demo_s0__20260507_1200")
    adapter.save_config(run_dir, _exp_cfg())
    expected_key = f"{_PREFIX}/s3_demo_s0__20260507_1200/input/config.json"
    listed = mocked_s3.list_objects_v2(Bucket=_BUCKET).get("Contents", [])
    keys = [obj["Key"] for obj in listed]
    assert expected_key in keys


def test_sync_to_local_downloads_all_keys(tmp_path: Path, mocked_s3) -> None:
    """sync_to_local pulls every key under ``<prefix>/<run_dir.name>/`` to local fs."""
    adapter = S3StorageAdapter(_config(), client=mocked_s3)
    run_dir = Path("s3_demo_s0__20260507_1200")
    adapter.save_config(run_dir, _exp_cfg())
    adapter.save_provenance(run_dir, _provenance())
    adapter.save_result(run_dir, _result())
    adapter.save_run_state(run_dir, _run_state())

    local_run = tmp_path / run_dir.name
    adapter.sync_to_local(run_dir, local_run)

    assert (local_run / "input" / "config.json").is_file()
    assert (local_run / "input" / "provenance.json").is_file()
    assert (local_run / "output" / "metrics.json").is_file()
    assert (local_run / "run_state.json").is_file()


def test_sync_from_local_uploads_all_files(tmp_path: Path, mocked_s3) -> None:
    """sync_from_local uploads every file under local_dir to S3, mirroring tree."""
    adapter = S3StorageAdapter(_config(), client=mocked_s3)
    local_run = tmp_path / "s3_demo_s0__20260507_1200"
    (local_run / "input").mkdir(parents=True)
    (local_run / "input" / "config.json").write_text("{}", encoding="utf-8")
    (local_run / "output").mkdir()
    (local_run / "output" / "metrics.json").write_text('{"x": 1}', encoding="utf-8")
    (local_run / "logs").mkdir()
    (local_run / "logs" / "run.log").write_text("log line\n", encoding="utf-8")

    adapter.sync_from_local(local_run, Path("s3_demo_s0__20260507_1200"))

    listed = mocked_s3.list_objects_v2(Bucket=_BUCKET).get("Contents", [])
    keys = sorted(obj["Key"] for obj in listed)
    expected = sorted(
        [
            f"{_PREFIX}/s3_demo_s0__20260507_1200/input/config.json",
            f"{_PREFIX}/s3_demo_s0__20260507_1200/output/metrics.json",
            f"{_PREFIX}/s3_demo_s0__20260507_1200/logs/run.log",
        ]
    )
    assert keys == expected


def test_s3_storage_config_yaml_round_trip(tmp_path: Path) -> None:
    """``S3StorageConfig.from_yaml`` round-trips cleanly."""
    cfg_path = tmp_path / "s3.yaml"
    cfg_path.write_text(
        "bucket: b\nprefix: p\nregion: gra\nendpoint_url: https://s3.gra.io.cloud.ovh.net\n",
        encoding="utf-8",
    )
    cfg = S3StorageConfig.from_yaml(cfg_path)
    assert cfg.bucket == "b"
    assert cfg.endpoint_url == "https://s3.gra.io.cloud.ovh.net"
