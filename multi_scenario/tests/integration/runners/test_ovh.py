"""F6.2 tests: OvhRunner — submission + polling + result loading (mocked)."""

# Test fakes legitimately bundle test-state knobs as kwargs; the protocol-fake
# methods don't all use their args (Logger / OvhClient stand-ins).
# pylint: disable=too-many-arguments,too-many-positional-arguments,unused-argument
# pylint: disable=missing-function-docstring,missing-class-docstring

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from multi_scenario.adapters.runners.ovh import OvhJobError, OvhRunner
from multi_scenario.adapters.runners.ovh_cli import JobInfo, OvhClient
from multi_scenario.adapters.secrets.fernet import FernetSecretsAdapter
from multi_scenario.adapters.storage.local import LocalStorageAdapter
from multi_scenario.domain.models import (
    ExperimentConfig,
    ExperimentResult,
    OvhJobConfig,
    RunState,
    RunStateRecord,
    RunStateTransition,
)
from multi_scenario.domain.ports import Runner


def _ovh_cfg() -> OvhJobConfig:
    return OvhJobConfig(
        region="GRA",
        image="ovhcom/ai-training-pytorch:latest",
        gpu_type="V100S",
        n_gpu=1,
        bucket_code="code-bucket",
        bucket_results="results-bucket",
        poll_interval_sec=0.0,
        timeout_sec=10.0,
    )


def _exp_cfg(storage_path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "experiment": {"id": "ovh_demo", "seed": 0},
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
                "runner": {"type": "ovh", "params": {}},
                "storage": {"type": "fs", "path": str(storage_path), "params": {}},
            },
        }
    )


def _result_dict(run_id: str = "ovh_demo_s0") -> dict[str, Any]:
    return {
        "run_id": run_id,
        "exp_id": "ovh_demo",
        "scenario": "discovery",
        "algorithm": "mappo",
        "seed": 0,
        "run_timestamp": "20260507_1500",
        "metrics": {
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
        "config_snapshot": {"n_agents": 2},
        "n_envs": 1,
        "n_eval_episodes": 1,
    }


class _StubClient:
    """OvhClient stand-in that returns scripted submit/get/logs responses."""

    def __init__(self, states_after_submit: list[str], log_text: str = "") -> None:
        self._states = list(states_after_submit)
        self._log_text = log_text
        self.submit_calls: list[list[str]] = []

    def submit(self, args) -> str:
        self.submit_calls.append(list(args))
        return "job_42"

    def get(self, job_id: str) -> JobInfo:  # noqa: ARG002
        state = self._states.pop(0) if self._states else "DONE"
        return JobInfo(id="job_42", state=state)

    def logs(self, job_id: str, tail: int = 100) -> str:  # noqa: ARG002
        return self._log_text


class _NoopLogger:
    def info(self, msg: str) -> None: ...
    def debug(self, msg: str) -> None: ...
    def warning(self, msg: str) -> None: ...
    def error(self, msg: str) -> None: ...


def _seed_done_run_dir(tmp_path: Path) -> Path:
    """Mock the post-S3-sync state: write a metrics.json + run_state.json."""
    run_dir = tmp_path / "ovh_demo_s0__test"
    run_dir.mkdir(parents=True)
    storage = LocalStorageAdapter()
    storage.save_result(run_dir, ExperimentResult.model_validate(_result_dict()))
    started = datetime(2026, 5, 7, 10, 0, tzinfo=timezone.utc)
    finished = datetime(2026, 5, 7, 10, 5, tzinfo=timezone.utc)
    storage.save_run_state(
        run_dir,
        RunStateRecord(
            state=RunState.DONE,
            transitions=[
                RunStateTransition(state=RunState.INITIALIZING, ts=started),
                RunStateTransition(state=RunState.DONE, ts=finished),
            ],
        ),
    )
    return run_dir


def test_implements_runner_protocol() -> None:
    """OvhRunner satisfies the Runner Protocol with supports_resume=False."""
    runner = OvhRunner(
        ovh_config=_ovh_cfg(),
        client=OvhClient(runner=lambda *a, **kw: None),  # never called in this check
        secrets=FernetSecretsAdapter(),
        logger=_NoopLogger(),
    )
    assert isinstance(runner, Runner)
    assert OvhRunner.supports_resume is False


def test_run_with_resume_from_raises(tmp_path: Path) -> None:
    """Defensive: even though supports_resume=False, calling with resume_from raises."""
    runner = OvhRunner(
        ovh_config=_ovh_cfg(),
        client=_StubClient(states_after_submit=["DONE"]),
        secrets=FernetSecretsAdapter(),
        logger=_NoopLogger(),
        sleep=lambda _: None,
    )
    with pytest.raises(OvhJobError, match="resume"):
        runner.run(_exp_cfg(tmp_path), tmp_path, resume_from=Path("/nope.pt"))


def test_run_happy_path_returns_loaded_result(tmp_path: Path) -> None:
    """Submit → poll until DONE → load metrics.json from run_dir → return ExperimentResult."""
    run_dir = _seed_done_run_dir(tmp_path)
    client = _StubClient(states_after_submit=["RUNNING", "RUNNING", "DONE"])
    runner = OvhRunner(
        ovh_config=_ovh_cfg(),
        client=client,
        secrets=FernetSecretsAdapter(),
        logger=_NoopLogger(),
        sleep=lambda _: None,
    )
    result = runner.run(_exp_cfg(tmp_path), run_dir)
    assert result.run_id == "ovh_demo_s0"
    assert result.scenario == "discovery"
    # One submit call.
    assert len(client.submit_calls) == 1


def test_submit_args_include_per_run_s3_prefix(tmp_path: Path) -> None:
    """Per-experiment S3 prefix isolation — `bucket_results/<run_id>` (no trailing slash)."""
    run_dir = _seed_done_run_dir(tmp_path)
    client = _StubClient(states_after_submit=["DONE"])
    runner = OvhRunner(
        ovh_config=_ovh_cfg(),
        client=client,
        secrets=FernetSecretsAdapter(),
        logger=_NoopLogger(),
        sleep=lambda _: None,
    )
    runner.run(_exp_cfg(tmp_path), run_dir)
    args = " ".join(client.submit_calls[0])
    # Per-experiment prefix appears, no trailing slash.
    assert "results-bucket@GRA/ovh_demo_s0:" in args
    assert "results-bucket@GRA/ovh_demo_s0/:" not in args


def test_run_failed_state_raises_with_logs_tail(tmp_path: Path) -> None:
    """Non-DONE terminal state → :class:`OvhJobError` including the log tail."""
    run_dir = _seed_done_run_dir(tmp_path)
    client = _StubClient(states_after_submit=["FAILED"], log_text="OOM at iter 12\n")
    runner = OvhRunner(
        ovh_config=_ovh_cfg(),
        client=client,
        secrets=FernetSecretsAdapter(),
        logger=_NoopLogger(),
        sleep=lambda _: None,
    )
    with pytest.raises(OvhJobError, match="OOM at iter 12"):
        runner.run(_exp_cfg(tmp_path), run_dir)


def test_secrets_shipped_via_env_when_configured(tmp_path: Path) -> None:
    """When ``secret_env`` + passphrase are set, the encrypted env-var pair appears in args."""
    run_dir = _seed_done_run_dir(tmp_path)
    client = _StubClient(states_after_submit=["DONE"])
    runner = OvhRunner(
        ovh_config=_ovh_cfg(),
        client=client,
        secrets=FernetSecretsAdapter(),
        logger=_NoopLogger(),
        secret_env={"OPENAI_API_KEY": "sk-xyz"},
        secret_passphrase="hunter2",
        sleep=lambda _: None,
    )
    runner.run(_exp_cfg(tmp_path), run_dir)
    args = client.submit_calls[0]
    # Both encrypted-blob and passphrase env vars appear as `--env KEY=VAL` flags.
    env_flags = [args[i + 1] for i, a in enumerate(args) if a == "--env"]
    keys = {flag.split("=", 1)[0] for flag in env_flags}
    assert "MS_ENCRYPTED_SECRETS" in keys
    assert "MS_SECRETS_PASSPHRASE" in keys


def test_run_times_out_when_never_terminal(tmp_path: Path) -> None:
    """Timeout exhausted before terminal state → :class:`OvhJobError`."""
    run_dir = _seed_done_run_dir(tmp_path)
    # poll_interval=0 + timeout=0 → first iteration: get returns RUNNING; check
    # deadline elapsed; raise. The stub returns RUNNING forever.
    cfg = _ovh_cfg()
    cfg.timeout_sec = 0.0
    client = _StubClient(states_after_submit=["RUNNING"] * 100)
    runner = OvhRunner(
        ovh_config=cfg,
        client=client,
        secrets=FernetSecretsAdapter(),
        logger=_NoopLogger(),
        sleep=lambda _: None,
    )
    with pytest.raises(OvhJobError, match="terminal state"):
        runner.run(_exp_cfg(tmp_path), run_dir)
