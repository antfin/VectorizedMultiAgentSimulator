"""F6.7 tests: ``multi-scenario sweep --runner ovh`` (mock-only)."""

# Pytest fixtures + protocol-fake stubs share the noisy lint shape used elsewhere.
# pylint: disable=redefined-outer-name,too-many-arguments,too-many-positional-arguments
# pylint: disable=missing-function-docstring,unused-argument

from pathlib import Path
from unittest.mock import patch

import yaml
from typer.testing import CliRunner

from multi_scenario.adapters.runners.ovh_cli import JobInfo
from multi_scenario.cli import app


def _smoke_yaml_dict(exp_id: str, storage_path: Path) -> dict:
    return {
        "experiment": {"id": exp_id, "seed": 0},
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


def _ovh_yaml_dict() -> dict:
    return {
        "region": "GRA",
        "image": "pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime",
        "flavor": "ai1-1-cpu",
        "n_gpu": 0,
        "bucket_code": "ms-code",
        "bucket_results": "ms-results",
        "mount_code": "/workspace/code",
        "mount_results": "/workspace/results",
        "poll_interval_sec": 0.0,
        "timeout_sec": 60.0,
        "cost_cap_eur": 5.0,
    }


def _write_yamls(tmp_path: Path, exp_ids: list[str]) -> tuple[list[Path], Path]:
    """Lay down per-cell smoke YAMLs + an OVH config; return paths."""
    cfgs_dir = tmp_path / "configs"
    cfgs_dir.mkdir()
    yaml_paths = []
    for exp_id in exp_ids:
        p = cfgs_dir / f"{exp_id}.yaml"
        p.write_text(yaml.safe_dump(_smoke_yaml_dict(exp_id, tmp_path)), encoding="utf-8")
        yaml_paths.append(p)
    ovh_path = tmp_path / "ovh.yaml"
    ovh_path.write_text(yaml.safe_dump(_ovh_yaml_dict()), encoding="utf-8")
    return yaml_paths, ovh_path


class _CountingStubClient:
    """Records every submit's args; ``get`` returns scripted JobInfos by job_id."""

    def __init__(self, terminal_after: int = 1) -> None:
        self.submit_calls: list[list[str]] = []
        self.next_id = 0
        self._gets_by_id: dict[str, int] = {}
        self._terminal_after = terminal_after

    def ensure_available(self) -> None:
        """No-op stand-in for OvhClient.ensure_available (F6.7.1)."""
        return None

    def submit(self, args: list[str]) -> str:
        self.submit_calls.append(list(args))
        self.next_id += 1
        return f"job_{self.next_id}"

    def get(self, job_id: str) -> JobInfo:
        self._gets_by_id[job_id] = self._gets_by_id.get(job_id, 0) + 1
        state = "RUNNING" if self._gets_by_id[job_id] < self._terminal_after else "DONE"
        return JobInfo(id=job_id, state=state)

    def logs(self, job_id: str, tail: int = 100) -> str:  # noqa: ARG002
        return ""


def test_dry_run_ovh_prints_cells_without_submitting(tmp_path: Path) -> None:
    """``--runner ovh --dry-run`` lists cells but does not submit anything."""
    yaml_paths, ovh_path = _write_yamls(tmp_path, ["smoke_a", "smoke_b"])
    stub = _CountingStubClient()

    with patch("multi_scenario.adapters.runners.ovh_cli.OvhClient", return_value=stub):
        result = CliRunner().invoke(
            app,
            [
                "sweep",
                str(yaml_paths[0].parent),
                "--runner",
                "ovh",
                "--ovh-config",
                str(ovh_path),
                "--repo-root",
                str(tmp_path),
                "--dry-run",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "smoke_a_s0" in result.output
    assert "smoke_b_s0" in result.output
    assert not stub.submit_calls


def test_parallel_sweep_uses_distinct_s3_prefixes_per_cell(tmp_path: Path) -> None:
    """F6.7 anti-collision invariant: 4 cells → 4 distinct results-bucket prefixes.

    Locks in the rendezvous_comm 2026-04-16 fix: parallel jobs must each get
    their own per-cell ``<bucket>@<region>/<run_id>:<mount>:rwd`` prefix so
    OVH FINALIZING can't overwrite earlier cells' results.
    """
    yaml_paths, ovh_path = _write_yamls(tmp_path, ["smoke_a", "smoke_b"])
    stub = _CountingStubClient()

    with patch("multi_scenario.adapters.runners.ovh_cli.OvhClient", return_value=stub):
        result = CliRunner().invoke(
            app,
            [
                "sweep",
                str(yaml_paths[0].parent),
                "--seeds",
                "0,1",
                "--runner",
                "ovh",
                "--ovh-config",
                str(ovh_path),
                "--repo-root",
                str(tmp_path),
            ],
        )
    assert result.exit_code == 0, result.output
    assert len(stub.submit_calls) == 4  # 2 yamls × 2 seeds
    # Pull out the results-bucket --volume entry from each submit args list.
    prefixes = []
    for args in stub.submit_calls:
        results_volume = next(a for a in args if a.startswith("ms-results@GRA/"))
        prefix = results_volume.split(":")[0]  # bucket@region/<run_id>
        prefixes.append(prefix)
    # 4 cells → 4 distinct per-run prefixes (no collisions).
    assert len(set(prefixes)) == 4, f"collision risk! prefixes={prefixes}"
    # Sanity: each prefix matches the run_id pattern.
    assert all(
        "smoke_a_s0" in p or "smoke_a_s1" in p or "smoke_b_s0" in p or "smoke_b_s1" in p
        for p in prefixes
    ), prefixes


def test_unknown_runner_kind_exits_nonzero(tmp_path: Path) -> None:
    """Bad ``--runner`` value → exit 2 with a helpful error."""
    yaml_paths, _ = _write_yamls(tmp_path, ["smoke_a"])
    result = CliRunner().invoke(
        app,
        ["sweep", str(yaml_paths[0]), "--runner", "modal"],
    )
    assert result.exit_code == 2
    assert "modal" in result.output


def test_ovh_runner_without_config_path_exits_nonzero(tmp_path: Path) -> None:
    """``--runner ovh`` without ``--ovh-config`` → exit 2."""
    yaml_paths, _ = _write_yamls(tmp_path, ["smoke_a"])
    result = CliRunner().invoke(app, ["sweep", str(yaml_paths[0]), "--runner", "ovh"])
    assert result.exit_code == 2
    assert "ovh-config" in result.output


def test_sweep_ovh_friendly_error_when_ovhai_missing(tmp_path: Path) -> None:
    """F6.7.1: missing ``ovhai`` binary → exit 2 + install URL in stderr."""
    yaml_paths, ovh_path = _write_yamls(tmp_path, ["smoke_a"])
    # Force ``check_available`` to report False — simulates a machine without
    # the ``ovhai`` binary on PATH.
    with patch(
        "multi_scenario.adapters.runners.ovh_cli.OvhClient.check_available",
        return_value=False,
    ):
        result = CliRunner().invoke(
            app,
            [
                "sweep",
                str(yaml_paths[0]),
                "--runner",
                "ovh",
                "--ovh-config",
                str(ovh_path),
                "--repo-root",
                str(tmp_path),
            ],
        )
    assert result.exit_code == 2, result.output
    assert "ovhai CLI not found" in result.output
    assert "cli.bhs.ai.cloud.ovh.net/install.sh" in result.output


def test_follow_polls_until_all_jobs_terminal(tmp_path: Path) -> None:
    """``--follow``: every submitted job is polled; final state appears in output."""
    yaml_paths, ovh_path = _write_yamls(tmp_path, ["smoke_a"])
    stub = _CountingStubClient(terminal_after=2)

    with (
        patch("multi_scenario.adapters.runners.ovh_cli.OvhClient", return_value=stub),
        patch("multi_scenario.cli.sweep.time.sleep"),  # don't actually sleep
    ):
        result = CliRunner().invoke(
            app,
            [
                "sweep",
                str(yaml_paths[0]),
                "--runner",
                "ovh",
                "--ovh-config",
                str(ovh_path),
                "--repo-root",
                str(tmp_path),
                "--follow",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "submitted smoke_a_s0" in result.output
    assert "DONE smoke_a_s0" in result.output
