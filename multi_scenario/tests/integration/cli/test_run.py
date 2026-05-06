"""F2.8 integration tests: `multi-scenario run <yaml>` end-to-end."""

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from multi_scenario.cli import app


def _smoke_yaml(storage_path: Path) -> dict:
    return {
        "experiment": {"id": "mappo_smoke", "seed": 0},
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
            "runner": {"type": "local", "params": {}},
            "storage": {
                "type": "fs",
                "path": str(storage_path),
                "params": {},
            },
        },
    }


def test_run_command_with_missing_yaml_returns_nonzero():
    """Pointing the CLI at a non-existent YAML exits non-zero (fast)."""
    result = CliRunner().invoke(app, ["run", "/tmp/does_not_exist.yaml"])
    assert result.exit_code != 0


@pytest.mark.slow
def test_run_command_succeeds(tmp_path: Path):
    """End-to-end: CLI runs a smoke config and writes the §3.5.2 layout."""
    cfg_path = tmp_path / "smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(_smoke_yaml(tmp_path)), encoding="utf-8")

    result = CliRunner().invoke(app, ["run", str(cfg_path)])
    assert result.exit_code == 0, result.output

    # Exactly one run folder under storage path, named <run_id>__<ts>.
    run_folders = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(run_folders) == 1
    run_dir = run_folders[0]
    assert run_dir.name.startswith("mappo_smoke_s0__")

    # All expected files materialise.
    assert (run_dir / "input" / "config.json").is_file()
    assert (run_dir / "input" / "provenance.json").is_file()
    assert (run_dir / "output" / "metrics.json").is_file()
    assert (run_dir / "logs" / "run.log").is_file()
    assert (run_dir / "run_state.json").is_file()

    # Final run state on disk is DONE.
    assert "DONE" in (run_dir / "run_state.json").read_text(encoding="utf-8")
