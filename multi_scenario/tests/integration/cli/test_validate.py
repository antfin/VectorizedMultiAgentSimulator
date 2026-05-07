"""F5.1 integration tests: ``multi-scenario validate <yaml>``."""

from pathlib import Path

import yaml
from typer.testing import CliRunner

from multi_scenario.cli import app


def _valid_config_dict() -> dict:
    return {
        "experiment": {"id": "ok_smoke", "seed": 0},
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
            "storage": {"type": "fs", "path": "/tmp/x", "params": {}},
        },
    }


def test_validate_returns_zero_for_valid_yaml(tmp_path: Path) -> None:
    """Valid YAML: exit 0 with an OK confirmation message."""
    cfg_path = tmp_path / "good.yaml"
    cfg_path.write_text(yaml.safe_dump(_valid_config_dict()), encoding="utf-8")

    result = CliRunner().invoke(app, ["validate", str(cfg_path)])
    assert result.exit_code == 0, result.output
    assert "OK" in result.output


def test_validate_returns_nonzero_on_missing_required_field(tmp_path: Path) -> None:
    """Missing required field: exit 1 with the field path in the error output."""
    raw = _valid_config_dict()
    del raw["experiment"]["id"]  # drop required field
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    result = CliRunner().invoke(app, ["validate", str(cfg_path)])
    assert result.exit_code == 1
    # Error output points to the offending field path.
    assert "experiment.id" in result.output


def test_validate_returns_nonzero_on_unknown_field(tmp_path: Path) -> None:
    """Unknown field (extra='forbid'): exit 1 with the field path called out."""
    raw = _valid_config_dict()
    raw["experiment"]["typo_field"] = "oops"
    cfg_path = tmp_path / "extra.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    result = CliRunner().invoke(app, ["validate", str(cfg_path)])
    assert result.exit_code == 1
    assert "typo_field" in result.output


def test_validate_returns_nonzero_on_wrong_type(tmp_path: Path) -> None:
    """Wrong type: exit 1 with the field path."""
    raw = _valid_config_dict()
    raw["experiment"]["seed"] = "not_an_int"
    cfg_path = tmp_path / "wrong_type.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    result = CliRunner().invoke(app, ["validate", str(cfg_path)])
    assert result.exit_code == 1
    assert "experiment.seed" in result.output


def test_validate_with_missing_yaml_returns_nonzero() -> None:
    """Pointing at a non-existent file exits non-zero."""
    result = CliRunner().invoke(app, ["validate", "/tmp/does_not_exist.yaml"])
    assert result.exit_code != 0
