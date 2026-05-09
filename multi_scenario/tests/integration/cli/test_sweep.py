"""F5.6 integration tests: ``multi-scenario sweep <input>``."""

from pathlib import Path

import pytest
import yaml

from multi_scenario.cli import app
from typer.testing import CliRunner


def _smoke_config_dict(exp_id: str, seed: int, storage_path: Path) -> dict:
    return {
        "experiment": {"id": exp_id, "seed": seed},
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
            "storage": {"type": "fs", "path": str(storage_path), "params": {}},
        },
    }


def _write_yaml(path: Path, exp_id: str, seed: int, storage_path: Path) -> Path:
    payload = yaml.safe_dump(_smoke_config_dict(exp_id, seed, storage_path))
    path.write_text(payload, encoding="utf-8")
    return path


def test_sweep_dry_run_single_yaml_with_seeds(tmp_path: Path) -> None:
    """Single yaml × 3 seeds → dry-run prints 3 expanded cells."""
    cfg_path = _write_yaml(tmp_path / "smoke.yaml", "x_smoke", 0, tmp_path)

    result = CliRunner().invoke(
        app, ["sweep", str(cfg_path), "--seeds", "0,1,2", "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    # All three seed-tagged exp_ids appear in the preview.
    assert "x_smoke_s0" in result.output
    assert "x_smoke_s1" in result.output
    assert "x_smoke_s2" in result.output
    # No actual run folders were created.
    assert not any(
        p.name.startswith("x_smoke_s") for p in tmp_path.iterdir() if p.is_dir()
    )


def test_sweep_dry_run_directory_input(tmp_path: Path) -> None:
    """Directory input → glob *.yaml; preview shows one cell per yaml."""
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    _write_yaml(configs_dir / "a_smoke.yaml", "a_smoke", 0, tmp_path)
    _write_yaml(configs_dir / "b_smoke.yaml", "b_smoke", 0, tmp_path)
    _write_yaml(configs_dir / "c_smoke.yaml", "c_smoke", 0, tmp_path)
    # Non-yaml file should be ignored.
    (configs_dir / "readme.txt").write_text("hi", encoding="utf-8")

    result = CliRunner().invoke(app, ["sweep", str(configs_dir), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "a_smoke_s0" in result.output
    assert "b_smoke_s0" in result.output
    assert "c_smoke_s0" in result.output


def test_sweep_dry_run_glob_pattern(tmp_path: Path) -> None:
    """Glob pattern selects only matching yamls."""
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    _write_yaml(configs_dir / "abs_one.yaml", "abs_one", 0, tmp_path)
    _write_yaml(configs_dir / "abs_two.yaml", "abs_two", 0, tmp_path)
    _write_yaml(configs_dir / "other.yaml", "other", 0, tmp_path)

    result = CliRunner().invoke(
        app, ["sweep", str(configs_dir / "abs_*.yaml"), "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "abs_one_s0" in result.output
    assert "abs_two_s0" in result.output
    assert "other_s0" not in result.output


def test_sweep_directory_with_seeds_cartesian(tmp_path: Path) -> None:
    """Directory × seeds → cartesian (2 yamls × 3 seeds = 6 cells)."""
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    _write_yaml(configs_dir / "a_smoke.yaml", "a_smoke", 0, tmp_path)
    _write_yaml(configs_dir / "b_smoke.yaml", "b_smoke", 0, tmp_path)

    result = CliRunner().invoke(
        app, ["sweep", str(configs_dir), "--seeds", "0,1,2", "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    # 6 cells: 2 yamls × 3 seeds.
    assert (
        "6 cell" in result.output or "6 runs" in result.output or "(6)" in result.output
    )
    expected_ids = (
        "a_smoke_s0",
        "a_smoke_s1",
        "a_smoke_s2",
        "b_smoke_s0",
        "b_smoke_s1",
        "b_smoke_s2",
    )
    for exp_id in expected_ids:
        assert exp_id in result.output


def test_sweep_size_cap_exceeded_exits_nonzero(tmp_path: Path) -> None:
    """Expansion exceeding --max-runs exits non-zero with a helpful message."""
    cfg_path = _write_yaml(tmp_path / "smoke.yaml", "x_smoke", 0, tmp_path)

    # 5 seeds with max-runs=3 → cap exceeded.
    result = CliRunner().invoke(
        app,
        [
            "sweep",
            str(cfg_path),
            "--seeds",
            "0,1,2,3,4",
            "--max-runs",
            "3",
            "--dry-run",
        ],
    )
    assert result.exit_code == 2
    assert "5" in result.output  # actual cell count
    assert "3" in result.output  # cap


def test_sweep_no_matches_exits_nonzero(tmp_path: Path) -> None:
    """Glob pattern with zero matches → non-zero exit."""
    result = CliRunner().invoke(app, ["sweep", str(tmp_path / "*.yaml"), "--dry-run"])
    assert result.exit_code != 0
    assert "no" in result.output.lower() or "0" in result.output


def test_sweep_seconds_per_run_prints_estimate(tmp_path: Path) -> None:
    """``--seconds-per-run`` produces a wall-time estimate string."""
    cfg_path = _write_yaml(tmp_path / "smoke.yaml", "x_smoke", 0, tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "sweep",
            str(cfg_path),
            "--seeds",
            "0,1,2,3",
            "--seconds-per-run",
            "30",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    # 4 cells × 30 sec = 120 sec ≈ 2 min — must reflect the multiplication somewhere.
    assert "120" in result.output or "2 min" in result.output or "2:00" in result.output


@pytest.mark.slow
def test_sweep_executes_each_cell_when_not_dry_run(tmp_path: Path) -> None:
    """Without --dry-run: 1 yaml × 2 seeds → 2 run folders produced under storage path."""
    cfg_path = _write_yaml(tmp_path / "smoke.yaml", "x_smoke", 0, tmp_path)

    result = CliRunner().invoke(app, ["sweep", str(cfg_path), "--seeds", "0,1"])
    assert result.exit_code == 0, result.output

    # Two run folders should exist with the expected prefix patterns.
    s0_dirs = [
        p
        for p in tmp_path.iterdir()
        if p.is_dir() and p.name.startswith("x_smoke_s0__")
    ]
    s1_dirs = [
        p
        for p in tmp_path.iterdir()
        if p.is_dir() and p.name.startswith("x_smoke_s1__")
    ]
    assert len(s0_dirs) == 1
    assert len(s1_dirs) == 1
    # Each has the §3.5.2 layout.
    for run_dir in (s0_dirs[0], s1_dirs[0]):
        assert (run_dir / "input" / "config.json").is_file()
        assert (run_dir / "output" / "metrics.json").is_file()
        assert (run_dir / "run_state.json").is_file()
