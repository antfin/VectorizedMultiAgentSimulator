"""F2.4 + F5.4 integration tests: MappoAdapter Protocol + training + long-format CSV."""

from pathlib import Path

import pandas as pd
import pytest

from multi_scenario.adapters.algorithms.mappo import MappoAdapter
from multi_scenario.domain.models import ExperimentConfig
from multi_scenario.domain.ports import Algorithm


def test_mappo_implements_algorithm_protocol():
    """MappoAdapter satisfies the Algorithm port."""
    assert isinstance(MappoAdapter(), Algorithm)


def _smoke_config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "experiment": {"id": "mappo_smoke", "seed": 0},
            "scenario": {
                "type": "discovery",
                "params": {
                    "n_agents": 2,
                    "n_targets": 2,
                    "agents_per_target": 2,
                    "targets_respawn": False,
                    "shared_reward": True,
                    "max_steps": 20,
                },
            },
            "algorithm": {"type": "mappo", "params": {}},
            "training": {
                "max_iters": 2,
                "num_envs": 1,
                "device": "cpu",
                "frames_per_batch": 100,
                "minibatch_size": 50,
                "n_minibatch_iters": 1,
            },
            "evaluation": {"interval_iters": 1, "episodes": 2},
            "runtime": {
                "runner": {"type": "local", "params": {}},
                "storage": {
                    "type": "fs",
                    "path": str(tmp_path),
                    "params": {},
                },
            },
        }
    )


@pytest.mark.slow
def test_smoke_training_2_iter(tmp_path: Path):
    """train() completes 2 iters; evaluate() returns the documented rollout shape."""
    adapter = MappoAdapter()
    cfg = _smoke_config(tmp_path)

    # Pass run_dir so BenchMARL's native output stays inside tmp_path (else it
    # leaks to CWD — see §3.5.2). Pytest auto-cleans tmp_path after the test.
    artifact = adapter.train(env=None, cfg=cfg, run_dir=tmp_path)
    assert artifact is not None

    rollout = adapter.evaluate(artifact, env=None, cfg=cfg, run_dir=tmp_path)

    # Universal rollout fields, sized to evaluation.episodes, with real values.
    assert rollout["episode_returns"].shape[0] == cfg.evaluation.episodes
    assert rollout["episode_lengths"].shape[0] == cfg.evaluation.episodes
    assert rollout["episode_collisions"].shape[0] == cfg.evaluation.episodes
    # episode_lengths is the rollout horizon — strictly positive.
    assert (rollout["episode_lengths"] > 0).all()

    # Discovery-specific: targets_covered is now a real cumsum tensor.
    assert rollout["n_targets"] == 2
    tc = rollout["targets_covered"]
    assert tc.dim() == 2  # [n_episodes, T]
    assert tc.shape[0] == cfg.evaluation.episodes
    # Cumsum is monotone non-decreasing along T.
    assert (tc[:, 1:] >= tc[:, :-1]).all()


@pytest.mark.slow
def test_evaluate_writes_long_format_csv_when_flag_on(tmp_path: Path):
    """F5.4: with ``storage.params.long_format: true``, evaluate() writes eval_steps.csv."""
    adapter = MappoAdapter()
    cfg = _smoke_config(tmp_path)
    cfg.runtime.storage.params["long_format"] = True

    artifact = adapter.train(env=None, cfg=cfg, run_dir=tmp_path)
    adapter.evaluate(artifact, env=None, cfg=cfg, run_dir=tmp_path)

    out = tmp_path / "output" / "eval_steps.csv"
    assert out.is_file()
    df = pd.read_csv(out)
    # Universal columns + at least one action dim.
    assert {"env_idx", "step", "agent", "reward", "done", "terminated"} <= set(
        df.columns
    )
    assert any(c.startswith("action_d") for c in df.columns)
    # Rows = num_envs × T × n_agents (smoke: 1 × max_steps × 2).
    assert len(df) > 0


@pytest.mark.slow
def test_evaluate_does_not_write_long_format_when_flag_off(tmp_path: Path):
    """F5.4: without the flag, no eval_steps.csv produced."""
    adapter = MappoAdapter()
    cfg = _smoke_config(tmp_path)
    # No long_format flag set (default False).

    artifact = adapter.train(env=None, cfg=cfg, run_dir=tmp_path)
    adapter.evaluate(artifact, env=None, cfg=cfg, run_dir=tmp_path)

    assert not (tmp_path / "output" / "eval_steps.csv").exists()
