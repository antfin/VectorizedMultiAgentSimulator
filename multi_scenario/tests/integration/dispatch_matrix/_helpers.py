"""Dispatch-matrix test helpers — shared YAML factories + assertions.

The four (ER1, LERO) × (local, OVH) cells share a lot of fixture shape
and assertion logic. Centralising it keeps each test focused on the one
behaviour it pins.

Smoke-test budget: max_iters=1, frames_per_batch=50, episodes=1, 5 steps
per episode. End-to-end submission completes in ~10-15s wall on CPU.
"""

# pylint: disable=missing-function-docstring,too-many-arguments,too-many-positional-arguments

from pathlib import Path
from typing import Any

import yaml


# ── YAML factories ──────────────────────────────────────────────────


def er1_smoke_cfg(storage_path: str) -> dict[str, Any]:
    """Minimal valid ER1-style ExperimentConfig dict for smoke testing.

    Tight budgets (max_iters=1, frames_per_batch=50) keep wall-time
    under ~10s on CPU. ``record_video=False`` avoids pyglet on CI.
    """
    return {
        "experiment": {"id": "er1_smoke", "seed": 0},
        "scenario": {
            "type": "discovery",
            "params": {
                "n_agents": 2,
                "n_targets": 2,
                "agents_per_target": 2,
                "max_steps": 5,
                "covering_range": 0.35,
                "targets_respawn": False,
                "shared_reward": True,
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
            "runner": {"type": "local", "params": {"record_video": False}},
            "storage": {"type": "fs", "path": storage_path, "params": {}},
        },
    }


def lero_smoke_cfg(storage_path: str) -> dict[str, Any]:
    """Minimal valid LERO-style ExperimentConfig dict.

    Same tight budget as :func:`er1_smoke_cfg` plus LERO knobs:
    1 iter × 1 candidate × 100-frame inner eval. Uses the
    ``v2_fewshot_k2_local`` prompt version (renders cleanly with
    the default Discovery scenario.params).
    """
    cfg = er1_smoke_cfg(storage_path)
    cfg["experiment"]["id"] = "lero_smoke"
    # The Jinja prompt template needs these extra knobs to render.
    cfg["scenario"]["params"].update(
        {
            "n_lidar_rays_entities": 15,
            "n_lidar_rays_agents": 12,
            "use_agent_lidar": True,
            "lidar_range": 0.35,
            "agent_collision_penalty": -0.01,
            "covering_rew_coeff": 1.0,
            "time_penalty": -0.01,
        }
    )
    cfg["lero"] = {
        "prompt_version": "v2_fewshot_k2_local",
        "n_iterations": 1,
        "n_candidates": 1,
        "eval_frames_per_candidate": 100,
        "evolve_reward": True,
        "evolve_observation": False,
    }
    cfg["llm"] = {"model": "gpt-4o-mini"}
    return cfg


def ovh_smoke_cfg(base: dict[str, Any]) -> dict[str, Any]:
    """Flip a local-runner cfg to OVH for the dispatch-mocking tests."""
    out = {**base}
    out["runtime"] = {
        **base["runtime"],
        "runner": {"type": "ovh", "params": {"record_video": False}},
        # The OVH path mounts the bucket at /workspace/results; the
        # local-runner-on-host pre-validation would reject the path,
        # but the dispatch tests mock OvhRunner before that check.
        "storage": {"type": "fs", "path": "/workspace/results", "params": {}},
    }
    return out


def write_smoke_yaml(
    tmp_path: Path,
    cfg_dict: dict[str, Any],
    *,
    scenario: str = "discovery",
    folder: str = "baseline",
    name: str = "smoke.yaml",
) -> Path:
    """Stage a YAML under ``tmp_path/<scenario>/<folder>/configs/<name>``.

    Matches the layout the Submit page's picker expects (and the
    ``tmp_experiments`` fixture pattern in test_submit_page_e2e.py).
    """
    cfg_dir = tmp_path / scenario / folder / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = cfg_dir / name
    yaml_path.write_text(yaml.dump(cfg_dict, sort_keys=False), encoding="utf-8")
    return yaml_path


# ── Run-dir assertion helpers ───────────────────────────────────────


def assert_er1_run_dir_complete(run_dir: Path) -> None:
    """Verify an ER1 run produced all expected files in the canonical layout."""
    assert run_dir.is_dir(), f"run_dir missing: {run_dir}"
    assert (run_dir / "input" / "config.json").is_file(), "input/config.json absent"
    assert (
        run_dir / "input" / "provenance.json"
    ).is_file(), "input/provenance.json absent"
    assert (run_dir / "output" / "metrics.json").is_file(), "output/metrics.json absent"
    assert (run_dir / "output" / "report.json").is_file(), "output/report.json absent"
    assert (
        run_dir / "output" / "eval_episodes.json"
    ).is_file(), "output/eval_episodes.json absent"
    assert (run_dir / "run_state.json").is_file(), "run_state.json absent"


def assert_lero_run_dir_complete(run_dir: Path) -> None:
    """Verify a LERO run produced ER1-shape outputs PLUS the LERO sub-tree."""
    assert_er1_run_dir_complete(run_dir)
    lero_root = run_dir / "output" / "lero"
    assert lero_root.is_dir(), f"output/lero/ absent under {run_dir}"
    assert (
        lero_root / "final_summary.json"
    ).is_file(), "lero/final_summary.json absent"
    assert (
        lero_root / "evolution_history.json"
    ).is_file(), "lero/evolution_history.json absent"
    assert (lero_root / "evolution_doc.md").is_file(), "lero/evolution_doc.md absent"
    # The prompts/ folder gets created on the first per-iter write.
    assert (lero_root / "prompts").is_dir(), "lero/prompts/ absent"
    # iter_0 always exists for a 1-iter smoke.
    assert (lero_root / "prompts" / "iter_0").is_dir(), "lero/prompts/iter_0/ absent"
    assert (
        lero_root / "prompts" / "iter_0" / "system.md"
    ).is_file(), "lero/prompts/iter_0/system.md absent"
