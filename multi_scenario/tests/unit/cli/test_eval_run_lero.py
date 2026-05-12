"""Phase 7 — LERO awareness in ``multi-scenario eval``.

Pure unit tests for the new helpers in :mod:`cli.eval_run`. The
end-to-end ``BenchMARL.reload_from_file`` path is exercised by the
existing CLI tests at ``tests/integration/cli/test_eval.py``; here we
just pin the helper contracts.
"""

# pylint: disable=missing-function-docstring

import json
from pathlib import Path

from multi_scenario.adapters.lero.scenario_env_fun_factory import (
    ScenarioEnvFunFactory,
)
from multi_scenario.cli.eval_run import (
    _build_experiment_patch,
    _install_patched_factory_for_lero_reload,
)
from multi_scenario.domain.models import ExperimentConfig


def _cfg(*, with_lero: bool, device: str = "cpu") -> ExperimentConfig:
    base = {
        "experiment": {"id": "x", "seed": 0},
        "scenario": {
            "type": "discovery",
            "params": {
                "n_agents": 2,
                "n_targets": 2,
                "agents_per_target": 2,
                "covering_range": 0.25,
                "n_lidar_rays_entities": 15,
                "n_lidar_rays_agents": 12,
                "use_agent_lidar": True,
                "lidar_range": 0.35,
            },
        },
        "algorithm": {"type": "mappo", "params": {}},
        "training": {"max_iters": 1, "device": device},
        "evaluation": {"interval_iters": 1, "episodes": 1},
    }
    if with_lero:
        base["lero"] = {
            "n_iterations": 1,
            "n_candidates": 1,
            "reward_clip": 50.0,
            "whitelist_strict": True,
        }
        base["llm"] = {"model": "gpt-4o-mini"}
    return ExperimentConfig.model_validate(base)


def test_build_experiment_patch_overrides_devices_for_cpu_host(tmp_path: Path):
    """CPU-only host → all BenchMARL device fields routed to cpu so torch.load
    doesn't try to materialise CUDA tensors that don't exist locally."""
    (tmp_path / "output" / "benchmarl").mkdir(parents=True)
    patch = _build_experiment_patch(tmp_path, _cfg(with_lero=True, device="cpu"))
    assert patch["sampling_device"] == "cpu"
    assert patch["train_device"] == "cpu"
    assert patch["buffer_device"] == "cpu"
    assert patch["restore_map_location"] == "cpu"
    # loggers cleared so eval doesn't clobber training-time scalars.
    assert patch["loggers"] == []


def test_build_experiment_patch_save_folder_points_to_local_benchmarl_dir(
    tmp_path: Path,
):
    """The patched ``save_folder`` is the LOCAL parent dir of the exp folder
    — overrides the container-side ``/workspace/results`` from the pickle."""
    bench_root = tmp_path / "output" / "benchmarl"
    (bench_root / "mappo_xyz").mkdir(parents=True)
    patch = _build_experiment_patch(tmp_path, _cfg(with_lero=True))
    assert patch["save_folder"] == str(bench_root.resolve())


def test_install_patched_factory_is_noop_when_no_lero_trace(tmp_path: Path):
    """No final_summary.json → helper returns silently (legacy non-LERO runs)."""
    cfg = _cfg(with_lero=True)
    _install_patched_factory_for_lero_reload(tmp_path, cfg)
    # No exception; nothing on disk to patch from.


def test_install_patched_factory_restores_state_via_setstate(tmp_path: Path):
    """With a winning candidate on disk, ``__setstate__`` injects the
    patched class + cfg.scenario.params into any unpickled factory.

    This makes ``multi-scenario eval`` work on Phase 5a legacy runs
    whose pickled factory had the dummy-state bug (Phase 2 regression).
    """
    # Stage minimal final_summary.json + evolution_history.json on disk.
    lero_dir = tmp_path / "output" / "lero"
    lero_dir.mkdir(parents=True)
    summary = {
        "exp_id": "test",
        "seed": 0,
        "n_iterations_completed": 1,
        "n_candidates_total": 1,
        "total_cost_usd": 0.0,
        "best_candidate_metrics": {},
        "best_candidate_verdict": "progress",
        "fallback_chain": [
            {
                "rank": 0,
                "iteration": 0,
                "candidate_idx": 0,
                "eval_metrics": {},
                "outcome": "success",
                "error": None,
                "full_train_metrics": None,
            }
        ],
        "full_training_succeeded": True,
        "best_candidate_full_metrics": None,
    }
    history = [
        {
            "candidate": {
                "iteration": 0,
                "candidate_idx": 0,
                "code": {
                    "reward_source": None,
                    "obs_source": "import torch\ndef enhance_observation(s): return torch.zeros((1,1))\n",
                },
            },
            "metrics": {},
            "verdict": "progress",
            "note": "",
        }
    ]
    (lero_dir / "final_summary.json").write_text(json.dumps(summary))
    (lero_dir / "evolution_history.json").write_text(json.dumps(history))

    cfg = _cfg(with_lero=True)
    _install_patched_factory_for_lero_reload(tmp_path, cfg)

    # Now pickle round-trip a "broken" factory (state stripped) — the
    # restore helper installed __setstate__ that recovers from on-disk
    # metadata.
    factory = ScenarioEnvFunFactory.__new__(ScenarioEnvFunFactory)
    factory.__setstate__({})  # simulates legacy dummy-state pickle
    assert factory.scenario_class is not None
    assert factory.scenario_class.__name__ == "PatchedDiscoveryScenario"
    # cfg.scenario.params injected (minus the prompt-only obs_lidar_agents key).
    assert "covering_range" in factory.config
    assert "obs_lidar_agents" not in factory.config


def test_install_patched_factory_skips_when_no_success_in_chain(tmp_path: Path):
    """All ranks crashed → no winning code to load → no-op."""
    lero_dir = tmp_path / "output" / "lero"
    lero_dir.mkdir(parents=True)
    (lero_dir / "final_summary.json").write_text(
        json.dumps(
            {
                "exp_id": "x",
                "seed": 0,
                "n_iterations_completed": 1,
                "n_candidates_total": 0,
                "total_cost_usd": 0.0,
                "best_candidate_metrics": {},
                "best_candidate_verdict": "invalid",
                "fallback_chain": [
                    {
                        "rank": 0,
                        "iteration": 0,
                        "candidate_idx": 0,
                        "eval_metrics": {},
                        "outcome": "crashed",
                        "error": "OOM",
                        "full_train_metrics": None,
                    }
                ],
                "full_training_succeeded": False,
                "best_candidate_full_metrics": None,
            }
        )
    )
    (lero_dir / "evolution_history.json").write_text("[]")
    cfg = _cfg(with_lero=True)
    # Restore the default __setstate__ so this test's no-op assertion
    # isn't polluted by other tests in the same process.
    original_setstate = ScenarioEnvFunFactory.__setstate__
    _install_patched_factory_for_lero_reload(tmp_path, cfg)
    # Helper returned early; __setstate__ unchanged.
    assert ScenarioEnvFunFactory.__setstate__ is original_setstate
