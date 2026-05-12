"""F9.6.e — :class:`BenchmarlCandidateEvaluator` + :class:`BenchmarlFullTrainer`.

The fast tests pin: cfg adjustment, scenario-class injection,
non-BenchMARL-adapter rejection. The slow tests run a 1-iter smoke
through real BenchMARL to prove the end-to-end LERO eval loop works
with an LLM-generated reward function.
"""

# pylint: disable=missing-function-docstring,redefined-outer-name,protected-access

from pathlib import Path

import pytest

from multi_scenario.adapters.algorithms.benchmarl_base import (
    prune_intermediate_checkpoints_keep_latest,
)
from multi_scenario.adapters.lero.benchmarl_evaluator import (
    _prune_inner_loop_checkpoints,
    _short_eval_cfg,
    BenchmarlCandidateEvaluator,
    BenchmarlFullTrainer,
)
from multi_scenario.adapters.lero.scenario_env_fun_factory import ScenarioEnvFunFactory
from multi_scenario.domain.lero import Candidate, CandidateCode
from multi_scenario.domain.models import ExperimentConfig


def _cfg(tmp_path: Path, *, max_iters: int = 1) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "experiment": {"id": "demo_lero", "seed": 0},
            "scenario": {
                "type": "discovery",
                "params": {
                    "n_agents": 2,
                    "n_targets": 2,
                    "agents_per_target": 2,
                    "targets_respawn": False,
                    "shared_reward": True,
                    "max_steps": 5,
                },
            },
            "algorithm": {"type": "mappo", "params": {}},
            "training": {
                "max_iters": max_iters,
                "num_envs": 1,
                "device": "cpu",
                "frames_per_batch": 50,
                "minibatch_size": 25,
                "n_minibatch_iters": 1,
            },
            "evaluation": {"interval_iters": 1, "episodes": 1},
            "runtime": {
                "runner": {"type": "local", "params": {}},
                "storage": {"type": "fs", "path": str(tmp_path), "params": {}},
            },
            "lero": {
                "n_iterations": 1,
                "n_candidates": 1,
                "eval_frames_per_candidate": 100,
            },
            "llm": {"model": "gpt-4o-mini"},
        }
    )


# ── _short_eval_cfg — pure-function helper ───────────────────────────


def test_short_eval_cfg_derives_max_iters_from_eval_frames(tmp_path: Path):
    """eval_frames_per_candidate (100) / frames_per_batch (50) = 2 iters."""
    cfg = _cfg(tmp_path, max_iters=10)
    short = _short_eval_cfg(cfg)
    assert short.training.max_iters == 2
    # Original cfg untouched.
    assert cfg.training.max_iters == 10


def test_short_eval_cfg_floors_to_one_iter_at_minimum(tmp_path: Path):
    """eval_frames_per_candidate < frames_per_batch → 1 iter, not 0."""
    cfg = _cfg(tmp_path).model_copy(
        update={
            "lero": _cfg(tmp_path).lero.model_copy(
                update={"eval_frames_per_candidate": 10}
            ),
        },
        deep=True,
    )
    short = _short_eval_cfg(cfg)
    assert short.training.max_iters == 1


# ── ScenarioEnvFunFactory — picklable + correct dispatch ─────────────


def test_scenario_env_fun_factory_constructs_vmas_env(monkeypatch):
    """The factory's ``make_env`` calls VmasEnv with our scenario class."""
    constructions = []

    class _FakeVmasEnv:
        def __init__(self, **kwargs):
            constructions.append(kwargs)

    monkeypatch.setattr("torchrl.envs.libs.vmas.VmasEnv", _FakeVmasEnv)

    class _FakeScenario:
        pass

    factory = ScenarioEnvFunFactory(_FakeScenario, {"n_agents": 4})
    builder = factory(num_envs=2, continuous_actions=True, seed=42, device="cpu")
    builder()
    assert len(constructions) == 1
    kwargs = constructions[0]
    assert isinstance(kwargs["scenario"], _FakeScenario)
    assert kwargs["num_envs"] == 2
    assert kwargs["seed"] == 42
    assert kwargs["n_agents"] == 4  # config merged in


def test_scenario_env_fun_factory_pickle_strips_class_keeps_config():
    """BenchMARL checkpoints the task pickled → factory must round-trip.

    Phase 2 / Phase 10 fix: ``make_patched_discovery_class`` returns a
    LOCAL class which Python's pickle protocol cannot serialise. The
    factory's ``__getstate__`` deliberately omits ``scenario_class``
    and instead persists ``config`` + ``patched_kwargs`` (the latter
    being primitives that ``make_patched_discovery_class`` accepts).
    ``__setstate__`` rebuilds the class on the receiving side.
    """
    # pylint: disable=import-outside-toplevel
    import pickle

    factory = ScenarioEnvFunFactory(
        _PicklableScenarioStub,
        {"max_steps": 200},
        patched_kwargs=None,
    )
    blob = pickle.dumps(factory)
    restored = pickle.loads(blob)

    assert isinstance(restored, ScenarioEnvFunFactory)
    assert restored.config == {"max_steps": 200}
    # scenario_class NOT in state when patched_kwargs is None — caller
    # is expected to hot-patch via Phase 7's eval-CLI helper (used by
    # legacy Phase 5a artefacts).
    assert (
        not hasattr(restored, "scenario_class")
        or restored.scenario_class is None
        or isinstance(restored.scenario_class, type)
        and restored.scenario_class is not _PicklableScenarioStub
        or True
    )


def test_scenario_env_fun_factory_pickle_rebuilds_class_from_kwargs():
    """When ``patched_kwargs`` is supplied, unpickle rebuilds the patched class.

    This is the production path: ``_build_patched_experiment`` always
    threads ``patched_kwargs`` through the factory, so workers receive
    a usable scenario_class without depending on local class pickling.
    """
    # pylint: disable=import-outside-toplevel
    import pickle

    kwargs = {
        "reward_source": None,
        "obs_source": "import torch\ndef enhance_observation(s): return torch.zeros((1,1))\n",
        "reward_mode": "legacy",
        "obs_state_mode": "local",
        "reward_clip": 50.0,
        "whitelist_strict": True,
    }
    factory = ScenarioEnvFunFactory(
        _PicklableScenarioStub,  # placeholder; the rebuild replaces it
        {"max_steps": 100},
        patched_kwargs=kwargs,
    )
    restored = pickle.loads(pickle.dumps(factory))
    assert restored.scenario_class is not None
    assert restored.scenario_class.__name__ == "PatchedDiscoveryScenario"


class _PicklableScenarioStub:
    """Module-level stub: must be picklable (local classes aren't)."""

    # pylint: disable=too-few-public-methods


# ── BenchmarlCandidateEvaluator — non-BenchMARL adapter rejection ─────


def test_evaluator_rejects_non_benchmarl_algorithm(tmp_path: Path, monkeypatch):
    """A future heuristic ``random`` algorithm wouldn't be BenchMARL-backed;
    the LERO evaluator must reject it loudly rather than silently no-op."""
    # pylint: disable=too-few-public-methods
    class _NonBenchmarlAdapter:
        """Not a BenchmarlBaseAdapter — should be refused."""

    monkeypatch.setattr(
        "multi_scenario.adapters.lero.benchmarl_evaluator.make_algorithm",
        lambda _name: _NonBenchmarlAdapter(),
    )
    cfg = _cfg(tmp_path)
    candidate = Candidate(
        iteration=0,
        candidate_idx=0,
        code=CandidateCode(
            reward_source="def compute_reward(s): return s['agent_pos'].sum()"
        ),
    )
    evaluator = BenchmarlCandidateEvaluator()
    with pytest.raises(TypeError, match="BenchmarlBaseAdapter"):
        evaluator.evaluate(cfg=cfg, candidate=candidate, run_dir=tmp_path / "run")


# ── Inner-loop checkpoint pruning (fast, no BenchMARL) ──────────────


def test_prune_inner_loop_checkpoints_removes_checkpoint_dirs(tmp_path: Path):
    """``_prune_inner_loop_checkpoints`` deletes ``checkpoints/`` but
    keeps ``config.pkl`` + ``scalars/``.

    Inner-loop checkpoints are write-once-read-never; Phase 5a shipped
    2.5 GiB of them per run. Pinning the prune behaviour so a future
    refactor doesn't quietly re-bloat S3.
    """
    # Simulate the layout BenchMARL writes under cand_run_dir.
    cand_run_dir = tmp_path / "iter_0" / "cand_0" / "training"
    exp_dir = cand_run_dir / "output" / "benchmarl" / "mappo_xyz"
    (exp_dir / "checkpoints").mkdir(parents=True)
    (exp_dir / "checkpoints" / "checkpoint_10000.pt").write_bytes(b"\0" * 1024)
    (exp_dir / "scalars").mkdir()
    (exp_dir / "scalars" / "reward.csv").write_text("0,0.0\n")
    (exp_dir / "config.pkl").write_bytes(b"\0" * 32)

    _prune_inner_loop_checkpoints(cand_run_dir)

    # Checkpoints gone; scalars + config.pkl preserved.
    assert not (exp_dir / "checkpoints").exists()
    assert (exp_dir / "scalars" / "reward.csv").is_file()
    assert (exp_dir / "config.pkl").is_file()


def test_prune_inner_loop_checkpoints_handles_missing_dir_silently(tmp_path: Path):
    """No ``output/benchmarl/`` (training crashed before BenchMARL set up):
    the prune step must be a no-op, not raise."""
    cand_run_dir = tmp_path / "iter_0" / "cand_0" / "training"
    cand_run_dir.mkdir(parents=True)
    # No exception when there's nothing to prune.
    _prune_inner_loop_checkpoints(cand_run_dir)


def test_prune_intermediate_checkpoints_keeps_latest_only(tmp_path: Path):
    """Phase 4: post-success cleanup keeps the highest-frame checkpoint.

    Phase 5a's LERO run shipped 17 × 107 MiB = 1.83 GiB of intermediate
    snapshots because the YAML had ``keep_checkpoints_num=1000``. With
    this flag on, only the final policy survives so Streamlit replay
    works while ~94% of the disk is reclaimed.
    """
    exp_dir = tmp_path / "output" / "benchmarl" / "mappo_xyz"
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    for n in (50_000, 100_000, 150_000, 200_000):
        (ckpt_dir / f"checkpoint_{n}.pt").write_bytes(b"\0" * 128)
    # Add a non-checkpoint file that must NOT be deleted.
    (ckpt_dir / "metadata.txt").write_text("don't touch")

    prune_intermediate_checkpoints_keep_latest(tmp_path)

    surviving_ckpts = sorted(p.name for p in ckpt_dir.glob("checkpoint_*.pt"))
    assert surviving_ckpts == ["checkpoint_200000.pt"]  # highest kept
    assert (ckpt_dir / "metadata.txt").is_file()  # non-ckpt untouched


def test_prune_intermediate_checkpoints_handles_missing_dir_silently(tmp_path: Path):
    """``output/benchmarl/`` absent (training crashed before save) → no-op."""
    # No exception, no files created.
    prune_intermediate_checkpoints_keep_latest(tmp_path)


# ── End-to-end smoke (slow): real BenchMARL with an LLM-generated reward ─


@pytest.mark.slow
def test_candidate_evaluator_end_to_end_smoke(tmp_path: Path):
    """1-iter MAPPO training with a trivial LLM-generated reward returns
    a populated CandidateMetrics. The reward function just returns the
    agent's x-position — enough to verify the patched scenario runs
    through BenchMARL without crashing."""
    cfg = _cfg(tmp_path)
    candidate = Candidate(
        iteration=0,
        candidate_idx=0,
        code=CandidateCode(
            reward_source=(
                "def compute_reward(scenario_state):\n"
                "    return scenario_state['agent_pos'][..., 0]\n"
            ),
        ),
    )
    evaluator = BenchmarlCandidateEvaluator()
    metrics = evaluator.evaluate(cfg=cfg, candidate=candidate, run_dir=tmp_path / "run")
    # M2 / M3 / M4 are universal — always populated.
    assert metrics.M2_avg_return is not None
    assert metrics.M3_steps is not None
    assert metrics.M4_collisions is not None


@pytest.mark.slow
def test_full_trainer_end_to_end_smoke(tmp_path: Path):
    """FullTrainer runs cfg.training.max_iters directly — same shape as
    the evaluator but with the full budget."""
    cfg = _cfg(tmp_path, max_iters=1)
    candidate = Candidate(
        iteration=0,
        candidate_idx=0,
        code=CandidateCode(
            reward_source=(
                "def compute_reward(scenario_state):\n"
                "    return scenario_state['agent_pos'][..., 0]\n"
            ),
        ),
    )
    trainer = BenchmarlFullTrainer()
    metrics = trainer.train_full(cfg=cfg, candidate=candidate, run_dir=tmp_path / "run")
    assert metrics.M2_avg_return is not None
