"""F9.6.e — :class:`BenchmarlCandidateEvaluator` + :class:`BenchmarlFullTrainer`.

The fast tests pin: cfg adjustment, scenario-class injection,
non-BenchMARL-adapter rejection. The slow tests run a 1-iter smoke
through real BenchMARL to prove the end-to-end LERO eval loop works
with an LLM-generated reward function.
"""

# pylint: disable=missing-function-docstring,redefined-outer-name,protected-access

from pathlib import Path

import pytest

from multi_scenario.adapters.lero.benchmarl_evaluator import (
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


def test_scenario_env_fun_factory_is_picklable():
    """BenchMARL checkpoints the task pickled → factory must round-trip
    without dragging the live scenario class along."""
    # pylint: disable=import-outside-toplevel
    import pickle

    class _S:
        pass

    factory = ScenarioEnvFunFactory(_S, {"x": 1})
    blob = pickle.dumps(factory)
    restored = pickle.loads(blob)
    assert isinstance(restored, ScenarioEnvFunFactory)


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
