"""F9.7.A — :class:`MetaPromptComposer` stub contract.

The stub proves the seam holds: the orchestrator + default-adapter
factory work end-to-end with a non-default composer. F9.7.B's full
round-table replaces the stub without orchestrator changes.
"""

# pylint: disable=missing-function-docstring,redefined-outer-name

from pathlib import Path

import pytest
from multi_scenario.adapters.lero import FilesystemTraceWriter

from multi_scenario.adapters.llm import FakeLlmClient
from multi_scenario.adapters.prompt_composers import MetaPromptComposer
from multi_scenario.adapters.prompts import JinjaPromptRenderer
from multi_scenario.application.lero_orchestrator import LeroOrchestrator
from multi_scenario.domain.lero import CandidateMetrics, LlmCompletion, StrategyCard
from multi_scenario.domain.models import ExperimentConfig


_VALID_REWARD = (
    "```python\nimport torch\n"
    "def compute_reward(scenario_state):\n"
    "    return scenario_state['agent_pos'].sum()\n"
    "```"
)


@pytest.fixture
def composer():
    return MetaPromptComposer(
        renderer=JinjaPromptRenderer(),
        prompt_version="v2_fewshot_k2_local",
        n_candidates=2,
    )


@pytest.fixture
def task_params():
    return {
        "n_agents": 4,
        "n_targets": 4,
        "agents_per_target": 2,
        "covering_range": 0.35,
        "n_lidar_rays_entities": 15,
        "n_lidar_rays_agents": 12,
        "obs_lidar_agents": "lidar_agents: …",
    }


# ── Placeholder injection ────────────────────────────────────────────


def test_compose_appends_placeholder_marker(composer, task_params):
    out = composer.compose(iteration=0, history=[], task_params=task_params)
    last = out.messages[-1]["content"]
    assert "[meta-prompt placeholder]" in last


def test_compose_preserves_initial_user_body(composer, task_params):
    """Stub mutation is purely additive — the original initial_user
    body is fully present, just with the marker appended."""
    out = composer.compose(iteration=0, history=[], task_params=task_params)
    last = out.messages[-1]["content"]
    assert "4 agents" in last  # from the initial_user template


def test_render_context_records_meta_placeholder_flag(composer, task_params):
    """Trace files should reflect that this was a meta-composer call."""
    out = composer.compose(iteration=0, history=[], task_params=task_params)
    assert out.render_context.get("meta_placeholder") is True


def test_compose_accepts_strategy_card_without_crashing(composer, task_params):
    """F9.7.B will consume the card; F9.7.A just passes it through."""
    out = composer.compose(
        iteration=0,
        history=[],
        task_params=task_params,
        strategy_card=StrategyCard(rationale="ignored for now"),
    )
    assert out.messages  # rendered without error


# ── Orchestrator end-to-end with the meta stub ───────────────────────


class _FakeLogger:
    # pylint: disable=missing-function-docstring,unused-argument
    def info(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        pass

    def debug(self, msg):
        pass


class _Evaluator:
    def evaluate(self, *, cfg, candidate, run_dir):  # noqa: ARG002
        return CandidateMetrics(M1_success_rate=0.5)


class _FullTrainer:
    def train_full(self, *, cfg, candidate, run_dir):  # noqa: ARG002
        return CandidateMetrics(M1_success_rate=1.0)


def test_orchestrator_runs_end_to_end_with_meta_composer(tmp_path: Path):
    """The whole point of F9.7.A: prove the seam.

    Run the orchestrator with the meta-prompt composer and assert the
    placeholder lands in the persisted prompt.json traces.
    """
    cfg = ExperimentConfig.model_validate(
        {
            "experiment": {"id": "meta_test", "seed": 0},
            "scenario": {
                "type": "discovery",
                "params": {
                    "n_agents": 4,
                    "n_targets": 4,
                    "agents_per_target": 2,
                    "covering_range": 0.35,
                    "n_lidar_rays_entities": 15,
                    "n_lidar_rays_agents": 12,
                    "obs_lidar_agents": "lidar_agents: …",
                },
            },
            "algorithm": {"type": "mappo", "params": {}},
            "training": {"max_iters": 1, "device": "cpu"},
            "evaluation": {"interval_iters": 1, "episodes": 1},
            "runtime": {
                "runner": {"type": "local", "params": {}},
                "storage": {"type": "fs", "path": str(tmp_path), "params": {}},
            },
            "lero": {
                "n_iterations": 1,
                "n_candidates": 1,
                "meta_prompting": True,
            },
            "llm": {"model": "gpt-4o-mini"},
        }
    )
    run_dir = tmp_path / "run"
    orch = LeroOrchestrator(
        llm=FakeLlmClient().register_always(LlmCompletion(text=_VALID_REWARD)),
        composer=MetaPromptComposer(
            renderer=JinjaPromptRenderer(),
            prompt_version="v2_fewshot_k2_local",
            n_candidates=1,
        ),
        trace_writer=FilesystemTraceWriter(),
        evaluator=_Evaluator(),
        full_trainer=_FullTrainer(),
        logger=_FakeLogger(),
    )
    summary = orch.run(cfg=cfg, run_dir=run_dir)
    assert summary.n_candidates_total == 1

    # Verify the persisted prompt trace carries the placeholder.
    prompt_path = (
        run_dir / "output" / "lero" / "iter_0" / "cand_0" / "attempt_0" / "prompt.json"
    )
    assert prompt_path.is_file()
    blob = prompt_path.read_text("utf-8")
    assert "[meta-prompt placeholder]" in blob


# ── Factory routes by cfg.lero.meta_prompting flag ────────────────────


def test_factory_picks_default_composer_when_meta_prompting_false(tmp_path: Path):
    from multi_scenario.adapters.prompt_composers import InitialAndFeedbackComposer
    from multi_scenario.application.lero_factory import build_default_lero_orchestrator

    cfg = ExperimentConfig.model_validate(
        {
            "experiment": {"id": "x", "seed": 0},
            "scenario": {"type": "discovery", "params": {}},
            "algorithm": {"type": "mappo", "params": {}},
            "training": {"max_iters": 1, "device": "cpu"},
            "evaluation": {"interval_iters": 1, "episodes": 1},
            "runtime": {
                "runner": {"type": "local", "params": {}},
                "storage": {"type": "fs", "path": str(tmp_path), "params": {}},
            },
            "lero": {"meta_prompting": False, "n_iterations": 1, "n_candidates": 1},
            "llm": {"model": "gpt-4o-mini"},
        }
    )
    orch = build_default_lero_orchestrator(cfg=cfg, logger=_FakeLogger())
    assert isinstance(
        orch._composer, InitialAndFeedbackComposer
    )  # pylint: disable=protected-access


def test_factory_picks_meta_composer_when_flag_true(tmp_path: Path):
    from multi_scenario.application.lero_factory import build_default_lero_orchestrator

    cfg = ExperimentConfig.model_validate(
        {
            "experiment": {"id": "x", "seed": 0},
            "scenario": {"type": "discovery", "params": {}},
            "algorithm": {"type": "mappo", "params": {}},
            "training": {"max_iters": 1, "device": "cpu"},
            "evaluation": {"interval_iters": 1, "episodes": 1},
            "runtime": {
                "runner": {"type": "local", "params": {}},
                "storage": {"type": "fs", "path": str(tmp_path), "params": {}},
            },
            "lero": {"meta_prompting": True, "n_iterations": 1, "n_candidates": 1},
            "llm": {"model": "gpt-4o-mini"},
        }
    )
    orch = build_default_lero_orchestrator(cfg=cfg, logger=_FakeLogger())
    assert isinstance(
        orch._composer, MetaPromptComposer
    )  # pylint: disable=protected-access
