"""F9.6.b — :class:`LeroOrchestrator` end-to-end contract.

Exercises the full loop without torch / BenchMARL / VMAS by injecting
fake adapters for :class:`LlmClient`, :class:`CandidateEvaluator`, and
:class:`FullTrainer`. The byte-parity tests for codegen + prompts are
isolated to F9.4 / F9.2 — this file is about the orchestrator's
control flow (iteration loop, cost-cap exit, fallback chain).
"""

# pylint: disable=missing-function-docstring,redefined-outer-name

from pathlib import Path

import pytest
from multi_scenario.adapters.lero import FilesystemTraceWriter

from multi_scenario.adapters.llm import FakeLlmClient
from multi_scenario.adapters.prompt_composers import InitialAndFeedbackComposer
from multi_scenario.adapters.prompts import JinjaPromptRenderer
from multi_scenario.application.lero_orchestrator import LeroOrchestrator
from multi_scenario.domain.lero import (
    CandidateMetrics,
    LlmCompletion,
    LlmCostCapExceeded,
)
from multi_scenario.domain.models import ExperimentConfig


# ── shared fixtures ──────────────────────────────────────────────────


def _cfg(tmp_path: Path, *, n_iter: int = 2, n_cand: int = 2) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "experiment": {"id": "test_lero", "seed": 0},
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
            "lero": {"n_iterations": n_iter, "n_candidates": n_cand},
            "llm": {"model": "gpt-4o-mini"},
        }
    )


_VALID_REWARD = (
    "```python\nimport torch\n"
    "def compute_reward(scenario_state):\n"
    "    return scenario_state['agent_pos'].sum()\n"
    "```"
)


class _FakeLogger:
    """Capture log lines so tests can assert on warnings."""

    def __init__(self) -> None:
        self.lines: list[tuple[str, str]] = []

    def info(self, msg: str) -> None:
        self.lines.append(("info", msg))

    def warning(self, msg: str) -> None:
        self.lines.append(("warning", msg))

    def error(self, msg: str) -> None:
        self.lines.append(("error", msg))

    def debug(self, msg: str) -> None:
        self.lines.append(("debug", msg))


class _RecordingEvaluator:
    """Returns predetermined M1 per (iter, cand) so we can assert ranking."""

    def __init__(self, m1_table: dict[tuple[int, int], float]) -> None:
        self.m1_table = m1_table
        self.calls: list[tuple[int, int]] = []

    def evaluate(self, *, cfg, candidate, run_dir):  # noqa: ARG002
        self.calls.append((candidate.iteration, candidate.candidate_idx))
        m1 = self.m1_table.get((candidate.iteration, candidate.candidate_idx), 0.0)
        return CandidateMetrics(M1_success_rate=m1, M2_avg_return=m1 * 100)


class _RecordingFullTrainer:
    """Records which candidates the fallback chain tried; configurable crashes."""

    def __init__(self, crash_until_rank: int = -1) -> None:
        # By default never crash (-1 = no rank crashes).
        self.crash_until_rank = crash_until_rank
        self.attempts: list[tuple[int, int]] = []

    def train_full(self, *, cfg, candidate, run_dir):  # noqa: ARG002
        self.attempts.append((candidate.iteration, candidate.candidate_idx))
        rank = len(self.attempts) - 1
        if rank <= self.crash_until_rank:
            raise RuntimeError(f"simulated full-training crash at rank {rank}")
        return CandidateMetrics(M1_success_rate=1.0)


# ── happy path: 2 iters × 2 candidates, all valid ───────────────────


def test_run_persists_history_and_summary(tmp_path: Path):
    run_dir = tmp_path / "run"
    cfg = _cfg(tmp_path)
    llm = FakeLlmClient().register_always(LlmCompletion(text=_VALID_REWARD))
    evaluator = _RecordingEvaluator(
        {(0, 0): 0.3, (0, 1): 0.5, (1, 0): 0.7, (1, 1): 0.2}
    )
    full = _RecordingFullTrainer()
    orch = LeroOrchestrator(
        llm=llm,
        composer=InitialAndFeedbackComposer(
            renderer=JinjaPromptRenderer(),
            prompt_version="v2_fewshot_k2_local",
            n_candidates=2,
        ),
        trace_writer=FilesystemTraceWriter(),
        evaluator=evaluator,
        full_trainer=full,
        logger=_FakeLogger(),
    )
    summary = orch.run(cfg=cfg, run_dir=run_dir)
    assert summary.n_iterations_completed == 2
    assert summary.n_candidates_total == 4
    # Best inner-loop M1 = 0.7 (iter 1, cand 0).
    assert summary.best_candidate_metrics.M1_success_rate == pytest.approx(0.7)
    assert summary.best_candidate_verdict == "progress"
    assert summary.full_training_succeeded is True
    # Filesystem layout populated.
    lero_root = run_dir / "output" / "lero"
    assert (lero_root / "final_summary.json").is_file()
    assert (lero_root / "evolution_history.json").is_file()
    assert (lero_root / "iter_0" / "cand_0" / "result.json").is_file()
    assert (lero_root / "iter_1" / "cand_0" / "attempt_0" / "prompt.json").is_file()


def test_run_seeds_distinct_per_candidate(tmp_path: Path):
    """Sibling candidates within the same iter MUST get distinct seeds."""
    cfg = _cfg(tmp_path, n_iter=1, n_cand=3)
    llm = FakeLlmClient().register_always(LlmCompletion(text=_VALID_REWARD))
    orch = LeroOrchestrator(
        llm=llm,
        composer=InitialAndFeedbackComposer(
            renderer=JinjaPromptRenderer(),
            prompt_version="v2_fewshot_k2_local",
            n_candidates=3,
        ),
        trace_writer=FilesystemTraceWriter(),
        evaluator=_RecordingEvaluator({}),
        full_trainer=_RecordingFullTrainer(),
        logger=_FakeLogger(),
    )
    orch.run(cfg=cfg, run_dir=tmp_path / "run")
    seeds = [c["seed"] for c in llm.calls]
    assert len(set(seeds)) == len(seeds), f"siblings got duplicate seeds: {seeds}"


# ── codegen failure ──────────────────────────────────────────────────


def test_run_marks_invalid_when_response_has_no_code_blocks(tmp_path: Path):
    """When the LLM returns plain text (no ```python fence), the candidate
    is marked invalid but the loop keeps going."""
    cfg = _cfg(tmp_path)
    llm = FakeLlmClient().register_always(LlmCompletion(text="No code, sorry!"))
    evaluator = _RecordingEvaluator({})
    orch = LeroOrchestrator(
        llm=llm,
        composer=InitialAndFeedbackComposer(
            renderer=JinjaPromptRenderer(),
            prompt_version="v2_fewshot_k2_local",
            n_candidates=2,
        ),
        trace_writer=FilesystemTraceWriter(),
        evaluator=evaluator,
        full_trainer=_RecordingFullTrainer(),
        logger=_FakeLogger(),
    )
    summary = orch.run(cfg=cfg, run_dir=tmp_path / "run")
    # All 4 candidates produced; all marked invalid.
    assert summary.n_candidates_total == 4
    assert evaluator.calls == []  # never reached the evaluator
    assert summary.full_training_succeeded is False


def test_run_marks_invalid_when_evaluator_crashes(tmp_path: Path):
    """Evaluator exception → candidate's verdict is ``invalid``, loop continues."""

    class _CrashingEval:
        def evaluate(self, *, cfg, candidate, run_dir):  # noqa: ARG002
            raise RuntimeError("simulated NaN actions")

    cfg = _cfg(tmp_path)
    llm = FakeLlmClient().register_always(LlmCompletion(text=_VALID_REWARD))
    orch = LeroOrchestrator(
        llm=llm,
        composer=InitialAndFeedbackComposer(
            renderer=JinjaPromptRenderer(),
            prompt_version="v2_fewshot_k2_local",
            n_candidates=2,
        ),
        trace_writer=FilesystemTraceWriter(),
        evaluator=_CrashingEval(),
        full_trainer=_RecordingFullTrainer(),
        logger=_FakeLogger(),
    )
    summary = orch.run(cfg=cfg, run_dir=tmp_path / "run")
    # All candidates invalid → no full-training attempt → not successful.
    # (fallback chain skips invalid entries.)
    assert summary.full_training_succeeded is False
    assert summary.best_candidate_verdict == "invalid"


# ── Cost cap short-circuit ───────────────────────────────────────────


def test_run_falls_back_gracefully_on_cost_cap(tmp_path: Path):
    """LlmCostCapExceeded mid-loop → use the history we have so far."""

    class _LlmCapHit:
        """Returns valid candidates iter 0, raises in iter 1."""

        def __init__(self):
            self.iter_index = 0

        def generate(
            self, *, messages, n=1, seed=None, response_format=None
        ):  # noqa: ARG002
            # Count calls (one per candidate); the second iter's first call trips.
            if self.iter_index >= 2:
                raise LlmCostCapExceeded("over budget", spent_usd=11.0, cap_usd=10.0)
            self.iter_index += 1
            return [LlmCompletion(text=_VALID_REWARD)]

    cfg = _cfg(tmp_path)
    evaluator = _RecordingEvaluator({(0, 0): 0.4, (0, 1): 0.2})
    full = _RecordingFullTrainer()
    orch = LeroOrchestrator(
        llm=_LlmCapHit(),
        composer=InitialAndFeedbackComposer(
            renderer=JinjaPromptRenderer(),
            prompt_version="v2_fewshot_k2_local",
            n_candidates=2,
        ),
        trace_writer=FilesystemTraceWriter(),
        evaluator=evaluator,
        full_trainer=full,
        logger=_FakeLogger(),
    )
    summary = orch.run(cfg=cfg, run_dir=tmp_path / "run")
    # Iter 0 completed (2 candidates); iter 1 short-circuited.
    assert summary.n_candidates_total == 2
    # Full training still runs on whatever we got.
    assert len(full.attempts) >= 1


# ── Fallback chain ──────────────────────────────────────────────────


def test_resume_skips_iterations_already_on_disk(tmp_path: Path):
    """F9.6.d: a crashed-and-resumed run picks up at the next iteration.

    Seed the run-dir with a completed iter 0 (two cand_*/result.json
    files), then call ``run(resume=True)``. The LLM is fired only for
    iter 1 — the cost budget from iter 0 is structurally preserved.
    """
    cfg = _cfg(tmp_path, n_iter=2, n_cand=2)
    run_dir = tmp_path / "run"
    # Plant iter 0 results on disk.
    writer = FilesystemTraceWriter()
    from multi_scenario.domain.lero import (
        Candidate,
        CandidateCode,
        CandidateMetrics,
        CandidateResult,
    )

    for cand_idx in range(2):
        writer.write_candidate_result(
            run_dir=run_dir,
            result=CandidateResult(
                candidate=Candidate(
                    iteration=0,
                    candidate_idx=cand_idx,
                    code=CandidateCode(
                        reward_source="def compute_reward(s): return s['agent_pos']"
                    ),
                ),
                metrics=CandidateMetrics(M1_success_rate=0.4),
                verdict="progress",
            ),
        )

    llm = FakeLlmClient().register_always(LlmCompletion(text=_VALID_REWARD))
    evaluator = _RecordingEvaluator({(1, 0): 0.5, (1, 1): 0.6})
    full = _RecordingFullTrainer()
    orch = LeroOrchestrator(
        llm=llm,
        composer=InitialAndFeedbackComposer(
            renderer=JinjaPromptRenderer(),
            prompt_version="v2_fewshot_k2_local",
            n_candidates=2,
        ),
        trace_writer=writer,
        evaluator=evaluator,
        full_trainer=full,
        logger=_FakeLogger(),
    )
    summary = orch.run(cfg=cfg, run_dir=run_dir, resume=True)

    # Iter 0 was NOT re-evaluated (no calls with iteration=0).
    assert all(call[0] != 0 for call in evaluator.calls)
    # Iter 1 ran fully (2 evaluator calls).
    assert sorted(evaluator.calls) == [(1, 0), (1, 1)]
    # History contains all 4 (recovered + fresh).
    assert summary.n_candidates_total == 4


def test_full_training_falls_back_when_rank_0_crashes(tmp_path: Path):
    """Crash at rank 0 → try rank 1 → eventually succeed."""
    cfg = _cfg(tmp_path)
    llm = FakeLlmClient().register_always(LlmCompletion(text=_VALID_REWARD))
    evaluator = _RecordingEvaluator(
        {(0, 0): 0.9, (0, 1): 0.5, (1, 0): 0.3, (1, 1): 0.1}
    )
    full = _RecordingFullTrainer(crash_until_rank=0)  # only rank 0 crashes
    orch = LeroOrchestrator(
        llm=llm,
        composer=InitialAndFeedbackComposer(
            renderer=JinjaPromptRenderer(),
            prompt_version="v2_fewshot_k2_local",
            n_candidates=2,
        ),
        trace_writer=FilesystemTraceWriter(),
        evaluator=evaluator,
        full_trainer=full,
        logger=_FakeLogger(),
    )
    summary = orch.run(cfg=cfg, run_dir=tmp_path / "run")
    # Rank 0 = (iter 0, cand 0) with M1=0.9; crashes. Rank 1 = (iter 0, cand 1)
    # with M1=0.5 succeeds.
    assert len(full.attempts) == 2
    assert full.attempts[0] == (0, 0)
    assert full.attempts[1] == (0, 1)
    assert summary.full_training_succeeded is True
    assert summary.fallback_chain[0].outcome == "crashed"
    assert summary.fallback_chain[1].outcome == "success"
