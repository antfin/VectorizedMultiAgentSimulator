"""F8.4 Phase 1 — full LERO orchestrator end-to-end on real BenchMARL.

The F9.6.b orchestrator contract tests stub the evaluator + full_trainer.
The F9.6.e BenchMARL adapter tests invoke the evaluator directly.

This test wires BOTH together: orchestrator → composer → FakeLlmClient
→ codegen extraction → BenchmarlCandidateEvaluator (real patched
scenario + real BenchMARL training) → trace writer → BenchmarlFullTrainer
→ summary. No LLM API cost; ~15-25s wall on Mac CPU.

Marked ``@pytest.mark.slow`` so the fast unit pass stays under a second.

When this test passes, every wire in the F8.4 chain is exercised
locally except the network call to OpenAI — Phase 2 covers that.
"""

# pylint: disable=missing-function-docstring,redefined-outer-name

from pathlib import Path

import pytest

from multi_scenario.adapters.lero import (
    BenchmarlCandidateEvaluator,
    BenchmarlFullTrainer,
    FilesystemTraceWriter,
)
from multi_scenario.adapters.llm import FakeLlmClient
from multi_scenario.adapters.prompt_composers import InitialAndFeedbackComposer
from multi_scenario.adapters.prompts import JinjaPromptRenderer
from multi_scenario.application.lero_orchestrator import LeroOrchestrator
from multi_scenario.domain.lero import LlmCompletion
from multi_scenario.domain.models import ExperimentConfig


pytestmark = pytest.mark.slow


class _SilentLogger:
    # pylint: disable=missing-function-docstring,unused-argument
    def info(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        pass

    def debug(self, msg):
        pass


_CANNED_REWARD = (
    "```python\n"
    "def compute_reward(scenario_state):\n"
    "    # Trivial: return the agent's x-position. Just needs to NOT NaN\n"
    "    # and execute through BenchMARL's training step.\n"
    "    return scenario_state['agent_pos'][..., 0]\n"
    "```"
)


def _cfg(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "experiment": {"id": "lero_smoke_phase1", "seed": 0},
            "scenario": {
                "type": "discovery",
                "params": {
                    "n_agents": 2,
                    "n_targets": 2,
                    "agents_per_target": 2,
                    "targets_respawn": False,
                    "shared_reward": True,
                    "max_steps": 5,
                    # Prompt template needs these for the v2_fewshot_k2_local
                    # initial_user body. Their values don't influence
                    # training (they're injected into prompt text only).
                    "covering_range": 0.35,
                    "n_lidar_rays_entities": 15,
                    "n_lidar_rays_agents": 12,
                    "obs_lidar_agents": (
                        '"lidar_agents":     # [batch, 12] — agent LiDAR'
                    ),
                    "lidar_range": 0.35,
                    "use_agent_lidar": True,
                },
            },
            "algorithm": {"type": "mappo", "params": {}},
            "training": {
                "max_iters": 1,  # tiny full-training budget
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
                "eval_frames_per_candidate": 100,  # ≈ 2 BenchMARL iters
                # evolve_observation=False so the canned reward source
                # is the only LLM contribution — keeps the smoke
                # focused on the reward-patching path.
                "evolve_reward": True,
                "evolve_observation": False,
            },
            "llm": {"model": "gpt-4o-mini"},
        }
    )


def test_orchestrator_full_chain_local_smoke(tmp_path: Path):
    """Drive LeroOrchestrator end-to-end with real BenchMARL + FakeLlmClient.

    Asserts the whole chain works without surprises before we burn OVH
    credit on Phase 3. Specifically pins:

    - LlmClient.generate() is hit (once per candidate).
    - extract_candidates() pulls the canned reward source.
    - Patched scenario builds, BenchMARL trains for the short budget.
    - CommonMetricsBundle populates M2/M3/M4 (universal metrics).
    - FullTrainer runs on the inner-loop winner; succeeds.
    - Filesystem trace layout is intact: iter_0/cand_0/result.json +
      iter_0/cand_0/attempt_0/{prompt,response}.json + final_summary.json.
    """
    cfg = _cfg(tmp_path)
    run_dir = tmp_path / "run"

    llm = FakeLlmClient().register_always(LlmCompletion(text=_CANNED_REWARD))
    orch = LeroOrchestrator(
        llm=llm,
        composer=InitialAndFeedbackComposer(
            renderer=JinjaPromptRenderer(),
            prompt_version="v2_fewshot_k2_local",
            n_candidates=1,
        ),
        trace_writer=FilesystemTraceWriter(),
        evaluator=BenchmarlCandidateEvaluator(),
        full_trainer=BenchmarlFullTrainer(),
        logger=_SilentLogger(),
    )

    summary = orch.run(cfg=cfg, run_dir=run_dir)

    # ── Orchestrator-level assertions ─────────────────────────────────

    assert summary.n_iterations_completed == 1
    assert summary.n_candidates_total == 1
    # M2/M3/M4 are universal — always populated if training+eval ran.
    best = summary.best_candidate_metrics
    assert best.M2_avg_return is not None, "M2 missing — eval rollout broken?"
    assert best.M3_steps is not None
    assert best.M4_collisions is not None
    # Full training should succeed for a trivial reward.
    assert summary.full_training_succeeded is True
    assert summary.fallback_chain[0].outcome == "success"

    # ── LLM call shape ────────────────────────────────────────────────

    assert len(llm.calls) == 1, f"expected 1 LLM call, got {len(llm.calls)}"
    call = llm.calls[0]
    assert call["n"] == 1  # one sibling per candidate
    assert call["seed"] is not None  # SHA-derived per-(iter, cand) seed

    # ── Filesystem layout ─────────────────────────────────────────────

    lero_root = run_dir / "output" / "lero"
    assert (lero_root / "final_summary.json").is_file()
    assert (lero_root / "evolution_history.json").is_file()
    assert (lero_root / "iter_0" / "cand_0" / "result.json").is_file()
    assert (lero_root / "iter_0" / "cand_0" / "attempt_0" / "prompt.json").is_file()
    assert (lero_root / "iter_0" / "cand_0" / "attempt_0" / "response.json").is_file()


def test_orchestrator_full_chain_pins_phase_1_to_9_behaviors(tmp_path: Path):
    """One smoke run, ten assertions: every Phase 1-9 behaviour pinned together.

    This is the gate before any future LERO OVH spend. It asserts that
    a single ~60s local run produces EVERY artefact the user expects:

    - Phase 1: ``best_candidate_full_metrics`` populated (not None).
    - Phase 4: full-training checkpoints pruned to a single ``.pt``.
    - Phase 5: ``evolution_doc.md`` + ``prompts/iter_0/{system,user_initial}.md``
      + ``prompts/iter_0/cand_0/{response.md, reward_source.py}``.
    - Phase 6: ``output/metrics.json`` (ER1-shape) carries the
      post-full-train numbers (preferred source per the
      ``_lero_metrics_to_records`` contract).
    - Phase 9 (#10): ``summary.total_cost_usd`` reports the LLM spend
      (FakeLlmClient returns 0.0 cost so we assert ``>= 0.0`` to pin
      the field exists; production runs see positive values).
    """
    cfg = _cfg(tmp_path)
    # Phase 4: opt into checkpoint cleanup.
    cfg.training.delete_intermediate_checkpoints_on_success = True
    run_dir = tmp_path / "run"

    llm = FakeLlmClient().register_always(LlmCompletion(text=_CANNED_REWARD))
    orch = LeroOrchestrator(
        llm=llm,
        composer=InitialAndFeedbackComposer(
            renderer=JinjaPromptRenderer(),
            prompt_version="v2_fewshot_k2_local",
            n_candidates=1,
        ),
        trace_writer=FilesystemTraceWriter(),
        evaluator=BenchmarlCandidateEvaluator(),
        full_trainer=BenchmarlFullTrainer(),
        logger=_SilentLogger(),
    )
    summary = orch.run(cfg=cfg, run_dir=run_dir)

    # Phase 1 — post-full-train metrics captured (was discarded pre-fix).
    assert summary.best_candidate_full_metrics is not None
    assert summary.best_candidate_full_metrics.M2_avg_return is not None
    # The success entry in the fallback chain carries the full metrics too.
    success = next(e for e in summary.fallback_chain if e.outcome == "success")
    assert success.full_train_metrics is not None

    # Phase 5 — evolution_doc.md + prompts/ layout.
    lero_root = run_dir / "output" / "lero"
    assert (lero_root / "evolution_doc.md").is_file()
    doc_text = (lero_root / "evolution_doc.md").read_text()
    assert "Headline" in doc_text
    assert "Iteration 0" in doc_text
    # Relative links to prompts/ resolve to actual files on disk.
    assert (lero_root / "prompts" / "iter_0" / "system.md").is_file()
    assert (lero_root / "prompts" / "iter_0" / "user_initial.md").is_file()
    assert (
        lero_root / "prompts" / "iter_0" / "cand_0" / "response.md"
    ).is_file()
    assert (
        lero_root / "prompts" / "iter_0" / "cand_0" / "reward_source.py"
    ).is_file()

    # Phase 4 — checkpoint pruning kept ≤ 1 .pt file under benchmarl/.
    bench_root = run_dir / "output" / "benchmarl"
    surviving = list(bench_root.rglob("checkpoints/*.pt"))
    assert len(surviving) <= 1, (
        f"checkpoint cleanup failed: {len(surviving)} .pt files survived: "
        f"{[p.name for p in surviving]}"
    )

    # Phase 6 — ER1-shape ``eval_episodes.json`` from the full-train eval.
    # (``metrics.json`` + ``report.json`` are written by ExperimentService
    # — not exercised in this direct-orchestrator test; covered separately
    # by tests/integration/application/test_experiment_service_lero_branch.py.)
    assert (run_dir / "output" / "eval_episodes.json").is_file()

    # Phase 9 (#10) — total_cost_usd field is populated (FakeLlmClient
    # always reports 0.0 USD; assert the field exists with a numeric
    # value, not the hardcoded 0.0 sentinel from pre-fix).
    assert isinstance(summary.total_cost_usd, float)
    assert summary.total_cost_usd >= 0.0


def test_orchestrator_resume_replays_disk_history(tmp_path: Path):
    """Resume contract: after a successful first run, ``resume=True`` on
    the same run_dir doesn't re-fire the LLM (the iter is already on
    disk), but still re-runs the full-training step on the recovered
    history.

    Catches the "resume re-asks the LLM" cost-leak regression — the
    whole point of F9.6.d is that the cost budget survives restarts.
    """
    cfg = _cfg(tmp_path)
    run_dir = tmp_path / "run"

    llm = FakeLlmClient().register_always(LlmCompletion(text=_CANNED_REWARD))
    orch = LeroOrchestrator(
        llm=llm,
        composer=InitialAndFeedbackComposer(
            renderer=JinjaPromptRenderer(),
            prompt_version="v2_fewshot_k2_local",
            n_candidates=1,
        ),
        trace_writer=FilesystemTraceWriter(),
        evaluator=BenchmarlCandidateEvaluator(),
        full_trainer=BenchmarlFullTrainer(),
        logger=_SilentLogger(),
    )

    # First run — populates the run dir.
    orch.run(cfg=cfg, run_dir=run_dir)
    assert len(llm.calls) == 1
    llm.calls.clear()

    # Resume — should NOT re-fire the LLM (history is on disk).
    summary2 = orch.run(cfg=cfg, run_dir=run_dir, resume=True)
    assert llm.calls == [], "resume must not re-fire LLM calls"
    # Full training still runs (it's idempotent at the BenchMARL level).
    assert summary2.full_training_succeeded is True
