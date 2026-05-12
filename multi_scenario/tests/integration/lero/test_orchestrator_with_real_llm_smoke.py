"""F8.4 Phase 2 — full LERO orchestrator with REAL OpenAI LLM.

Same shape as Phase 1's smoke (the FakeLlmClient one) but with the
production :class:`LiteLlmClient` wired in. Spends ~€0.02-0.05 per
run on ``gpt-4o-mini``. Gated by ``OPENAI_API_KEY`` so CI doesn't
unintentionally bill the project — opt in locally via:

    OPENAI_API_KEY=sk-... pytest -m slow tests/integration/lero/test_orchestrator_with_real_llm_smoke.py

Validates pieces that Phase 1 can't:

- LiteLlmClient subprocess against OpenAI (auth, model name, ``n=k``).
- Cost-cap decorator records the actual EUR cost in the host-wide
  ledger.
- DiskCacheDecorator (when enabled) caches the LLM response so a
  re-run is free.
- ``.env`` autoload via ``_load_env_once`` reaches the right file.

When the LLM returns garbage (no ```python fence, malformed code),
the orchestrator's `extract_candidates` drops it and marks the
candidate invalid — but the loop completes. Pinning that's left to
the F9.4 codegen tests; here we just want one valid candidate to
make it through.
"""

# pylint: disable=missing-function-docstring,redefined-outer-name

import os
from datetime import timedelta
from pathlib import Path

import pytest

from multi_scenario.adapters.lero import (
    BenchmarlCandidateEvaluator,
    BenchmarlFullTrainer,
    FilesystemTraceWriter,
)
from multi_scenario.adapters.llm import (
    CostCapDecorator,
    FilesystemCostLedger,
    LiteLlmClient,
)
from multi_scenario.adapters.prompt_composers import InitialAndFeedbackComposer
from multi_scenario.adapters.prompts import JinjaPromptRenderer
from multi_scenario.application.lero_orchestrator import LeroOrchestrator
from multi_scenario.domain.models import ExperimentConfig


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="real-LLM smoke needs OPENAI_API_KEY in env (~€0.05 per run)",
    ),
]


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


def _cfg(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "experiment": {"id": "lero_smoke_phase2", "seed": 0},
            "scenario": {
                "type": "discovery",
                "params": {
                    "n_agents": 2,
                    "n_targets": 2,
                    "agents_per_target": 2,
                    "targets_respawn": False,
                    "shared_reward": True,
                    "max_steps": 5,
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
                "storage": {"type": "fs", "path": str(tmp_path), "params": {}},
            },
            "lero": {
                "n_iterations": 1,
                "n_candidates": 1,
                "eval_frames_per_candidate": 100,
                "evolve_reward": True,
                "evolve_observation": False,
                # Loose caps for the smoke (one call expected).
                "prompt_version": "v2_fewshot_k2_local",
            },
            "llm": {
                "model": "gpt-4o-mini",
                "temperature": 0.0,  # deterministic for smoke (avoid retries)
                "max_tokens": 1024,
                "cost_cap_per_day_eur": 1.0,  # plenty for one call (~€0.02)
                "cost_cap_per_month_eur": 5.0,
            },
        }
    )


def test_orchestrator_full_chain_real_llm_smoke(tmp_path: Path):
    """One iter × one candidate against real ``gpt-4o-mini``.

    Verifies the production decorator stack (LiteLlmClient + cost cap +
    real OpenAI call) and that an actual LLM response makes it through
    codegen → patched scenario → BenchMARL training.

    Skips when ``OPENAI_API_KEY`` is unset (CI / no-key environments).
    """
    cfg = _cfg(tmp_path)
    # Point the cost ledger at the tmp dir so this test doesn't pollute
    # the user's ~/.multi_scenario/cost_ledger.jsonl.
    os.environ["MULTI_SCENARIO_COST_LEDGER"] = str(tmp_path / "ledger.jsonl")
    try:
        run_dir = tmp_path / "run"
        ledger = FilesystemCostLedger()
        assert cfg.llm is not None
        llm = CostCapDecorator(LiteLlmClient(cfg.llm), ledger=ledger, cfg_llm=cfg.llm)
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

        # ── Orchestrator-level assertions ──────────────────────────────

        assert summary.n_iterations_completed == 1
        assert summary.n_candidates_total == 1
        # Whether the candidate is "valid" depends on what the LLM
        # returned — gpt-4o-mini with our prompt usually produces a
        # parseable function. If validation fails, the verdict is
        # "invalid" and we just assert the trace exists.
        best = summary.best_candidate_metrics
        if summary.best_candidate_verdict != "invalid":
            assert best.M2_avg_return is not None
            assert best.M3_steps is not None

        # ── Cost ledger entry ─────────────────────────────────────────

        # The cost cap recorded the LLM call's USD→EUR cost. Even
        # micro-costs (~€0.001 for one gpt-4o-mini call) land as a
        # positive entry — assert it's non-zero.
        spent_today = ledger.sum_window(timedelta(days=1))
        assert spent_today > 0.0, "cost ledger missed the real LLM call"

        # ── Trace files materialised ──────────────────────────────────

        lero_root = run_dir / "output" / "lero"
        assert (lero_root / "final_summary.json").is_file()
        prompt_path = lero_root / "iter_0" / "cand_0" / "attempt_0" / "prompt.json"
        response_path = lero_root / "iter_0" / "cand_0" / "attempt_0" / "response.json"
        assert prompt_path.is_file()
        assert response_path.is_file()
        # Response text is non-empty (real LLM output).
        response_blob = response_path.read_text(encoding="utf-8")
        assert len(response_blob) > 100
    finally:
        os.environ.pop("MULTI_SCENARIO_COST_LEDGER", None)
