"""F9.6.e — BenchMARL-backed :class:`CandidateEvaluator` + :class:`FullTrainer`.

These wire the LERO orchestrator into the existing
:class:`BenchmarlBaseAdapter` training stack:

1. Compile the candidate's reward / observation code via F9.5's
   :func:`make_patched_discovery_class`.
2. Build a BenchMARL ``Experiment`` whose task uses our
   :class:`ScenarioEnvFunFactory` so every worker creates a fresh
   instance of the patched class.
3. Run ``experiment.run()`` for the configured frame budget.
4. Roll out an eval episode via the algorithm adapter's ``evaluate``.
5. Compute M1–M9 via :class:`CommonMetricsBundle` and return
   :class:`CandidateMetrics`.

The evaluator and the full trainer differ only in the frame budget:
- Evaluator: ``cfg.lero.eval_frames_per_candidate`` (default 1M frames
  per candidate).
- FullTrainer: ``cfg.training.max_iters * cfg.training.frames_per_batch``
  (full ER1/LERO budget — typically 10M frames).

Both reuse the same patched-experiment helper so the only difference
is one cfg-derived integer.

This adapter holds the only torch / VMAS / BenchMARL coupling in the
LERO loop — the orchestrator + 6 of the 8 ports stay pure-Python.
"""

import logging
from pathlib import Path
from typing import Any

from multi_scenario.adapters.algorithms.benchmarl_base import BenchmarlBaseAdapter
from multi_scenario.adapters.lero.scenario_env_fun_factory import ScenarioEnvFunFactory
from multi_scenario.adapters.metrics.common import CommonMetricsBundle
from multi_scenario.adapters.scenarios.discovery import VmasDiscoveryAdapter
from multi_scenario.adapters.scenarios.patched_discovery import (
    make_patched_discovery_class,
)
from multi_scenario.application.factories import make_algorithm
from multi_scenario.domain.lero import Candidate, CandidateMetrics
from multi_scenario.domain.models import ExperimentConfig


_log = logging.getLogger(__name__)


# ── Shared patched-experiment construction ────────────────────────────


def _build_patched_experiment(
    *,
    cfg: ExperimentConfig,
    candidate: Candidate,
    run_dir: Path,
) -> tuple[Any, BenchmarlBaseAdapter]:
    """Build a BenchMARL ``Experiment`` with the candidate's code spliced in.

    Returns ``(experiment, adapter)`` so the caller can reuse the
    adapter's ``evaluate(...)`` path (which already knows how to
    convert BenchMARL rollouts → the universal rollout dict
    CommonMetricsBundle expects).

    Per cfg.lero settings:
    - ``reward_mode`` = "replace" by default (paper-faithful).
      ``cfg.lero.evolve_reward=False`` → reward_source=None → no patch.
    - ``obs_state_mode`` = "global" by default. ``"local"`` lands when
      the prompt version is one of the local-only variants (heuristic:
      the prompt version's name ends with ``_local``).
    - ``whitelist_strict`` = cfg.lero.whitelist_strict.
    - ``reward_clip`` = cfg.lero.reward_clip.
    """
    assert cfg.lero is not None
    is_local_obs = cfg.lero.prompt_version.endswith("_local")
    patched_cls = make_patched_discovery_class(
        reward_source=(
            candidate.code.reward_source if cfg.lero.evolve_reward else None
        ),
        obs_source=(candidate.code.obs_source if cfg.lero.evolve_observation else None),
        reward_mode="replace",
        obs_state_mode="local" if is_local_obs else "global",
        reward_clip=cfg.lero.reward_clip,
        whitelist_strict=cfg.lero.whitelist_strict,
    )

    # Build the adapter via the factory so the algorithm-type
    # registry is the single source of truth (mappo / ippo / etc.).
    adapter = make_algorithm(cfg.algorithm.type)
    if not isinstance(adapter, BenchmarlBaseAdapter):
        raise TypeError(
            f"LERO evaluator requires a BenchmarlBaseAdapter; got "
            f"{type(adapter).__name__}. Configure cfg.algorithm.type to "
            "a BenchMARL-backed algorithm (mappo/ippo/iddpg/maddpg/isac/masac)."
        )

    # Standard build_experiment, but inject the patched env factory
    # AFTER the adapter constructs the task so our get_env_fun override
    # wins. Need to repeat a tiny bit of build_experiment's body
    # (saving the patched task) to keep the swap intact.
    # pylint: disable=import-outside-toplevel
    from benchmarl.experiment import Experiment

    save_folder_path = run_dir / "output" / "benchmarl"
    save_folder_path.mkdir(parents=True, exist_ok=True)

    task = adapter._task(cfg)  # pylint: disable=protected-access
    task.get_env_fun = ScenarioEnvFunFactory(patched_cls, dict(task.config))

    bm_cfg = adapter._experiment_config(  # pylint: disable=protected-access
        cfg, save_folder=str(save_folder_path)
    )
    experiment = Experiment(
        task=task,
        algorithm_config=adapter._algorithm_config(
            cfg
        ),  # pylint: disable=protected-access
        model_config=adapter._model_config(cfg),  # pylint: disable=protected-access
        seed=cfg.experiment.seed,
        config=bm_cfg,
    )
    return experiment, adapter


def _short_eval_cfg(cfg: ExperimentConfig) -> ExperimentConfig:
    """Derive a shortened cfg for the candidate-eval phase.

    The LERO inner loop trains each candidate for
    ``cfg.lero.eval_frames_per_candidate`` total frames before scoring
    it. Convert to ``max_iters`` (= frames / frames_per_batch) and
    return a deep-copied cfg with the override applied so we don't
    mutate the caller's config.
    """
    assert cfg.lero is not None
    iters = max(1, cfg.lero.eval_frames_per_candidate // cfg.training.frames_per_batch)
    return cfg.model_copy(
        update={"training": cfg.training.model_copy(update={"max_iters": iters})},
        deep=True,
    )


def _compute_metrics(
    adapter: BenchmarlBaseAdapter, experiment: Any, cfg: ExperimentConfig, run_dir: Path
) -> CandidateMetrics:
    """Roll out the trained policy and convert metrics to CandidateMetrics."""
    rollout = adapter.evaluate(experiment, env=None, cfg=cfg, run_dir=run_dir)
    # CommonMetricsBundle wants a Scenario port. For Discovery (the only
    # patchable scenario today) we reuse the standard VmasDiscoveryAdapter
    # — its success_predicate / coverage_progress / utilization_predicate
    # operate on the rollout dict, not on the patched class instance.
    metrics_dict = CommonMetricsBundle().compute(rollout, VmasDiscoveryAdapter())
    return CandidateMetrics(**metrics_dict)


# ── CandidateEvaluator (short training) ──────────────────────────────


class BenchmarlCandidateEvaluator:
    """LERO inner-loop short-training evaluator.

    Runs ``cfg.lero.eval_frames_per_candidate`` frames per candidate
    (rendezvous_comm Phase 4 default: 1M frames). Raises on training
    crashes — the orchestrator catches and marks the candidate
    ``invalid``.
    """

    def evaluate(
        self,
        *,
        cfg: ExperimentConfig,
        candidate: Candidate,
        run_dir: Path,
    ) -> CandidateMetrics:
        short_cfg = _short_eval_cfg(cfg)
        # Each candidate runs under its own subdir so BenchMARL's logs +
        # checkpoints don't clobber siblings.
        cand_run_dir = (
            run_dir
            / "output"
            / "lero"
            / f"iter_{candidate.iteration}"
            / f"cand_{candidate.candidate_idx}"
            / "training"
        )
        experiment, adapter = _build_patched_experiment(
            cfg=short_cfg, candidate=candidate, run_dir=cand_run_dir
        )
        _log.info(
            f"LERO eval starting: iter={candidate.iteration} "
            f"cand={candidate.candidate_idx} iters={short_cfg.training.max_iters}"
        )
        experiment.run()
        return _compute_metrics(adapter, experiment, short_cfg, cand_run_dir)


# ── FullTrainer (full budget) ────────────────────────────────────────


class BenchmarlFullTrainer:
    """LERO post-loop full-training runner.

    Uses ``cfg.training.max_iters`` directly (the same budget a non-LERO
    ER1-style run would use). Persists output under
    ``run_dir/output/benchmarl/`` — the same layout the standard
    BenchmarlBaseAdapter writes — so post-hoc tools (eval-only,
    regenerate-videos) work without LERO-specific adaptations.
    """

    def train_full(
        self,
        *,
        cfg: ExperimentConfig,
        candidate: Candidate,
        run_dir: Path,
    ) -> CandidateMetrics:
        experiment, adapter = _build_patched_experiment(
            cfg=cfg, candidate=candidate, run_dir=run_dir
        )
        _log.info(
            f"LERO full training starting: iter={candidate.iteration} "
            f"cand={candidate.candidate_idx} iters={cfg.training.max_iters}"
        )
        experiment.run()
        return _compute_metrics(adapter, experiment, cfg, run_dir)
