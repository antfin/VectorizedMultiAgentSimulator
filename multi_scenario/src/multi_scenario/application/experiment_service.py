"""ExperimentService — the in-process use-case orchestrator.

Wires the five domain ports (Scenario, Algorithm, MetricsBundle, Storage,
Logger) together to execute one experiment run end-to-end. Provenance is
*injected* by the caller so this class stays free of git / package-version
I/O. Crash handling lands at F5.7.

The optional ``eval_episodes_writer`` callable is the F2.10.1 escape hatch:
``LocalRunner`` injects ``LocalStorageAdapter.save_eval_episodes`` so the
service can persist the raw eval rollout without growing the ``Storage``
Protocol surface (consistent with the F1.9 minimalism rule).
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from multi_scenario.domain.models import (
    ExperimentConfig,
    ExperimentResult,
    Provenance,
    RunId,
    RunState,
    RunStateRecord,
)
from multi_scenario.domain.ports import (
    Algorithm,
    Logger,
    MetricsBundle,
    Scenario,
    Storage,
)


class ExperimentService:
    """In-process orchestrator for one experiment run."""

    # Five injected ports (DI) + the optional eval-episodes writer → 7 ctor args;
    # single public `run` method is the whole point of a use-case orchestrator.
    # Pylint's defaults don't fit this pattern.
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-few-public-methods

    def __init__(
        self,
        scenario: Scenario,
        algorithm: Algorithm,
        metrics: MetricsBundle,
        storage: Storage,
        logger: Logger,
        eval_episodes_writer: Callable[[Path, Any], None] | None = None,
    ) -> None:
        self._scenario = scenario
        self._algorithm = algorithm
        self._metrics = metrics
        self._storage = storage
        self._logger = logger
        self._eval_episodes_writer = eval_episodes_writer

    def run(
        self,
        cfg: ExperimentConfig,
        run_dir: Path,
        provenance: Provenance,
        resume_from: Path | None = None,
    ) -> ExperimentResult:
        """Execute the full lifecycle: init → save inputs → train → eval → save result → done.

        F5.7: when ``resume_from`` is set, this is a resume of a previously-crashed
        run — the run-state record is loaded from disk (caller already transitioned
        it to ``RESUMED``) and ``config.json`` / ``provenance.json`` are NOT
        re-written (they're immutable for the run's identity).

        F9.6.c: when ``cfg.lero is not None``, the run is a LERO loop —
        delegate to :class:`LeroOrchestrator` instead of the
        single-train+eval path. The orchestrator handles its own input
        persistence (prompts/responses/traces under
        ``output/lero/``) and emits its own summary; we wrap that into
        an :class:`ExperimentResult` so the caller-facing return type
        stays uniform.
        """
        run_id = RunId(exp_id=cfg.experiment.id, seed=cfg.experiment.seed)
        is_resume = resume_from is not None

        if is_resume:
            state = self._storage.load_run_state(run_dir)
            started_at = state.transitions[0].ts
        else:
            started_at = self._now()
            state = RunStateRecord.initial(started_at)
            self._storage.save_run_state(run_dir, state)
            self._storage.save_config(run_dir, cfg)
            self._storage.save_provenance(run_dir, provenance)

        # F9.6.c: LERO dispatch fork. Lives here (not at a higher layer)
        # so the same run-state lifecycle + config/provenance persistence
        # applies to LERO runs — the only difference is *what training
        # loop* executes between INITIALIZING and DONE.
        if cfg.lero is not None:
            return self._run_lero(
                cfg=cfg,
                run_dir=run_dir,
                run_id=run_id,
                state=state,
                started_at=started_at,
            )

        env = self._scenario.make_env(
            cfg.scenario, cfg.training.num_envs, cfg.experiment.seed
        )

        state = state.transition_to(RunState.RUNNING, self._now())
        self._storage.save_run_state(run_dir, state)

        self._logger.info(f"training {run_id}{' (resume)' if is_resume else ''}")
        artifact = self._algorithm.train(
            env, cfg, run_dir=run_dir, resume_from=resume_from
        )

        self._logger.info(f"evaluating {run_id}")
        rollout = self._algorithm.evaluate(artifact, env, cfg, run_dir=run_dir)

        # F2.10.1: persist raw per-episode eval data when an opt-in writer is
        # wired (LocalRunner does this). Off the Storage Protocol per F1.9.
        if self._eval_episodes_writer is not None:
            self._eval_episodes_writer(run_dir, rollout)

        metric_dict = self._metrics.compute(rollout, self._scenario)

        result = ExperimentResult(
            run_id=str(run_id),
            exp_id=cfg.experiment.id,
            scenario=cfg.scenario.type,
            algorithm=cfg.algorithm.type,
            seed=cfg.experiment.seed,
            run_timestamp=started_at.strftime("%Y%m%d_%H%M%S"),
            metrics=metric_dict,
            config_snapshot={**cfg.scenario.params, **cfg.algorithm.params},
            n_envs=cfg.training.num_envs,
            n_eval_episodes=cfg.evaluation.episodes,
        )
        self._storage.save_result(run_dir, result)

        state = state.transition_to(RunState.DONE, self._now())
        self._storage.save_run_state(run_dir, state)

        self._logger.info(f"done {run_id}")
        return result

    def _run_lero(
        self,
        *,
        cfg: ExperimentConfig,
        run_dir: Path,
        run_id: RunId,
        state: RunStateRecord,
        started_at: datetime,
    ) -> ExperimentResult:
        """F9.6.c LERO dispatch: assemble the orchestrator, run, map back.

        Returns an :class:`ExperimentResult` whose ``metrics`` carry the
        best inner-loop candidate's M1–M9. The full-training-after-loop
        artefact (output/benchmarl) is produced by the orchestrator's
        :class:`FullTrainer` (F9.6.e); the eval rollout it returns isn't
        threaded back here because LERO's "result of the run" is the
        chosen-candidate's metrics, not a separate eval rollout.
        """
        # pylint: disable=import-outside-toplevel  # circular avoidance + lazy import
        from multi_scenario.application.lero_factory import (
            build_default_lero_orchestrator,
        )

        state = state.transition_to(RunState.RUNNING, self._now())
        self._storage.save_run_state(run_dir, state)

        self._logger.info(f"training {run_id} (LERO loop)")
        orchestrator = build_default_lero_orchestrator(cfg=cfg, logger=self._logger)
        summary = orchestrator.run(cfg=cfg, run_dir=run_dir)

        metric_dict = [
            # Mirror the standard path's MetricRecord shape — flatten the
            # CandidateMetrics fields into the existing schema.
            *_lero_metrics_to_records(summary),
        ]
        result = ExperimentResult(
            run_id=str(run_id),
            exp_id=cfg.experiment.id,
            scenario=cfg.scenario.type,
            algorithm=cfg.algorithm.type,
            seed=cfg.experiment.seed,
            run_timestamp=started_at.strftime("%Y%m%d_%H%M%S"),
            metrics=metric_dict,
            config_snapshot={**cfg.scenario.params, **cfg.algorithm.params},
            n_envs=cfg.training.num_envs,
            n_eval_episodes=cfg.evaluation.episodes,
        )
        self._storage.save_result(run_dir, result)
        state = state.transition_to(RunState.DONE, self._now())
        self._storage.save_run_state(run_dir, state)
        self._logger.info(
            f"done {run_id} (LERO best M1={summary.best_candidate_metrics.M1_success_rate})"
        )
        return result

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)


def _lero_metrics_to_records(summary) -> list:
    """Map :class:`CandidateMetrics` → the existing MetricRecord list shape.

    ExperimentResult's persistence layer expects a list of records, not
    a Pydantic CandidateMetrics. Hand-flatten the fields here.
    """
    # pylint: disable=import-outside-toplevel
    from multi_scenario.domain.models import MetricRecord

    m = summary.best_candidate_metrics
    fields = (
        ("M1_success_rate", m.M1_success_rate),
        ("M2_avg_return", m.M2_avg_return),
        ("M3_steps", m.M3_steps),
        ("M4_collisions", m.M4_collisions),
        ("M5_tokens", m.M5_tokens),
        ("M6_coverage_progress", m.M6_coverage_progress),
        ("M7_sample_efficiency", m.M7_sample_efficiency),
        ("M8_agent_utilization", m.M8_agent_utilization),
        ("M9_spatial_spread", m.M9_spatial_spread),
    )
    return [MetricRecord(name=name, value=value) for name, value in fields]
