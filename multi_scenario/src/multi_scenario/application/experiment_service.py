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
    ) -> ExperimentResult:
        """Execute the full lifecycle: init → save inputs → train → eval → save result → done."""
        run_id = RunId(exp_id=cfg.experiment.id, seed=cfg.experiment.seed)
        started_at = self._now()

        state = RunStateRecord.initial(started_at)
        self._storage.save_run_state(run_dir, state)
        self._storage.save_config(run_dir, cfg)
        self._storage.save_provenance(run_dir, provenance)

        env = self._scenario.make_env(cfg.scenario, cfg.training.num_envs, cfg.experiment.seed)

        state = state.transition_to(RunState.RUNNING, self._now())
        self._storage.save_run_state(run_dir, state)

        self._logger.info(f"training {run_id}")
        artifact = self._algorithm.train(env, cfg, run_dir=run_dir)

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
            run_timestamp=started_at.strftime("%Y%m%d_%H%M"),
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

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)
