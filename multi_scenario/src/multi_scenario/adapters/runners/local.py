"""LocalRunner — in-process Runner adapter wrapping ExperimentService.

Wires concrete adapters via the application-layer factories, then delegates
to ``ExperimentService.run``. Provenance is built via an injected
``provenance_factory`` so this module stays free of git / package-version
I/O — the real ``ProvenanceWriter`` lands at F2.7 and plugs in here.
"""

from pathlib import Path
from typing import Callable

from multi_scenario.adapters.metrics.common import CommonMetricsBundle
from multi_scenario.application.experiment_service import ExperimentService
from multi_scenario.application.factories import (
    make_algorithm,
    make_scenario,
    make_storage,
)
from multi_scenario.domain.models import (
    ExperimentConfig,
    ExperimentResult,
    Provenance,
)
from multi_scenario.domain.ports import Logger, MetricsBundle


class LocalRunner:
    """In-process runner: builds deps via factories, delegates to ExperimentService."""

    # Three injected deps + the no-comm metrics bundle is the design; pylint
    # flags too-few-public-methods on this single-method runner.
    # pylint: disable=too-few-public-methods

    name = "local"

    def __init__(
        self,
        logger: Logger,
        provenance_factory: Callable[[ExperimentConfig], Provenance],
        metrics: MetricsBundle | None = None,
    ) -> None:
        self._logger = logger
        self._provenance_factory = provenance_factory
        self._metrics: MetricsBundle = metrics or CommonMetricsBundle()

    def run(self, cfg: ExperimentConfig, run_dir: Path) -> ExperimentResult:
        """Resolve adapter names, build the service, delegate, and return the result."""
        scenario = make_scenario(cfg.scenario.type)
        algorithm = make_algorithm(cfg.algorithm.type)
        storage_name = cfg.runtime.storage.type if cfg.runtime is not None else "fs"
        storage = make_storage(storage_name)

        service = ExperimentService(
            scenario=scenario,
            algorithm=algorithm,
            metrics=self._metrics,
            storage=storage,
            logger=self._logger,
        )
        provenance = self._provenance_factory(cfg)
        return service.run(cfg, run_dir, provenance)
