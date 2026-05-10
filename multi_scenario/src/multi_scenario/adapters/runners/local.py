"""LocalRunner — in-process Runner adapter wrapping ExperimentService.

Wires concrete adapters via the application-layer factories, then delegates
to ``ExperimentService.run``. Provenance is built via an injected
``provenance_factory`` so this module stays free of git / package-version
I/O — the real ``ProvenanceWriter`` lands at F2.7 and plugs in here.

After the service returns, the runner builds the run-end manifest
(``output/report.json``, F2.10) and saves it via the concrete storage
adapter. The report writer is off the ``Storage`` Protocol surface (per
F1.9) so we narrow to the concrete adapter that supports it.
"""

from pathlib import Path
from typing import Callable

from multi_scenario.adapters.metrics.common import CommonMetricsBundle
from multi_scenario.adapters.provenance.writer import ProvenanceWriter
from multi_scenario.adapters.storage.local import LocalStorageAdapter
from multi_scenario.adapters.storage.report_builder import ReportBuilder
from multi_scenario.application.experiment_service import ExperimentService
from multi_scenario.application.factories import (
    make_algorithm,
    make_scenario,
    make_storage,
)
from multi_scenario.domain.models import ExperimentConfig, ExperimentResult, Provenance
from multi_scenario.domain.ports import Logger, MetricsBundle


class LocalRunner:
    """In-process runner: builds deps via factories, delegates to ExperimentService."""

    # Three injected deps + the no-comm metrics bundle is the design; pylint
    # flags too-few-public-methods on this single-method runner.
    # pylint: disable=too-few-public-methods

    name = "local"
    # F5.7: local fs gives direct checkpoint access; resume is supported.
    supports_resume: bool = True

    def __init__(
        self,
        logger: Logger,
        provenance_factory: Callable[[ExperimentConfig], Provenance] | None = None,
        metrics: MetricsBundle | None = None,
    ) -> None:
        self._logger = logger
        self._provenance_factory: Callable[[ExperimentConfig], Provenance] = (
            provenance_factory or ProvenanceWriter()
        )
        self._metrics: MetricsBundle = metrics or CommonMetricsBundle()

    def run(
        self,
        cfg: ExperimentConfig,
        run_dir: Path,
        resume_from: Path | None = None,
    ) -> ExperimentResult:
        """Resolve adapter names, build the service, delegate, and return the result."""
        # F7.7.A4: fail-fast when YAML asks for CUDA but the host has none.
        # Without this check, BenchMARL crashes ~30s in with a deep TorchRL
        # traceback that doesn't mention CUDA. Cheaper to surface here.
        if cfg.training.device == "cuda":
            self._assert_cuda_available()
        scenario = make_scenario(cfg.scenario.type)
        algorithm = make_algorithm(cfg.algorithm.type)
        storage_name = cfg.runtime.storage.type if cfg.runtime is not None else "fs"
        storage = make_storage(storage_name)

        # F2.10.1: opt in the LocalStorageAdapter writer so eval_episodes.json
        # gets produced. Off the Storage Protocol on purpose (F1.9 minimalism).
        eval_writer = (
            storage.save_eval_episodes
            if isinstance(storage, LocalStorageAdapter)
            else None
        )

        service = ExperimentService(
            scenario=scenario,
            algorithm=algorithm,
            metrics=self._metrics,
            storage=storage,
            logger=self._logger,
            eval_episodes_writer=eval_writer,
        )
        provenance = self._provenance_factory(cfg)
        result = service.run(cfg, run_dir, provenance, resume_from=resume_from)

        # F2.10: build + save the run-end manifest. ``save_report`` is off the
        # ``Storage`` Protocol surface, so narrow to the concrete adapter.
        # Future S3 adapter would either grow its own ``save_report`` or this
        # branch would dispatch via a mixin.
        if isinstance(storage, LocalStorageAdapter):
            run_state = storage.load_run_state(run_dir)
            report = ReportBuilder().build(run_dir, result, run_state)
            storage.save_report(run_dir, report)

        return result

    @staticmethod
    def _assert_cuda_available() -> None:
        """Raise immediately if ``training.device=cuda`` but no CUDA on this host.

        This is the local-runner equivalent of the OVH-side preflight ``GPU
        flavor`` check. Inside the OVH container the same call also fires —
        if the OVH flavor is wrongly picked as CPU-only, BenchMARL would
        crash later; this surfaces the misconfig at minute 0.
        """
        # pylint: disable=import-outside-toplevel
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "training.device=cuda but torch.cuda.is_available()=False on "
                "this host. Either run on a CUDA-capable machine, set "
                "training.device=cpu in the YAML, or pass --runner ovh to "
                "submit to a GPU-bearing OVH node."
            )
