"""Runner port — the 'where to run' abstraction.

Concrete adapters: ``LocalRunner`` (in-process, F2.6) wraps ``ExperimentService``
directly; ``OvhRunner`` (F6.2) submits a job that runs ``ExperimentService``
remotely. The Protocol stays infra-agnostic.
"""

from pathlib import Path
from typing import Protocol, runtime_checkable

from multi_scenario.domain.models import ExperimentConfig, ExperimentResult


@runtime_checkable
class Runner(Protocol):
    """Domain port for run execution."""

    # Single-method Protocol — pylint flags as candidate-for-function but that
    # doesn't apply to structurally-typed ports.
    # pylint: disable=too-few-public-methods

    name: str
    supports_resume: bool

    def run(
        self,
        cfg: ExperimentConfig,
        run_dir: Path,
        resume_from: Path | None = None,
    ) -> ExperimentResult:
        """Execute one experiment run end-to-end and return its result.

        ``resume_from`` (F5.7): optional path to a BenchMARL checkpoint to
        load before continuing training. Only meaningful for runners with
        ``supports_resume = True``; others raise / ignore.
        """
