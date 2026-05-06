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

    def run(self, cfg: ExperimentConfig, run_dir: Path) -> ExperimentResult:
        """Execute one experiment run end-to-end and return its result."""
