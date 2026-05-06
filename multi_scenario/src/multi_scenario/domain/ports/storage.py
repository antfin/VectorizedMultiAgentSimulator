"""Storage port — Protocol for per-run persistence (config, provenance, result, run_state).

Cross-run aggregations (``runs.csv`` / ``runs.json``) are a separate concern;
they live with the consolidator features at F5.2/F5.3 and aren't part of this
port. Optional artefacts (eval_episodes, report, videos, log) are added later
on the concrete adapter when each writer feature lands.
"""

from pathlib import Path
from typing import Protocol, runtime_checkable

from multi_scenario.domain.models import (
    ExperimentConfig,
    ExperimentResult,
    Provenance,
    RunStateRecord,
)


@runtime_checkable
class Storage(Protocol):
    """Domain port for per-run persistence."""

    name: str

    def save_config(self, run_dir: Path, config: ExperimentConfig) -> None:
        """Persist the resolved config to ``input/config.json`` under ``run_dir``."""

    def save_provenance(self, run_dir: Path, provenance: Provenance) -> None:
        """Persist ``input/provenance.json`` under ``run_dir``."""

    def save_result(self, run_dir: Path, result: ExperimentResult) -> None:
        """Persist ``output/metrics.json`` under ``run_dir``."""

    def save_run_state(self, run_dir: Path, state: RunStateRecord) -> None:
        """Persist ``run_state.json`` under ``run_dir`` (called at every transition)."""

    def load_config(self, run_dir: Path) -> ExperimentConfig:
        """Read back the resolved config that produced this run."""

    def load_provenance(self, run_dir: Path) -> Provenance:
        """Read back the provenance record."""

    def load_result(self, run_dir: Path) -> ExperimentResult:
        """Read back the final result; used by Streamlit and consolidate."""

    def load_run_state(self, run_dir: Path) -> RunStateRecord:
        """Read the current run state; used by resume detection (F5.7)."""
