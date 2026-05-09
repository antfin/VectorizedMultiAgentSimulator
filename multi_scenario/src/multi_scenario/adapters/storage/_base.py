"""Shared scaffolding for the ``Storage`` Protocol implementations.

Both ``LocalStorageAdapter`` and ``S3StorageAdapter`` implement the same eight
Protocol methods (save/load × {config, provenance, result, run_state}) over
different backing stores. This base puts the §3.5.2 layout in ONE place;
subclasses override the ``_write_text`` / ``_read_text`` primitives to talk to
their backing store (filesystem / S3 / future variants).
"""

from pathlib import Path

from multi_scenario.domain.models import (
    ExperimentConfig,
    ExperimentResult,
    Provenance,
    RunStateRecord,
)

# §3.5.2 layout — single source of truth. Forward-slash separated so the same
# string works as a relative ``Path`` join AND as an S3 key suffix.
_CONFIG_REL = "input/config.json"
_PROVENANCE_REL = "input/provenance.json"
_RESULT_REL = "output/metrics.json"
_RUN_STATE_REL = "run_state.json"


class StorageAdapterBase:
    """Template-method scaffolding for the eight ``Storage`` Protocol methods.

    Subclasses provide the I/O primitives (``_write_text`` / ``_read_text``);
    the Protocol-surface methods are inherited unchanged.
    """

    name: str = "base"

    def save_config(self, run_dir: Path, config: ExperimentConfig) -> None:
        """Persist the resolved config to ``input/config.json``."""
        self._write_text(run_dir, _CONFIG_REL, config.model_dump_json(indent=2))

    def save_provenance(self, run_dir: Path, provenance: Provenance) -> None:
        """Persist provenance to ``input/provenance.json``."""
        self._write_text(run_dir, _PROVENANCE_REL, provenance.model_dump_json(indent=2))

    def save_result(self, run_dir: Path, result: ExperimentResult) -> None:
        """Persist the final aggregate to ``output/metrics.json``."""
        self._write_text(run_dir, _RESULT_REL, result.model_dump_json(indent=2))

    def save_run_state(self, run_dir: Path, state: RunStateRecord) -> None:
        """Persist the run-state record to ``run_state.json``."""
        self._write_text(run_dir, _RUN_STATE_REL, state.model_dump_json(indent=2))

    def load_config(self, run_dir: Path) -> ExperimentConfig:
        """Read back ``input/config.json`` as :class:`ExperimentConfig`."""
        return ExperimentConfig.model_validate_json(
            self._read_text(run_dir, _CONFIG_REL)
        )

    def load_provenance(self, run_dir: Path) -> Provenance:
        """Read back ``input/provenance.json`` as :class:`Provenance`."""
        return Provenance.model_validate_json(self._read_text(run_dir, _PROVENANCE_REL))

    def load_result(self, run_dir: Path) -> ExperimentResult:
        """Read back ``output/metrics.json`` as :class:`ExperimentResult`."""
        return ExperimentResult.model_validate_json(
            self._read_text(run_dir, _RESULT_REL)
        )

    def load_run_state(self, run_dir: Path) -> RunStateRecord:
        """Read back ``run_state.json`` as :class:`RunStateRecord`."""
        return RunStateRecord.model_validate_json(
            self._read_text(run_dir, _RUN_STATE_REL)
        )

    def _write_text(self, run_dir: Path, rel: str, body: str) -> None:
        raise NotImplementedError

    def _read_text(self, run_dir: Path, rel: str) -> str:
        raise NotImplementedError
