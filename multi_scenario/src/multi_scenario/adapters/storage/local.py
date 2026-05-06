"""LocalStorageAdapter — filesystem implementation of the Storage port.

Writes the §3.5.2 layout to disk via JSON for everything we own. Pydantic's
``model_dump_json`` / ``model_validate_json`` handles datetime fields
(Provenance, RunStateRecord) cleanly via built-in ISO-8601 (de)serialization.

Optional artefacts (eval_episodes, report, videos, log) and the cross-run
``runs.csv`` / ``runs.json`` are not in the Storage Protocol surface and are
added on later concrete adapters when each writer feature lands (F2.7 /
F2.10 / F2.11 / F5.2 / F5.3).
"""

from pathlib import Path

from multi_scenario.domain.models import (
    ExperimentConfig,
    ExperimentResult,
    Provenance,
    RunStateRecord,
)


class LocalStorageAdapter:
    """Persists per-run JSON files to the local filesystem under ``run_dir``."""

    name = "fs"

    def save_config(self, run_dir: Path, config: ExperimentConfig) -> None:
        """Write the resolved config to ``input/config.json``."""
        self._write(run_dir / "input" / "config.json", config.model_dump_json(indent=2))

    def save_provenance(self, run_dir: Path, provenance: Provenance) -> None:
        """Write provenance to ``input/provenance.json``."""
        self._write(
            run_dir / "input" / "provenance.json",
            provenance.model_dump_json(indent=2),
        )

    def save_result(self, run_dir: Path, result: ExperimentResult) -> None:
        """Write the final aggregate to ``output/metrics.json``."""
        self._write(run_dir / "output" / "metrics.json", result.model_dump_json(indent=2))

    def save_run_state(self, run_dir: Path, state: RunStateRecord) -> None:
        """Write the run-state record to ``run_state.json`` at the run-folder root."""
        self._write(run_dir / "run_state.json", state.model_dump_json(indent=2))

    def load_config(self, run_dir: Path) -> ExperimentConfig:
        """Read back ``input/config.json`` as ``ExperimentConfig``."""
        return ExperimentConfig.model_validate_json(
            (run_dir / "input" / "config.json").read_text(encoding="utf-8")
        )

    def load_provenance(self, run_dir: Path) -> Provenance:
        """Read back ``input/provenance.json`` as ``Provenance``."""
        return Provenance.model_validate_json(
            (run_dir / "input" / "provenance.json").read_text(encoding="utf-8")
        )

    def load_result(self, run_dir: Path) -> ExperimentResult:
        """Read back ``output/metrics.json`` as ``ExperimentResult``."""
        return ExperimentResult.model_validate_json(
            (run_dir / "output" / "metrics.json").read_text(encoding="utf-8")
        )

    def load_run_state(self, run_dir: Path) -> RunStateRecord:
        """Read back ``run_state.json`` as ``RunStateRecord``."""
        return RunStateRecord.model_validate_json(
            (run_dir / "run_state.json").read_text(encoding="utf-8")
        )

    @staticmethod
    def _write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
