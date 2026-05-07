"""RunsJsonWriter — slim cross-run manifest (F5.3) per §3.5.3.

Walks the same DONE-run folders as ``RunsCsvWriter``, builds a
``RunsManifest`` with scope summary, per-metric rankings, and pointer-only
``runs[]`` entries linking each run's ``output/report.json`` (relative to
``exp_type_dir``). Atomic write-rename with a one-step ``runs.previous.json``
backup.

Rankings are sorted descending by raw value — consumers know which metrics
are minimize-vs-maximize (M1/M2/M6/M7 = higher better; M3/M4/M5/M8/M9 =
context-dependent). None values are filtered out per metric.
"""

import os
import shutil
from pathlib import Path
from typing import Any

from multi_scenario.adapters.storage.local import LocalStorageAdapter
from multi_scenario.domain.models import (
    ManifestRunEntry,
    ManifestScope,
    RankingEntry,
    RunsManifest,
)

# All M1-M9 metric names — the manifest emits a ranking for any of these
# that has at least one non-None value across the consolidated runs.
_METRIC_NAMES = (
    "M1_success_rate",
    "M2_avg_return",
    "M3_steps",
    "M4_collisions",
    "M5_tokens",
    "M6_coverage_progress",
    "M7_sample_efficiency",
    "M8_agent_utilization",
    "M9_spatial_spread",
)


class RunsJsonWriter:
    """Builds ``runs.json`` from all DONE runs under an ``<exp_type_dir>``."""

    # Single public method by design; pylint defaults flag this even though
    # the class is the natural unit for the rankings + scope assembly logic.
    # pylint: disable=too-few-public-methods

    def consolidate(self, exp_type_dir: Path) -> Path:
        """Write ``<exp_type_dir>/runs.json`` summarising all DONE runs.

        Atomic write-rename via ``runs.json.tmp`` → ``os.replace``. If
        ``runs.json`` exists, copies it to ``runs.previous.json`` first.
        """
        if not exp_type_dir.is_dir():
            raise FileNotFoundError(f"exp_type_dir not found: {exp_type_dir}")

        runs_data = list(_collect_runs(exp_type_dir))
        manifest = RunsManifest(
            scope=_build_scope(runs_data),
            csv="runs.csv",
            rankings=_build_rankings(exp_type_dir, runs_data),
            runs=[_run_entry(exp_type_dir, run_dir) for (run_dir, _) in runs_data],
        )

        target = exp_type_dir / "runs.json"
        if target.exists():
            shutil.copy2(target, exp_type_dir / "runs.previous.json")
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, target)
        return target


def _collect_runs(exp_type_dir: Path) -> list[tuple[Path, Any]]:
    """Yield ``(run_dir, ExperimentResult)`` for every DONE run."""
    storage = LocalStorageAdapter()
    for child in sorted(exp_type_dir.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "output" / "metrics.json").is_file():
            continue
        if not (child / "run_state.json").is_file():
            continue
        try:
            state = storage.load_run_state(child)
        except (OSError, ValueError):
            continue
        if state.state.value != "DONE":
            continue
        try:
            result = storage.load_result(child)
        except (OSError, ValueError):
            continue
        yield child, result


def _build_scope(runs_data: list[tuple[Path, Any]]) -> ManifestScope:
    """Aggregate scope counts/lists across the consolidated runs."""
    exp_ids: set[str] = set()
    seeds: set[int] = set()
    algorithms: set[str] = set()
    for _, result in runs_data:
        exp_ids.add(result.exp_id)
        seeds.add(result.seed)
        algorithms.add(result.algorithm)
    return ManifestScope(
        n_runs=len(runs_data),
        exp_ids=sorted(exp_ids),
        seeds=sorted(seeds),
        algorithms=sorted(algorithms),
    )


def _build_rankings(
    exp_type_dir: Path, runs_data: list[tuple[Path, Any]]
) -> dict[str, list[RankingEntry]]:
    """Per-metric rankings: descending raw value, None values dropped."""
    rankings: dict[str, list[RankingEntry]] = {}
    for metric in _METRIC_NAMES:
        entries: list[RankingEntry] = []
        for run_dir, result in runs_data:
            value = next(
                (m.value for m in result.metrics if m.name == metric and m.value is not None),
                None,
            )
            if value is None:
                continue
            entries.append(
                RankingEntry(
                    run_id=result.run_id,
                    value=float(value),
                    report=_report_rel_path(exp_type_dir, run_dir),
                )
            )
        if entries:
            entries.sort(key=lambda e: e.value, reverse=True)
            rankings[metric] = entries
    return rankings


def _run_entry(exp_type_dir: Path, run_dir: Path) -> ManifestRunEntry:
    """Build one ``runs[]`` entry pointing at the run's ``report.json`` if present."""
    return ManifestRunEntry(
        run_id=run_dir.name.split("__")[0],
        report=_report_rel_path(exp_type_dir, run_dir),
    )


def _report_rel_path(exp_type_dir: Path, run_dir: Path) -> str | None:
    """Return ``<run_folder>/output/report.json`` relative to ``exp_type_dir`` or None."""
    report = run_dir / "output" / "report.json"
    if not report.is_file():
        return None
    return report.relative_to(exp_type_dir).as_posix()
