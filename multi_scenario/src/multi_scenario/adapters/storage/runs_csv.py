"""RunsCsvWriter — cross-run leaderboard CSV (F5.2).

Walks ``<exp_type_dir>/<run_folder>/`` matching the §3.5.2 layout, filters
runs in ``DONE`` state, and emits one ``record_type=final`` row per run to
``<exp_type_dir>/runs.csv``. Atomic write-rename with a one-step backup
(``runs.previous.csv``) per §3.5.3.

Eval-step rows (``record_type=eval``) are deferred to F5.2.1 — they need
either custom BenchMARL eval callbacks or scalar-CSV aggregation mapping
to M1-M9, neither of which exist yet.
"""

import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from multi_scenario.adapters.storage.local import LocalStorageAdapter

# Canonical columns always present on a final row (independent of the
# scenario's config_snapshot keys). Used as the header for empty CSVs and
# as the leading column order; config_snapshot keys append after these.
_CANONICAL_COLUMNS = (
    "record_type",
    "run_id",
    "exp_id",
    "scenario",
    "algorithm",
    "seed",
    "run_timestamp",
    "M1_success_rate",
    "M2_avg_return",
    "M3_steps",
    "M4_collisions",
    "M5_tokens",
    "M6_coverage_progress",
    "M7_sample_efficiency",
    "M8_agent_utilization",
    "M9_spatial_spread",
    "n_envs",
    "n_eval_episodes",
    "convergence_frame",
    "duration_seconds",
)


class RunsCsvWriter:
    """Builds ``runs.csv`` from all DONE runs under an ``<exp_type_dir>``."""

    # One public method is the whole point of a consolidator; pylint defaults
    # flag this even though the class will gain helpers / state for F5.2.1
    # eval-row aggregation later.
    # pylint: disable=too-few-public-methods

    def consolidate(self, exp_type_dir: Path) -> Path:
        """Write ``<exp_type_dir>/runs.csv`` with one final row per DONE run.

        Atomic write-rename: writes to ``runs.csv.tmp`` then ``os.replace``s
        onto ``runs.csv``. If a previous ``runs.csv`` exists, copies it to
        ``runs.previous.csv`` first (one-step rollback per §3.5.3).
        """
        if not exp_type_dir.is_dir():
            raise FileNotFoundError(f"exp_type_dir not found: {exp_type_dir}")

        rows = list(self._collect_rows(exp_type_dir))
        # Empty result → still write the canonical header so consumers can
        # parse the file without an EmptyDataError.
        df = pd.DataFrame(rows, columns=list(_CANONICAL_COLUMNS) if not rows else None)
        if rows:
            df = _reorder_columns(df)

        target = exp_type_dir / "runs.csv"
        if target.exists():
            shutil.copy2(target, exp_type_dir / "runs.previous.csv")

        tmp = target.with_suffix(".csv.tmp")
        df.to_csv(tmp, index=False, na_rep="N/A")
        os.replace(tmp, target)
        return target

    def _collect_rows(self, exp_type_dir: Path):
        """Yield one final-row dict per completed run under ``exp_type_dir``."""
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
            yield _build_final_row(result, state)


def _build_final_row(result: Any, state: Any) -> dict[str, Any]:
    """Compose one ``record_type=final`` row from a result + run-state record."""
    started = state.transitions[0].ts
    finished = state.transitions[-1].ts
    duration = (finished - started).total_seconds()
    return {
        "record_type": "final",
        **result.to_flat_dict(),
        "duration_seconds": duration,
    }


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Place canonical columns first; appended config_snapshot keys after."""
    leading = [c for c in _CANONICAL_COLUMNS if c in df.columns]
    trailing = [c for c in df.columns if c not in _CANONICAL_COLUMNS]
    return df[leading + trailing]
