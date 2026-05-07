"""Walk an experiments tree and assemble a flat per-run DataFrame (F7.1).

Used by the Streamlit landing page: zero-setup data source — no need to
``multi-scenario consolidate`` first. Walks recursively for any
``output/metrics.json`` file and reads each via :class:`LocalStorageAdapter`
+ :meth:`ExperimentResult.to_flat_dict`.

Best-effort ``state`` lookup (``DONE`` / ``CRASHED`` / ``RUNNING`` / ``UNKNOWN``)
so the page can tag rows even before consolidator output exists.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from multi_scenario.adapters.storage.local import LocalStorageAdapter


def load_runs(experiments_dir: Path) -> pd.DataFrame:
    """Return a flat per-run DataFrame for every run under ``experiments_dir``.

    Columns: ``run_id, exp_id, scenario, algorithm, seed, run_timestamp,
    n_envs, n_eval_episodes, convergence_frame, M1_*..M9_*``, plus any
    ``config_snapshot`` keys, plus ``run_dir`` (string) and ``state``
    (``DONE`` / ``CRASHED`` / ``RUNNING`` / ``UNKNOWN``).

    Returns an empty DataFrame if ``experiments_dir`` doesn't exist or
    contains no metrics files. Reads are best-effort — any single corrupt
    metrics.json or run_state.json is skipped, never raised.
    """
    if not experiments_dir.is_dir():
        return pd.DataFrame()

    storage = LocalStorageAdapter()
    rows: list[dict[str, Any]] = []
    for metrics_file in sorted(experiments_dir.rglob("output/metrics.json")):
        run_dir = metrics_file.parent.parent
        try:
            result = storage.load_result(run_dir)
        except (OSError, ValueError):
            continue
        row = result.to_flat_dict()
        row["run_dir"] = str(run_dir)
        row["state"] = _safe_state(storage, run_dir)
        rows.append(row)
    return pd.DataFrame(rows)


def _safe_state(storage: LocalStorageAdapter, run_dir: Path) -> str:
    """Read ``run_state.json`` best-effort; return the state name or ``UNKNOWN``."""
    try:
        return storage.load_run_state(run_dir).state.value
    except (OSError, ValueError):
        return "UNKNOWN"
