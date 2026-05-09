"""Pure helpers for the F7.4 Compare page — aggregations, config flattening, diffs.

Streamlit-free so each helper is unit-testable. Used by ``pages/comparison.py``
to build the cross-experiment leaderboard, distribution, and config-diff
sections.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from multi_scenario.adapters.storage.local import LocalStorageAdapter
from multi_scenario.domain.models import ExperimentConfig


def aggregate_metric(
    df: pd.DataFrame,
    metric: str,
    by: list[str],
    how: str = "mean",
) -> pd.DataFrame:
    """Group ``df`` by ``by`` columns and aggregate ``metric`` per group.

    Returns a DataFrame with the ``by`` columns plus ``mean`` (the aggregated
    metric) and ``sem`` (standard error across rows in the group, useful for
    error-bar plotting). Rows where the metric is NaN are dropped per group.

    Args:
        df: per-run DataFrame (typically from ``load_runs``).
        metric: column name to aggregate (e.g. ``M1_success_rate``).
        by: grouping columns (e.g. ``["scenario", "algorithm"]``).
        how: ``"mean"`` / ``"median"`` / ``"max"`` / ``"min"``.
    """
    if metric not in df.columns or not all(c in df.columns for c in by):
        return pd.DataFrame(columns=[*by, "mean", "sem"])
    sub = df.dropna(subset=[metric])
    if sub.empty:
        return pd.DataFrame(columns=[*by, "mean", "sem"])
    grouped = sub.groupby(by)[metric]
    agg_fn = {"mean": "mean", "median": "median", "max": "max", "min": "min"}.get(
        how, "mean"
    )
    out = grouped.agg(agg_fn).reset_index().rename(columns={metric: "mean"})
    sem = grouped.sem().reset_index().rename(columns={metric: "sem"})
    out = out.merge(sem, on=by, how="left")
    out["sem"] = out["sem"].fillna(0.0)
    return out


def flatten_config(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict to dotted keys (``a.b.c`` → ``"a.b.c"``).

    Lists are kept as-is (rendered as ``str(value)`` downstream); only dicts
    recurse. ``None`` values flatten to ``"—"`` so the diff table reads cleanly.
    """
    out: dict[str, Any] = {}
    for key, value in payload.items():
        full = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten_config(value, full))
        else:
            out[full] = "—" if value is None else value
    return out


def build_config_diff_table(
    run_dirs: list[Path],
) -> "tuple[pd.DataFrame, list[str]]":
    """Build a per-run config table where rows are flattened param paths.

    Each column is one run's ``run_id``; cells are the parameter value for
    that run. Returns ``(df, diff_keys)`` where ``diff_keys`` is the list of
    rows whose values differ across columns — used by the page to highlight
    those rows.

    Returns an empty DataFrame and empty diff list if ``run_dirs`` is empty
    or no run yields a loadable config.
    """
    if not run_dirs:
        return pd.DataFrame(), []
    storage = LocalStorageAdapter()
    columns: dict[str, dict[str, Any]] = {}
    for run_dir in run_dirs:
        try:
            cfg: ExperimentConfig = storage.load_config(run_dir)
        except (OSError, ValueError):
            continue
        flat = flatten_config(cfg.model_dump())
        columns[run_dir.name] = flat
    if not columns:
        return pd.DataFrame(), []
    all_keys = sorted({k for col in columns.values() for k in col})
    table = pd.DataFrame(
        {name: [flat.get(k, "—") for k in all_keys] for name, flat in columns.items()},
        index=all_keys,
    )
    # A row "differs" if any pair of columns has different stringified values.
    diff_keys = [
        k
        for k in all_keys
        if len({stringify_diff_value(table.loc[k, c]) for c in columns}) > 1
    ]
    return table, diff_keys


def stringify_diff_value(value: Any) -> str:
    """Stringify a config value for diff comparison.

    Lists become tuple-of-strings so list ordering matters; everything else
    falls back to ``str(value)``. Public because the F7.4 page reuses this
    when computing per-cell highlighting (otherwise we'd duplicate).
    """
    if isinstance(value, list):
        return str(tuple(value))
    return str(value)
