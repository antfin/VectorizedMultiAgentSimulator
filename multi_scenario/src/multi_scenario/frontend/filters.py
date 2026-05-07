"""Pure filter helpers for the F7.2 Experiments Browser table.

Convention: an empty list means "no filter on this axis"; passing
``scenarios=[]`` returns rows for *all* scenarios. Search is a case-insensitive
substring match on ``run_id``. All filters compose (AND).
"""

import pandas as pd


def filter_runs(
    df: pd.DataFrame,
    scenarios: list[str] | None = None,
    algorithms: list[str] | None = None,
    states: list[str] | None = None,
    search: str = "",
) -> pd.DataFrame:
    """Filter ``df`` by intersecting axis selections.

    Args:
        df: per-run DataFrame as produced by ``load_runs``.
        scenarios: keep rows whose ``scenario`` is in this list. Empty/None → no filter.
        algorithms: keep rows whose ``algorithm`` is in this list. Empty/None → no filter.
        states: keep rows whose ``state`` is in this list. Empty/None → no filter.
        search: case-insensitive substring match on ``run_id``. Empty → no filter.
    """
    out = df
    if scenarios and "scenario" in out.columns:
        out = out[out["scenario"].isin(scenarios)]
    if algorithms and "algorithm" in out.columns:
        out = out[out["algorithm"].isin(algorithms)]
    if states and "state" in out.columns:
        out = out[out["state"].isin(states)]
    if search and "run_id" in out.columns:
        out = out[out["run_id"].str.contains(search, case=False, na=False)]
    return out
