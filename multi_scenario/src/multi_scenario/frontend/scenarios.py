"""Per-scenario render functions for the Streamlit landing page (F7.1).

Each ``render_*`` function takes a per-scenario DataFrame (rows = runs of
that scenario) and writes its tab contents directly via Streamlit. The
imperative shape is deliberate — *conversational updates* are the agreed
update mechanism, so each scenario's KPIs / Top Runs / charts read
top-to-bottom and are easy to tweak in place.

KPI / chart picks per scenario are documented inline; the rationale lives
in :doc:`/docs/dashboard_spec` (Phase 7 design notes).
"""

from typing import Callable

import pandas as pd
import streamlit as st

from .charts import bar_by_algo_with_stderr, box_by_algo, scatter_xy


# ── Shared helpers ────────────────────────────────────────────────────


_KpiSpec = tuple[str, Callable[[pd.DataFrame], float], str]


def _kpi_row(df: pd.DataFrame, kpis: list[_KpiSpec]) -> None:
    """Render a row of KPI tiles. ``kpis`` is a list of ``(label, agg_fn, fmt)``."""
    cols = st.columns(len(kpis))
    for col, (label, agg_fn, fmt) in zip(cols, kpis):
        try:
            value = agg_fn(df)
        except (KeyError, ValueError, TypeError):
            value = None
        if value is None or (isinstance(value, float) and pd.isna(value)):
            col.metric(label, "—")
        else:
            col.metric(label, fmt.format(value))


def _top_runs(
    df: pd.DataFrame,
    sort_cols: list[tuple[str, bool]],
    metric_cols: list[str],
    n: int = 10,
) -> None:
    """Render the top-``n`` runs table, sorted by ``sort_cols`` (col, ascending)."""
    st.subheader("Top Runs")
    base_cols = ["run_id", "algorithm", "seed", "state"]
    display_cols = base_cols + [c for c in metric_cols if c in df.columns]
    valid_sort = [(c, asc) for c, asc in sort_cols if c in df.columns]
    if valid_sort:
        cols, ascending = zip(*valid_sort)
        sorted_df = df.sort_values(list(cols), ascending=list(ascending))
    else:
        sorted_df = df
    top_df = sorted_df[display_cols].head(n)

    fmt: dict = {}
    for col in top_df.columns:
        if "rate" in col or "progress" in col:
            fmt[col] = "{:.0%}".format
        elif col.startswith("M"):
            fmt[col] = "{:.2f}".format

    st.dataframe(
        top_df.style.format(fmt, na_rep="—"),
        use_container_width=True,
        hide_index=True,
    )


def _safe_max(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns or df[col].isna().all():
        return None
    return float(df[col].max())


def _safe_min(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns or df[col].isna().all():
        return None
    return float(df[col].min())


def _safe_mean(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns or df[col].isna().all():
        return None
    return float(df[col].mean())


# ── Scenario render functions ────────────────────────────────────────


def render_discovery(df: pd.DataFrame) -> None:
    """Discovery — cover N targets with K agents-per-target. Primary: M1 + M6."""
    _kpi_row(
        df,
        [
            ("Total Runs", len, "{:d}"),
            ("Best M1 Success", lambda d: _safe_max(d, "M1_success_rate"), "{:.0%}"),
            ("Mean M6 Coverage", lambda d: _safe_mean(d, "M6_coverage_progress"), "{:.0%}"),
            ("Mean M3 Steps", lambda d: _safe_mean(d, "M3_steps"), "{:.1f}"),
        ],
    )
    st.markdown("---")
    _top_runs(
        df,
        sort_cols=[("M1_success_rate", False), ("M6_coverage_progress", False), ("M3_steps", True)],
        metric_cols=["M1_success_rate", "M6_coverage_progress", "M3_steps", "M4_collisions"],
    )
    st.markdown("---")
    left, right = st.columns(2)
    with left:
        scatter_xy(df, "M1_success_rate", "M3_steps",
                   "M1 vs M3 — Success vs Speed",
                   xlabel="Success Rate", ylabel="Avg Steps")
        box_by_algo(df, "M6_coverage_progress",
                    "M6 Coverage by Algorithm", ylabel="Coverage Progress")
    with right:
        scatter_xy(df, "M1_success_rate", "M4_collisions",
                   "M1 vs M4 — Success vs Safety",
                   xlabel="Success Rate", ylabel="Collisions")
        scatter_xy(df, "M9_spatial_spread", "M1_success_rate",
                   "Spatial Spread vs Success",
                   xlabel="Spatial Spread (M9)", ylabel="Success Rate")


def render_navigation(df: pd.DataFrame) -> None:
    """Navigation — point-to-point with obstacles. Primary: M1 + M3 + M4."""
    _kpi_row(
        df,
        [
            ("Total Runs", len, "{:d}"),
            ("Best M1 Success", lambda d: _safe_max(d, "M1_success_rate"), "{:.0%}"),
            ("Mean M3 Steps", lambda d: _safe_mean(d, "M3_steps"), "{:.1f}"),
            ("Worst M4 Collisions", lambda d: _safe_max(d, "M4_collisions"), "{:.1f}"),
        ],
    )
    st.markdown("---")
    _top_runs(
        df,
        sort_cols=[("M1_success_rate", False), ("M4_collisions", True), ("M3_steps", True)],
        metric_cols=["M1_success_rate", "M3_steps", "M4_collisions", "M2_avg_return"],
    )
    st.markdown("---")
    left, right = st.columns(2)
    with left:
        scatter_xy(df, "M1_success_rate", "M3_steps",
                   "M1 vs M3 — Success vs Speed",
                   xlabel="Success Rate", ylabel="Avg Steps")
        box_by_algo(df, "M4_collisions",
                    "M4 Collisions by Algorithm", ylabel="Collisions")
    with right:
        scatter_xy(df, "M3_steps", "M4_collisions",
                   "M3 vs M4 — Speed–Safety Pareto",
                   xlabel="Avg Steps", ylabel="Collisions")
        bar_by_algo_with_stderr(df, "M1_success_rate",
                                "M1 by Algorithm (mean ± SE across seeds)",
                                ylabel="Success Rate")


def render_transport(df: pd.DataFrame) -> None:
    """Transport — cooperative carrying. Primary: M2 + M8."""
    _kpi_row(
        df,
        [
            ("Total Runs", len, "{:d}"),
            ("Best M2 Avg Return", lambda d: _safe_max(d, "M2_avg_return"), "{:.2f}"),
            ("Mean M8 Utilization", lambda d: _safe_mean(d, "M8_agent_utilization"), "{:.2f}"),
            ("Mean M4 Collisions", lambda d: _safe_mean(d, "M4_collisions"), "{:.1f}"),
        ],
    )
    st.markdown("---")
    _top_runs(
        df,
        sort_cols=[
            ("M2_avg_return", False),
            ("M8_agent_utilization", False),
            ("M4_collisions", True),
        ],
        metric_cols=["M2_avg_return", "M8_agent_utilization", "M4_collisions", "M3_steps"],
    )
    st.markdown("---")
    left, right = st.columns(2)
    with left:
        scatter_xy(df, "M2_avg_return", "M3_steps",
                   "M2 vs M3 — Return vs Speed",
                   xlabel="Avg Return", ylabel="Avg Steps")
        scatter_xy(df, "M8_agent_utilization", "M2_avg_return",
                   "M8 vs M2 — Cooperation vs Return",
                   xlabel="Agent Utilization", ylabel="Avg Return")
    with right:
        box_by_algo(df, "M8_agent_utilization",
                    "M8 Utilization by Algorithm", ylabel="Agent Utilization")
        bar_by_algo_with_stderr(df, "M2_avg_return",
                                "M2 by Algorithm (mean ± SE across seeds)",
                                ylabel="Avg Return")


def render_flocking(df: pd.DataFrame) -> None:
    """Flocking — cohesion + separation + alignment. Primary: M2 + M9 + M4."""
    _kpi_row(
        df,
        [
            ("Total Runs", len, "{:d}"),
            ("Best M2 Avg Return", lambda d: _safe_max(d, "M2_avg_return"), "{:.2f}"),
            ("Mean M9 Spread", lambda d: _safe_mean(d, "M9_spatial_spread"), "{:.2f}"),
            ("Worst M4 Collisions", lambda d: _safe_max(d, "M4_collisions"), "{:.1f}"),
        ],
    )
    st.markdown("---")
    _top_runs(
        df,
        sort_cols=[("M2_avg_return", False), ("M9_spatial_spread", True), ("M4_collisions", True)],
        metric_cols=["M2_avg_return", "M9_spatial_spread", "M4_collisions", "M3_steps"],
    )
    st.markdown("---")
    left, right = st.columns(2)
    with left:
        scatter_xy(df, "M2_avg_return", "M9_spatial_spread",
                   "M2 vs M9 — Return vs Cohesion",
                   xlabel="Avg Return", ylabel="Spatial Spread")
        box_by_algo(df, "M4_collisions",
                    "M4 Collisions by Algorithm", ylabel="Collisions")
    with right:
        box_by_algo(df, "M9_spatial_spread",
                    "M9 Spatial Spread by Algorithm", ylabel="Spatial Spread")
        bar_by_algo_with_stderr(df, "M2_avg_return",
                                "M2 by Algorithm (mean ± SE across seeds)",
                                ylabel="Avg Return")


# ── Dispatch table ────────────────────────────────────────────────────

SCENARIO_RENDERERS: dict[str, Callable[[pd.DataFrame], None]] = {
    "discovery": render_discovery,
    "navigation": render_navigation,
    "transport": render_transport,
    "flocking": render_flocking,
}
