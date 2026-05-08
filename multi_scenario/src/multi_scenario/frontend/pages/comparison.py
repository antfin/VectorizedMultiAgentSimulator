"""F7.4 — Cross-experiment Compare page.

One screen for comparing every run on disk. Sidebar filters narrow the
candidate set (scenario, algorithm, metric, aggregation); the main area is
top-down:

1. Headline KPIs — run / scenario / algorithm / seed counts in selection.
2. Top-N runs table with per-row checkboxes — picks which runs feed the
   config-diff section at the bottom. Form row above: N input + select-all /
   deselect-all helpers.
3. Algorithm leaderboard by scenario — grouped bar (mean ± SE).
4. Metric distribution — box plot grouped by algorithm.
5. Pareto by scenario — scatter on the chosen X/Y metrics.
6. Config-diff accordion — ALL flattened config params for the
   checkbox-selected runs, with the differing CELLS highlighted (rows where
   every column matches stay quiet).
"""

# pylint: disable=wrong-import-position,invalid-name

from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

from multi_scenario.frontend.aggregations import (
    aggregate_metric,
    build_config_diff_table,
    stringify_diff_value,
)
from multi_scenario.frontend.charts import (
    box_by_algo,
    grouped_bar_with_se,
    scatter_xy,
)
from multi_scenario.frontend.sidebar import (
    load_runs_with_cache,
    persist_widget_state,
    render_active_root_caption,
)
from multi_scenario.frontend.theme import POLIMI_LIGHT_BLUE, apply_theme

apply_theme(title="Compare", subtitle="Cross-experiment leaderboard")

experiments_dir = render_active_root_caption()
runs_df = load_runs_with_cache(experiments_dir)

if runs_df.empty:
    st.info(f"No runs under `{experiments_dir}`.")
    st.stop()

# ── Sidebar: filters ──────────────────────────────────────────────────
ALL_M_METRICS = (
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
SELECTED_KEY = "compare_selected_run_ids"


def _options(col: str) -> list[str]:
    if col not in runs_df.columns:
        return []
    return sorted(runs_df[col].dropna().unique().tolist())


def _available_metrics() -> list[str]:
    return [m for m in ALL_M_METRICS if m in runs_df.columns and runs_df[m].notna().any()]


# Filter selections persist across page nav via shadow keys (Streamlit's
# multipage nav silently drops widget state otherwise). ``compare_*``
# namespace keeps this page's filters independent from Browse's.
st.sidebar.header("Filters")

_p_scen = persist_widget_state("compare_scenarios", [])
scenarios = st.sidebar.multiselect(
    "Scenario", _options("scenario"), key="compare_scenarios"
)
st.session_state[_p_scen] = scenarios

_p_algo = persist_widget_state("compare_algorithms", [])
algorithms = st.sidebar.multiselect(
    "Algorithm", _options("algorithm"), key="compare_algorithms"
)
st.session_state[_p_algo] = algorithms

metrics = _available_metrics() or ["M1_success_rate"]
_p_metric = persist_widget_state("compare_metric", metrics[0])
metric = st.sidebar.selectbox("Primary metric", metrics, key="compare_metric")
st.session_state[_p_metric] = metric

# Y options depend on X choice; default to a value other than X.
y_options = [m for m in metrics if m != metric] or metrics
_p_metric_y = persist_widget_state("compare_metric_y", y_options[0])
metric_y = st.sidebar.selectbox(
    "Pareto Y metric",
    y_options,
    help="Y-axis for the Pareto scatter (X is the primary metric).",
    key="compare_metric_y",
)
st.session_state[_p_metric_y] = metric_y

_p_how = persist_widget_state("compare_how", "mean")
how = st.sidebar.selectbox(
    "Aggregation", ["mean", "median", "max", "min"], key="compare_how"
)
st.session_state[_p_how] = how

filtered = runs_df.copy()
if scenarios and "scenario" in filtered.columns:
    filtered = filtered[filtered["scenario"].isin(scenarios)]
if algorithms and "algorithm" in filtered.columns:
    filtered = filtered[filtered["algorithm"].isin(algorithms)]

if filtered.empty:
    st.warning("No runs match these filters. Loosen the selection on the sidebar.")
    st.stop()

# Sort once — used by the top-N table below and as the default-pick source
# for the config-diff accordion.
sort_asc = metric in ("M3_steps", "M4_collisions", "M5_tokens")
sorted_df = filtered.sort_values(metric, ascending=sort_asc).reset_index(drop=True)

# ── 1. Headline KPIs ─────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Runs", len(filtered))
k2.metric("Scenarios", filtered["scenario"].nunique() if "scenario" in filtered.columns else 0)
k3.metric("Algorithms", filtered["algorithm"].nunique() if "algorithm" in filtered.columns else 0)
k4.metric("Seeds", filtered["seed"].nunique() if "seed" in filtered.columns else 0)
st.markdown("---")

# ── 2. Top-N table with per-row checkboxes (drives the config diff) ──
st.subheader("Top runs in selection")

# Initialise persistent selection (run_ids checkboxed in earlier reruns).
if SELECTED_KEY not in st.session_state:
    st.session_state[SELECTED_KEY] = []

# Form row: select-all + deselect-all + (Top-N label + input) on one line.
# Inline label goes in its own column with ``label_visibility="collapsed"``
# on the number_input so the row reads as a tidy horizontal toolbar instead
# of a stack with floating labels.
form_all, form_none, form_n_label, form_n, _ = st.columns([1, 1, 1, 1, 4])

if form_all.button("Select all", use_container_width=True):
    # We don't yet know N → use the slider's stored value (or 10 default).
    pending_n = int(st.session_state.get("compare_top_n", 10))
    st.session_state[SELECTED_KEY] = sorted_df.head(pending_n)["run_id"].tolist()
if form_none.button("Deselect all", use_container_width=True):
    st.session_state[SELECTED_KEY] = []

form_n_label.markdown(
    "<div style='padding-top: 0.4rem; text-align: right;'>Top-N</div>",
    unsafe_allow_html=True,
)
N = form_n.number_input(
    "Top-N",
    min_value=1,
    max_value=100,
    value=10,
    step=5,
    label_visibility="collapsed",
    key="compare_top_n",
)
top_view = sorted_df.head(int(N)).copy()

# Compose the editable view: ✓ checkbox + open-link + identity + metric cols.
selected_set = set(st.session_state[SELECTED_KEY])
top_view.insert(0, "✓", top_view["run_id"].isin(selected_set))
if "run_id" in top_view.columns:
    top_view["open"] = top_view["run_id"].apply(lambda rid: f"/run_detail?run_id={rid}")
display_cols = [
    c
    for c in (
        "✓",
        "open",
        "run_id",
        "scenario",
        "algorithm",
        "seed",
        metric,
        *[m for m in metrics if m != metric][:3],
    )
    if c in top_view.columns
]
fmt: dict = {}
for col in display_cols:
    if "rate" in col or "progress" in col:
        fmt[col] = "{:.0%}".format
    elif col.startswith("M"):
        fmt[col] = "{:.2f}".format

# Use ``st.data_editor`` so the ✓ column is interactive; everything else is
# read-only. Edits flow back into session_state for the config-diff section.
edited = st.data_editor(
    top_view[display_cols],
    column_config={
        "✓": st.column_config.CheckboxColumn("✓", width="small"),
        "open": st.column_config.LinkColumn(
            "", display_text=":material/open_in_new:", width="small"
        ),
    },
    disabled=[c for c in display_cols if c != "✓"],
    hide_index=True,
    use_container_width=True,
    key="compare_top_editor",
)
st.session_state[SELECTED_KEY] = edited[edited["✓"]]["run_id"].tolist()

# ── 3. Config-diff accordion (sits right under the table) ────────────
with st.expander("Config differences across selected runs", expanded=False):
    selected_run_ids = list(st.session_state[SELECTED_KEY])
    if not selected_run_ids:
        st.info("Tick rows in the table above to compare their configs here.")
    elif "run_dir" not in filtered.columns:
        st.info("Need at least one run with a known directory to diff.")
    else:
        # Show every selected run as its own column — no cap. Streamlit's
        # dataframe widget scrolls horizontally if there are many.
        picked_dirs = []
        for rid in selected_run_ids:
            match = runs_df[runs_df["run_id"] == rid]
            if not match.empty:
                picked_dirs.append(Path(match.iloc[0]["run_dir"]))
        diff_table, _ = build_config_diff_table(picked_dirs)
        if diff_table.empty:
            st.info("No configs could be loaded for the selected runs.")
        else:
            cell_highlight = (
                f"background-color: {POLIMI_LIGHT_BLUE}30;"
                f"border-left: 3px solid {POLIMI_LIGHT_BLUE};"
            )

            def _cell_style(row: pd.Series) -> list[str]:
                """Highlight cells whose value differs from the row's most-common.

                A row where every cell matches → no highlight (silent identical
                params). A row with a 2-1 split → only the outlier highlighted.
                A row where every cell differs → all but the (arbitrarily
                tie-broken) most-common are highlighted.
                """
                values = [stringify_diff_value(v) for v in row]
                if len(set(values)) == 1:
                    return [""] * len(row)
                most_common, _ = Counter(values).most_common(1)[0]
                return [cell_highlight if v != most_common else "" for v in values]

            styled = diff_table.style.apply(_cell_style, axis=1)
            st.dataframe(styled, use_container_width=True)
            n_diff_cells = sum(
                1
                for k in diff_table.index
                for v in (stringify_diff_value(x) for x in diff_table.loc[k])
                if v != Counter(stringify_diff_value(x) for x in diff_table.loc[k]).most_common(1)[0][0]
            )
            st.caption(
                f"{n_diff_cells} cell(s) differ from their row's most-common value "
                f"across {len(picked_dirs)} run(s)."
            )
st.markdown("---")

# ── 4. Algorithm leaderboard by scenario ─────────────────────────────
agg = aggregate_metric(filtered, metric, by=["scenario", "algorithm"], how=how)
grouped_bar_with_se(
    agg,
    group_col="scenario",
    bar_col="algorithm",
    title=f"{metric} by scenario × algorithm ({how} ± SE)",
    ylabel=metric,
)

# ── 5. Metric distribution by algorithm ──────────────────────────────
st.markdown("---")
box_by_algo(filtered, metric, title=f"{metric} distribution by algorithm")

# ── 6. Pareto by scenario ────────────────────────────────────────────
st.markdown("---")
scatter_xy(
    filtered,
    x=metric,
    y=metric_y,
    title=f"{metric} vs {metric_y} (Pareto)",
    xlabel=metric,
    ylabel=metric_y,
)
