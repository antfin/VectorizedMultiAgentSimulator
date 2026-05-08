"""F7.2 — Experiments Browser.

Filterable table of every run under the experiments root. Sidebar exposes
multi-select filters for scenario / algorithm / state plus a free-text search
on ``run_id``. State is shown as an emoji-prefixed badge per run.

The detail-link column is intentionally absent for now — it lands with F7.3
(per-run detail page) where ``?run_id=...`` deep-links resolve. Until then
this page is a stand-alone browse view.
"""

# Streamlit pages are scripts loaded by ``st.navigation`` at runtime; the
# file-level execution model triggers pylint's import-position rules. Disable
# them rather than restructure into a function (the script *is* the page).
# pylint: disable=wrong-import-position,invalid-name

import streamlit as st

from multi_scenario.frontend.filters import filter_runs
from multi_scenario.frontend.sidebar import (
    load_runs_with_cache,
    persist_widget_state,
    render_active_root_caption,
)
from multi_scenario.frontend.theme import apply_theme

apply_theme(
    title="Experiments Browser",
    subtitle="Filter and inspect every run on disk",
)

# ── Data ──────────────────────────────────────────────────────────────
experiments_dir = render_active_root_caption()
runs_df = load_runs_with_cache(experiments_dir)

if runs_df.empty:
    st.info(
        f"No runs found under `{experiments_dir}`. "
        "Run an experiment via `multi-scenario run <yaml>` first."
    )
    st.stop()

# ── Sidebar filters ───────────────────────────────────────────────────
st.sidebar.header("Filters")


def _options(col: str) -> list[str]:
    """Sorted unique values from a column (empty list if column missing)."""
    if col not in runs_df.columns:
        return []
    return sorted(runs_df[col].dropna().unique().tolist())


# Filter selections persist across page nav via shadow keys (Streamlit's
# multipage nav silently drops widget state otherwise — see
# ``persist_widget_state``). Namespaced ``browse_*`` so the same filter on
# Compare keeps independent state.
_p_scen = persist_widget_state("browse_scenarios", [])
scenarios = st.sidebar.multiselect(
    "Scenario", _options("scenario"), key="browse_scenarios"
)
st.session_state[_p_scen] = scenarios

_p_algo = persist_widget_state("browse_algorithms", [])
algorithms = st.sidebar.multiselect(
    "Algorithm", _options("algorithm"), key="browse_algorithms"
)
st.session_state[_p_algo] = algorithms

_p_state = persist_widget_state("browse_states", [])
states = st.sidebar.multiselect("State", _options("state"), key="browse_states")
st.session_state[_p_state] = states

_p_search = persist_widget_state("browse_search", "")
search = st.sidebar.text_input(
    "Search run_id",
    help="Case-insensitive substring match.",
    key="browse_search",
)
st.session_state[_p_search] = search

filtered = filter_runs(
    runs_df,
    scenarios=scenarios,
    algorithms=algorithms,
    states=states,
    search=search,
)

# ── Main ──────────────────────────────────────────────────────────────
total = len(runs_df)
shown = len(filtered)
if shown == total:
    st.markdown(f"### {total} run(s)")
else:
    st.markdown(f"### {shown} of {total} runs match filters")

if shown == 0:
    st.warning("No runs match the current filters. Clear some to widen results.")
    st.stop()

_STATE_BADGES = {
    "DONE": "🟢 DONE",
    "RUNNING": "🔵 RUNNING",
    "CRASHED": "🔴 CRASHED",
    "RESUMED": "🟠 RESUMED",
    "INITIALIZING": "⚪ INITIALIZING",
    "UNKNOWN": "⚪ UNKNOWN",
}

# Build the displayed view: state badge + identity columns + metric columns.
view = filtered.copy()
if "state" in view.columns:
    view["state"] = view["state"].map(_STATE_BADGES).fillna(view["state"])

# Deep-link column → opens F7.3 with the run preselected via ``?run_id=``.
# ``LinkColumn`` works with absolute or relative URLs; ``/run_detail`` matches
# st.navigation's auto-generated path for ``pages/run_detail.py``.
if "run_id" in view.columns:
    view["open"] = view["run_id"].apply(lambda rid: f"/run_detail?run_id={rid}")

display_cols = [
    c
    for c in (
        "open",
        "state",
        "run_id",
        "scenario",
        "algorithm",
        "seed",
        "run_timestamp",
        "M1_success_rate",
        "M2_avg_return",
        "M3_steps",
        "M4_collisions",
        "M6_coverage_progress",
    )
    if c in view.columns
]

fmt: dict = {}
for col in display_cols:
    if "rate" in col or "progress" in col:
        fmt[col] = "{:.0%}".format
    elif col.startswith("M"):
        fmt[col] = "{:.2f}".format

st.dataframe(
    view[display_cols].style.format(fmt, na_rep="—"),
    use_container_width=True,
    hide_index=True,
    column_config={
        # Empty label hides the column header; the "small" width is the
        # narrowest preset Streamlit exposes (the icon is the only content).
        # Material ":open_in_new:" renders a button-like square-arrow glyph
        # so the cell reads as "click me".
        "open": st.column_config.LinkColumn(
            "",
            display_text=":material/open_in_new:",
            width="small",
        ),
    },
)
