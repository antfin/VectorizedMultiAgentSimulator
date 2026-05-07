"""F7.2 — Experiments Browser.

Filterable table of every run under the experiments root. Sidebar exposes
multi-select filters for scenario / algorithm / state plus a free-text search
on ``run_id``. State is shown as an emoji-prefixed badge per run.

The detail-link column is intentionally absent for now — it lands with F7.3
(per-run detail page) where ``?run_id=...`` deep-links resolve. Until then
this page is a stand-alone browse view.
"""

# Streamlit scripts are top-level scripts, not importable modules — Pylint's
# default rules around module-level setup don't fit. Suppress the noisy ones.
# pylint: disable=wrong-import-position,invalid-name

import sys
from pathlib import Path

import streamlit as st

# Make the package importable when launched via `streamlit run Dashboard.py`.
# Streamlit auto-loads page files but doesn't add our src/ root to sys.path.
_SRC_ROOT = Path(__file__).resolve().parents[3]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

st.set_page_config(
    page_title="Experiments — Multi-Robot",
    page_icon=":mag:",
    layout="wide",
    initial_sidebar_state="expanded",
)

from multi_scenario.frontend.filters import filter_runs
from multi_scenario.frontend.sidebar import (
    load_runs_with_cache,
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


scenarios = st.sidebar.multiselect("Scenario", _options("scenario"))
algorithms = st.sidebar.multiselect("Algorithm", _options("algorithm"))
states = st.sidebar.multiselect("State", _options("state"))
search = st.sidebar.text_input("Search run_id", help="Case-insensitive substring match.")

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

display_cols = [
    c
    for c in (
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
)
