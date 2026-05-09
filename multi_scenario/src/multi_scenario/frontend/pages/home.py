"""Home — landing dashboard (F7.1).

Per-scenario detail view with KPIs, top-runs table and scenario-specific
charts. Scenario is picked from the sidebar (single-select, defaults to the
first scenario that has runs on disk). The browse-all-runs view lives on the
sibling :mod:`.experiments` page (F7.2).

Per-scenario tweaks (KPIs, charts, sort order) are conversational — edit
:mod:`.scenarios` directly. Update mechanism documented in the F7.1 spec.
"""

# Streamlit pages are scripts loaded by ``st.navigation`` at runtime; the
# file-level execution model triggers pylint's import-position rules. Disable
# them rather than restructure into a function (the script *is* the page).
# pylint: disable=wrong-import-position,invalid-name

import streamlit as st

from multi_scenario.frontend.scenarios import SCENARIO_RENDERERS
from multi_scenario.frontend.sidebar import (
    load_runs_with_cache,
    persist_widget_state,
    render_active_root_caption,
)
from multi_scenario.frontend.theme import apply_theme

apply_theme(
    title="Multi-Robot Experiments",
    subtitle="Cooperative MARL — Politecnico di Milano",
)

# ── Sidebar: read-only data-source caption + scenario selector ───────
experiments_dir = render_active_root_caption()
runs_df = load_runs_with_cache(experiments_dir)

SCENARIO_ORDER = ("discovery", "navigation", "transport", "flocking")
present_scenarios = (
    set(runs_df["scenario"].dropna().unique())
    if "scenario" in runs_df.columns
    else set()
)
# Default to the first scenario that has data, falling back to the first option.
default_idx = next(
    (i for i, s in enumerate(SCENARIO_ORDER) if s in present_scenarios),
    0,
)

st.sidebar.header("Scenario")
_picker_persist = persist_widget_state(
    "home_scenario_picker", SCENARIO_ORDER[default_idx]
)
selected_scenario = st.sidebar.selectbox(
    "Pick a scenario",
    SCENARIO_ORDER,
    format_func=str.capitalize,
    key="home_scenario_picker",
)
st.session_state[_picker_persist] = selected_scenario

# ── Main: render selected scenario ───────────────────────────────────
if "scenario" in runs_df.columns:
    sub_df = runs_df[runs_df["scenario"] == selected_scenario]
else:
    sub_df = runs_df.iloc[0:0]  # empty with same columns

if sub_df.empty:
    st.info(
        f"No runs yet for **{selected_scenario}**. "
        f"Run via `multi-scenario run experiments/{selected_scenario}/...`."
    )
else:
    SCENARIO_RENDERERS[selected_scenario](sub_df)
