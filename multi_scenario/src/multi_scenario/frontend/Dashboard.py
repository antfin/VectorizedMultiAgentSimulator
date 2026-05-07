"""multi_scenario — Streamlit landing dashboard (F7.1).

Launch: ``streamlit run src/multi_scenario/frontend/Dashboard.py``.

Per-scenario detail view with KPIs, top-runs table and scenario-specific
charts. The scenario is picked from the sidebar (single-select, defaults to
the first scenario that has runs on disk). The browse-all-runs view lives
on the sibling F7.2 page (``pages/1_Experiments.py``).

Per-scenario tweaks (KPIs, charts, sort order) are conversational — edit
:mod:`.scenarios` directly. Update mechanism documented in the F7.1 spec.
"""

# Streamlit scripts are top-level scripts, not importable modules — Pylint's
# default rules around module-level setup don't fit. Suppress the noisy ones.
# pylint: disable=wrong-import-position,invalid-name

import sys
from pathlib import Path

import streamlit as st

# Make the package importable when launched via `streamlit run <path>`. The
# entrypoint runs at file-level, so we resolve our src/ root and prepend it.
_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

st.set_page_config(
    page_title="Multi-Robot Experiments",
    page_icon=":robot_face:",
    layout="wide",
    initial_sidebar_state="expanded",
)

from multi_scenario.frontend.scenarios import SCENARIO_RENDERERS
from multi_scenario.frontend.sidebar import (
    load_runs_with_cache,
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
    set(runs_df["scenario"].dropna().unique()) if "scenario" in runs_df.columns else set()
)
# Default to the first scenario that has data, falling back to the first option.
default_idx = next(
    (i for i, s in enumerate(SCENARIO_ORDER) if s in present_scenarios),
    0,
)

st.sidebar.header("Scenario")
selected_scenario = st.sidebar.selectbox(
    "Pick a scenario",
    SCENARIO_ORDER,
    index=default_idx,
    format_func=str.capitalize,
)

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
