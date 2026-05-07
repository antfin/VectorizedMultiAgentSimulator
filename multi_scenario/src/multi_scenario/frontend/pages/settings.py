"""Settings — central config page for the dashboard.

The experiments root path lives here (single source of truth via
``st.session_state[EXPERIMENTS_ROOT_KEY]``). All other pages display a
read-only ``📁 path`` caption and link back here to change it.

Future config (OVH cluster path, S3 endpoint overrides, plot theme toggles)
will land on this page so users have one place to look.

Numbered ``9_`` so it appears at the bottom of the page nav — meta config
shouldn't compete with the analysis pages for first attention.
"""

# pylint: disable=wrong-import-position,invalid-name

import sys
from pathlib import Path

import streamlit as st

_SRC_ROOT = Path(__file__).resolve().parents[3]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

st.set_page_config(
    page_title="Settings — Multi-Robot",
    page_icon=":gear:",
    layout="wide",
    initial_sidebar_state="expanded",
)

from multi_scenario.frontend.charts import pie_by_category
from multi_scenario.frontend.sidebar import (
    EXPERIMENTS_ROOT_KEY,
    active_experiments_dir,
    default_experiments_dir,
    load_runs_with_cache,
)
from multi_scenario.frontend.theme import apply_theme

apply_theme(
    title="Settings",
    subtitle="Dashboard configuration",
)

# ── Active root (read from session state, default fallback) ──────────
current = active_experiments_dir()

st.subheader("Experiments root")
st.markdown(
    "Folder containing run subdirectories with ``output/metrics.json`` files. "
    "Change here to point the dashboard at a different experiments tree."
)

# Seed the session_state key with the resolved default on first visit.
if EXPERIMENTS_ROOT_KEY not in st.session_state:
    st.session_state[EXPERIMENTS_ROOT_KEY] = str(current)

st.text_input(
    "Path",
    key=EXPERIMENTS_ROOT_KEY,
    help="Absolute or ``~``-relative path. Changes apply across all pages immediately.",
)

resolved = active_experiments_dir()
st.caption(f"Resolved → `{resolved}`  &nbsp;·&nbsp;  exists: **{'yes' if resolved.is_dir() else 'no'}**")

if resolved != default_experiments_dir():
    if st.button("Reset to default"):
        st.session_state[EXPERIMENTS_ROOT_KEY] = str(default_experiments_dir())
        st.rerun()

# ── Cache controls ────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Cache")
st.markdown(
    "The dashboard auto-invalidates its cache when a new run lands on disk "
    "(content-aware fingerprint of `output/metrics.json` files). Manual clear "
    "is only needed if a run was edited in place."
)
col_a, col_b = st.columns([1, 3])
if col_a.button("Clear cache & reload"):
    st.cache_data.clear()
    st.rerun()

# ── Sanity-check preview ──────────────────────────────────────────────
st.markdown("---")
st.subheader("Detected runs")
runs_df = load_runs_with_cache(resolved)
if runs_df.empty:
    st.info(f"No runs detected under `{resolved}`.")
else:
    st.metric("Total runs", len(runs_df))
    if "scenario" in runs_df.columns:
        pie_by_category(
            runs_df["scenario"].value_counts().to_dict(),
            "Runs by scenario",
        )
