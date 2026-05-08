"""Streamlit entrypoint — owns navigation + global page config.

Launch: ``streamlit run src/multi_scenario/frontend/streamlit_app.py``.

Pages are explicitly registered with :func:`st.navigation` (which disables
Streamlit's auto-``pages/`` discovery) so we can group "Browse" + "Detail"
under a non-clickable "Experiments" section header. The dict's empty-string
keys carry pages that should appear ungrouped (Dashboard at top, Settings
at bottom).

This file is the only place that calls :func:`st.set_page_config` and the
only place that mutates ``sys.path`` to make ``multi_scenario.*`` imports
resolve when launched via ``streamlit run``. Per-page modules import the
already-on-path package directly.
"""

# pylint: disable=wrong-import-position,invalid-name

import sys
from pathlib import Path

# Make the package importable when launched via `streamlit run <path>`. The
# entrypoint runs at file-level, so we resolve our src/ root once and prepend
# it; pages loaded later by st.navigation share this interpreter and inherit
# the patched sys.path.
_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import streamlit as st

st.set_page_config(
    page_title="Multi-Robot Experiments",
    page_icon=":robot_face:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Page registration. Keep file paths relative to this entrypoint so Streamlit
# resolves them correctly. Titles control the sidebar label; section keys
# (``""`` vs ``"Experiments"``) control grouping. The "Pages" heading above
# the nav is injected via CSS ``::before`` in :mod:`.theme` because
# ``st.navigation`` always pins itself to the top of the sidebar — sidebar
# widgets added before ``nav.run()`` still render *below* it.
home = st.Page("pages/home.py", title="Dashboard", icon=":material/home:", default=True)
browse = st.Page("pages/experiments.py", title="Browse", icon=":material/list:")
detail = st.Page("pages/run_detail.py", title="Detail", icon=":material/insights:")
compare = st.Page("pages/comparison.py", title="Compare", icon=":material/compare_arrows:")
settings = st.Page("pages/settings.py", title="Settings", icon=":material/settings:")

nav = st.navigation(
    {
        "": [home],
        "Experiments": [browse, detail, compare],
        " ": [settings],  # space differentiates from the other empty-string section
    },
    expanded=True,  # keep all sections always-expanded; combined with the
    # ``pointer-events: none`` rule on the section header, the user can't
    # accidentally collapse "Experiments" by clicking it.
)
nav.run()

# Path footer — pinned at the very bottom of the sidebar regardless of page.
# Imported lazily so the entry script runs ``set_page_config`` before any
# package internals touch ``st`` (Streamlit requires that ordering).
# pylint: disable=wrong-import-position
from multi_scenario.frontend.sidebar import render_path_footer  # noqa: E402

render_path_footer()
