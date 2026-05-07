"""multi_scenario — Streamlit landing dashboard (F7.1).

Launch: ``streamlit run src/multi_scenario/frontend/Dashboard.py``.

Title-level summary across 4 scenario tabs (Discovery / Navigation / Transport
/ Flocking). Each tab renders KPIs + Top Runs + scenario-specific charts via
:mod:`.scenarios`. The cache key is content-aware (number-of-files +
max-mtime) so adding a new run on disk auto-invalidates without a manual
refresh; a sidebar 🔄 button is also wired for explicit clears.

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

import pandas as pd  # noqa: E402

from multi_scenario.frontend.runs_loader import load_runs
from multi_scenario.frontend.scenarios import SCENARIO_RENDERERS
from multi_scenario.frontend.theme import apply_theme

apply_theme(
    title="Multi-Robot Experiments",
    subtitle="Cooperative MARL — Politecnico di Milano",
)

# ── Sidebar: experiments root + refresh ──────────────────────────────
DEFAULT_EXPERIMENTS_DIR = Path.cwd() / "experiments"
st.sidebar.header("Data source")
experiments_dir_str = st.sidebar.text_input(
    "Experiments root",
    value=str(DEFAULT_EXPERIMENTS_DIR),
    help="Folder containing run subdirectories with output/metrics.json files.",
)
experiments_dir = Path(experiments_dir_str).expanduser()
if st.sidebar.button("🔄 Refresh"):
    st.cache_data.clear()
    st.rerun()


def _experiments_signature(path: Path) -> tuple[int, float]:
    """Cheap fingerprint: ``(file_count, max_mtime)`` over ``output/metrics.json`` files.

    Used as part of the cache key so the cache invalidates the moment a new
    run lands on disk — no manual refresh needed for fresh data.
    """
    if not path.is_dir():
        return (0, 0.0)
    files = list(path.rglob("output/metrics.json"))
    if not files:
        return (0, 0.0)
    return (len(files), max(f.stat().st_mtime for f in files))


@st.cache_data(show_spinner="Loading runs…")
def _cached_load(path_str: str, signature: tuple[int, float]) -> pd.DataFrame:
    """Cache wrapper for ``load_runs``; ``signature`` varies the key on disk changes."""
    del signature  # only present to vary the cache key — load_runs walks fresh
    return load_runs(Path(path_str))


runs_df = _cached_load(str(experiments_dir), _experiments_signature(experiments_dir))

# ── Tabs (4 scenarios, always visible) ───────────────────────────────
SCENARIO_ORDER = ("discovery", "navigation", "transport", "flocking")
tabs = st.tabs([s.capitalize() for s in SCENARIO_ORDER])

for tab, scenario in zip(tabs, SCENARIO_ORDER):
    with tab:
        if "scenario" in runs_df.columns:
            sub_df = runs_df[runs_df["scenario"] == scenario]
        else:
            sub_df = pd.DataFrame()
        if sub_df.empty:
            st.info(
                f"No runs yet for **{scenario}**. "
                f"Run via `multi-scenario run experiments/{scenario}/...`."
            )
            continue
        SCENARIO_RENDERERS[scenario](sub_df)
