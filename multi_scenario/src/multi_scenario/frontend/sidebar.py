"""Shared sidebar helpers — keeps the F7.x pages visually consistent.

The editable "Experiments root" lives on a dedicated **Settings** page; all
other pages just **read** the active root via :func:`active_experiments_dir`,
which is backed by ``st.session_state``. This keeps the per-page sidebar
clean (just a small read-only ``📁 path`` caption + a link to Settings).

Streamlit's ``st.session_state`` survives reruns within the same browser tab
but resets on full page reload — adequate for a "set once per session" knob.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from .runs_loader import load_runs

#: ``st.session_state`` key under which the active experiments root lives.
#: Settings page writes it; other pages read it via :func:`active_experiments_dir`.
EXPERIMENTS_ROOT_KEY = "experiments_root_path"


def default_experiments_dir() -> Path:
    """Sensible default — ``./experiments`` resolved against the CWD."""
    return Path.cwd() / "experiments"


def active_experiments_dir() -> Path:
    """Read the current experiments root from session state (with default fallback)."""
    raw = st.session_state.get(EXPERIMENTS_ROOT_KEY) or str(default_experiments_dir())
    return Path(raw).expanduser()


def render_active_root_caption() -> Path:
    """Render the read-only ``📁 path`` caption in the sidebar.

    Streamlit's auto-page nav at the top of the sidebar already exposes the
    Settings page as a clickable link, so we don't render a redundant in-body
    link here (which also avoids ``st.page_link`` blowing up under AppTest).

    Returns the resolved active path so callers can immediately load runs.
    """
    path = active_experiments_dir()
    st.sidebar.caption(f"📁 {path}")
    return path


def _experiments_signature(path: Path) -> tuple[int, float]:
    """Cheap fingerprint: ``(file_count, max_mtime)`` over ``output/metrics.json`` files.

    Used as part of the cache key so the cache invalidates the moment a new
    run lands on disk — no manual refresh needed.
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


def load_runs_with_cache(experiments_dir: Path) -> pd.DataFrame:
    """One-call helper: cached load with content-aware auto-invalidation."""
    return _cached_load(str(experiments_dir), _experiments_signature(experiments_dir))
