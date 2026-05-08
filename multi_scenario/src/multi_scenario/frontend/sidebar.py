"""Shared sidebar helpers — keeps the F7.x pages visually consistent.

The editable "Experiments root" lives on a dedicated **Settings** page; all
other pages just **read** the active root via :func:`active_experiments_dir`,
which is backed by ``st.session_state``. This keeps the per-page sidebar
clean (just a small read-only ``📁 path`` caption + a link to Settings).

Streamlit's ``st.session_state`` survives reruns within the same browser tab
but resets on full page reload — adequate for a "set once per session" knob.

**Filter persistence across page nav** uses :func:`persist_widget_state`.
Streamlit clears widget keys from ``session_state`` whenever the widget
isn't rendered — including while you're on a different page in a multipage
app. The shadow ``_persist_*`` key bypasses that and re-seeds the widget on
the next render so filter selections survive navigation.
"""

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from .runs_loader import load_runs


def persist_widget_state(widget_key: str, default: Any) -> str:
    """Bridge a widget's state to a persistent session_state shadow key.

    Workaround for Streamlit's auto-clearing of widget keys when the widget
    isn't rendered (e.g. during multipage nav). Call BEFORE rendering the
    widget; assign the widget's return value to the returned shadow key
    AFTER rendering. The widget itself uses ``key=widget_key``.

    Example::

        persist_key = persist_widget_state("browse_scenarios", [])
        scenarios = st.sidebar.multiselect(
            "Scenario", options, key="browse_scenarios"
        )
        st.session_state[persist_key] = scenarios
    """
    # Avoid a leading underscore in the key — Streamlit treats ``_*`` keys
    # as internal and clears them between page navigations in some versions.
    persist_key = f"persist__{widget_key}"
    if persist_key not in st.session_state:
        st.session_state[persist_key] = default
    # ONLY re-seed when the widget's own key isn't present in session_state.
    # That happens exactly once per page entry: Streamlit drops widget keys
    # on nav and re-creates them on first render. On *subsequent* reruns
    # within the same page, the widget key persists, so we don't touch it
    # (otherwise we'd clobber the user's interactive selection on every click).
    if widget_key not in st.session_state:
        st.session_state[widget_key] = st.session_state[persist_key]
    return persist_key

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
    """Return the resolved active path. (No sidebar render — that's owned by
    :func:`render_path_footer` which streamlit_app.py calls after the page,
    so the caption pins to the bottom of the sidebar regardless of which
    page is active.)
    """
    return active_experiments_dir()


def render_path_footer() -> None:
    """Append a single-line, truncated ``📁 path`` caption at the sidebar bottom.

    Uses raw HTML so we get ``text-overflow: ellipsis`` and a native
    ``title=`` tooltip on hover (the full path stays one keystroke away).
    Styled via the ``.ms-path-caption`` class in :mod:`.theme`.
    """
    path = str(active_experiments_dir())
    # ``html.escape`` keeps weird path chars from breaking the markup; we
    # render via st.sidebar.markdown with unsafe_allow_html for the tooltip.
    import html  # pylint: disable=import-outside-toplevel

    safe = html.escape(path)
    st.sidebar.markdown(
        f"<span class='ms-path-caption' title='{safe}'>📁 {safe}</span>",
        unsafe_allow_html=True,
    )


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
