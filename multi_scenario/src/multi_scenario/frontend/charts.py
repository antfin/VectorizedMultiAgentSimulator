"""Streamlit/matplotlib chart helpers used by the per-scenario render functions.

Each helper renders directly to the current Streamlit container — no figure
return values — so call sites are one-liners. Helpers handle empty / missing
columns by rendering a small ``st.info`` notice instead of failing.

Colour palette is anchored to the Polimi brand (see :mod:`.theme`); algorithms
get stable per-name colours so the same algorithm appears in the same colour
across charts and tabs.
"""

from typing import Iterable

# Set the non-GUI matplotlib backend BEFORE importing pyplot — once pyplot
# is imported the backend is locked. Hence the imports below sit after this
# call (intentionally; the pylint position warning is suppressed below).
import matplotlib

matplotlib.use("Agg")

# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from .theme import (  # noqa: E402
    POLIMI_DARK_BLUE,
    POLIMI_GRAY,
    POLIMI_GREEN,
    POLIMI_LIGHT_BLUE,
    POLIMI_ORANGE,
    POLIMI_RED,
)
# pylint: enable=wrong-import-position

# Cycle order matches ``rendezvous_comm/src/plotting.py`` so the two
# dashboards stay visually consistent for side-by-side analysis.
_POLIMI_CYCLE = (
    POLIMI_DARK_BLUE,
    POLIMI_RED,
    POLIMI_GREEN,
    POLIMI_ORANGE,
    POLIMI_LIGHT_BLUE,
    POLIMI_GRAY,
)

# Stable algorithm → colour mapping. Keeps the same algo the same colour across
# scenarios and chart types — meaningful when the eye scans the page top-to-bottom.
ALGO_COLORS = {
    "mappo": POLIMI_DARK_BLUE,
    "ippo": POLIMI_LIGHT_BLUE,
    "masac": POLIMI_GREEN,
    "isac": POLIMI_ORANGE,
    "iddpg": POLIMI_RED,
    "maddpg": POLIMI_GRAY,
}


def set_style() -> None:
    """Apply the Polimi-branded matplotlib rc params (call before each plot).

    Values mirror ``rendezvous_comm/src/plotting.py:set_style`` so charts
    look the same across both dashboards.
    """
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 12,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.prop_cycle": plt.cycler(color=list(_POLIMI_CYCLE)),
        }
    )


def _color_for(algo: str, fallback_idx: int = 0) -> str:
    """Return a stable colour for an algo name; fallback to the Polimi cycle."""
    return ALGO_COLORS.get(algo, _POLIMI_CYCLE[fallback_idx % len(_POLIMI_CYCLE)])


def _check(df: pd.DataFrame, cols: Iterable[str], title: str) -> bool:
    """Render an info notice + return False if any required column is missing or all-NaN."""
    missing = [c for c in cols if c not in df.columns or df[c].isna().all()]
    if missing:
        st.subheader(title)
        st.info(f"Not enough data to plot — missing or all-NaN: {', '.join(missing)}")
        return False
    return True


# Six args is the natural shape (df + x/y cols + title + axis labels);
# bundling them into a config object would obscure call sites.
# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def scatter_xy(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    """Scatter ``y`` vs ``x``, one point per run, coloured by ``algorithm``."""
    if not _check(df, [x, y, "algorithm"], title):
        return
    set_style()
    fig, ax = plt.subplots()
    sub = df.dropna(subset=[x, y])
    for i, (algo, grp) in enumerate(sub.groupby("algorithm")):
        ax.scatter(
            grp[x], grp[y],
            label=algo,
            color=_color_for(algo, i),
            s=70, alpha=0.85, edgecolors="white",
        )
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if sub["algorithm"].nunique() > 1:
        ax.legend(title="Algorithm", fontsize=9)
    fig.tight_layout()
    st.subheader(title)
    st.pyplot(fig)
    plt.close(fig)


def box_by_algo(df: pd.DataFrame, metric: str, title: str, ylabel: str | None = None) -> None:
    """Box plot of ``metric`` grouped by ``algorithm``."""
    if not _check(df, [metric, "algorithm"], title):
        return
    set_style()
    sub = df.dropna(subset=[metric])
    if sub.empty:
        st.subheader(title)
        st.info(f"No non-NaN values for {metric}.")
        return
    algos = sorted(sub["algorithm"].unique())
    data = [sub[sub["algorithm"] == a][metric].values for a in algos]
    fig, ax = plt.subplots()
    bp = ax.boxplot(
        data, tick_labels=algos, patch_artist=True, widths=0.55,
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(_color_for(algos[i], i))
        patch.set_alpha(0.7)
    ax.set_ylabel(ylabel or metric)
    fig.tight_layout()
    st.subheader(title)
    st.pyplot(fig)
    plt.close(fig)


def bar_by_algo_with_stderr(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str | None = None,
) -> None:
    """Bar of mean ``metric`` per algorithm, error bars = stderr across seeds."""
    if not _check(df, [metric, "algorithm"], title):
        return
    set_style()
    sub = df.dropna(subset=[metric])
    grouped = sub.groupby("algorithm")[metric]
    means = grouped.mean().sort_values(ascending=False)
    sem = grouped.sem().reindex(means.index).fillna(0)
    fig, ax = plt.subplots()
    colors = [_color_for(a, i) for i, a in enumerate(means.index)]
    bars = ax.bar(
        range(len(means)), means.values, yerr=sem.values,
        capsize=4, color=colors, alpha=0.85,
    )
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(means.index, rotation=20, ha="right")
    ax.set_ylabel(ylabel or metric)
    for rect, val in zip(bars, means.values):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height(), f"{val:.2f}",
            ha="center", va="bottom", fontsize=9,
        )
    fig.tight_layout()
    st.subheader(title)
    st.pyplot(fig)
    plt.close(fig)
