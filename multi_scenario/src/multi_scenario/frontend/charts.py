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
            grp[x],
            grp[y],
            label=algo,
            color=_color_for(algo, i),
            s=70,
            alpha=0.85,
            edgecolors="white",
        )
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if sub["algorithm"].nunique() > 1:
        ax.legend(title="Algorithm", fontsize=9)
    fig.tight_layout()
    st.subheader(title)
    st.pyplot(fig)
    plt.close(fig)


def box_by_algo(
    df: pd.DataFrame, metric: str, title: str, ylabel: str | None = None
) -> None:
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
        data,
        tick_labels=algos,
        patch_artist=True,
        widths=0.55,
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(_color_for(algos[i], i))
        patch.set_alpha(0.7)
    ax.set_ylabel(ylabel or metric)
    fig.tight_layout()
    st.subheader(title)
    st.pyplot(fig)
    plt.close(fig)


def line_plot_csvs(csv_paths: list["Path"], title: str) -> None:  # noqa: F821
    """Plot one line per CSV (BenchMARL scalar dumps).

    BenchMARL writes headerless ``step,value`` CSVs under
    ``output/benchmarl/.../scalars/``; we read with ``header=None`` and
    synthesize column names. CSVs that already have a header row (e.g. with
    ``step`` / ``iteration`` column) are auto-detected. Each CSV becomes a
    single labelled line; series are coloured from the Polimi cycle.
    """
    # pylint: disable=import-outside-toplevel
    from pathlib import Path  # noqa: F401  (re-imported for clarity in helper)

    if not csv_paths:
        st.subheader(title)
        st.info("No scalar CSVs to plot.")
        return
    set_style()
    # Use the same defaults as Dashboard scatter/box charts so axis text and
    # plot area stay proportional. The actual rendered width is capped by the
    # caller's ``st.columns`` wrapper rather than the figure's intrinsic size.
    fig, ax = plt.subplots()
    plotted = 0
    for i, path in enumerate(csv_paths):
        df = _read_scalar_csv(path)
        if df is None or df.empty:
            continue
        x_col = next((c for c in ("step", "iteration") if c in df.columns), None)
        numeric_cols = [
            c for c in df.columns if c != x_col and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not numeric_cols:
            continue
        y_col = numeric_cols[0]
        x = df[x_col] if x_col else df.index
        ax.plot(
            x,
            df[y_col],
            label=path.stem,
            color=_color_for(path.stem, i),
            linewidth=1.6,
        )
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        st.subheader(title)
        st.info("No usable numeric columns in the supplied CSVs.")
        return
    ax.set_xlabel("step")
    ax.set_ylabel("value")
    ax.legend(fontsize=9)
    fig.tight_layout()
    st.subheader(title)
    st.pyplot(fig)
    plt.close(fig)


def _read_scalar_csv(path: "Path") -> "pd.DataFrame | None":  # noqa: F821
    """Read a scalar CSV, auto-detecting whether it has a header row.

    BenchMARL emits headerless ``step,value`` lines; legacy / future formats
    may include a header. Strategy: peek at the first cell — if it parses as
    a float, the file is headerless and we synthesize ``step,value``.
    """
    try:
        with path.open("r", encoding="utf-8") as fh:
            first = fh.readline().strip()
    except OSError:
        return None
    first_token = first.split(",", 1)[0] if "," in first else first
    try:
        float(first_token)
        has_header = False
    except ValueError:
        has_header = True
    try:
        if has_header:
            return pd.read_csv(path)
        return pd.read_csv(path, header=None, names=["step", "value"])
    except (OSError, pd.errors.ParserError):
        return None


def grouped_bar_with_se(
    agg_df: "pd.DataFrame",  # noqa: F821
    *,
    group_col: str,
    bar_col: str,
    title: str,
    ylabel: str | None = None,
) -> None:
    """Grouped bar chart with SE error bars.

    ``agg_df`` must have columns ``group_col`` (e.g. ``"scenario"``,
    one row group per cluster), ``bar_col`` (e.g. ``"algorithm"``,
    one bar per cluster), ``mean``, and ``sem`` — i.e. the schema
    produced by :func:`aggregations.aggregate_metric`.

    Used by F7.4's "Algorithm leaderboard by scenario" section.
    """
    if agg_df.empty:
        st.subheader(title)
        st.info("No aggregated rows to plot.")
        return
    set_style()
    groups = sorted(agg_df[group_col].unique())
    bars = sorted(agg_df[bar_col].unique())
    fig, ax = plt.subplots()
    width = 0.8 / max(len(bars), 1)
    for i, b in enumerate(bars):
        sub = agg_df[agg_df[bar_col] == b].set_index(group_col).reindex(groups)
        x = [groups.index(g) + i * width - 0.4 + width / 2 for g in groups]
        ax.bar(
            x,
            sub["mean"].fillna(0).values,
            width=width,
            yerr=sub["sem"].fillna(0).values,
            capsize=3,
            label=str(b),
            color=_color_for(str(b), i),
            alpha=0.85,
        )
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=15, ha="right")
    ax.set_ylabel(ylabel or "mean")
    ax.legend(title=bar_col, fontsize=9)
    fig.tight_layout()
    st.subheader(title)
    st.pyplot(fig)
    plt.close(fig)


def pie_by_category(counts: dict[str, int], title: str) -> None:
    """Pie chart of category → count, coloured from the Polimi cycle.

    Used by the Settings page (and future cross-experiment summaries) to
    show share of runs per scenario / algorithm.
    """
    if not counts:
        st.subheader(title)
        st.info("No data to plot.")
        return
    set_style()
    labels = list(counts.keys())
    values = [counts[k] for k in labels]
    colors = [_color_for(label, i) for i, label in enumerate(labels)]
    # Compact layout — Settings page is a meta config screen, not a chart-first
    # page, so the pie shouldn't dominate. ``figsize`` × ``dpi`` controls the
    # intrinsic pixel size; ``use_container_width=False`` keeps Streamlit from
    # upscaling it to fill the column.
    fig, ax = plt.subplots(figsize=(3.0, 3.0), dpi=80)
    ax.pie(
        values,
        labels=labels,
        colors=colors,
        autopct=lambda p: f"{int(round(p * sum(values) / 100))}",
        startangle=90,
        textprops={"fontsize": 9, "color": "#002F6C"},
        wedgeprops={"edgecolor": "white", "linewidth": 1.0},
    )
    ax.set_aspect("equal")
    fig.tight_layout()
    st.subheader(title)
    st.pyplot(fig, use_container_width=False)
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
        range(len(means)),
        means.values,
        yerr=sem.values,
        capsize=4,
        color=colors,
        alpha=0.85,
    )
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(means.index, rotation=20, ha="right")
    ax.set_ylabel(ylabel or metric)
    for rect, val in zip(bars, means.values):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    st.subheader(title)
    st.pyplot(fig)
    plt.close(fig)
