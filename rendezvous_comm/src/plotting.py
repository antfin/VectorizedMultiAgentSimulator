"""Plotting utilities for experiment analysis and comparison.

All plot functions return matplotlib figure objects so notebooks
can display them inline or save to files.
"""
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


# ── Style defaults ──────────────────────────────────────────────────
COLORS = {
    "er1": "#1f77b4",   # blue  - no comm
    "er2": "#ff7f0e",   # orange - engineered schema
    "er3": "#2ca02c",   # green  - symbolic intent
    "er4": "#d62728",   # red    - event triggered
    "e1":  "#9467bd",   # purple - static LLM
}

LABELS = {
    "er1": "ER1: No Comm",
    "er2": "ER2: Eng. Schema",
    "er3": "ER3: Symbolic Intent",
    "er4": "ER4: Event-Triggered",
    "e1":  "E1: Static LLM",
}

METRIC_LABELS = {
    "M1_success_rate": "Success Rate",
    "M2_avg_return": "Avg Return",
    "M3_avg_steps": "Avg Steps to Complete",
    "M4_avg_collisions": "Collisions / Episode",
    "M5_avg_tokens": "Tokens / Episode",
    "M6_coverage_progress": "Coverage Progress",
    "M8_agent_utilization": "Agent Utilization",
    "M9_spatial_spread": "Spatial Spread",
}


def set_style():
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "font.size": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


# ── Single experiment plots ────────────────────────────────────────

def plot_sweep_heatmap(
    df,
    metric: str = "M1_success_rate",
    row_param: str = "n_agents",
    col_param: str = "n_targets",
    title: Optional[str] = None,
) -> plt.Figure:
    """Heatmap of a metric across two swept parameters.

    Args:
        df: DataFrame from ExperimentStorage.to_dataframe()
        metric: which metric to plot
        row_param, col_param: swept params for axes
    """
    set_style()
    pivot = df.groupby([row_param, col_param])[metric].mean().unstack()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel(col_param)
    ax.set_ylabel(row_param)
    ax.set_title(title or f"{METRIC_LABELS.get(metric, metric)}")

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="black", fontsize=10)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def plot_training_curves(
    training_logs: Dict[str, "pd.DataFrame"],
    metric: str = "episode_reward_mean",
    title: Optional[str] = None,
) -> plt.Figure:
    """Training curves from multiple runs (mean + std band).

    Args:
        training_logs: {label: DataFrame with 'step' and metric columns}
    """
    set_style()
    fig, ax = plt.subplots()

    for label, df in training_logs.items():
        exp_id = label.split("_")[0] if "_" in label else label
        color = COLORS.get(exp_id, None)
        ax.plot(df["step"], df[metric], label=label, color=color)

    ax.set_xlabel("Training Frames")
    ax.set_ylabel(metric)
    ax.set_title(title or "Training Curves")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_seed_variance(
    df,
    metric: str = "M1_success_rate",
    group_by: str = "algorithm",
    title: Optional[str] = None,
) -> plt.Figure:
    """Bar chart with error bars showing variance across seeds."""
    set_style()
    grouped = df.groupby(group_by)[metric]
    means = grouped.mean()
    stds = grouped.std()

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(means))
    ax.bar(x, means, yerr=stds, capsize=5, color="#4a90d9", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(means.index, rotation=45, ha="right")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(title or f"{metric} by {group_by}")
    fig.tight_layout()
    return fig


# ── Cross-experiment comparison ────────────────────────────────────

def plot_baseline_comparison(
    cross_df,
    metric: str = "M1_success_rate",
    group_col: str = "experiment",
    title: Optional[str] = None,
) -> plt.Figure:
    """Compare a metric across experiments (bar + error bars).

    Args:
        cross_df: DataFrame from load_cross_experiment()
    """
    set_style()
    grouped = cross_df.groupby(group_col)[metric]
    means = grouped.mean()
    stds = grouped.std()

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(means))
    colors = [COLORS.get(exp, "#999") for exp in means.index]
    labels = [LABELS.get(exp, exp) for exp in means.index]

    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(title or f"Baseline Comparison: {METRIC_LABELS.get(metric, metric)}")

    # Value labels on bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    return fig


def plot_metric_radar(
    experiment_metrics: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Radar/spider chart comparing multiple experiments across metrics.

    Args:
        experiment_metrics: {exp_id: {metric: value}}
    """
    if metrics is None:
        metrics = ["M1_success_rate", "M2_avg_return", "M3_avg_steps",
                    "M4_avg_collisions"]

    set_style()
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for exp_id, mvals in experiment_metrics.items():
        values = [mvals.get(m, 0) for m in metrics]
        values += values[:1]
        color = COLORS.get(exp_id, "#999")
        label = LABELS.get(exp_id, exp_id)
        ax.plot(angles, values, "o-", label=label, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_thetagrids(
        [a * 180 / np.pi for a in angles[:-1]],
        [METRIC_LABELS.get(m, m) for m in metrics],
    )
    ax.set_title(title or "Experiment Comparison", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    return fig


def plot_success_vs_tokens(
    experiment_metrics: Dict[str, Dict[str, float]],
    title: Optional[str] = None,
) -> plt.Figure:
    """Scatter: success rate vs communication cost (tokens/ep).

    This is the core Pareto frontier plot for comm efficiency.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    for exp_id, mvals in experiment_metrics.items():
        color = COLORS.get(exp_id, "#999")
        label = LABELS.get(exp_id, exp_id)
        ax.scatter(
            mvals.get("M5_avg_tokens", 0),
            mvals.get("M1_success_rate", 0),
            c=color, s=150, label=label, zorder=5, edgecolors="white",
        )

    ax.set_xlabel("Tokens / Episode")
    ax.set_ylabel("Success Rate")
    ax.set_title(title or "Success vs Communication Cost")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_training_dashboard(
    scalars: Dict[str, list],
    title: str = "Training Progress",
    heuristic_reward: Optional[float] = None,
) -> plt.Figure:
    """2x2 dashboard of training curves from BenchMARL CSV scalars.

    Args:
        scalars: {csv_name: [(step, value), ...]} from RunStorage.load_benchmarl_scalars()
        title: figure suptitle
        heuristic_reward: if given, draw horizontal reference line on reward plot
    """
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    panels = [
        ("eval_reward_episode_reward_mean", "Eval Reward", axes[0, 0], "#1f77b4"),
        ("collection_agents_info_targets_covered", "Targets Covered / Step", axes[0, 1], "#2ca02c"),
        ("collection_agents_info_covering_reward", "Covering Reward", axes[1, 0], "#ff7f0e"),
        ("train_agents_entropy", "Policy Entropy", axes[1, 1], "#9467bd"),
    ]

    for csv_name, label, ax, color in panels:
        data = scalars.get(csv_name)
        if data:
            steps, values = zip(*data)
            ax.plot(steps, values, color=color, linewidth=1.8)
            ax.fill_between(steps, values, alpha=0.1, color=color)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Iteration", fontsize=9)
        ax.grid(True, alpha=0.3)

    # Add heuristic reference line on reward plot
    if heuristic_reward is not None:
        axes[0, 0].axhline(
            y=heuristic_reward, color="#e74c3c", linestyle="--",
            linewidth=1.5, alpha=0.7, label=f"Heuristic ({heuristic_reward:.1f})",
        )
        axes[0, 0].legend(fontsize=9)

    fig.tight_layout()
    return fig


def plot_baseline_comparison_bars(
    metrics_dict: Dict[str, Dict[str, float]],
    metric_key: str = "M1_success_rate",
    title: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
) -> plt.Figure:
    """Horizontal bar chart comparing multiple policies on one metric.

    Args:
        metrics_dict: {"Random": {...}, "Trained": {...}, "Heuristic": {...}}
        metric_key: which metric to compare
    """
    set_style()
    default_colors = {"Random": "#e74c3c", "Trained": "#1f77b4", "Heuristic": "#27ae60"}
    if colors is None:
        colors = default_colors

    labels = list(metrics_dict.keys())
    values = [metrics_dict[k].get(metric_key, 0) for k in labels]
    bar_colors = [colors.get(k, "#999") for k in labels]

    fig, ax = plt.subplots(figsize=(8, max(2.5, len(labels) * 0.8)))
    bars = ax.barh(labels, values, color=bar_colors, height=0.5, edgecolor="white")

    is_pct = "rate" in metric_key or "progress" in metric_key
    for bar, v in zip(bars, values):
        label = f"{v:.0%}" if is_pct else f"{v:.2f}"
        ax.text(
            max(v + 0.02, 0.05) if is_pct else v + abs(v) * 0.05,
            bar.get_y() + bar.get_height() / 2,
            label, va="center", fontsize=11, fontweight="bold",
        )

    if is_pct:
        ax.set_xlim(0, 1.15)
    metric_label = METRIC_LABELS.get(metric_key, metric_key)
    ax.set_xlabel(metric_label)
    ax.set_title(title or f"Comparison: {metric_label}", fontsize=13)
    ax.grid(True, alpha=0.2, axis="x")
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, path: str, dpi: int = 150):
    """Save figure to file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}")
