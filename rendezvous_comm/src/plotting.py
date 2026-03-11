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
    "e1": "#9467bd",    # purple - static LLM
}

LABELS = {
    "er1": "ER1: No Comm",
    "er2": "ER2: Eng. Schema",
    "er3": "ER3: Symbolic Intent",
    "er4": "ER4: Event-Triggered",
    "e1": "E1: Static LLM",
}

METRIC_LABELS = {
    "M1_success_rate": "M1: Success Rate",
    "M2_avg_return": "M2: Avg Return",
    "M3_avg_steps": "M3: Avg Steps to Complete",
    "M4_avg_collisions": "M4: Collisions / Episode",
    "M5_avg_tokens": "M5: Tokens / Episode",
    "M6_coverage_progress": "M6: Coverage Progress",
    "M8_agent_utilization": "M8: Agent Utilization",
    "M9_spatial_spread": "M9: Spatial Spread",
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
    training_logs: Dict[str, object],
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
        metrics = [
            "M1_success_rate", "M2_avg_return", "M3_avg_steps",
            "M4_avg_collisions",
        ]

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

    ax.set_xlabel("M5: Tokens / Episode")
    ax.set_ylabel("M1: Success Rate")
    ax.set_title(title or "M1 vs M5: Success vs Communication Cost")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_training_dashboard(
    scalars: Dict[str, list],
    title: str = "Training Progress",
    heuristic_reward: Optional[float] = None,
) -> plt.Figure:
    """3x2 dashboard of training curves from BenchMARL CSV scalars.

    Shows 6 panels: Eval Reward, Targets Covered, Covering Reward,
    Collisions, Episode Length, and Policy Entropy.

    Args:
        scalars: {csv_name: [(step, value), ...]} from RunStorage.load_benchmarl_scalars()
        title: figure suptitle
        heuristic_reward: if given, draw horizontal reference line on reward plot
    """
    set_style()
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    panels = [
        ("eval_reward_episode_reward_mean", "Eval Reward", axes[0, 0], "#1f77b4"),
        ("collection_agents_info_targets_covered", "Targets Covered / Step", axes[0, 1], "#2ca02c"),
        ("collection_agents_info_covering_reward", "Covering Reward", axes[1, 0], "#ff7f0e"),
        ("collection_agents_info_collision_rew", "Collision Penalty", axes[1, 1], "#e74c3c"),
        ("eval_reward_episode_len_mean", "Eval Episode Length", axes[2, 0], "#17becf"),
        ("train_agents_entropy", "Policy Entropy", axes[2, 1], "#9467bd"),
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


def plot_baseline_grouped_bars(
    heuristic_metrics: Dict[str, float],
    random_metrics: Dict[str, float],
) -> plt.Figure:
    """Grouped horizontal bar chart comparing Heuristic vs Random.

    Shows four metrics side-by-side: M1_success_rate,
    M6_coverage_progress, M2_avg_return, M9_spatial_spread.

    Args:
        heuristic_metrics: metric dict for the heuristic policy
        random_metrics: metric dict for the random policy

    Returns:
        matplotlib Figure
    """
    set_style()
    metrics = [
        "M1_success_rate",
        "M6_coverage_progress",
        "M2_avg_return",
        "M9_spatial_spread",
    ]
    labels = [METRIC_LABELS.get(m, m) for m in metrics]
    heur_vals = [heuristic_metrics.get(m, 0) for m in metrics]
    rand_vals = [random_metrics.get(m, 0) for m in metrics]

    fig, ax = plt.subplots(figsize=(7, 5))
    y = np.arange(len(metrics))
    bar_h = 0.35

    bars_h = ax.barh(
        y - bar_h / 2, heur_vals, bar_h,
        label="Heuristic", color="#27ae60", edgecolor="white",
    )
    bars_r = ax.barh(
        y + bar_h / 2, rand_vals, bar_h,
        label="Random", color="#e74c3c", edgecolor="white",
    )

    all_vals = heur_vals + rand_vals
    x_min = min(0, min(all_vals))
    x_max = max(all_vals)
    x_pad = (x_max - x_min) * 0.2 + 0.1

    for bars in (bars_h, bars_r):
        for bar in bars:
            v = bar.get_width()
            offset = x_pad * 0.1 if v >= 0 else -x_pad * 0.1
            ax.text(
                v + offset, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}", va="center", fontsize=10, fontweight="bold",
            )

    ax.set_xlim(x_min - x_pad * 0.3, x_max + x_pad)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_title("Baseline Comparison: Heuristic vs Random", fontsize=13)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.2, axis="x")
    fig.tight_layout()
    return fig


def plot_results_comparison(
    comparison_dict: Dict[str, Dict[str, float]],
    colors: Optional[Dict[str, str]] = None,
) -> plt.Figure:
    """1x3 subplot figure comparing policies across three metrics.

    Each subplot is a horizontal bar chart for one metric:
    M1_success_rate, M2_avg_return, M6_coverage_progress.

    Args:
        comparison_dict: {"Random": metrics, "Heuristic": metrics,
                          "Trained (MAPPO)": metrics}
        colors: optional dict mapping labels to hex colour strings

    Returns:
        matplotlib Figure
    """
    set_style()
    if colors is None:
        colors = {
            "Random": "#e74c3c",
            "Heuristic": "#27ae60",
            "Trained (MAPPO)": "#1f77b4",
        }

    metrics = ["M1_success_rate", "M2_avg_return", "M6_coverage_progress"]
    policy_labels = list(comparison_dict.keys())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, metric in zip(axes, metrics):
        values = [
            comparison_dict[p].get(metric, 0) for p in policy_labels
        ]
        bar_colors = [colors.get(p, "#999") for p in policy_labels]

        bars = ax.barh(
            policy_labels, values, color=bar_colors,
            height=0.5, edgecolor="white",
        )

        is_pct = "rate" in metric or "progress" in metric
        v_min = min(0, min(values)) if values else 0
        v_max = max(values) if values else 1
        v_pad = (v_max - v_min) * 0.2 + 0.1

        for bar, v in zip(bars, values):
            label = f"{v:.0%}" if is_pct else f"{v:.2f}"
            offset = v_pad * 0.1
            ax.text(
                v + offset, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=10, fontweight="bold",
            )

        ax.set_xlim(
            v_min - v_pad * 0.3 if v_min < 0 else 0,
            v_max + v_pad,
        )
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=12)
        ax.grid(True, alpha=0.2, axis="x")

    fig.suptitle("Policy Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_sweep_overview(
    df,
    title: str = "Sweep Overview",
) -> plt.Figure:
    """2x2 multi-metric overview across all runs in a sweep.

    Panels:
      [0,0] M1 vs M4  — success rate vs collisions (safety tradeoff)
      [0,1] M1 vs M3  — success rate vs completion speed
      [1,0] M6 vs M8  — coverage vs agent utilization balance
      [1,1] M1 vs M9  — success rate vs spatial spread

    Points are colored by n_agents if available, otherwise uniform.

    Args:
        df: DataFrame from ExperimentStorage.to_dataframe()
        title: figure suptitle
    """
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Color by n_agents if available
    if "n_agents" in df.columns:
        agents_vals = sorted(df["n_agents"].unique())
        cmap = plt.cm.viridis
        norm = plt.Normalize(min(agents_vals), max(agents_vals))
        colors = [cmap(norm(n)) for n in df["n_agents"]]
        # Build legend handles
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=cmap(norm(n)), markersize=8,
                   label=f"N={n}")
            for n in agents_vals
        ]
    else:
        colors = "#1f77b4"
        handles = None

    panels = [
        ("M1_success_rate", "M4_avg_collisions", axes[0, 0]),
        ("M1_success_rate", "M3_avg_steps", axes[0, 1]),
        ("M6_coverage_progress", "M8_agent_utilization", axes[1, 0]),
        ("M1_success_rate", "M9_spatial_spread", axes[1, 1]),
    ]

    for x_key, y_key, ax in panels:
        if x_key not in df.columns or y_key not in df.columns:
            ax.set_visible(False)
            continue

        ax.scatter(
            df[x_key], df[y_key],
            c=colors, s=60, alpha=0.7, edgecolors="white",
            linewidth=0.5,
        )
        ax.set_xlabel(METRIC_LABELS.get(x_key, x_key), fontsize=10)
        ax.set_ylabel(METRIC_LABELS.get(y_key, y_key), fontsize=10)
        ax.grid(True, alpha=0.2)

        # Format percentage axes
        if "rate" in x_key or "progress" in x_key:
            ax.set_xlim(-0.05, 1.05)
        if "rate" in y_key or "progress" in y_key:
            ax.set_ylim(-0.05, 1.05)

    if handles:
        fig.legend(
            handles=handles, loc="upper right",
            bbox_to_anchor=(0.99, 0.99), fontsize=9,
            title="Agents", title_fontsize=10,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def save_figure(fig: plt.Figure, path: str, dpi: int = 150):
    """Save figure to file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}")
