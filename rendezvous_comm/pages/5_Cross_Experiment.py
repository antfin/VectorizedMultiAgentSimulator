"""Cross-experiment comparison, statistical tests, and Pareto frontiers."""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import CONFIGS_DIR
from src.storage import ExperimentStorage, load_cross_experiment
from src.plotting import (
    plot_baseline_comparison, plot_metric_radar, plot_success_vs_tokens,
    set_style, METRIC_LABELS, COLORS, LABELS,
)
from src.stats import compare_experiments, bootstrap_ci, pareto_frontier

st.set_page_config(page_title="Cross-Experiment", layout="wide")
st.title("Cross-Experiment Comparison")

# Multi-select experiments
all_exp_ids = sorted(
    [d.name for d in CONFIGS_DIR.iterdir() if d.is_dir()],
)
selected_exps = st.sidebar.multiselect(
    "Experiments", all_exp_ids, default=all_exp_ids[:2],
)

if len(selected_exps) < 1:
    st.info("Select at least one experiment.")
    st.stop()

cross_df = load_cross_experiment(selected_exps)

if cross_df.empty:
    st.info("No completed runs found for selected experiments.")
    st.stop()

st.sidebar.metric("Total runs", len(cross_df))

metric_cols = [c for c in cross_df.columns if c.startswith("M")]

# ── Bar Comparison ──
st.subheader("Metric Comparison")

metric = st.selectbox(
    "Metric",
    metric_cols,
    format_func=lambda m: METRIC_LABELS.get(m, m),
    key="compare_metric",
)

fig = plot_baseline_comparison(
    cross_df, metric=metric, group_col="experiment",
    title=f"{METRIC_LABELS.get(metric, metric)} — Across Experiments",
)
st.pyplot(fig)
plt.close(fig)

# ── Statistical Significance ──
if len(selected_exps) >= 2:
    st.subheader("Statistical Significance (Mann-Whitney U)")

    col1, col2 = st.columns(2)
    with col1:
        exp_a = st.selectbox("Experiment A", selected_exps, index=0, key="exp_a")
    with col2:
        exp_b = st.selectbox(
            "Experiment B", selected_exps,
            index=min(1, len(selected_exps) - 1), key="exp_b",
        )

    if exp_a != exp_b:
        sig_metric = st.selectbox(
            "Metric for test",
            metric_cols,
            format_func=lambda m: METRIC_LABELS.get(m, m),
            key="sig_metric",
        )

        vals_a = cross_df[cross_df["experiment"] == exp_a][sig_metric].dropna().tolist()
        vals_b = cross_df[cross_df["experiment"] == exp_b][sig_metric].dropna().tolist()

        if vals_a and vals_b:
            result = compare_experiments(vals_a, vals_b)
            ci_a = bootstrap_ci(vals_a)
            ci_b = bootstrap_ci(vals_b)

            col1, col2, col3 = st.columns(3)
            col1.metric("p-value", f"{result['p_value']:.4f}")
            col2.metric("Effect size", f"{result['effect_size']:.3f}")
            col3.metric(
                "Significant",
                "Yes" if result["significant"] else "No",
            )

            st.write(
                f"**{exp_a.upper()}**: mean={np.mean(vals_a):.3f}, "
                f"95% CI=[{ci_a[0]:.3f}, {ci_a[1]:.3f}], n={len(vals_a)}"
            )
            st.write(
                f"**{exp_b.upper()}**: mean={np.mean(vals_b):.3f}, "
                f"95% CI=[{ci_b[0]:.3f}, {ci_b[1]:.3f}], n={len(vals_b)}"
            )
        else:
            st.warning("Not enough data for statistical test.")
    else:
        st.info("Select two different experiments to compare.")

# ── Radar Chart ──
st.subheader("Multi-Metric Radar")

# Compute per-experiment mean metrics
radar_metrics = {}
for eid in selected_exps:
    subset = cross_df[cross_df["experiment"] == eid]
    means = {}
    for m in metric_cols:
        if m in subset.columns:
            val = subset[m].mean()
            if not np.isnan(val):
                means[m] = val
    if means:
        radar_metrics[eid] = means

if len(radar_metrics) >= 1:
    available = sorted(
        set.intersection(*(set(v.keys()) for v in radar_metrics.values()))
    )
    if len(available) >= 3:
        fig = plot_metric_radar(
            radar_metrics, metrics=available[:6],
            title="Experiment Comparison",
        )
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Need at least 3 common metrics for radar chart.")

# ── Pareto Frontier ──
st.subheader("Pareto Frontier: M1 vs M5")

if "M1_success_rate" in cross_df.columns and "M5_avg_tokens" in cross_df.columns:
    fig = plot_success_vs_tokens(radar_metrics, title="M1 vs M5 — Pareto Frontier")

    # Compute and overlay Pareto frontier
    points = []
    point_labels = []
    for eid, mvals in radar_metrics.items():
        if "M5_avg_tokens" in mvals and "M1_success_rate" in mvals:
            points.append((mvals["M5_avg_tokens"], mvals["M1_success_rate"]))
            point_labels.append(eid)

    if len(points) >= 2:
        frontier_idx = pareto_frontier(
            points, maximize_x=False, maximize_y=True,
        )
        if frontier_idx:
            frontier_pts = sorted(
                [points[i] for i in frontier_idx], key=lambda p: p[0],
            )
            fx, fy = zip(*frontier_pts)
            fig.axes[0].plot(
                fx, fy, "--", color="gray", alpha=0.6,
                linewidth=1.5, label="Pareto frontier",
            )
            fig.axes[0].legend()

    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("M1 and M5 data needed for Pareto frontier plot.")

# ── Raw data ──
with st.expander("Full Cross-Experiment Table"):
    st.dataframe(cross_df, use_container_width=True)
