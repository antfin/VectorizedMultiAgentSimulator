"""Sweep analysis: heatmaps, seed variance, and overview scatter plots."""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import CONFIGS_DIR, load_experiment, find_configs
from src.storage import ExperimentStorage
from src.plotting import (
    plot_sweep_heatmap, plot_seed_variance, plot_sweep_overview,
    set_style, METRIC_LABELS,
)

st.set_page_config(page_title="Sweep Analysis", layout="wide")
st.title("Sweep Analysis")

# Experiment selector
exp_ids = sorted(
    [d.name for d in CONFIGS_DIR.iterdir() if d.is_dir()],
)
if not exp_ids:
    st.warning("No experiment configs found.")
    st.stop()

exp_id = st.sidebar.selectbox("Experiment", exp_ids)
storage = ExperimentStorage(exp_id)
df = storage.to_dataframe()

if df.empty:
    st.info(f"No completed runs for {exp_id.upper()}.")
    st.stop()

st.sidebar.metric("Completed runs", len(df))

# Available metrics
metric_cols = [c for c in df.columns if c.startswith("M")]
metric = st.sidebar.selectbox(
    "Metric",
    metric_cols,
    format_func=lambda m: METRIC_LABELS.get(m, m),
)

# ── Heatmap ──
st.subheader("Sweep Heatmap")

sweep_params = [
    c for c in ["n_agents", "n_targets", "agents_per_target", "lidar_range"]
    if c in df.columns and df[c].nunique() > 1
]

if len(sweep_params) >= 2:
    col1, col2 = st.columns(2)
    with col1:
        row_param = st.selectbox("Row axis", sweep_params, index=0)
    with col2:
        remaining = [p for p in sweep_params if p != row_param]
        col_param = st.selectbox("Column axis", remaining, index=0)

    try:
        fig = plot_sweep_heatmap(
            df, metric=metric,
            row_param=row_param, col_param=col_param,
            title=f"{METRIC_LABELS.get(metric, metric)} — {exp_id.upper()}",
        )
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.warning(f"Cannot render heatmap: {e}")
elif len(sweep_params) == 1:
    st.info(
        f"Only one swept parameter ({sweep_params[0]}). "
        "Need at least two for a heatmap."
    )
else:
    st.info("No parameter variation in this experiment — heatmap needs a sweep.")

# ── Seed Variance ──
st.subheader("Seed Variance")

group_options = [
    c for c in ["algorithm", "n_agents", "n_targets", "agents_per_target"]
    if c in df.columns and df[c].nunique() > 1
]
if not group_options:
    group_options = ["algorithm"] if "algorithm" in df.columns else []

if group_options and "seed" in df.columns and df["seed"].nunique() > 1:
    group_by = st.selectbox("Group by", group_options)
    fig = plot_seed_variance(
        df, metric=metric, group_by=group_by,
        title=f"Seed Variance: {METRIC_LABELS.get(metric, metric)}",
    )
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("Need multiple seeds and a grouping parameter for variance analysis.")

# ── Overview Scatter ──
st.subheader("Cross-Metric Overview")
try:
    fig = plot_sweep_overview(df, title=f"Sweep Overview — {exp_id.upper()}")
    st.pyplot(fig)
    plt.close(fig)
except Exception as e:
    st.warning(f"Cannot render overview: {e}")

# ── Raw data table ──
with st.expander("Raw Metrics Table"):
    display_cols = ["run_id"] + metric_cols
    display_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(
        df[display_cols].sort_values(metric, ascending=False),
        use_container_width=True,
    )
