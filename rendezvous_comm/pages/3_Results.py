"""Results Dashboard — merged view for sweep overview, run detail, and training curves."""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.theme import apply_theme
from src.config import CONFIGS_DIR, RESULTS_DIR
from src.storage import ExperimentStorage
from src.consolidate import load_latest_csv, consolidate_csvs, list_experiments_with_data
from src.plotting import (
    plot_sweep_heatmap, plot_seed_variance, plot_sweep_overview,
    plot_training_dashboard, set_style, METRIC_LABELS,
    POLIMI_DARK_BLUE, POLIMI_RED, POLIMI_LIGHT_BLUE,
)

st.set_page_config(page_title="Results Dashboard", layout="wide")
apply_theme(title="Results Dashboard")

# ── Sidebar ──────────────────────────────────────────────────────────
exp_ids = list_experiments_with_data()
if not exp_ids:
    st.info("No experiments with completed runs.")
    st.stop()

exp_id = st.sidebar.selectbox("Experiment", exp_ids)
results_dir = RESULTS_DIR / exp_id

# Rebuild CSVs button
if st.sidebar.button("Rebuild CSVs"):
    with st.spinner("Consolidating CSVs..."):
        paths = consolidate_csvs(exp_id)
    if paths:
        st.sidebar.success(
            f"Built {len(paths)} CSVs: "
            + ", ".join(p.name for p in paths.values())
        )
    else:
        st.sidebar.warning("No data to consolidate.")

# View mode
view_mode = st.sidebar.radio(
    "View",
    ["Overview", "Run Detail", "Training Curves"],
    index=0,
)

# ── Load data ────────────────────────────────────────────────────────
sweep_df = load_latest_csv(results_dir, "sweep_results")
iter_df = load_latest_csv(results_dir, "training_iter")
eval_df = load_latest_csv(results_dir, "training_eval")

# Fallback to ExperimentStorage if no consolidated CSV
storage = ExperimentStorage(exp_id)
if sweep_df is None or sweep_df.empty:
    sweep_df = storage.to_dataframe()

if sweep_df is None or sweep_df.empty:
    st.info(f"No completed runs for {exp_id.upper()}. Run an experiment first.")
    st.stop()

metric_cols = sorted(
    [c for c in sweep_df.columns if c.startswith("M")]
)

# ─────────────────────────────────────────────────────────────────────
# OVERVIEW MODE
# ─────────────────────────────────────────────────────────────────────
if view_mode == "Overview":
    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Runs", len(sweep_df))
    if "M1_success_rate" in sweep_df.columns:
        c2.metric("Best M1", f"{sweep_df['M1_success_rate'].max():.0%}")
    if "M6_coverage_progress" in sweep_df.columns:
        c3.metric("Mean M6", f"{sweep_df['M6_coverage_progress'].mean():.0%}")
    if "M4_avg_collisions" in sweep_df.columns:
        c4.metric("Mean M4", f"{sweep_df['M4_avg_collisions'].mean():.1f}")

    st.markdown("---")

    # Heatmap
    st.subheader("Sweep Heatmap")
    metric = st.selectbox(
        "Metric",
        metric_cols,
        format_func=lambda m: METRIC_LABELS.get(m, m),
        key="heatmap_metric",
    )

    sweep_params = [
        c for c in [
            "n_agents", "n_targets", "agents_per_target", "lidar_range",
        ]
        if c in sweep_df.columns and sweep_df[c].nunique() > 1
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
                sweep_df, metric=metric,
                row_param=row_param, col_param=col_param,
                title=f"{METRIC_LABELS.get(metric, metric)} — {exp_id.upper()}",
            )
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.warning(f"Cannot render heatmap: {e}")
    else:
        st.info("Need at least 2 swept parameters for a heatmap.")

    # 2x2 overview scatter
    st.subheader("Cross-Metric Overview")
    try:
        fig = plot_sweep_overview(
            sweep_df, title=f"Sweep Overview — {exp_id.upper()}",
        )
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.warning(f"Cannot render overview: {e}")

    # Raw data table
    with st.expander("Raw Sweep Data"):
        display_cols = ["run_id"] + metric_cols
        display_cols = [c for c in display_cols if c in sweep_df.columns]
        st.dataframe(
            sweep_df[display_cols].sort_values(
                metric_cols[0] if metric_cols else "run_id",
                ascending=False,
            ),
            use_container_width=True,
        )

# ─────────────────────────────────────────────────────────────────────
# RUN DETAIL MODE
# ─────────────────────────────────────────────────────────────────────
elif view_mode == "Run Detail":
    completed = storage.list_runs()
    if not completed:
        st.info("No completed runs.")
        st.stop()

    run_id = st.sidebar.selectbox("Run", completed)
    rs = storage.get_run(run_id)
    metrics = rs.load_metrics()

    # KPI cards
    st.subheader("Key Metrics")
    if metrics:
        kpi_keys = [
            ("M1_success_rate", "M1: Success Rate", "%.0f%%", 100),
            ("M2_avg_return", "M2: Avg Return", "%.2f", 1),
            ("M3_avg_steps", "M3: Avg Steps", "%.1f", 1),
            ("M4_avg_collisions", "M4: Collisions", "%.2f", 1),
            ("M5_avg_tokens", "M5: Tokens/Ep", "%.0f", 1),
            ("M6_coverage_progress", "M6: Coverage", "%.0f%%", 100),
            ("M8_agent_utilization", "M8: Utilization CV", "%.3f", 1),
            ("M9_spatial_spread", "M9: Spatial Spread", "%.3f", 1),
        ]
        cols = st.columns(4)
        for i, (key, label, fmt, mult) in enumerate(kpi_keys):
            val = metrics.get(key)
            if val is not None:
                cols[i % 4].metric(label, fmt % (val * mult))
    else:
        st.warning("No metrics found.")

    st.markdown("---")

    # M1/M4 eval curves (from eval CSV or raw scalars)
    st.subheader("Evaluation Curves")
    scalars = rs.load_benchmarl_scalars()

    m1_data = scalars.get("eval_M1_success_rate") if scalars else None
    m4_data = scalars.get("eval_M4_avg_collisions") if scalars else None

    if m1_data or m4_data:
        set_style()
        fig, (ax1, ax4) = plt.subplots(1, 2, figsize=(12, 3.5))

        if m1_data:
            s, v = zip(*m1_data)
            ax1.plot(s, v, color=POLIMI_DARK_BLUE, linewidth=1.8,
                     marker="o", markersize=4)
            ax1.fill_between(s, v, alpha=0.1, color=POLIMI_DARK_BLUE)
        ax1.set_ylim(0, 1.05)
        ax1.set_title("M1 — Success Rate")
        ax1.set_xlabel("Iteration")
        ax1.grid(True, alpha=0.3)

        if m4_data:
            s, v = zip(*m4_data)
            ax4.plot(s, v, color=POLIMI_RED, linewidth=1.8,
                     marker="o", markersize=4)
            ax4.fill_between(s, v, alpha=0.1, color=POLIMI_RED)
        ax4.set_title("M4 — Avg Collisions")
        ax4.set_xlabel("Iteration")
        ax4.grid(True, alpha=0.3)

        fig.suptitle(f"Eval Metrics — {run_id}",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # 6-panel training dashboard
    if scalars:
        st.subheader("Training Dashboard")
        fig = plot_training_dashboard(
            scalars, title=f"Training Progress — {run_id}",
        )
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No BenchMARL scalars found for this run.")

    # Config + policy info
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Run Config")
        config_path = rs.input_dir / "config.yaml"
        if config_path.exists():
            st.code(config_path.read_text(), language="yaml")
        else:
            st.info("No config snapshot.")
    with col_r:
        st.subheader("Policy")
        if rs.has_policy():
            policy_path = rs.output_dir / "policy.pt"
            st.success(f"Saved ({policy_path.stat().st_size / 1024:.1f} KB)")
        else:
            st.info("No saved policy.")

        st.subheader("Report")
        report_path = rs.run_dir / "report.md"
        if report_path.exists():
            with st.expander("View Report"):
                st.markdown(report_path.read_text())

# ─────────────────────────────────────────────────────────────────────
# TRAINING CURVES MODE
# ─────────────────────────────────────────────────────────────────────
elif view_mode == "Training Curves":
    completed = storage.list_runs()
    if not completed:
        st.info("No completed runs.")
        st.stop()

    selected_runs = st.sidebar.multiselect(
        "Runs", completed, default=completed[:3],
    )
    if not selected_runs:
        st.info("Select at least one run.")
        st.stop()

    # Per-run dashboards
    for run_id in selected_runs:
        rs = storage.get_run(run_id)
        scalars = rs.load_benchmarl_scalars()
        if not scalars:
            st.warning(f"No scalars for {run_id}")
            continue

        with st.expander(run_id, expanded=len(selected_runs) == 1):
            fig = plot_training_dashboard(
                scalars, title=f"Training — {run_id}",
            )
            st.pyplot(fig)
            plt.close(fig)

    # M1 overlay across runs
    if len(selected_runs) > 1:
        st.subheader("M1 Overlay — Selected Runs")
        set_style()
        fig, ax = plt.subplots(figsize=(10, 4))
        for run_id in selected_runs:
            rs = storage.get_run(run_id)
            scalars = rs.load_benchmarl_scalars()
            m1 = scalars.get("eval_M1_success_rate") if scalars else None
            if m1:
                s, v = zip(*m1)
                label = run_id.split(f"{exp_id}_")[-1].rsplit("_s", 1)[0]
                ax.plot(s, v, linewidth=1.8, marker="o",
                        markersize=3, label=label)
        ax.set_ylim(0, 1.05)
        ax.set_title("M1 Success Rate — Comparison")
        ax.set_xlabel("Iteration")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Eval reward overlay
    if len(selected_runs) > 1:
        st.subheader("Eval Reward Overlay")
        set_style()
        fig, ax = plt.subplots(figsize=(10, 4))
        for run_id in selected_runs:
            rs = storage.get_run(run_id)
            scalars = rs.load_benchmarl_scalars()
            rew = scalars.get("eval_reward_episode_reward_mean") if scalars else None
            if rew:
                s, v = zip(*rew)
                label = run_id.split(f"{exp_id}_")[-1].rsplit("_s", 1)[0]
                ax.plot(s, v, linewidth=1.8, label=label)
        ax.set_title("Eval Reward — Comparison")
        ax.set_xlabel("Iteration")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
