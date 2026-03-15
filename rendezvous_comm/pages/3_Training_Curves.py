"""Training curve visualization for individual runs."""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.storage import ExperimentStorage
from src.plotting import plot_training_dashboard, set_style

st.set_page_config(page_title="Training Curves", layout="wide")
st.title("Training Curves")

# Experiment selector
exp_ids = ["er1", "er2", "er3", "er4"]
exp_id = st.sidebar.selectbox("Experiment", exp_ids)

storage = ExperimentStorage(exp_id)
completed = storage.list_runs()

if not completed:
    st.info(f"No completed runs for {exp_id.upper()}.")
    st.stop()

# Run selector (multi-select for overlay)
selected_runs = st.sidebar.multiselect(
    "Runs", completed, default=completed[:1],
)

if not selected_runs:
    st.info("Select at least one run.")
    st.stop()

for run_id in selected_runs:
    rs = storage.get_run(run_id)
    scalars = rs.load_benchmarl_scalars()

    if not scalars:
        st.warning(f"No scalars for {run_id}")
        continue

    st.subheader(run_id)

    # M1 + M4 training curves
    m1_data = scalars.get("eval_M1_success_rate")
    m4_data = scalars.get("eval_M4_avg_collisions")

    if m1_data or m4_data:
        set_style()
        fig, (ax1, ax4) = plt.subplots(1, 2, figsize=(12, 3.5))

        if m1_data:
            s, v = zip(*m1_data)
            ax1.plot(s, v, color="#1f77b4", linewidth=1.8,
                     marker="o", markersize=4)
            ax1.fill_between(s, v, alpha=0.1, color="#1f77b4")
        ax1.set_ylim(0, 1.05)
        ax1.set_title("M1 — Success Rate")
        ax1.set_xlabel("Iteration")
        ax1.grid(True, alpha=0.3)

        if m4_data:
            s, v = zip(*m4_data)
            ax4.plot(s, v, color="#e74c3c", linewidth=1.8,
                     marker="o", markersize=4)
            ax4.fill_between(s, v, alpha=0.1, color="#e74c3c")
        ax4.set_title("M4 — Avg Collisions")
        ax4.set_xlabel("Iteration")
        ax4.grid(True, alpha=0.3)

        fig.suptitle(f"Eval Metrics — {run_id}",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # 6-panel dashboard
    fig = plot_training_dashboard(
        scalars, title=f"Training Progress — {run_id}",
    )
    st.pyplot(fig)
    plt.close(fig)

# Overlay mode: M1 across all selected runs
if len(selected_runs) > 1:
    st.subheader("M1 Overlay — All Selected Runs")
    set_style()
    fig, ax = plt.subplots(figsize=(10, 4))
    for run_id in selected_runs:
        rs = storage.get_run(run_id)
        scalars = rs.load_benchmarl_scalars()
        m1 = scalars.get("eval_M1_success_rate") if scalars else None
        if m1:
            s, v = zip(*m1)
            # Short label from run_id
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
