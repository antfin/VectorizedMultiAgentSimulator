"""Per-run deep dive: KPI cards, config, provenance, training curves."""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import CONFIGS_DIR, load_experiment
from src.storage import ExperimentStorage
from src.provenance import check_freshness, Freshness
from src.plotting import plot_training_dashboard, set_style, METRIC_LABELS

st.set_page_config(page_title="Run Detail", layout="wide")
st.title("Run Detail")

# Experiment selector
exp_ids = sorted(
    [d.name for d in CONFIGS_DIR.iterdir() if d.is_dir()],
)
exp_id = st.sidebar.selectbox("Experiment", exp_ids)
storage = ExperimentStorage(exp_id)
completed = storage.list_runs()

if not completed:
    st.info(f"No completed runs for {exp_id.upper()}.")
    st.stop()

run_id = st.sidebar.selectbox("Run", completed)
rs = storage.get_run(run_id)
metrics = rs.load_metrics()

# ── KPI Cards ──
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
    st.warning("No metrics found for this run.")

# ── Provenance / Freshness ──
st.subheader("Provenance")

config_dir = CONFIGS_DIR / exp_id
yamls = sorted(config_dir.glob("*.yaml")) if config_dir.exists() else []

if yamls:
    # Try to match run to a config
    freshness_info = []
    for yp in yamls:
        try:
            spec = load_experiment(yp)
            f = check_freshness(rs.run_dir, spec)
            freshness_info.append((yp.name, f.value))
        except Exception:
            freshness_info.append((yp.name, "ERROR"))

    if freshness_info:
        for cfg_name, badge in freshness_info:
            if badge == Freshness.VALID.value:
                st.success(f"{cfg_name}: {badge}")
            elif badge == "ERROR":
                st.error(f"{cfg_name}: {badge}")
            else:
                st.warning(f"{cfg_name}: {badge}")
else:
    st.info("No config files found for freshness check.")

# ── Saved Config ──
st.subheader("Run Config")

config_path = rs.input_dir / "config.yaml"
if config_path.exists():
    st.code(config_path.read_text(), language="yaml")
else:
    st.info("No saved config snapshot (input/config.yaml).")

# ── Training Curves ──
st.subheader("Training Curves")

scalars = rs.load_benchmarl_scalars()
if scalars:
    # M1 + M4 quick view
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

    # Full 6-panel dashboard
    fig = plot_training_dashboard(
        scalars, title=f"Training Progress — {run_id}",
    )
    st.pyplot(fig)
    plt.close(fig)

    # Scalar keys available
    with st.expander("Available Scalars"):
        for name, data in sorted(scalars.items()):
            st.write(f"**{name}**: {len(data)} points")
else:
    st.info("No BenchMARL scalars found.")

# ── Policy Info ──
st.subheader("Policy")

if rs.has_policy():
    st.success("Trained policy saved (policy.pt)")
    policy_path = rs.output_dir / "policy.pt"
    st.write(f"Size: {policy_path.stat().st_size / 1024:.1f} KB")
else:
    st.info("No saved policy.")

# ── Report ──
st.subheader("Report")

report_path = rs.run_dir / "report.md"
if report_path.exists():
    st.markdown(report_path.read_text())
else:
    st.info("No report.md generated for this run.")
