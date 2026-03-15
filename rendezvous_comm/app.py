"""Rendezvous Experiment Manager — Streamlit app.

Launch: streamlit run app.py
"""
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Rendezvous Experiments",
    page_icon="🤖",  # noqa: RUF001
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure src is importable
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.config import RESULTS_DIR, CONFIGS_DIR
from src.storage import ExperimentStorage

st.title("Rendezvous Experiment Manager")

st.markdown("""
Multi-agent coordination experiments using VMAS Discovery + BenchMARL.
Use the sidebar to navigate between experiment setup, OVH job management,
and results analysis.
""")

# Quick status
st.subheader("Status")
col1, col2, col3 = st.columns(3)

exp_ids = ["er1", "er2", "er3", "er4"]
total_runs = 0
total_exps_with_data = 0
for eid in exp_ids:
    es = ExperimentStorage(eid)
    runs = es.list_runs()
    if runs:
        total_exps_with_data += 1
        total_runs += len(runs)

col1.metric("Experiments with data", f"{total_exps_with_data}/{len(exp_ids)}")
col2.metric("Total completed runs", total_runs)
col3.metric("Results dir", str(RESULTS_DIR))

# Config overview
st.subheader("Available Configs")
for eid in exp_ids:
    config_dir = CONFIGS_DIR / eid
    if config_dir.exists():
        yamls = sorted(config_dir.glob("*.yaml"))
        if yamls:
            names = [y.name for y in yamls]
            st.write(f"**{eid.upper()}**: {', '.join(names)}")
