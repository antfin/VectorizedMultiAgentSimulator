"""Experiment config browser and run status."""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.theme import apply_theme
from src.config import CONFIGS_DIR, load_experiment
from src.storage import ExperimentStorage
from src.provenance import check_freshness, Freshness

st.set_page_config(page_title="Experiments", layout="wide")
apply_theme(title="Experiment Setup")

# Experiment selector
exp_ids = sorted(
    [d.name for d in CONFIGS_DIR.iterdir() if d.is_dir()],
)
exp_id = st.sidebar.selectbox("Experiment", exp_ids, index=0)

config_dir = CONFIGS_DIR / exp_id
yamls = sorted(config_dir.glob("*.yaml")) if config_dir.exists() else []

if not yamls:
    st.warning(f"No configs found in {config_dir}")
    st.stop()

# Config selector
config_name = st.sidebar.selectbox(
    "Config file",
    [y.name for y in yamls],
)
config_path = config_dir / config_name

# Load and display
spec = load_experiment(config_path)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Config")
    config_text = config_path.read_text()
    st.code(config_text, language="yaml")

with col2:
    st.subheader("Sweep Dimensions")
    st.write(f"**Agents:** {spec.sweep.n_agents}")
    st.write(f"**Targets:** {spec.sweep.n_targets}")
    st.write(f"**K (agents/target):** {spec.sweep.agents_per_target}")
    st.write(f"**LiDAR range:** {spec.sweep.lidar_range}")
    st.write(f"**Algorithms:** {spec.sweep.algorithms}")
    st.write(f"**Seeds:** {spec.sweep.seeds}")

    n_runs = len(list(spec.iter_runs()))
    st.metric("Total sweep runs", n_runs)
    st.write(f"**Frames/run:** {spec.train.max_n_frames:,}")
    st.write(f"**Device:** {spec.train.train_device}")

# Run status
st.subheader("Run Status")
storage = ExperimentStorage(exp_id)

status_rows = []
for run_id, overrides, algo, seed in spec.iter_runs():
    rs = storage.get_run(run_id)
    complete = rs.is_complete()
    has_policy = rs.has_policy()

    # Freshness
    if complete:
        freshness = check_freshness(rs.run_dir, spec)
        badge = freshness.value
    else:
        badge = "NEW"

    status_rows.append({
        "Run ID": run_id,
        "Status": "DONE" if complete else "PENDING",
        "Freshness": badge,
        "Policy": "Yes" if has_policy else "No",
        **{k: v for k, v in overrides.items()},
    })

if status_rows:
    st.dataframe(status_rows, use_container_width=True)
