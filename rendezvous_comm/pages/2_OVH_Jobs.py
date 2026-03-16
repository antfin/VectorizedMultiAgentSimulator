"""OVH AI Training job management."""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import CONFIGS_DIR, RESULTS_DIR
from src.ovh import (
    GPU_MODELS, check_cli_available, submit_training_job,
    list_jobs, get_job, get_job_logs, stop_job,
    list_buckets, upload_code, download_results, estimate_cost,
    default_bucket_code, default_bucket_results, default_region,
    default_gpu, default_image,
)

from src.theme import apply_theme

st.set_page_config(page_title="OVH Jobs", layout="wide")
apply_theme(title="OVH Job Management")

# CLI check
cli_ok = check_cli_available()
if not cli_ok:
    st.error(
        "**ovhai CLI not found.** Install it from "
        "[OVH docs](https://docs.ovh.com/gb/en/ai-training/install-client/) "
        "and run `ovhai login` to authenticate."
    )
    st.stop()

st.success("ovhai CLI available")

tab_launch, tab_monitor, tab_download, tab_cost = st.tabs([
    "Launch Job", "Monitor Jobs", "Download Results", "Cost Estimator",
])

# ── Launch Tab ──
with tab_launch:
    st.subheader("Submit Training Job")

    col1, col2 = st.columns(2)

    with col1:
        # Config selection
        exp_ids = sorted(
            [d.name for d in CONFIGS_DIR.iterdir() if d.is_dir()],
        )
        exp_id = st.selectbox("Experiment", exp_ids, key="launch_exp")
        config_dir = CONFIGS_DIR / exp_id
        yamls = sorted(config_dir.glob("*.yaml")) if config_dir.exists() else []
        config_name = st.selectbox(
            "Config", [y.name for y in yamls], key="launch_config",
        )
        config_rel = f"rendezvous_comm/configs/{exp_id}/{config_name}"

    with col2:
        gpu_keys = list(GPU_MODELS.keys())
        gpu_default_idx = (
            gpu_keys.index(default_gpu())
            if default_gpu() in gpu_keys else 0
        )
        gpu_type = st.selectbox(
            "GPU Model",
            gpu_keys,
            index=gpu_default_idx,
            format_func=lambda g: f"{g} ({GPU_MODELS[g]['vram_gb']}GB, {GPU_MODELS[g]['eur_per_hr']} EUR/hr)",
        )
        bucket_code = st.text_input("Code bucket", default_bucket_code())
        bucket_results = st.text_input("Results bucket", default_bucket_results())
        region = st.text_input("Region", default_region())

    # Upload code button
    col_up, col_sub = st.columns(2)
    with col_up:
        if st.button("Upload Code to OVH"):
            code_dir = str(Path(__file__).parent.parent)
            with st.spinner("Uploading..."):
                ok = upload_code(code_dir, bucket_code, region)
            if ok:
                st.success("Code uploaded")
            else:
                st.error("Upload failed — check logs")

    with col_sub:
        if st.button("Submit Job", type="primary"):
            with st.spinner("Submitting..."):
                job_id = submit_training_job(
                    config_yaml=config_rel,
                    gpu_type=gpu_type,
                    bucket_code=bucket_code,
                    bucket_results=bucket_results,
                    region=region,
                )
            if job_id:
                st.success(f"Job submitted: `{job_id}`")
                st.session_state["last_job_id"] = job_id
            else:
                st.error("Submission failed — check ovhai auth and bucket names")

# ── Monitor Tab ──
with tab_monitor:
    st.subheader("Job Status")

    status_filter = st.selectbox(
        "Filter by status",
        [None, "RUNNING", "DONE", "ERROR", "FINALIZING", "INITIALIZING"],
        format_func=lambda x: "All" if x is None else x,
    )

    if st.button("Refresh", key="refresh_jobs"):
        st.session_state["jobs_cache"] = None

    jobs = list_jobs(status_filter)
    if jobs:
        rows = []
        for j in jobs:
            dur_m = j.duration_seconds // 60 if j.duration_seconds else 0
            rows.append({
                "ID": j.id[:12],
                "Name": j.name,
                "Status": j.status,
                "GPU": j.gpu_type,
                "Duration": f"{dur_m}m",
                "Created": j.created_at[:19] if j.created_at else "",
            })
        st.dataframe(rows, use_container_width=True)

        # Job detail
        job_ids = [j.id for j in jobs]
        selected_job = st.selectbox("Select job for details", job_ids,
                                     format_func=lambda x: x[:12])
        if selected_job:
            col_log, col_stop = st.columns([4, 1])
            with col_log:
                if st.button("Show Logs"):
                    logs = get_job_logs(selected_job, tail=50)
                    st.code(logs, language="text")
            with col_stop:
                if st.button("Stop Job", type="secondary"):
                    if stop_job(selected_job):
                        st.success("Job stopped")
                    else:
                        st.error("Failed to stop job")
    else:
        st.info("No jobs found")

# ── Download Tab ──
with tab_download:
    st.subheader("Download Results from OVH")

    dl_bucket = st.text_input("Results bucket", default_bucket_results(),
                               key="dl_bucket")
    dl_prefix = st.text_input(
        "Prefix (e.g., er1/)",
        placeholder="Leave empty for all",
        key="dl_prefix",
    )
    dl_region = st.text_input("Region", default_region(), key="dl_region")
    dl_dir = st.text_input("Local directory", str(RESULTS_DIR))

    if st.button("Download", type="primary", key="dl_btn"):
        with st.spinner("Downloading..."):
            ok = download_results(
                dl_bucket, dl_dir,
                prefix=dl_prefix, region=dl_region,
            )
        if ok:
            st.success(f"Downloaded to {dl_dir}")
        else:
            st.error("Download failed")

# ── Cost Tab ──
with tab_cost:
    st.subheader("Cost Estimator")

    col1, col2 = st.columns(2)
    with col1:
        est_gpu = st.selectbox(
            "GPU Model", list(GPU_MODELS.keys()), key="est_gpu",
        )
        est_runs = st.number_input("Number of runs", 1, 500, 4)
    with col2:
        est_min = st.number_input(
            "Est. minutes per run", 5, 600, 30,
        )
        est_storage = st.number_input(
            "Storage (GB)", 0.1, 100.0, 1.0,
        )

    cost = estimate_cost(est_gpu, est_runs, est_min, est_storage)
    st.metric("GPU Hours", f"{cost['gpu_hours']:.1f}h")
    st.metric("GPU Cost", f"{cost['gpu_cost_eur']:.2f} EUR")
    st.metric("Storage Cost", f"{cost['storage_cost_eur']:.4f} EUR/month")
    st.metric("Total", f"{cost['total_eur']:.2f} EUR")
