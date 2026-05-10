"""F7.3 — Per-run Detail page.

Sections (top → bottom): header card, M1–M9 metrics tiles, config viewer
(collapsible), training curves (BenchMARL scalar CSVs), before/after videos,
log tail. Sections quietly self-skip when their artefact is absent so a
half-finished run still renders cleanly.

Entry: deep link from F7.2 (`?run_id=<x>`) → URL query param. If the param
is missing or doesn't match a known run, fall back to the most-recent run
(by ``run_timestamp`` desc) and surface a sidebar selectbox so the user can
switch.
"""

# pylint: disable=wrong-import-position,invalid-name

import streamlit as st

from multi_scenario.frontend.charts import line_plot_csvs
from multi_scenario.frontend.run_detail_loader import load_run_detail, RunDetail
from multi_scenario.frontend.sidebar import (
    load_runs_with_cache,
    render_active_root_caption,
)
from multi_scenario.frontend.theme import (
    apply_theme,
    POLIMI_DARK_BLUE,
    POLIMI_GREEN,
    POLIMI_ORANGE,
    POLIMI_RED,
)

apply_theme(title="Run Detail", subtitle="Per-run drill-down")

experiments_dir = render_active_root_caption()
runs_df = load_runs_with_cache(experiments_dir)

if runs_df.empty:
    st.info(f"No runs under `{experiments_dir}`. Run an experiment first.")
    st.stop()

# ── Resolve which run to show ────────────────────────────────────────
# Sort most-recent-first so the default fallback is the latest run.
sorted_df = runs_df.sort_values("run_timestamp", ascending=False).reset_index(drop=True)
run_ids = sorted_df["run_id"].tolist()

requested = st.query_params.get("run_id")
default_run_id = requested if requested in run_ids else run_ids[0]

st.sidebar.header("Run")
selected_run_id = st.sidebar.selectbox(
    "Pick a run",
    run_ids,
    index=run_ids.index(default_run_id),
    label_visibility="collapsed",
)
if selected_run_id != requested:
    # Sync URL so the page is shareable / back-button-friendly.
    st.query_params["run_id"] = selected_run_id

# Look up the run_dir for the chosen run_id.
matching = sorted_df[sorted_df["run_id"] == selected_run_id]
if matching.empty:
    st.error(f"No run with id `{selected_run_id}` under `{experiments_dir}`.")
    st.stop()

from pathlib import Path  # noqa: E402  (kept inline so the page reads top-to-bottom)

run_dir = Path(matching.iloc[0]["run_dir"])
detail: RunDetail | None = load_run_detail(run_dir)
if detail is None:
    st.error(f"Could not load run artefacts at `{run_dir}`.")
    st.stop()

# ── 1. Header card ───────────────────────────────────────────────────
_STATE_BADGES = {
    "DONE": (POLIMI_GREEN, "DONE"),
    "RUNNING": (POLIMI_DARK_BLUE, "RUNNING"),
    "CRASHED": (POLIMI_RED, "CRASHED"),
    "RESUMED": (POLIMI_ORANGE, "RESUMED"),
    "INITIALIZING": ("#888888", "INITIALIZING"),
}


def _state_badge(state: str | None) -> str:
    color, label = _STATE_BADGES.get(state or "", ("#888888", state or "UNKNOWN"))
    return (
        f"<span style='background:{color};color:white;padding:2px 8px;"
        f"border-radius:4px;font-size:0.8rem;font-weight:600;'>{label}</span>"
    )


state_str = detail.run_state.state.value if detail.run_state else "UNKNOWN"
duration_str = "—"
if detail.run_state and len(detail.run_state.transitions) >= 2:
    started = detail.run_state.transitions[0].ts
    finished = detail.run_state.transitions[-1].ts
    secs = int((finished - started).total_seconds())
    mins, secs = divmod(secs, 60)
    duration_str = f"{mins}m{secs:02d}s"

st.markdown(
    f"### `{detail.result.run_id}` &nbsp;{_state_badge(state_str)}",
    unsafe_allow_html=True,
)
meta_cols = st.columns(5)
meta_cols[0].caption(f"**Scenario** &nbsp;{detail.result.scenario}")
meta_cols[1].caption(f"**Algorithm** &nbsp;{detail.result.algorithm}")
meta_cols[2].caption(f"**Seed** &nbsp;{detail.result.seed}")
meta_cols[3].caption(f"**Duration** &nbsp;{duration_str}")
meta_cols[4].caption(f"**Timestamp** &nbsp;{detail.result.run_timestamp}")
st.markdown("---")

# ── 2. Metrics tiles ─────────────────────────────────────────────────
st.subheader("Metrics")
present = [m for m in detail.result.metrics if m.value is not None]
if not present:
    st.info("No metric values recorded for this run.")
else:
    # Three-column grid; wraps if more than 3.
    for row_start in range(0, len(present), 3):
        row = present[row_start : row_start + 3]
        cols = st.columns(3)
        for col, metric in zip(cols, row):
            label = metric.name.replace("_", " ").title()
            if "rate" in metric.name or "progress" in metric.name:
                col.metric(label, f"{metric.value:.0%}")
            else:
                col.metric(label, f"{metric.value:.2f}")

# ── 3. Config viewer ─────────────────────────────────────────────────
with st.expander("Config", expanded=False):
    st.json(detail.cfg.model_dump())

# ── 4. Training curves ───────────────────────────────────────────────
st.markdown("---")
st.subheader("Training curves")
if not detail.scalar_csvs:
    st.info(
        "No BenchMARL scalar CSVs found under "
        f"`{detail.run_dir}/output/benchmarl/.../scalars/`."
    )
else:
    csv_names = [p.name for p in detail.scalar_csvs]
    picked_names = st.multiselect(
        "Pick scalars to plot",
        csv_names,
        default=csv_names[: min(3, len(csv_names))],
    )
    picked_paths = [p for p in detail.scalar_csvs if p.name in set(picked_names)]
    # Constrain to ~half of the page width on big screens — pyplot's
    # ``use_container_width`` flag isn't enough; the column wrapper enforces
    # an absolute cap.
    chart_col, _ = st.columns([1, 1])
    with chart_col:
        line_plot_csvs(picked_paths, title="Selected scalars")

# ── 5. Videos ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Videos")
if not detail.videos:
    # Local runs normally produce videos during training (begin + end). When
    # they're missing it's usually an OVH job whose container was headless
    # and Pyglet failed quietly, OR a local run whose recorder hit an error.
    # Either way the fix is the same: regenerate from the trained policy.
    msg_col, btn_col = st.columns([3, 1])
    with msg_col:
        st.info("No videos found for this run.")
    with btn_col:
        if st.button(
            "🎬 Regenerate videos",
            key="regen_videos_btn",
            use_container_width=True,
            help="Replays the trained policy to render before/after videos.",
        ):
            # pylint: disable=import-outside-toplevel
            import subprocess
            import sys

            # ``sys.executable -m`` works whether or not the
            # ``multi-scenario`` console script is on PATH (the Streamlit
            # server isn't always launched inside the project venv).
            with st.spinner("Regenerating videos…"):
                try:
                    proc = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "multi_scenario.cli",
                            "regenerate-videos",
                            str(detail.run_dir),
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                except OSError as exc:
                    st.error(f"Could not invoke regenerate-videos: {exc}")
                    proc = None
            if proc is not None:
                if proc.returncode == 0:
                    st.success("Videos regenerated.")
                    st.rerun()
                else:
                    st.error("Regenerate-videos failed.")
                    with st.expander("Show error output"):
                        st.code((proc.stderr or proc.stdout) or "(no output)")
else:
    video_cols = st.columns(2)
    if "before" in detail.videos:
        with video_cols[0]:
            st.caption("Before training (random-init policy)")
            st.video(str(detail.videos["before"]))
    if "after" in detail.videos:
        with video_cols[1]:
            st.caption("After training (trained policy)")
            st.video(str(detail.videos["after"]))

# ── 6. Logs tail ─────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Logs")
if detail.log_path is None:
    st.info("No log file found for this run.")
else:
    with st.expander(f"`{detail.log_path.name}` (last 200 lines)", expanded=False):
        try:
            lines = detail.log_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            st.warning(f"Could not read log: {exc}")
        else:
            tail = lines[-200:]
            st.code("\n".join(tail) if tail else "(empty log)", language=None)
