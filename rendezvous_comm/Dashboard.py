"""Multi-Robot Discovery K-N Experiment — Streamlit dashboard.

Launch: streamlit run Dashboard.py
"""
import streamlit as st
import sys
from pathlib import Path

st.set_page_config(
    page_title="Multi-Robot Discovery K-N Experiment",
    page_icon=":robot_face:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.theme import apply_theme
from src.config import RESULTS_DIR
from src.storage import ExperimentStorage
from src.consolidate import load_latest_csv, list_experiments_with_data
from src.plotting import (
    set_style, POLIMI_DARK_BLUE, POLIMI_LIGHT_BLUE,
    POLIMI_GREEN, POLIMI_RED, POLIMI_ORANGE,
)

apply_theme(title="Multi-Robot Discovery K-N Experiment")

# ── Load sweep data ──────────────────────────────────────────────────
exp_ids = list_experiments_with_data()

if not exp_ids:
    st.info("No completed runs found. Run an experiment first.")
    st.stop()

sweep_df = None
for eid in exp_ids:
    results_dir = RESULTS_DIR / eid
    df = load_latest_csv(results_dir, "sweep_results")
    if df is not None and not df.empty:
        df["exp_id"] = df.get("exp_id", eid)
        if sweep_df is None:
            sweep_df = df
        else:
            import pandas as pd
            sweep_df = pd.concat([sweep_df, df], ignore_index=True)

# Fallback to ExperimentStorage
if sweep_df is None:
    for eid in exp_ids:
        es = ExperimentStorage(eid)
        df = es.to_dataframe()
        if not df.empty:
            df["exp_id"] = eid
            if sweep_df is None:
                sweep_df = df
            else:
                import pandas as pd
                sweep_df = pd.concat([sweep_df, df], ignore_index=True)

if sweep_df is None or sweep_df.empty:
    st.info("No completed runs found. Run an experiment first.")
    st.stop()

# ── KPI Row ──────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Runs", len(sweep_df))

if "M1_success_rate" in sweep_df.columns:
    best_m1 = sweep_df["M1_success_rate"].max()
    c2.metric("Best M1 (Success)", f"{best_m1:.0%}")

if "M6_coverage_progress" in sweep_df.columns:
    mean_m6 = sweep_df["M6_coverage_progress"].mean()
    c3.metric("Mean M6 (Coverage)", f"{mean_m6:.0%}")

if "M4_avg_collisions" in sweep_df.columns:
    mean_m4 = sweep_df["M4_avg_collisions"].mean()
    c4.metric("Mean M4 (Collisions)", f"{mean_m4:.1f}")

st.markdown("---")

# ── Row 1: M1 bar + M1 vs M3 scatter ────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("M1 Success Rate by Configuration")
    if "M1_success_rate" in sweep_df.columns:
        set_style()
        fig, ax = plt.subplots(figsize=(8, 5))

        if "n_agents" in sweep_df.columns and "n_targets" in sweep_df.columns:
            sweep_df["config"] = (
                "N=" + sweep_df["n_agents"].astype(str)
                + " T=" + sweep_df["n_targets"].astype(str)
            )
            grouped = sweep_df.groupby("config")["M1_success_rate"]
            means = grouped.mean().sort_values(ascending=False)
            stds = grouped.std().reindex(means.index).fillna(0)

            bars = ax.bar(
                range(len(means)), means, yerr=stds,
                capsize=4, color=POLIMI_DARK_BLUE, alpha=0.85,
            )
            ax.set_xticks(range(len(means)))
            ax.set_xticklabels(means.index, rotation=30, ha="right")

            for bar, m in zip(bars, means):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{m:.0%}", ha="center", va="bottom", fontsize=10,
                )
        else:
            ax.bar(
                range(len(sweep_df)), sweep_df["M1_success_rate"],
                color=POLIMI_DARK_BLUE, alpha=0.85,
            )

        ax.set_ylabel("M1: Success Rate")
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.2, axis="y")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with col_right:
    st.subheader("M1 vs M3: Success vs Speed")
    if (
        "M1_success_rate" in sweep_df.columns
        and "M3_avg_steps" in sweep_df.columns
    ):
        set_style()
        fig, ax = plt.subplots(figsize=(8, 5))

        if "n_agents" in sweep_df.columns:
            agents_vals = sorted(sweep_df["n_agents"].unique())
            cmap = plt.cm.viridis
            norm = plt.Normalize(min(agents_vals), max(agents_vals))
            colors = [cmap(norm(n)) for n in sweep_df["n_agents"]]
            ax.scatter(
                sweep_df["M1_success_rate"],
                sweep_df["M3_avg_steps"],
                c=colors, s=80, alpha=0.8, edgecolors="white",
            )
            from matplotlib.lines import Line2D
            handles = [
                Line2D(
                    [0], [0], marker="o", color="w",
                    markerfacecolor=cmap(norm(n)), markersize=8,
                    label=f"N={n}",
                )
                for n in agents_vals
            ]
            ax.legend(handles=handles, title="Agents", fontsize=9)
        else:
            ax.scatter(
                sweep_df["M1_success_rate"],
                sweep_df["M3_avg_steps"],
                c=POLIMI_DARK_BLUE, s=80, alpha=0.8, edgecolors="white",
            )

        ax.set_xlabel("M1: Success Rate")
        ax.set_ylabel("M3: Avg Steps")
        ax.set_xlim(-0.05, 1.05)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ── Row 2: M1 vs M4 tradeoff + M6 coverage bar ─────────────────────
st.markdown("---")
col_left2, col_right2 = st.columns(2)

with col_left2:
    st.subheader("M1 vs M4: Success vs Safety")
    if (
        "M1_success_rate" in sweep_df.columns
        and "M4_avg_collisions" in sweep_df.columns
    ):
        set_style()
        fig, ax = plt.subplots(figsize=(8, 5))

        if "agents_per_target" in sweep_df.columns:
            k_vals = sorted(sweep_df["agents_per_target"].unique())
            markers = {1: "o", 2: "s", 3: "^"}
            for k in k_vals:
                sub = sweep_df[sweep_df["agents_per_target"] == k]
                ax.scatter(
                    sub["M1_success_rate"],
                    sub["M4_avg_collisions"],
                    s=90, alpha=0.8, edgecolors="white",
                    marker=markers.get(k, "o"),
                    label=f"K={k}",
                )
            ax.legend(title="Agents/Target", fontsize=9)
        else:
            ax.scatter(
                sweep_df["M1_success_rate"],
                sweep_df["M4_avg_collisions"],
                c=POLIMI_DARK_BLUE, s=80, alpha=0.8, edgecolors="white",
            )

        ax.set_xlabel("M1: Success Rate")
        ax.set_ylabel("M4: Avg Collisions")
        ax.set_xlim(-0.05, 1.05)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with col_right2:
    st.subheader("M6 Coverage Progress by Configuration")
    if "M6_coverage_progress" in sweep_df.columns:
        set_style()
        fig, ax = plt.subplots(figsize=(8, 5))

        if "n_agents" in sweep_df.columns and "n_targets" in sweep_df.columns:
            if "config" not in sweep_df.columns:
                sweep_df["config"] = (
                    "N=" + sweep_df["n_agents"].astype(str)
                    + " T=" + sweep_df["n_targets"].astype(str)
                )
            grouped = sweep_df.groupby("config")["M6_coverage_progress"]
            means = grouped.mean().sort_values(ascending=False)
            stds = grouped.std().reindex(means.index).fillna(0)

            bar_colors = [
                POLIMI_GREEN if m >= 0.8
                else POLIMI_ORANGE if m >= 0.5
                else POLIMI_RED
                for m in means
            ]
            bars = ax.bar(
                range(len(means)), means, yerr=stds,
                capsize=4, color=bar_colors, alpha=0.85,
            )
            ax.set_xticks(range(len(means)))
            ax.set_xticklabels(means.index, rotation=30, ha="right")

            for bar, m in zip(bars, means):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{m:.0%}", ha="center", va="bottom", fontsize=10,
                )
        else:
            ax.bar(
                range(len(sweep_df)), sweep_df["M6_coverage_progress"],
                color=POLIMI_GREEN, alpha=0.85,
            )

        ax.set_ylabel("M6: Coverage Progress")
        ax.set_ylim(0, 1.15)
        ax.axhline(y=1.0, color=POLIMI_DARK_BLUE, linestyle="--",
                    alpha=0.4, linewidth=1)
        ax.grid(True, alpha=0.2, axis="y")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ── Best Runs Table ──────────────────────────────────────────────────
st.markdown("---")
st.subheader("Top Runs")

display_cols = ["run_id"]
for c in ["n_agents", "n_targets", "agents_per_target",
           "M1_success_rate", "M2_avg_return", "M3_avg_steps",
           "M4_avg_collisions", "M6_coverage_progress"]:
    if c in sweep_df.columns:
        display_cols.append(c)

top_df = sweep_df[display_cols].sort_values(
    "M1_success_rate", ascending=False,
).head(10)

# Format percentages
format_dict = {}
for c in top_df.columns:
    if "rate" in c or "progress" in c:
        format_dict[c] = "{:.0%}".format
    elif c.startswith("M"):
        format_dict[c] = "{:.2f}".format

st.dataframe(
    top_df.style.format(format_dict),
    use_container_width=True,
    hide_index=True,
)
