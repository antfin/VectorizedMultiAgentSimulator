"""Report generation for experiment runs.

Produces human-readable report.md files summarizing
configuration, training, and results for each run.
"""
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import ExperimentSpec


METRIC_DETAILS = {
    "M1_success_rate": {
        "label": "M1: Success Rate",
        "fmt": ".1%",
        "description": (
            "Fraction of evaluation episodes where ALL targets were covered. "
            "A target is 'covered' when K agents are within covering_range "
            "simultaneously. Episode ends when every target has been covered."
        ),
    },
    "M2_avg_return": {
        "label": "M2: Avg Return",
        "fmt": ".4f",
        "description": (
            "Mean cumulative reward per episode (covering + collision penalty + time penalty). "
            "Positive = covering outweighs penalties."
        ),
    },
    "M3_avg_steps": {
        "label": "M3: Avg Steps to Completion",
        "fmt": ".1f",
        "description": (
            "Mean steps until all targets covered. "
            "Equals max_steps if episode never completes. Lower = faster."
        ),
    },
    "M4_avg_collisions": {
        "label": "M4: Collisions/Episode",
        "fmt": ".2f",
        "description": "Mean agent-agent collisions per episode.",
    },
    "M5_avg_tokens": {
        "label": "M5: Tokens/Episode",
        "fmt": ".1f",
        "description": (
            "Communication tokens per episode. Always 0 for no-comm baselines (ER1)."
        ),
    },
    "M6_coverage_progress": {
        "label": "M6: Coverage Progress",
        "fmt": ".1%",
        "description": (
            "Fraction of targets covered by episode end (partial credit). "
            "100% = all targets covered in every episode."
        ),
    },
    "M7_sample_efficiency": {
        "label": "M7: Sample Efficiency",
        "fmt": ",.0f",
        "description": (
            "Training frames to reach 80% of final eval reward. "
            "Lower = faster learning."
        ),
    },
    "M8_agent_utilization": {
        "label": "M8: Agent Utilization",
        "fmt": ".3f",
        "description": (
            "Coefficient of variation of per-agent covering counts. "
            "0 = perfectly balanced workload."
        ),
    },
    "M9_spatial_spread": {
        "label": "M9: Spatial Spread",
        "fmt": ".3f",
        "description": (
            "Mean pairwise agent distance. "
            "Higher = exploring, lower = clumping. Field diagonal ≈ 2.83."
        ),
    },
}


def _find_images(output_dir: Path) -> List[Path]:
    """Find all .png images in the output directory."""
    if not output_dir.exists():
        return []
    return sorted(output_dir.glob("*.png"))


def generate_run_report(
    run_dir: Path,
    run_id: str,
    spec: ExperimentSpec,
    metrics: Dict[str, float],
    elapsed_seconds: float = 0.0,
    task_overrides: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a detailed markdown report for a single run.

    Writes report.md to run_dir and returns the text.
    """
    now = datetime.now()
    L = []  # lines accumulator

    # ── Header ──
    L.append(f"# Experiment Report: {spec.exp_id.upper()} — {spec.name}")
    L.append("")
    L.append(f"| | |")
    L.append(f"|---|---|")
    L.append(f"| **Run ID** | `{run_id}` |")
    L.append(f"| **Folder** | `{run_dir.name}` |")
    L.append(f"| **Generated** | {now.strftime('%Y-%m-%d %H:%M:%S')} |")
    L.append("")
    L.append(f"> {spec.description.strip()}")
    L.append("")

    # ── Task Configuration ──
    L.append("## Task Configuration (Discovery Scenario)")
    L.append("")
    task = spec.task.to_dict()
    effective_task = dict(task)
    if task_overrides:
        effective_task.update(task_overrides)

    key_params = [
        ("n_agents", "N", "Number of agents"),
        ("n_targets", "T", "Number of targets to cover"),
        ("agents_per_target", "K", "Agents required per target simultaneously"),
        ("covering_range", "", "Distance threshold to 'cover' a target"),
        ("lidar_range", "", "LiDAR sensing range"),
        ("targets_respawn", "", "Must be False for M1/M3"),
        ("max_steps", "", "Max steps per episode"),
    ]

    L.append("| Parameter | Value | Description |")
    L.append("|-----------|-------|-------------|")
    for param, short, desc in key_params:
        val = effective_task.get(param, "?")
        override = " ⚙" if task_overrides and param in task_overrides else ""
        name = f"`{param}`" + (f" ({short})" if short else "")
        L.append(f"| {name} | **{val}**{override} | {desc} |")
    L.append("")

    other_params = {k: v for k, v in effective_task.items()
                    if k not in {p for p, _, _ in key_params}}
    if other_params:
        L.append("<details><summary>Other task parameters</summary>")
        L.append("")
        L.append("| Parameter | Value |")
        L.append("|-----------|-------|")
        for k, v in other_params.items():
            override = " ⚙" if task_overrides and k in task_overrides else ""
            L.append(f"| `{k}` | {v}{override} |")
        L.append("")
        if task_overrides:
            L.append("⚙ = overridden by sweep")
        L.append("</details>")
        L.append("")

    # ── Training Configuration ──
    L.append("## Training Configuration")
    L.append("")
    train = spec.train.__dict__
    iterations = train["max_n_frames"] // train["on_policy_collected_frames_per_batch"]

    L.append("| Setting | Value |")
    L.append("|---------|-------|")
    L.append(f"| Algorithm | **{train['algorithm'].upper()}** |")
    L.append(f"| Total frames | {train['max_n_frames']:,} |")
    L.append(f"| Iterations | {iterations} |")
    L.append(f"| Frames/batch | {train['on_policy_collected_frames_per_batch']:,} |")
    L.append(f"| Parallel envs | {train['on_policy_n_envs_per_worker']} |")
    L.append(f"| SGD epochs/batch | {train['on_policy_n_minibatch_iters']} |")
    L.append(f"| Minibatch size | {train['on_policy_minibatch_size']:,} |")
    L.append(f"| Learning rate | {train['lr']} |")
    L.append(f"| Gamma | {train['gamma']} |")
    L.append(f"| Shared policy | {train['share_policy_params']} |")
    L.append(f"| Device | {train['train_device']} |")
    L.append(f"| Eval interval | every {train['evaluation_interval']:,} frames |")
    L.append(f"| Eval episodes | {train['evaluation_episodes']} |")
    L.append("")

    # ── Training Summary ──
    L.append("## Training Summary")
    L.append("")
    if elapsed_seconds > 0:
        m, s = divmod(int(elapsed_seconds), 60)
        h, m = divmod(m, 60)
        fps = train["max_n_frames"] / elapsed_seconds
        L.append(f"- **Wall time:** {h}h {m}m {s}s")
        L.append(f"- **Throughput:** {fps:,.0f} frames/sec")
        L.append(f"- **Time/iteration:** {elapsed_seconds / iterations:.1f}s")
    else:
        L.append("- Wall time: (not recorded)")
    L.append("")

    # ── Training Curves (images) ──
    output_dir = run_dir / "output"
    images = _find_images(output_dir)
    training_images = [img for img in images if "training" in img.name.lower()
                       or "dashboard" in img.name.lower()]
    if training_images:
        L.append("### Training Curves")
        L.append("")
        for img in training_images:
            rel = img.relative_to(run_dir)
            L.append(f"![{img.stem}]({rel})")
            L.append("")

    # ── Evaluation Results ──
    L.append("## Evaluation Results")
    L.append("")
    if metrics:
        n_eval = int(metrics.get("n_envs", 0))
        L.append(f"**{n_eval} episodes** — deterministic (no exploration)")
        L.append("")
        L.append("| Metric | Value | Description |")
        L.append("|--------|-------|-------------|")
        for key, value in metrics.items():
            detail = METRIC_DETAILS.get(key)
            if detail:
                formatted = f"{value:{detail['fmt']}}"
                L.append(f"| **{detail['label']}** | {formatted} | {detail['description']} |")
        L.append("")
    else:
        L.append("No metrics available (dry run or training failed).")
        L.append("")

    # ── Comparison plots (images) ──
    comparison_images = [img for img in images if img not in training_images]
    if comparison_images:
        L.append("### Comparison")
        L.append("")
        for img in comparison_images:
            rel = img.relative_to(run_dir)
            L.append(f"![{img.stem}]({rel})")
            L.append("")

    # ── Output Artifacts ──
    L.append("## Output Artifacts")
    L.append("")
    L.append(f"📁 `{run_dir}`")
    L.append("")
    artifacts = [
        ("input/config.yaml", "Frozen config snapshot"),
        ("logs/run.log", "Training log with timestamps"),
        ("output/metrics.json", "Final evaluation metrics (M1–M9)"),
        ("output/policy.pt", "Trained policy weights"),
        ("output/benchmarl/", "BenchMARL raw outputs"),
    ]
    L.append("| Path | Size | Description |")
    L.append("|------|------|-------------|")
    for path_str, desc in artifacts:
        full = run_dir / path_str
        if full.exists():
            if full.is_file():
                size = full.stat().st_size
                if size > 1_000_000:
                    sz = f"{size / 1_000_000:.1f} MB"
                elif size > 1000:
                    sz = f"{size / 1000:.1f} KB"
                else:
                    sz = f"{size} B"
            else:
                sz = "(dir)"
            L.append(f"| `{path_str}` | {sz} | {desc} |")
        else:
            L.append(f"| `{path_str}` | — | {desc} |")
    L.append("")

    # ── How to Reproduce ──
    L.append("## How to Reproduce / Reload")
    L.append("")
    L.append("```python")
    L.append("from src.runner import build_experiment")
    L.append("from src.storage import ExperimentStorage")
    L.append(f"storage = ExperimentStorage('{spec.exp_id}')")
    L.append(f"run = storage.get_run('{run_id}')")
    L.append("state_dict = run.load_policy_state_dict()")
    L.append("# Rebuild experiment with same config:")
    L.append("experiment = build_experiment(...)")
    L.append("experiment.policy.load_state_dict(state_dict)")
    L.append("```")
    L.append("")

    report_text = "\n".join(L)

    # Write to file
    report_path = run_dir / "report.md"
    report_path.write_text(report_text)

    return report_text


def generate_sweep_report(
    spec: ExperimentSpec,
    all_metrics: Dict[str, Dict[str, float]],
    elapsed_seconds: float = 0.0,
    results_dir: Optional[Path] = None,
) -> str:
    """Generate a summary report for a full sweep."""
    if results_dir is None:
        results_dir = spec.results_dir

    now = datetime.now()
    L = []

    L.append(f"# Sweep Report: {spec.exp_id.upper()} — {spec.name}")
    L.append("")
    L.append(f"- **Generated:** {now.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"- **Total runs:** {len(all_metrics)}")
    if elapsed_seconds > 0:
        m, s = divmod(int(elapsed_seconds), 60)
        h, m = divmod(m, 60)
        L.append(f"- **Wall time:** {h}h {m}m {s}s")
    L.append(f"- **Results dir:** `{results_dir}`")
    L.append("")

    if not all_metrics:
        L.append("No completed runs.")
        report_text = "\n".join(L)
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "sweep_report.md").write_text(report_text)
        return report_text

    # Aggregate stats
    import statistics

    metric_keys = [
        "M1_success_rate", "M2_avg_return", "M3_avg_steps",
        "M4_avg_collisions", "M6_coverage_progress",
        "M8_agent_utilization", "M9_spatial_spread",
    ]

    L.append("## Aggregate Results")
    L.append("")
    L.append("| Metric | Mean | Std |")
    L.append("|--------|------|-----|")
    for key in metric_keys:
        values = [m[key] for m in all_metrics.values() if key in m]
        if not values:
            continue
        detail = METRIC_DETAILS.get(
            key, {"label": key, "fmt": ".4f"}
        )
        label = detail["label"]  # already includes M-ID prefix
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        if "rate" in key or "progress" in key:
            L.append(f"| {label} | {mean:.1%} | ±{std:.1%} |")
        else:
            L.append(f"| {label} | {mean:.4f} | ±{std:.4f} |")
    L.append("")

    # Sweep images
    images = sorted(results_dir.glob("*.png"))
    if images:
        L.append("## Visualizations")
        L.append("")
        for img in images:
            L.append(f"![{img.stem}]({img.name})")
            L.append("")

    # Per-run table
    L.append("## Per-Run Results")
    L.append("")
    L.append(
        "| Run ID | M1: Success | M2: Return "
        "| M3: Steps | M4: Collisions "
        "| M6: Coverage | M8: Util | M9: Spread |"
    )
    L.append("|--------|------------|--------"
             "|----------|---------------"
             "|------------|---------|-----------|")
    for run_id, metrics in sorted(all_metrics.items()):
        m1 = f"{metrics.get('M1_success_rate', 0):.1%}"
        m2 = f"{metrics.get('M2_avg_return', 0):.2f}"
        m3 = f"{metrics.get('M3_avg_steps', 0):.0f}"
        m4 = f"{metrics.get('M4_avg_collisions', 0):.2f}"
        m6 = f"{metrics.get('M6_coverage_progress', 0):.1%}"
        m8 = f"{metrics.get('M8_agent_utilization', 0):.3f}"
        m9 = f"{metrics.get('M9_spatial_spread', 0):.3f}"
        L.append(
            f"| `{run_id}` | {m1} | {m2} "
            f"| {m3} | {m4} "
            f"| {m6} | {m8} | {m9} |"
        )
    L.append("")

    L.append("---")
    L.append(f"Each run folder contains its own `report.md` with full details.")
    L.append("")

    report_text = "\n".join(L)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "sweep_report.md").write_text(report_text)

    return report_text
