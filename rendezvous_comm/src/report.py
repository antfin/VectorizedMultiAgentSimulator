"""Report generation for experiment runs.

Produces human-readable report.txt files summarizing
configuration, training, and results for each run.
"""
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .config import ExperimentSpec


METRIC_LABELS = {
    "M1_success_rate": ("Success Rate", "Fraction of episodes where all targets covered"),
    "M1b_avg_targets_covered_per_step": ("Avg Targets Covered/Step", "Mean targets covered per step (respawn mode)"),
    "M2_avg_return": ("Avg Return", "Mean cumulative reward per episode"),
    "M3_avg_steps": ("Avg Steps to Completion", "Mean steps until done (max_steps if not done)"),
    "M4_avg_collisions": ("Collisions/Episode", "Mean agent-agent collisions per episode"),
    "M5_avg_tokens": ("Tokens/Episode", "Communication tokens used per episode"),
}


def generate_run_report(
    run_dir: Path,
    run_id: str,
    spec: ExperimentSpec,
    metrics: Dict[str, float],
    elapsed_seconds: float = 0.0,
    task_overrides: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a human-readable report for a single run.

    Writes report.txt to run_dir and returns the text.
    """
    lines = []
    sep = "=" * 70
    lines.append(sep)
    lines.append(f"  EXPERIMENT REPORT")
    lines.append(sep)
    lines.append(f"  Experiment:  {spec.exp_id.upper()} - {spec.name}")
    lines.append(f"  Run ID:      {run_id}")
    lines.append(f"  Generated:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Description
    lines.append(f"  Description:")
    for line in spec.description.strip().split("\n"):
        lines.append(f"    {line.strip()}")
    lines.append("")

    # Configuration
    lines.append("-" * 70)
    lines.append("  CONFIGURATION")
    lines.append("-" * 70)

    lines.append("")
    lines.append("  Task Parameters:")
    task = spec.task.to_dict()
    if task_overrides:
        task.update(task_overrides)
    for k, v in task.items():
        marker = " *" if task_overrides and k in task_overrides else ""
        lines.append(f"    {k:<35} {v}{marker}")
    if task_overrides:
        lines.append("    (* = overridden by sweep)")

    lines.append("")
    lines.append("  Training Parameters:")
    train = spec.train.__dict__
    for k, v in train.items():
        val = f"{v:,}" if isinstance(v, int) and v > 1000 else str(v)
        lines.append(f"    {k:<35} {val}")

    # Training summary
    lines.append("")
    lines.append("-" * 70)
    lines.append("  TRAINING SUMMARY")
    lines.append("-" * 70)
    lines.append(f"    Total frames:    {spec.train.max_n_frames:,}")
    lines.append(f"    Device:          {spec.train.train_device}")
    if elapsed_seconds > 0:
        m, s = divmod(int(elapsed_seconds), 60)
        h, m = divmod(m, 60)
        lines.append(f"    Wall time:       {h}h {m}m {s}s")
    lines.append(f"    Algorithm:       {spec.train.algorithm}")

    # Results
    lines.append("")
    lines.append("-" * 70)
    lines.append("  EVALUATION RESULTS")
    lines.append("-" * 70)
    if metrics:
        for key, value in metrics.items():
            label_info = METRIC_LABELS.get(key)
            if label_info:
                label, desc = label_info
                if "rate" in key:
                    formatted = f"{value:.1%}"
                elif isinstance(value, float):
                    formatted = f"{value:.4f}"
                else:
                    formatted = str(value)
                lines.append(f"    {label:<30} {formatted:>12}    {desc}")
            elif key != "n_envs":
                lines.append(f"    {key:<30} {value}")
        lines.append(f"    {'Eval Episodes':<30} {metrics.get('n_envs', 'N/A'):>12}")
    else:
        lines.append("    No metrics available (dry run or training failed)")

    # Artifacts
    lines.append("")
    lines.append("-" * 70)
    lines.append("  OUTPUT ARTIFACTS")
    lines.append("-" * 70)
    _list_artifacts(lines, run_dir)

    lines.append("")
    lines.append(sep)
    lines.append("")

    report_text = "\n".join(lines)

    # Write to file
    report_path = run_dir / "report.txt"
    report_path.write_text(report_text)

    return report_text


def generate_sweep_report(
    spec: ExperimentSpec,
    all_metrics: Dict[str, Dict[str, float]],
    elapsed_seconds: float = 0.0,
    results_dir: Optional[Path] = None,
) -> str:
    """Generate a summary report for a full sweep.

    Writes sweep_report.txt to the experiment results directory.
    """
    if results_dir is None:
        results_dir = spec.results_dir

    lines = []
    sep = "=" * 70
    lines.append(sep)
    lines.append(f"  SWEEP REPORT: {spec.exp_id.upper()} - {spec.name}")
    lines.append(sep)
    lines.append(f"  Generated:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Total runs:  {len(all_metrics)}")
    if elapsed_seconds > 0:
        m, s = divmod(int(elapsed_seconds), 60)
        h, m = divmod(m, 60)
        lines.append(f"  Wall time:   {h}h {m}m {s}s")
    lines.append("")

    if not all_metrics:
        lines.append("  No completed runs.")
        lines.append(sep)
        report_text = "\n".join(lines)
        (results_dir / "sweep_report.txt").write_text(report_text)
        return report_text

    # Aggregate stats
    import statistics

    metric_keys = ["M1_success_rate", "M2_avg_return", "M3_avg_steps", "M4_avg_collisions"]
    lines.append("-" * 70)
    lines.append("  AGGREGATE RESULTS (mean +/- std across all runs)")
    lines.append("-" * 70)

    for key in metric_keys:
        values = [m[key] for m in all_metrics.values() if key in m]
        if not values:
            continue
        label_info = METRIC_LABELS.get(key, (key, ""))
        label = label_info[0]
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        if "rate" in key:
            lines.append(f"    {label:<30} {mean:.1%} +/- {std:.1%}")
        else:
            lines.append(f"    {label:<30} {mean:.4f} +/- {std:.4f}")

    # Per-run table
    lines.append("")
    lines.append("-" * 70)
    lines.append("  PER-RUN RESULTS")
    lines.append("-" * 70)
    header = f"    {'Run ID':<50} {'M1':>8} {'M2':>10} {'M3':>8} {'M4':>8}"
    lines.append(header)
    lines.append("    " + "-" * 84)
    for run_id, metrics in sorted(all_metrics.items()):
        m1 = f"{metrics.get('M1_success_rate', 0):.1%}"
        m2 = f"{metrics.get('M2_avg_return', 0):.2f}"
        m3 = f"{metrics.get('M3_avg_steps', 0):.0f}"
        m4 = f"{metrics.get('M4_avg_collisions', 0):.2f}"
        lines.append(f"    {run_id:<50} {m1:>8} {m2:>10} {m3:>8} {m4:>8}")

    lines.append("")
    lines.append(sep)

    report_text = "\n".join(lines)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "sweep_report.txt").write_text(report_text)

    return report_text


def _list_artifacts(lines, run_dir: Path, indent: str = "    "):
    """List files in the run directory tree."""
    if not run_dir.exists():
        lines.append(f"{indent}(directory not found)")
        return
    for root, dirs, files in os.walk(run_dir):
        level = Path(root).relative_to(run_dir)
        prefix = indent + "  " * len(level.parts)
        if str(level) != ".":
            lines.append(f"{prefix}{level.name}/")
        for f in sorted(files):
            fpath = Path(root) / f
            size = fpath.stat().st_size
            if size > 1_000_000:
                size_str = f"{size / 1_000_000:.1f} MB"
            elif size > 1000:
                size_str = f"{size / 1000:.1f} KB"
            else:
                size_str = f"{size} B"
            lines.append(f"{prefix}  {f} ({size_str})")
