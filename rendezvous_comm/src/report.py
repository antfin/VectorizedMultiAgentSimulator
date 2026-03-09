"""Report generation for experiment runs.

Produces human-readable report.txt files summarizing
configuration, training, and results for each run.
"""
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .config import ExperimentSpec


METRIC_DETAILS = {
    "M1_success_rate": {
        "label": "Success Rate",
        "fmt": ".1%",
        "description": (
            "Fraction of evaluation episodes where ALL targets were covered.\n"
            "      A target is 'covered' when K agents are within covering_range\n"
            "      simultaneously in the same step. With targets_respawn=False, the\n"
            "      episode ends (done=True) when every target has been covered at\n"
            "      least once. M1 = episodes_done / total_episodes."
        ),
    },
    "M1b_avg_targets_covered_per_step": {
        "label": "Avg Targets Covered/Step",
        "fmt": ".4f",
        "description": (
            "Mean number of newly covered targets per step, averaged across\n"
            "      episodes. Useful for tracking partial progress even when\n"
            "      episodes don't complete."
        ),
    },
    "M2_avg_return": {
        "label": "Avg Return",
        "fmt": ".4f",
        "description": (
            "Mean cumulative reward per episode, summed across all agents.\n"
            "      Reward = covering_rew_coeff * targets_covered_by_agent\n"
            "             + agent_collision_penalty * collisions\n"
            "             + time_penalty * steps.\n"
            "      Positive return = covering outweighs penalties."
        ),
    },
    "M3_avg_steps": {
        "label": "Avg Steps to Completion",
        "fmt": ".1f",
        "description": (
            "Mean steps until done() fires (all targets covered). If the\n"
            "      episode never completes, this equals max_steps. Lower = faster\n"
            "      coordination."
        ),
    },
    "M4_avg_collisions": {
        "label": "Collisions/Episode",
        "fmt": ".2f",
        "description": (
            "Mean agent-agent collisions per episode. A collision is counted\n"
            "      when two agents are within min_collision_distance (0.005)."
        ),
    },
    "M5_avg_tokens": {
        "label": "Tokens/Episode",
        "fmt": ".1f",
        "description": (
            "Communication tokens used per episode. Always 0 for no-comm\n"
            "      baselines (ER1). Measures bandwidth cost for comm experiments."
        ),
    },
}


def generate_run_report(
    run_dir: Path,
    run_id: str,
    spec: ExperimentSpec,
    metrics: Dict[str, float],
    elapsed_seconds: float = 0.0,
    task_overrides: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a detailed human-readable report for a single run.

    Writes report.txt to run_dir and returns the text.
    """
    now = datetime.now()
    lines = []
    sep = "=" * 76
    thin = "-" * 76

    # ── Header ──────────────────────────────────────────────────────
    lines.append(sep)
    lines.append(f"  EXPERIMENT REPORT")
    lines.append(sep)
    lines.append(f"  Experiment:    {spec.exp_id.upper()} - {spec.name}")
    lines.append(f"  Run ID:        {run_id}")
    lines.append(f"  Folder:        {run_dir.name}")
    lines.append(f"  Generated:     {now.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"  Description:")
    for line in spec.description.strip().split("\n"):
        lines.append(f"    {line.strip()}")
    lines.append("")

    # ── Task Configuration ──────────────────────────────────────────
    lines.append(thin)
    lines.append("  TASK CONFIGURATION (Discovery Scenario)")
    lines.append(thin)
    lines.append("")
    task = spec.task.to_dict()
    effective_task = dict(task)
    if task_overrides:
        effective_task.update(task_overrides)

    # Group key params prominently
    key_params = [
        ("n_agents", "N", "Number of agents"),
        ("n_targets", "T", "Number of targets to cover"),
        ("agents_per_target", "K", "Agents required per target simultaneously"),
        ("covering_range", "", "Distance threshold to 'cover' a target"),
        ("lidar_range", "", "LiDAR sensing range"),
        ("targets_respawn", "", "Targets reappear after covered (must be False for M1)"),
        ("max_steps", "", "Max steps per episode"),
    ]
    for param, short, desc in key_params:
        val = effective_task.get(param, "?")
        marker = " *" if task_overrides and param in task_overrides else ""
        short_str = f" ({short})" if short else ""
        lines.append(f"    {param}{short_str:<30} = {val}{marker:<6}  {desc}")

    lines.append("")
    lines.append("    Other parameters:")
    shown = {p for p, _, _ in key_params}
    for k, v in effective_task.items():
        if k not in shown:
            marker = " *" if task_overrides and k in task_overrides else ""
            lines.append(f"      {k:<33} = {v}{marker}")
    if task_overrides:
        lines.append("    (* = overridden by sweep)")
    lines.append("")

    # ── Training Configuration ──────────────────────────────────────
    lines.append(thin)
    lines.append("  TRAINING CONFIGURATION")
    lines.append(thin)
    lines.append("")
    train = spec.train.__dict__
    iterations = train["max_n_frames"] // train["on_policy_collected_frames_per_batch"]
    lines.append(f"    Algorithm:                 {train['algorithm'].upper()}")
    lines.append(f"    Total frames:              {train['max_n_frames']:,}")
    lines.append(f"    Training iterations:       {iterations}")
    lines.append(f"    Frames per batch:          {train['on_policy_collected_frames_per_batch']:,}")
    lines.append(f"    Parallel envs:             {train['on_policy_n_envs_per_worker']}")
    lines.append(f"    SGD epochs per batch:      {train['on_policy_n_minibatch_iters']}")
    lines.append(f"    Minibatch size:            {train['on_policy_minibatch_size']:,}")
    lines.append(f"    Learning rate:             {train['lr']}")
    lines.append(f"    Discount (gamma):          {train['gamma']}")
    lines.append(f"    Shared policy:             {train['share_policy_params']}")
    lines.append(f"    Train device:              {train['train_device']}")
    lines.append(f"    Sampling device:           {train['sampling_device']}")
    lines.append(f"    Eval interval:             every {train['evaluation_interval']:,} frames")
    lines.append(f"    Eval episodes:             {train['evaluation_episodes']}")
    lines.append("")

    # ── Training Summary ────────────────────────────────────────────
    lines.append(thin)
    lines.append("  TRAINING SUMMARY")
    lines.append(thin)
    lines.append("")
    if elapsed_seconds > 0:
        m, s = divmod(int(elapsed_seconds), 60)
        h, m = divmod(m, 60)
        lines.append(f"    Wall time:                 {h}h {m}m {s}s")
        fps = train["max_n_frames"] / elapsed_seconds
        lines.append(f"    Throughput:                {fps:,.0f} frames/sec")
        lines.append(f"    Time per iteration:        {elapsed_seconds / iterations:.1f}s")
    else:
        lines.append("    Wall time:                 (not recorded)")
    lines.append("")

    # ── Evaluation Results ──────────────────────────────────────────
    lines.append(thin)
    lines.append("  EVALUATION RESULTS")
    lines.append(thin)
    lines.append("")
    if metrics:
        n_eval = int(metrics.get("n_envs", 0))
        lines.append(f"    Evaluation episodes:       {n_eval}")
        lines.append(f"    Mode:                      deterministic (no exploration)")
        lines.append("")

        for key, value in metrics.items():
            detail = METRIC_DETAILS.get(key)
            if detail:
                formatted = f"{value:{detail['fmt']}}"
                lines.append(f"    {detail['label']:<30} {formatted:>12}")
                lines.append(f"      {detail['description']}")
                lines.append("")
    else:
        lines.append("    No metrics available (dry run or training failed)")
    lines.append("")

    # ── Output Artifacts ────────────────────────────────────────────
    lines.append(thin)
    lines.append("  OUTPUT ARTIFACTS")
    lines.append(thin)
    lines.append("")
    lines.append(f"    Run directory:   {run_dir}")
    lines.append("")

    # List with descriptions
    artifacts = [
        ("input/config.yaml", "Frozen config snapshot (all params at run time)"),
        ("logs/run.log", "Full training log with timestamps"),
        ("output/metrics.json", "Final evaluation metrics (M1-M5)"),
        ("output/policy.pt", "Trained policy weights (load with torch.load)"),
        ("output/benchmarl/", "BenchMARL raw outputs:"),
    ]
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
                lines.append(f"    {path_str:<35} {sz:<12} {desc}")
            else:
                lines.append(f"    {path_str:<35} {'(dir)':<12} {desc}")
        else:
            lines.append(f"    {path_str:<35} {'(missing)':<12} {desc}")

    # List BenchMARL subdirectory contents
    benchmarl_dir = run_dir / "output" / "benchmarl"
    if benchmarl_dir.exists():
        for sub in sorted(benchmarl_dir.iterdir()):
            if sub.is_dir():
                lines.append(f"      {sub.name}/")
                csv_dir = sub / "csv"
                if csv_dir.exists():
                    csv_files = sorted(csv_dir.glob("*.csv"))
                    lines.append(f"        csv/  ({len(csv_files)} metric files)")
                    for cf in csv_files[:5]:
                        lines.append(f"          {cf.name}")
                    if len(csv_files) > 5:
                        lines.append(f"          ... and {len(csv_files) - 5} more")
                ckpt_dir = sub / "checkpoints"
                if ckpt_dir.exists():
                    ckpts = list(ckpt_dir.iterdir())
                    lines.append(f"        checkpoints/  ({len(ckpts)} files)")
                json_files = list(sub.glob("*.json"))
                for jf in json_files:
                    sz = jf.stat().st_size
                    sz_str = f"{sz / 1000:.1f} KB" if sz > 1000 else f"{sz} B"
                    lines.append(f"        {jf.name}  ({sz_str})")

    lines.append("")

    # ── How to Reproduce ────────────────────────────────────────────
    lines.append(thin)
    lines.append("  HOW TO REPRODUCE / RELOAD")
    lines.append(thin)
    lines.append("")
    lines.append("    Reload the trained policy:")
    lines.append("      from src.runner import build_experiment")
    lines.append("      from src.storage import ExperimentStorage")
    lines.append(f"      storage = ExperimentStorage('{spec.exp_id}')")
    lines.append(f"      run = storage.get_run('{run_id}')")
    lines.append("      state_dict = run.load_policy_state_dict()")
    lines.append("      # Rebuild experiment with same config:")
    lines.append("      experiment = build_experiment(...)")
    lines.append("      experiment.policy.load_state_dict(state_dict)")
    lines.append("")
    lines.append("    View training log:")
    lines.append(f"      cat {run_dir / 'logs' / 'run.log'}")
    lines.append("")
    lines.append("    View BenchMARL training curves (CSV):")
    lines.append(f"      ls {benchmarl_dir}/*/csv/")
    lines.append("")
    lines.append(sep)
    lines.append("")

    report_text = "\n".join(lines)

    # Write to file (no date in filename — folder already has the date)
    report_path = run_dir / "report.txt"
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
    lines = []
    sep = "=" * 76
    thin = "-" * 76

    lines.append(sep)
    lines.append(f"  SWEEP REPORT: {spec.exp_id.upper()} - {spec.name}")
    lines.append(sep)
    lines.append(f"  Generated:     {now.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Total runs:    {len(all_metrics)}")
    if elapsed_seconds > 0:
        m, s = divmod(int(elapsed_seconds), 60)
        h, m = divmod(m, 60)
        lines.append(f"  Wall time:     {h}h {m}m {s}s")
    lines.append(f"  Results dir:   {results_dir}")
    lines.append("")

    if not all_metrics:
        lines.append("  No completed runs.")
        lines.append(sep)
        report_text = "\n".join(lines)
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "sweep_report.txt").write_text(report_text)
        return report_text

    # Aggregate stats
    import statistics

    metric_keys = [
        "M1_success_rate", "M2_avg_return",
        "M3_avg_steps", "M4_avg_collisions",
    ]
    lines.append(thin)
    lines.append("  AGGREGATE RESULTS (mean +/- std across all runs)")
    lines.append(thin)
    lines.append("")

    for key in metric_keys:
        values = [m[key] for m in all_metrics.values() if key in m]
        if not values:
            continue
        detail = METRIC_DETAILS.get(key, {"label": key, "fmt": ".4f"})
        label = detail["label"]
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        if "rate" in key:
            lines.append(f"    {label:<30} {mean:.1%} +/- {std:.1%}")
        else:
            lines.append(f"    {label:<30} {mean:.4f} +/- {std:.4f}")

    # Per-run table
    lines.append("")
    lines.append(thin)
    lines.append("  PER-RUN RESULTS")
    lines.append(thin)
    header = f"    {'Run ID':<45} {'M1':>8} {'M2':>10} {'M3':>8} {'M4':>8}"
    lines.append(header)
    lines.append("    " + "-" * 79)
    for run_id, metrics in sorted(all_metrics.items()):
        m1 = f"{metrics.get('M1_success_rate', 0):.1%}"
        m2 = f"{metrics.get('M2_avg_return', 0):.2f}"
        m3 = f"{metrics.get('M3_avg_steps', 0):.0f}"
        m4 = f"{metrics.get('M4_avg_collisions', 0):.2f}"
        lines.append(f"    {run_id:<45} {m1:>8} {m2:>10} {m3:>8} {m4:>8}")

    # Individual run reports
    lines.append("")
    lines.append(thin)
    lines.append("  INDIVIDUAL RUN REPORTS")
    lines.append(thin)
    lines.append("")
    lines.append("    Each run folder contains its own report.txt with full details.")
    lines.append(f"    Location: {results_dir}/YYYYMMDD_HHMM__<run_id>/report.txt")

    lines.append("")
    lines.append(sep)

    report_text = "\n".join(lines)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "sweep_report.txt").write_text(report_text)

    return report_text
