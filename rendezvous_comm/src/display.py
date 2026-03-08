"""Formatted display utilities for experiment notebooks.

Renders configs, metrics, and summaries as readable tables.
Falls back to plain text when IPython is not available.
"""
from typing import Any, Dict, List, Optional

from .config import ExperimentSpec


METRIC_INFO = {
    "M1_success_rate": ("M1", "Success Rate", "Episodes where all targets covered", ".1%"),
    "M1b_avg_targets_covered_per_step": ("M1b", "Avg Targets Covered/Step", "Mean targets covered per step (respawn mode)", ".4f"),
    "M2_avg_return": ("M2", "Avg Return", "Mean cumulative reward per episode", ".2f"),
    "M3_avg_steps": ("M3", "Avg Steps", "Mean steps to completion (max_steps if not done)", ".1f"),
    "M4_avg_collisions": ("M4", "Collisions/Episode", "Mean agent-agent collisions", ".2f"),
    "M5_avg_tokens": ("M5", "Tokens/Episode", "Communication tokens used per episode", ".1f"),
    "n_envs": ("", "Eval Episodes", "Number of evaluation episodes", ".0f"),
}


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


def _html(content: str):
    """Display HTML in notebook or print plain text."""
    if _in_notebook():
        from IPython.display import display, HTML
        display(HTML(content))
    else:
        print(content)


# ── Config display ─────────────────────────────────────────────────

def display_config(spec: ExperimentSpec, profile: str = "complete"):
    """Display full experiment configuration as a formatted table."""
    task = spec.task.to_dict()
    train = spec.train.__dict__
    sweep = spec.sweep.__dict__
    total_runs = sum(1 for _ in spec.iter_runs())

    if _in_notebook():
        html = _build_config_html(spec, task, train, sweep, total_runs, profile)
        _html(html)
    else:
        _print_config_text(spec, task, train, sweep, total_runs, profile)


def _build_config_html(spec, task, train, sweep, total_runs, profile):
    rows = ""

    # Header
    rows += f"""
    <tr style="background:#2c3e50;color:white">
      <td colspan="3" style="padding:10px;font-size:16px">
        <b>{spec.exp_id.upper()}: {spec.name}</b>
        &nbsp;&mdash;&nbsp; Profile: <code>{profile}</code>
        &nbsp;&mdash;&nbsp; Total runs: <b>{total_runs}</b>
      </td>
    </tr>
    <tr><td colspan="3" style="padding:8px;font-style:italic;background:#ecf0f1">
      {spec.description}
    </td></tr>
    """

    # Task params
    rows += '<tr style="background:#3498db;color:white"><td colspan="3" style="padding:6px"><b>Task Parameters (Discovery Scenario)</b></td></tr>'
    for k, v in task.items():
        rows += f'<tr><td style="padding:4px 12px;font-family:monospace">{k}</td><td style="padding:4px 12px"><b>{v}</b></td><td style="padding:4px 12px;color:#666">{_task_param_desc(k)}</td></tr>'

    # Train params
    rows += '<tr style="background:#e67e22;color:white"><td colspan="3" style="padding:6px"><b>Training Parameters</b></td></tr>'
    for k, v in train.items():
        val = f"{v:,}" if isinstance(v, int) and v > 1000 else str(v)
        rows += f'<tr><td style="padding:4px 12px;font-family:monospace">{k}</td><td style="padding:4px 12px"><b>{val}</b></td><td style="padding:4px 12px;color:#666">{_train_param_desc(k)}</td></tr>'

    # Sweep params
    rows += '<tr style="background:#27ae60;color:white"><td colspan="3" style="padding:6px"><b>Sweep Dimensions</b></td></tr>'
    for k, v in sweep.items():
        rows += f'<tr><td style="padding:4px 12px;font-family:monospace">{k}</td><td style="padding:4px 12px"><b>{v}</b></td><td style="padding:4px 12px;color:#666">{len(v)} values</td></tr>'

    return f'<table style="border-collapse:collapse;width:100%;border:1px solid #ddd;margin:10px 0">{rows}</table>'


def _print_config_text(spec, task, train, sweep, total_runs, profile):
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  {spec.exp_id.upper()}: {spec.name}  |  Profile: {profile}  |  Runs: {total_runs}")
    print(f"{sep}")
    print(f"  {spec.description}\n")

    print("  TASK PARAMETERS")
    print("  " + "-" * 40)
    for k, v in task.items():
        print(f"    {k:<35} {v}")

    print("\n  TRAINING PARAMETERS")
    print("  " + "-" * 40)
    for k, v in train.items():
        val = f"{v:,}" if isinstance(v, int) and v > 1000 else str(v)
        print(f"    {k:<35} {val}")

    print("\n  SWEEP DIMENSIONS")
    print("  " + "-" * 40)
    for k, v in sweep.items():
        print(f"    {k:<35} {v}")
    print(sep)


def _task_param_desc(key: str) -> str:
    descs = {
        "n_agents": "Number of agents",
        "n_targets": "Number of targets to cover",
        "agents_per_target": "K: agents needed per target",
        "lidar_range": "LiDAR sensing range",
        "covering_range": "Distance to 'cover' a target",
        "use_agent_lidar": "Whether agents sense other agents",
        "targets_respawn": "Targets reappear after covered",
        "shared_reward": "All agents share same reward",
        "agent_collision_penalty": "Penalty for agent collisions",
        "covering_rew_coeff": "Reward coefficient for covering",
        "time_penalty": "Per-step time penalty",
        "max_steps": "Max steps per episode",
        "n_lidar_rays_entities": "LiDAR rays for targets",
        "n_lidar_rays_agents": "LiDAR rays for other agents",
        "x_semidim": "World half-width",
        "y_semidim": "World half-height",
        "min_dist_between_entities": "Min spawn distance",
    }
    return descs.get(key, "")


def _train_param_desc(key: str) -> str:
    descs = {
        "algorithm": "RL algorithm",
        "max_n_frames": "Total training frames",
        "gamma": "Discount factor",
        "on_policy_collected_frames_per_batch": "Frames per collection batch",
        "on_policy_n_envs_per_worker": "Parallel envs per worker",
        "on_policy_n_minibatch_iters": "SGD epochs per batch",
        "on_policy_minibatch_size": "Minibatch size",
        "lr": "Learning rate",
        "share_policy_params": "Shared policy across agents",
        "evaluation_interval": "Eval every N frames",
        "evaluation_episodes": "Episodes per evaluation",
        "train_device": "Device for training",
        "sampling_device": "Device for env sampling",
    }
    return descs.get(key, "")


# ── Metrics display ────────────────────────────────────────────────

def display_metrics(metrics: Dict[str, float], title: str = "Evaluation Metrics"):
    """Display metrics as a formatted table with descriptions."""
    if _in_notebook():
        _display_metrics_html(metrics, title)
    else:
        _display_metrics_text(metrics, title)


def _display_metrics_html(metrics: Dict[str, float], title: str):
    rows = f'<tr style="background:#2c3e50;color:white"><td colspan="4" style="padding:8px"><b>{title}</b></td></tr>'
    rows += '<tr style="background:#ecf0f1"><td style="padding:4px 8px"><b>ID</b></td><td style="padding:4px 8px"><b>Metric</b></td><td style="padding:4px 8px"><b>Value</b></td><td style="padding:4px 8px"><b>Description</b></td></tr>'

    for key, value in metrics.items():
        info = METRIC_INFO.get(key)
        if info:
            mid, name, desc, fmt = info
            formatted = f"{value:{fmt}}"
        else:
            mid, name, desc = "", key, ""
            formatted = f"{value:.4f}" if isinstance(value, float) else str(value)

        rows += f'<tr><td style="padding:4px 8px;font-family:monospace">{mid}</td><td style="padding:4px 8px">{name}</td><td style="padding:4px 8px;text-align:right"><b>{formatted}</b></td><td style="padding:4px 8px;color:#666">{desc}</td></tr>'

    _html(f'<table style="border-collapse:collapse;width:100%;border:1px solid #ddd;margin:10px 0">{rows}</table>')


def _display_metrics_text(metrics: Dict[str, float], title: str):
    print(f"\n  {title}")
    print("  " + "-" * 60)
    for key, value in metrics.items():
        info = METRIC_INFO.get(key)
        if info:
            mid, name, _, fmt = info
            formatted = f"{value:{fmt}}"
            label = f"{mid} {name}" if mid else name
        else:
            label = key
            formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
        print(f"    {label:<40} {formatted:>10}")
    print()


# ── Sweep summary ──────────────────────────────────────────────────

def display_sweep_summary(all_metrics: Dict[str, Dict[str, float]]):
    """Display a summary table of all sweep run results."""
    if not all_metrics:
        print("No completed runs to display.")
        return

    import pandas as pd

    rows = []
    for run_id, metrics in all_metrics.items():
        row = {"run_id": run_id}
        for key in ["M1_success_rate", "M2_avg_return", "M3_avg_steps", "M4_avg_collisions"]:
            info = METRIC_INFO.get(key)
            if info and key in metrics:
                _, name, _, fmt = info
                row[name] = f"{metrics[key]:{fmt}}"
        rows.append(row)

    df = pd.DataFrame(rows).set_index("run_id")

    if _in_notebook():
        from IPython.display import display
        display(df)
    else:
        print(df.to_string())


# ── Run status ─────────────────────────────────────────────────────

def display_run_status(spec: ExperimentSpec):
    """Show which runs are complete vs pending."""
    from .storage import ExperimentStorage

    storage = ExperimentStorage(spec.exp_id)
    completed = set(storage.list_runs())
    all_runs = [run_id for run_id, _, _, _ in spec.iter_runs()]
    total = len(all_runs)
    done = len(completed)
    pending = total - done

    if _in_notebook():
        color = "#27ae60" if done == total else "#e67e22"
        _html(f'<div style="padding:10px;border:1px solid #ddd;margin:10px 0;border-radius:4px">'
              f'<b>Run Status:</b> <span style="color:{color}">{done}/{total} complete</span>'
              f' &mdash; {pending} pending</div>')
    else:
        print(f"\n  Run Status: {done}/{total} complete, {pending} pending\n")
