"""Formatted display utilities for experiment notebooks.

Renders configs, metrics, and summaries as readable tables.
Falls back to plain text when IPython is not available.
"""
import math
from typing import Any, Dict, List, Optional

from .config import ExperimentSpec


METRIC_INFO = {
    "M1_success_rate": ("M1", "Success Rate", "Episodes where all targets covered", ".1%"),
    "M2_avg_return": ("M2", "Avg Return", "Mean cumulative reward per episode", ".2f"),
    "M3_avg_steps": ("M3", "Avg Steps", "Mean steps to completion (max_steps if not done)", ".1f"),
    "M4_avg_collisions": ("M4", "Collisions/Episode", "Mean agent-agent collisions", ".2f"),
    "M5_avg_tokens": ("M5", "Tokens/Episode", "Communication tokens used per episode", ".1f"),
    "M6_coverage_progress": ("M6", "Coverage Progress", "Fraction of targets covered (partial credit)", ".1%"),
    "M7_sample_efficiency": ("M7", "Sample Efficiency", "Frames to 80% of final reward", ",.0f"),
    "M8_agent_utilization": ("M8", "Agent Utilization", "CV of per-agent covering (0=balanced)", ".3f"),
    "M9_spatial_spread": ("M9", "Spatial Spread", "Mean pairwise agent distance (higher=exploring)", ".3f"),
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

def display_config(spec: ExperimentSpec):
    """Display full experiment configuration as a formatted table."""
    task = spec.task.to_dict()
    train = spec.train.__dict__
    sweep = spec.sweep.__dict__
    total_runs = sum(1 for _ in spec.iter_runs())

    if _in_notebook():
        html = _build_config_html(spec, task, train, sweep, total_runs)
        _html(html)
    else:
        _print_config_text(spec, task, train, sweep, total_runs)


def _build_config_html(spec, task, train, sweep, total_runs):
    rows = ""

    # Header
    rows += f"""
    <tr style="background:#2c3e50;color:white">
      <td colspan="3" style="padding:10px;font-size:16px">
        <b>{spec.exp_id.upper()}: {spec.name}</b>
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


def _print_config_text(spec, task, train, sweep, total_runs):
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  {spec.exp_id.upper()}: {spec.name}  |  Runs: {total_runs}")
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
        for key in ["M1_success_rate", "M2_avg_return", "M3_avg_steps", "M4_avg_collisions", "M6_coverage_progress", "M8_agent_utilization", "M9_spatial_spread"]:
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


# ── Environment dimensions ────────────────────────────────────────

def display_environment_info(spec: ExperimentSpec):
    """Display computed environment dimensions and physical intuition."""
    task = spec.task
    field_w = 2 * task.x_semidim
    field_h = 2 * task.y_semidim
    diag = math.sqrt(field_w**2 + field_h**2)
    agent_speed = 0.4  # approximate terminal velocity with drag=0.25
    crossing_steps = int(diag / agent_speed) + 1
    lidar_frac = task.lidar_range / diag

    rows = [
        ("Field size", f"{field_w:.1f} x {field_h:.1f}", f"Coords from -{task.x_semidim} to +{task.x_semidim}"),
        ("Field diagonal", f"~{diag:.2f} units", "Max distance between any two points"),
        ("Agent speed", f"~{agent_speed} units/step", "Terminal velocity (drag=0.25, force=1.0)"),
        ("Steps to cross", f"~{crossing_steps} steps", "Diagonal / speed"),
        ("LiDAR range", f"{task.lidar_range}", f"{lidar_frac:.0%} of diagonal -- very local"),
        ("Covering range", f"{task.covering_range}", f"K={task.agents_per_target} agents must be this close"),
        ("Max steps", f"{task.max_steps}", f"~{task.max_steps // crossing_steps}x crossing time"),
    ]

    if _in_notebook():
        html_rows = ""
        for label, value, note in rows:
            html_rows += (
                f'<tr><td style="padding:4px 12px;font-weight:bold">{label}</td>'
                f'<td style="padding:4px 12px;font-family:monospace">{value}</td>'
                f'<td style="padding:4px 12px;color:#666">{note}</td></tr>'
            )
        _html(
            '<table style="border-collapse:collapse;width:100%;border:1px solid #3498db;'
            'margin:10px 0;border-radius:4px;overflow:hidden">'
            '<tr style="background:#3498db;color:white">'
            '<td colspan="3" style="padding:8px"><b>Environment Dimensions</b></td></tr>'
            f'{html_rows}</table>'
        )
    else:
        print("\n  ENVIRONMENT DIMENSIONS")
        print("  " + "-" * 50)
        for label, value, note in rows:
            print(f"    {label:<20} {value:<20} {note}")


# ── Scrollable output ─────────────────────────────────────────────

def scrollable(text: str, height: int = 300, title: str = ""):
    """Display long text in a scrollable container."""
    escaped = text.replace("&", "&amp;").replace("<", "&lt;")
    header = ""
    if title:
        header = (
            f'<div style="padding:6px 10px;background:#34495e;color:white;'
            f'font-size:12px;font-weight:bold;border-radius:4px 4px 0 0">{title}</div>'
        )
    if _in_notebook():
        _html(
            f'{header}'
            f'<div style="max-height:{height}px;overflow-y:auto;'
            f'border:1px solid #ddd;padding:10px;font-family:monospace;'
            f'font-size:11px;background:#1e1e1e;color:#d4d4d4;white-space:pre-wrap;'
            f'line-height:1.5;border-radius:0 0 4px 4px">{escaped}</div>'
        )
    else:
        if title:
            print(f"\n  {title}")
            print("  " + "=" * 60)
        print(text)


# ── Metric cards ──────────────────────────────────────────────────

def display_metric_cards(metrics: Dict[str, float], title: str = ""):
    """Display key metrics as large colored KPI cards."""
    card_specs = [
        ("M1_success_rate", "Success Rate", ".0%", _m1_color),
        ("M6_coverage_progress", "Coverage", ".0%", lambda v: "#27ae60" if v > 0.7 else "#e67e22" if v > 0.3 else "#e74c3c"),
        ("M3_avg_steps", "Avg Steps", ".0f", lambda v: "#27ae60" if v < 100 else "#e67e22" if v < 150 else "#e74c3c"),
        ("M9_spatial_spread", "Spatial Spread", ".2f", lambda _: "#3498db"),
        ("M8_agent_utilization", "Agent Util (CV)", ".2f", lambda v: "#27ae60" if v < 0.5 else "#e67e22"),
        ("M4_avg_collisions", "Collisions", ".1f", lambda v: "#27ae60" if v < 5 else "#e67e22"),
    ]

    if not _in_notebook():
        if title:
            print(f"\n  {title}")
        for key, label, fmt, _ in card_specs:
            if key in metrics:
                print(f"    {label:<25} {metrics[key]:{fmt}}")
        return

    cards_html = ""
    for key, label, fmt, color_fn in card_specs:
        if key not in metrics:
            continue
        val = metrics[key]
        color = color_fn(val)
        cards_html += (
            f'<div style="display:inline-block;width:145px;margin:6px;padding:14px 10px;'
            f'border-radius:8px;background:{color};color:white;text-align:center;'
            f'vertical-align:top">'
            f'<div style="font-size:24px;font-weight:bold">{val:{fmt}}</div>'
            f'<div style="font-size:11px;margin-top:4px;opacity:0.9">{label}</div></div>'
        )

    header = f'<div style="font-size:14px;font-weight:bold;margin-bottom:4px">{title}</div>' if title else ""
    _html(f'{header}<div>{cards_html}</div>')


def _m1_color(v):
    if v > 0.8:
        return "#27ae60"
    if v > 0.4:
        return "#e67e22"
    return "#e74c3c"


# ── Verdict box ───────────────────────────────────────────────────

def display_verdict(success_rate: float, avg_return: float):
    """Display a color-coded verdict box based on success rate."""
    if success_rate > 0.50:
        color, verdict = "#e67e22", "TASK TOO EASY"
        msg = "Floor is >50%. Increase difficulty (more targets, smaller LiDAR, higher K) before running ER2+."
    elif success_rate >= 0.20:
        color, verdict = "#27ae60", "FLOOR ESTABLISHED"
        msg = "Proceed to ER2+. Any communication method must beat this floor to justify its complexity."
    else:
        color, verdict = "#e74c3c", "TASK TOO HARD"
        msg = "Floor is <20%. Consider easier configs or more training frames, or try the complete config."

    if _in_notebook():
        _html(
            f'<div style="padding:16px;border-left:6px solid {color};'
            f'background:#f9f9f9;margin:10px 0;border-radius:4px">'
            f'<div style="font-size:18px;font-weight:bold;color:{color}">{verdict}</div>'
            f'<div style="margin-top:8px">{msg}</div>'
            f'<div style="margin-top:10px;font-size:13px;color:#666">'
            f'Success Rate: {success_rate:.0%} &nbsp;|&nbsp; Avg Return: {avg_return:.2f}</div></div>'
        )
    else:
        print(f"\n  {verdict}")
        print(f"  {msg}")
        print(f"  Success: {success_rate:.0%}  |  Return: {avg_return:.2f}")


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


# ── Config selector ──────────────────────────────────────────────

_FRESHNESS_BADGE = {
    "valid": ('<span style="background:#27ae60;color:white;padding:2px 8px;'
              'border-radius:3px;font-size:11px">VALID</span>'),
    "config_changed": ('<span style="background:#e74c3c;color:white;padding:2px 8px;'
                       'border-radius:3px;font-size:11px">CONFIG CHANGED</span>'),
    "code_changed": ('<span style="background:#e67e22;color:white;padding:2px 8px;'
                     'border-radius:3px;font-size:11px">CODE CHANGED</span>'),
    "both_changed": ('<span style="background:#e74c3c;color:white;padding:2px 8px;'
                     'border-radius:3px;font-size:11px">STALE</span>'),
    "no_provenance": ('<span style="background:#95a5a6;color:white;padding:2px 8px;'
                      'border-radius:3px;font-size:11px">NO PROVENANCE</span>'),
    "new": ('<span style="background:#3498db;color:white;padding:2px 8px;'
            'border-radius:3px;font-size:11px">NEW</span>'),
}

_FRESHNESS_TEXT = {
    "valid": "VALID",
    "config_changed": "CONFIG CHANGED",
    "code_changed": "CODE CHANGED",
    "both_changed": "STALE",
    "no_provenance": "NO PROVENANCE",
    "new": "NEW",
}


def display_config_selector(exp_id: str):
    """Display available configs for an experiment with freshness status.

    Shows a table of all YAML configs in configs/<exp_id>/ with:
    - Config filename and key parameters
    - Number of runs defined
    - Completion status (done/total)
    - Freshness check against current config+code
    """
    from .config import find_configs
    from .provenance import check_freshness, Freshness
    from .storage import ExperimentStorage

    configs = find_configs(exp_id)
    if not configs:
        print(f"No configs found in configs/{exp_id}/")
        return

    storage = ExperimentStorage(exp_id)
    completed_runs = set(storage.list_runs())

    rows_data = []
    for i, (yaml_path, spec) in enumerate(configs):
        all_runs = list(spec.iter_runs())
        total = len(all_runs)
        done = sum(1 for rid, _, _, _ in all_runs if rid in completed_runs)

        # Freshness: check all completed runs
        if done == 0:
            freshness_key = "new"
        else:
            # Check freshness of first completed run
            worst = Freshness.VALID
            for rid, _, _, _ in all_runs:
                if rid in completed_runs:
                    run_dir = storage._find_run_dir(rid)
                    if run_dir:
                        f = check_freshness(run_dir, yaml_path)
                        if f.value != "valid":
                            worst = f
                            break
            freshness_key = worst.value

        algos = ", ".join(a.upper() for a in spec.sweep.algorithms)
        agents = spec.sweep.n_agents
        agents_str = str(agents[0]) if len(agents) == 1 else f"{agents[0]}-{agents[-1]}"
        lidar = spec.sweep.lidar_range
        lidar_str = str(lidar[0]) if len(lidar) == 1 else f"{lidar[0]}-{lidar[-1]}"

        rows_data.append({
            "idx": i + 1,
            "filename": yaml_path.name,
            "path": yaml_path,
            "algos": algos,
            "agents": agents_str,
            "lidar": lidar_str,
            "total": total,
            "done": done,
            "freshness": freshness_key,
        })

    if _in_notebook():
        _display_selector_html(exp_id, rows_data)
    else:
        _display_selector_text(exp_id, rows_data)


def _display_selector_html(exp_id, rows_data):
    table_rows = ""
    for r in rows_data:
        done_color = "#27ae60" if r["done"] == r["total"] else "#e67e22" if r["done"] > 0 else "#999"
        badge = _FRESHNESS_BADGE[r["freshness"]]
        table_rows += f"""
        <tr>
          <td style="padding:6px 10px;text-align:center">{r["idx"]}</td>
          <td style="padding:6px 10px;font-family:monospace;font-weight:bold">{r["filename"]}</td>
          <td style="padding:6px 10px">{r["algos"]}</td>
          <td style="padding:6px 10px;text-align:center">{r["agents"]}</td>
          <td style="padding:6px 10px;text-align:center">{r["lidar"]}</td>
          <td style="padding:6px 10px;text-align:center">{r["total"]}</td>
          <td style="padding:6px 10px;text-align:center;color:{done_color};font-weight:bold">
            {r["done"]}/{r["total"]}
          </td>
          <td style="padding:6px 10px;text-align:center">{badge}</td>
        </tr>"""

    _html(f"""
    <div style="margin:10px 0">
      <div style="padding:8px 12px;background:#2c3e50;color:white;border-radius:4px 4px 0 0;
                  font-size:14px;font-weight:bold">
        Available Configs: {exp_id.upper()}
      </div>
      <table style="border-collapse:collapse;width:100%;border:1px solid #ddd">
        <tr style="background:#ecf0f1">
          <th style="padding:6px 10px">#</th>
          <th style="padding:6px 10px;text-align:left">Config</th>
          <th style="padding:6px 10px;text-align:left">Algo</th>
          <th style="padding:6px 10px">N agents</th>
          <th style="padding:6px 10px">LiDAR</th>
          <th style="padding:6px 10px">Runs</th>
          <th style="padding:6px 10px">Complete</th>
          <th style="padding:6px 10px">Freshness</th>
        </tr>
        {table_rows}
      </table>
    </div>""")


def _display_selector_text(exp_id, rows_data):
    print(f"\n  Available Configs: {exp_id.upper()}")
    print(f"  {'=' * 80}")
    print(f"  {'#':<4} {'Config':<40} {'Algo':<12} {'N':<6} {'L':<8} "
          f"{'Runs':<6} {'Done':<8} {'Status'}")
    print(f"  {'-' * 80}")
    for r in rows_data:
        status = _FRESHNESS_TEXT[r["freshness"]]
        print(f"  {r['idx']:<4} {r['filename']:<40} {r['algos']:<12} "
              f"{r['agents']:<6} {r['lidar']:<8} {r['total']:<6} "
              f"{r['done']}/{r['total']:<6} {status}")
    print()


def select_config(exp_id: str, force_retrain: bool = False):
    """Interactive config selector. Returns (config_path, force_retrain).

    Uses ipywidgets if available, falls back to input().
    """
    from pathlib import Path
    from .config import find_configs

    configs = find_configs(exp_id)
    if not configs:
        raise FileNotFoundError(f"No configs in configs/{exp_id}/")

    display_config_selector(exp_id)

    try:
        import ipywidgets as widgets
        from IPython.display import display

        options = {f"{c[0].name}": c[0] for c in configs}
        dropdown = widgets.Dropdown(
            options=options,
            description="Config:",
            style={"description_width": "60px"},
        )
        retrain_cb = widgets.Checkbox(
            value=force_retrain,
            description="Force retrain (ignore existing results)",
            style={"description_width": "initial"},
        )
        box = widgets.VBox([dropdown, retrain_cb])
        display(box)

        class _Result:
            path = configs[0][0]
            retrain = force_retrain

        def _on_change(change):
            _Result.path = change["new"]

        def _on_retrain(change):
            _Result.retrain = change["new"]

        dropdown.observe(_on_change, names="value")
        retrain_cb.observe(_on_retrain, names="value")

        return _Result

    except ImportError:
        # Fallback: numbered input
        if len(configs) == 1:
            print(f"  → Using: {configs[0][0].name}")
            return configs[0][0], force_retrain

        choice = input(f"  Select config [1-{len(configs)}]: ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(configs):
                return configs[idx][0], force_retrain
        except ValueError:
            pass
        print(f"  → Default: {configs[0][0].name}")
        return configs[0][0], force_retrain
