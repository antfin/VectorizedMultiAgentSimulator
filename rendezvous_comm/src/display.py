"""Formatted display utilities for experiment notebooks.

Renders configs, metrics, and summaries as readable tables.
Falls back to plain text when IPython is not available.
All HTML uses dark-first styling for notebook compatibility.
"""
import base64
import math
from pathlib import Path
from typing import Dict

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

# ── Dark-first color tokens ──────────────────────────────────────
_BG_SURFACE = "#1e1e2e"
_BG_SURFACE2 = "#252535"
_BG_HEADER = "#2c3e50"
_BG_HEADER_BLUE = "#3498db"
_BG_HEADER_ORANGE = "#e67e22"
_BG_HEADER_GREEN = "#27ae60"
_FG_PRIMARY = "#cdd6f4"
_FG_SECONDARY = "#a6adc8"
_FG_MUTED = "#7f849c"
_BORDER = "#45475a"
_GREEN = "#27ae60"
_ORANGE = "#e67e22"
_RED = "#e74c3c"
_BLUE = "#3498db"


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


def _table_style():
    return (
        f"border-collapse:collapse;width:100%;border:1px solid {_BORDER};"
        f"margin:10px 0;background:{_BG_SURFACE};color:{_FG_PRIMARY}"
    )


def _header_row(text, colspan=3, bg=_BG_HEADER):
    return (
        f'<tr style="background:{bg};color:white">'
        f'<td colspan="{colspan}" style="padding:8px"><b>{text}</b></td></tr>'
    )


def _subheader_row(text, colspan=3):
    return (
        f'<tr style="background:{_BG_SURFACE2};color:{_FG_SECONDARY}">'
        f'<td colspan="{colspan}" style="padding:6px"><b>{text}</b></td></tr>'
    )


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
    rows += (
        f'<tr style="background:{_BG_HEADER};color:white">'
        f'<td colspan="3" style="padding:10px;font-size:16px">'
        f'<b>{spec.exp_id.upper()}: {spec.name}</b>'
        f' &nbsp;&mdash;&nbsp; Total runs: <b>{total_runs}</b></td></tr>'
    )
    rows += (
        f'<tr><td colspan="3" style="padding:8px;font-style:italic;'
        f'background:{_BG_SURFACE2};color:{_FG_SECONDARY}">'
        f'{spec.description}</td></tr>'
    )

    def _param_row(k, v, desc):
        val = f"{v:,}" if isinstance(v, int) and v > 1000 else str(v)
        return (
            f'<tr style="border-bottom:1px solid {_BORDER}">'
            f'<td style="padding:4px 12px;font-family:monospace;color:{_FG_PRIMARY}">{k}</td>'
            f'<td style="padding:4px 12px;color:{_FG_PRIMARY}"><b>{val}</b></td>'
            f'<td style="padding:4px 12px;color:{_FG_MUTED}">{desc}</td></tr>'
        )

    # Parameters that are swept — skip from task, show only in sweep
    # Sweep keys that overlap with task — show only in sweep section
    sweep_task_keys = set(sweep.keys()) - {"seeds", "algorithms"}

    rows += _subheader_row("Task Parameters (Discovery Scenario)")
    for k, v in task.items():
        if k in sweep_task_keys:
            continue  # shown in sweep section
        rows += _param_row(k, v, _task_param_desc(k))

    rows += (
        f'<tr style="background:{_BG_HEADER_ORANGE};color:white">'
        f'<td colspan="3" style="padding:6px"><b>Training Parameters</b></td></tr>'
    )
    for k, v in train.items():
        rows += _param_row(k, v, _train_param_desc(k))

    rows += (
        f'<tr style="background:{_BG_HEADER_GREEN};color:white">'
        f'<td colspan="3" style="padding:6px">'
        f'<b>Sweep Dimensions</b></td></tr>'
    )
    for k, v in sweep.items():
        desc = _task_param_desc(k) or _sweep_param_desc(k)
        n_vals = len(v) if isinstance(v, list) else 1
        if n_vals > 1:
            desc += f" ({n_vals} values)"
        rows += _param_row(k, v, desc)

    return f'<table style="{_table_style()}">{rows}</table>'


def _print_config_text(spec, task, train, sweep, total_runs):
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  {spec.exp_id.upper()}: {spec.name}  |  Runs: {total_runs}")
    print(f"{sep}")
    print(f"  {spec.description}\n")

    sweep_task_keys = set(sweep.keys()) - {"seeds", "algorithms"}

    print("  TASK PARAMETERS")
    print("  " + "-" * 40)
    for k, v in task.items():
        if k in sweep_task_keys:
            continue
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


def _sweep_param_desc(key: str) -> str:
    descs = {
        "seeds": "Random seeds for reproducibility",
        "algorithms": "RL algorithms to compare",
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
    rows = _header_row(title, colspan=4)
    rows += (
        f'<tr style="background:{_BG_SURFACE2};color:{_FG_SECONDARY}">'
        f'<td style="padding:4px 8px"><b>ID</b></td>'
        f'<td style="padding:4px 8px"><b>Metric</b></td>'
        f'<td style="padding:4px 8px"><b>Value</b></td>'
        f'<td style="padding:4px 8px"><b>Description</b></td></tr>'
    )

    for key, value in metrics.items():
        info = METRIC_INFO.get(key)
        if info:
            mid, name, desc, fmt = info
            formatted = f"{value:{fmt}}"
        else:
            mid, name, desc = "", key, ""
            formatted = f"{value:.4f}" if isinstance(value, float) else str(value)

        rows += (
            f'<tr style="border-bottom:1px solid {_BORDER}">'
            f'<td style="padding:4px 8px;font-family:monospace;color:{_FG_MUTED}">{mid}</td>'
            f'<td style="padding:4px 8px;color:{_FG_PRIMARY}">{name}</td>'
            f'<td style="padding:4px 8px;text-align:right;color:{_FG_PRIMARY}"><b>{formatted}</b></td>'
            f'<td style="padding:4px 8px;color:{_FG_MUTED}">{desc}</td></tr>'
        )

    _html(f'<table style="{_table_style()}">{rows}</table>')


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
        for key in [
            "M1_success_rate", "M2_avg_return", "M3_avg_steps",
            "M4_avg_collisions", "M6_coverage_progress",
            "M8_agent_utilization", "M9_spatial_spread",
        ]:
            info = METRIC_INFO.get(key)
            if info and key in metrics:
                mid, name, _, fmt = info
                col_name = f"{mid}: {name}" if mid else name
                row[col_name] = f"{metrics[key]:{fmt}}"
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
    agent_speed = 0.4
    crossing_steps = int(diag / agent_speed) + 1
    lidar_frac = task.lidar_range / diag

    info_rows = [
        ("Field size", f"{field_w:.1f} x {field_h:.1f}",
         f"Coords from -{task.x_semidim} to +{task.x_semidim}"),
        ("Field diagonal", f"~{diag:.2f} units",
         "Max distance between any two points"),
        ("Agent speed", f"~{agent_speed} units/step",
         "Terminal velocity (drag=0.25, force=1.0)"),
        ("Steps to cross", f"~{crossing_steps} steps",
         "Diagonal / speed"),
        ("LiDAR range", f"{task.lidar_range}",
         f"{lidar_frac:.0%} of diagonal -- very local"),
        ("Covering range", f"{task.covering_range}",
         f"K={task.agents_per_target} agents must be this close"),
        ("Max steps", f"{task.max_steps}",
         f"~{task.max_steps // crossing_steps}x crossing time"),
    ]

    if _in_notebook():
        html_rows = ""
        for label, value, note in info_rows:
            html_rows += (
                f'<tr style="border-bottom:1px solid {_BORDER}">'
                f'<td style="padding:4px 12px;font-weight:bold;'
                f'color:{_FG_PRIMARY}">{label}</td>'
                f'<td style="padding:4px 12px;font-family:monospace;'
                f'color:{_FG_PRIMARY}">{value}</td>'
                f'<td style="padding:4px 12px;color:{_FG_MUTED}">'
                f'{note}</td></tr>'
            )
        _html(
            f'<table style="{_table_style()};border-radius:4px;'
            f'overflow:hidden">'
            f'<tr style="background:{_BG_HEADER_BLUE};color:white">'
            f'<td colspan="3" style="padding:8px">'
            f'<b>Environment Dimensions</b></td></tr>'
            f'{html_rows}</table>'
        )
    else:
        print("\n  ENVIRONMENT DIMENSIONS")
        print("  " + "-" * 50)
        for label, value, note in info_rows:
            print(f"    {label:<20} {value:<20} {note}")


# ── Scrollable output ─────────────────────────────────────────────

def scrollable(text: str, height: int = 300, title: str = ""):
    """Display long text in a scrollable container."""
    escaped = text.replace("&", "&amp;").replace("<", "&lt;")
    header = ""
    if title:
        header = (
            f'<div style="padding:6px 10px;background:{_BG_HEADER};'
            f'color:white;font-size:12px;font-weight:bold;'
            f'border-radius:4px 4px 0 0">{title}</div>'
        )
    if _in_notebook():
        _html(
            f'{header}'
            f'<div style="max-height:{height}px;overflow-y:auto;'
            f'border:1px solid {_BORDER};padding:10px;'
            f'font-family:monospace;font-size:11px;'
            f'background:{_BG_SURFACE};color:{_FG_PRIMARY};'
            f'white-space:pre-wrap;line-height:1.5;'
            f'border-radius:0 0 4px 4px">{escaped}</div>'
        )
    else:
        if title:
            print(f"\n  {title}")
            print("  " + "=" * 60)
        print(text)


def scrollable_md(md_text: str, height: int = 400, title: str = ""):
    """Display markdown report using IPython.display.Markdown.

    Uses native notebook markdown rendering which VS Code supports.
    """
    if _in_notebook():
        from IPython.display import display, Markdown, HTML

        if title:
            display(HTML(
                f'<div style="padding:6px 10px;background:{_BG_HEADER};'
                f'color:white;font-size:12px;font-weight:bold;'
                f'border-radius:4px">{title}</div>'
            ))
        display(Markdown(md_text))
    else:
        if title:
            print(f"\n  {title}")
            print("  " + "=" * 60)
        print(md_text)


# ── Metric cards ──────────────────────────────────────────────────

def display_metric_cards(
    metrics: Dict[str, float], title: str = ""
):
    """Display key metrics as large colored KPI cards."""
    card_specs = [
        ("M1_success_rate", "M1: Success Rate", ".0%", _m1_color),
        ("M6_coverage_progress", "M6: Coverage", ".0%",
         lambda v: _GREEN if v > 0.7 else _ORANGE if v > 0.3 else _RED),
        ("M3_avg_steps", "M3: Avg Steps", ".0f",
         lambda v: _GREEN if v < 100 else _ORANGE if v < 150 else _RED),
        ("M9_spatial_spread", "M9: Spatial Spread", ".2f",
         lambda _: _BLUE),
        ("M8_agent_utilization", "M8: Agent Util (CV)", ".2f",
         lambda v: _GREEN if v < 0.5 else _ORANGE),
        ("M4_avg_collisions", "M4: Collisions", ".1f",
         lambda v: _GREEN if v < 5 else _ORANGE),
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
            f'<div style="display:inline-block;width:145px;margin:6px;'
            f'padding:14px 10px;border-radius:8px;background:{color};'
            f'color:white;text-align:center;vertical-align:top">'
            f'<div style="font-size:24px;font-weight:bold">'
            f'{val:{fmt}}</div>'
            f'<div style="font-size:11px;margin-top:4px;opacity:0.9">'
            f'{label}</div></div>'
        )

    header = (
        f'<div style="font-size:14px;font-weight:bold;margin-bottom:4px;'
        f'color:{_FG_PRIMARY}">{title}</div>' if title else ""
    )
    _html(f'{header}<div>{cards_html}</div>')


def _m1_color(v):
    if v > 0.8:
        return _GREEN
    if v > 0.4:
        return _ORANGE
    return _RED


# ── Verdict box ───────────────────────────────────────────────────

def display_verdict(success_rate: float, avg_return: float):
    """Display a color-coded verdict box based on success rate."""
    if success_rate > 0.50:
        color, verdict = _ORANGE, "TASK TOO EASY"
        msg = ("Floor is >50%. Increase difficulty (more targets, "
               "smaller LiDAR, higher K) before running ER2+.")
    elif success_rate >= 0.20:
        color, verdict = _GREEN, "FLOOR ESTABLISHED"
        msg = ("Proceed to ER2+. Any communication method must beat "
               "this floor to justify its complexity.")
    else:
        color, verdict = _RED, "TASK TOO HARD"
        msg = ("Floor is <20%. Consider easier configs or more "
               "training frames, or try the complete config.")

    if _in_notebook():
        _html(
            f'<div style="padding:16px;border-left:6px solid {color};'
            f'background:{_BG_SURFACE};margin:10px 0;border-radius:4px">'
            f'<div style="font-size:18px;font-weight:bold;color:{color}">'
            f'{verdict}</div>'
            f'<div style="margin-top:8px;color:{_FG_PRIMARY}">{msg}</div>'
            f'<div style="margin-top:10px;font-size:13px;color:{_FG_MUTED}">'
            f'M1 Success Rate: {success_rate:.0%} &nbsp;|&nbsp; '
            f'M2 Avg Return: {avg_return:.2f}</div></div>'
        )
    else:
        print(f"\n  {verdict}")
        print(f"  {msg}")
        print(f"  M1 Success: {success_rate:.0%}  |  M2 Return: {avg_return:.2f}")


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
        color = _GREEN if done == total else _ORANGE
        _html(
            f'<div style="padding:10px;border:1px solid {_BORDER};'
            f'margin:10px 0;border-radius:4px;background:{_BG_SURFACE};'
            f'color:{_FG_PRIMARY}">'
            f'<b>Run Status:</b> '
            f'<span style="color:{color}">{done}/{total} complete</span>'
            f' &mdash; {pending} pending</div>'
        )
    else:
        print(
            f"\n  Run Status: {done}/{total} complete, "
            f"{pending} pending\n"
        )


# ── Config selector ──────────────────────────────────────────────

_FRESHNESS_BADGE = {
    "valid": (
        f'<span style="background:{_GREEN};color:white;padding:2px 8px;'
        'border-radius:3px;font-size:11px">VALID</span>'
    ),
    "config_changed": (
        f'<span style="background:{_RED};color:white;padding:2px 8px;'
        'border-radius:3px;font-size:11px">CONFIG CHANGED</span>'
    ),
    "code_changed": (
        f'<span style="background:{_ORANGE};color:white;padding:2px 8px;'
        'border-radius:3px;font-size:11px">CODE CHANGED</span>'
    ),
    "both_changed": (
        f'<span style="background:{_RED};color:white;padding:2px 8px;'
        'border-radius:3px;font-size:11px">STALE</span>'
    ),
    "no_provenance": (
        '<span style="background:#95a5a6;color:white;padding:2px 8px;'
        'border-radius:3px;font-size:11px">NO PROVENANCE</span>'
    ),
    "new": (
        f'<span style="background:{_BLUE};color:white;padding:2px 8px;'
        'border-radius:3px;font-size:11px">NEW</span>'
    ),
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
    """Display available configs with freshness status."""
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
        done = sum(
            1 for rid, _, _, _ in all_runs if rid in completed_runs
        )

        if done == 0:
            freshness_key = "new"
        else:
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
        agents_str = (
            str(agents[0]) if len(agents) == 1
            else f"{agents[0]}-{agents[-1]}"
        )
        lidar = spec.sweep.lidar_range
        lidar_str = (
            str(lidar[0]) if len(lidar) == 1
            else f"{lidar[0]}-{lidar[-1]}"
        )

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
        done_color = (
            _GREEN if r["done"] == r["total"]
            else _ORANGE if r["done"] > 0 else _FG_MUTED
        )
        badge = _FRESHNESS_BADGE[r["freshness"]]
        table_rows += (
            f'<tr style="border-bottom:1px solid {_BORDER}">'
            f'<td style="padding:6px 10px;text-align:center;'
            f'color:{_FG_PRIMARY}">{r["idx"]}</td>'
            f'<td style="padding:6px 10px;text-align:left;font-family:monospace;'
            f'font-weight:bold;color:{_FG_PRIMARY}">'
            f'{r["filename"]}</td>'
            f'<td style="padding:6px 10px;text-align:left;color:{_FG_PRIMARY}">'
            f'{r["algos"]}</td>'
            f'<td style="padding:6px 10px;text-align:center;'
            f'color:{_FG_PRIMARY}">{r["agents"]}</td>'
            f'<td style="padding:6px 10px;text-align:center;'
            f'color:{_FG_PRIMARY}">{r["lidar"]}</td>'
            f'<td style="padding:6px 10px;text-align:center;'
            f'color:{_FG_PRIMARY}">{r["total"]}</td>'
            f'<td style="padding:6px 10px;text-align:center;'
            f'color:{done_color};font-weight:bold">'
            f'{r["done"]}/{r["total"]}</td>'
            f'<td style="padding:6px 10px;text-align:center">'
            f'{badge}</td></tr>'
        )

    _html(f"""
    <div style="margin:10px 0">
      <div style="padding:8px 12px;background:{_BG_HEADER};color:white;
                  border-radius:4px 4px 0 0;font-size:14px;
                  font-weight:bold">
        Available Configs: {exp_id.upper()}
      </div>
      <table style="{_table_style()}">
        <tr style="background:{_BG_SURFACE2};color:{_FG_SECONDARY}">
          <th style="padding:6px 10px;text-align:center">#</th>
          <th style="padding:6px 10px;text-align:left">Config</th>
          <th style="padding:6px 10px;text-align:left">Algo</th>
          <th style="padding:6px 10px;text-align:center">N agents</th>
          <th style="padding:6px 10px;text-align:center">LiDAR</th>
          <th style="padding:6px 10px;text-align:center">Runs</th>
          <th style="padding:6px 10px;text-align:center">Complete</th>
          <th style="padding:6px 10px;text-align:center">Freshness</th>
        </tr>
        {table_rows}
      </table>
    </div>""")


def _display_selector_text(exp_id, rows_data):
    print(f"\n  Available Configs: {exp_id.upper()}")
    print(f"  {'=' * 80}")
    print(
        f"  {'#':<4} {'Config':<40} {'Algo':<12} {'N':<6} "
        f"{'L':<8} {'Runs':<6} {'Done':<8} {'Status'}"
    )
    print(f"  {'-' * 80}")
    for r in rows_data:
        status = _FRESHNESS_TEXT[r["freshness"]]
        print(
            f"  {r['idx']:<4} {r['filename']:<40} {r['algos']:<12} "
            f"{r['agents']:<6} {r['lidar']:<8} {r['total']:<6} "
            f"{r['done']}/{r['total']:<6} {status}"
        )
    print()


def select_config(exp_id: str, force_retrain: bool = False):
    """Interactive config selector with ipywidgets dropdown.

    Returns a result object with .path and .retrain attributes
    that update live when the user changes the widget.
    """
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
            layout=widgets.Layout(width="400px"),
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
        if len(configs) == 1:
            print(f"  -> Using: {configs[0][0].name}")
            return configs[0][0], force_retrain

        choice = input(
            f"  Select config [1-{len(configs)}]: "
        ).strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(configs):
                return configs[idx][0], force_retrain
        except ValueError:
            pass
        print(f"  -> Default: {configs[0][0].name}")
        return configs[0][0], force_retrain


# ── Baseline comparison (chart + table side-by-side) ─────────────

def display_baseline_comparison(
    heuristic_metrics: Dict[str, float],
    random_metrics: Dict[str, float],
    fig=None,
):
    """Display baseline comparison as side-by-side chart + table.

    Args:
        heuristic_metrics: metrics from heuristic policy
        random_metrics: metrics from random policy
        fig: matplotlib figure (from plot_baseline_grouped_bars)
    """
    if not _in_notebook():
        display_metrics(heuristic_metrics, "Heuristic Baseline")
        display_metrics(random_metrics, "Random Baseline")
        return

    import matplotlib.pyplot as plt
    from io import BytesIO

    img_html = ""
    if fig is not None:
        buf = BytesIO()
        fig.savefig(
            buf, format="png", dpi=120, bbox_inches="tight",
            facecolor="white", edgecolor="none",
        )
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        img_html = (
            f'<img src="data:image/png;base64,{b64}" '
            f'style="max-width:100%;border-radius:4px">'
        )
        plt.close(fig)

    keys = [
        "M1_success_rate", "M2_avg_return", "M3_avg_steps",
        "M4_avg_collisions", "M6_coverage_progress", "M9_spatial_spread",
    ]
    table_rows = ""
    for k in keys:
        info = METRIC_INFO.get(k)
        if not info:
            continue
        mid, name, _, fmt = info
        h_val = heuristic_metrics.get(k, 0)
        r_val = random_metrics.get(k, 0)
        table_rows += (
            f'<tr style="border-bottom:1px solid {_BORDER}">'
            f'<td style="padding:4px 8px;color:{_FG_MUTED};'
            f'font-family:monospace">{mid}</td>'
            f'<td style="padding:4px 8px;color:{_FG_PRIMARY}">'
            f'{name}</td>'
            f'<td style="padding:4px 8px;text-align:right;'
            f'color:{_GREEN}"><b>{h_val:{fmt}}</b></td>'
            f'<td style="padding:4px 8px;text-align:right;'
            f'color:{_RED}"><b>{r_val:{fmt}}</b></td></tr>'
        )

    table_html = (
        f'<table style="border-collapse:collapse;width:100%;'
        f'background:{_BG_SURFACE};color:{_FG_PRIMARY}">'
        f'<tr style="background:{_BG_SURFACE2}">'
        f'<th style="padding:6px 8px;text-align:left;'
        f'color:{_FG_SECONDARY}">ID</th>'
        f'<th style="padding:6px 8px;text-align:left;'
        f'color:{_FG_SECONDARY}">Metric</th>'
        f'<th style="padding:6px 8px;text-align:right;'
        f'color:{_GREEN}">Heuristic</th>'
        f'<th style="padding:6px 8px;text-align:right;'
        f'color:{_RED}">Random</th>'
        f'</tr>{table_rows}</table>'
    )

    _html(f"""
    <div style="margin:10px 0">
      <div>{img_html}</div>
      <div style="margin-top:12px">{table_html}</div>
    </div>""")


# ── Results dashboard (KPI cards + consolidated chart) ────────────

def display_results_dashboard(
    trained_metrics: Dict[str, float],
    comparison_fig=None,
    run_id: str = "",
):
    """Display results: KPI cards on top, consolidated chart below.

    Args:
        trained_metrics: metrics from trained policy
        comparison_fig: matplotlib figure (from plot_results_comparison)
        run_id: run identifier for title
    """
    display_metric_cards(
        trained_metrics, title=f"Trained Policy — {run_id}"
    )

    if comparison_fig is not None and _in_notebook():
        import matplotlib.pyplot as plt
        from io import BytesIO

        buf = BytesIO()
        comparison_fig.savefig(
            buf, format="png", dpi=120, bbox_inches="tight",
            facecolor="white", edgecolor="none",
        )
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        _html(
            f'<div style="margin:10px 0">'
            f'<img src="data:image/png;base64,{b64}" '
            f'style="max-width:100%;border-radius:4px"></div>'
        )
        plt.close(comparison_fig)


# ── Training videos (first + last side-by-side) ──────────────────

def display_training_videos(run_storage):
    """Embed first and last eval videos side-by-side as HTML5 video.

    Args:
        run_storage: RunStorage instance
    """
    video_dir = None
    for d in run_storage.output_dir.rglob("videos"):
        if d.is_dir():
            video_dir = d
            break

    if not video_dir:
        print("No training videos found.")
        return

    videos = sorted(video_dir.glob("eval_video_*.mp4"))
    if len(videos) < 2:
        print(f"Only {len(videos)} video(s) found, need at least 2.")
        return

    first_video = videos[0]
    last_video = videos[-1]

    if not _in_notebook():
        print(f"First video: {first_video}")
        print(f"Last video:  {last_video}")
        return

    def _video_tag(path: Path, label: str) -> str:
        data = path.read_bytes()
        b64 = base64.b64encode(data).decode()
        return (
            f'<div style="flex:1;min-width:300px;text-align:center">'
            f'<div style="font-weight:bold;margin-bottom:6px;'
            f'color:{_FG_PRIMARY}">{label}</div>'
            f'<video controls loop autoplay muted '
            f'style="width:100%;border-radius:4px;'
            f'border:1px solid {_BORDER}">'
            f'<source src="data:video/mp4;base64,{b64}" '
            f'type="video/mp4"></video>'
            f'<div style="font-size:11px;color:{_FG_MUTED};'
            f'margin-top:4px">{path.name}</div></div>'
        )

    first_iter = first_video.stem.replace("eval_video_", "iter ")
    last_iter = last_video.stem.replace("eval_video_", "iter ")

    _html(f"""
    <div style="margin:10px 0">
      <div style="padding:8px 12px;background:{_BG_HEADER};color:white;
                  border-radius:4px 4px 0 0;font-size:14px;
                  font-weight:bold">
        Training Progress: First vs Last Evaluation
      </div>
      <div style="display:flex;gap:16px;padding:12px;
                  background:{_BG_SURFACE};border:1px solid {_BORDER};
                  border-radius:0 0 4px 4px">
        {_video_tag(first_video, f"Early ({first_iter})")}
        {_video_tag(last_video, f"Final ({last_iter})")}
      </div>
    </div>""")


# ── Artifact tree ─────────────────────────────────────────────────

def display_artifact_tree(run_storage):
    """Display output artifacts as a styled HTML tree.

    Args:
        run_storage: RunStorage instance
    """
    base = run_storage.run_dir

    if not _in_notebook():
        for f in sorted(base.rglob("*")):
            if f.is_file():
                size = f.stat().st_size
                sz = (
                    f"{size / 1024:.1f} KB" if size > 1024
                    else f"{size} B"
                )
                print(f"  {f.relative_to(base)}  ({sz})")
        return

    entries = []
    for f in sorted(base.rglob("*")):
        if f.is_file():
            size = f.stat().st_size
            if size > 1024 * 1024:
                sz = f"{size / (1024 * 1024):.1f} MB"
            elif size > 1024:
                sz = f"{size / 1024:.1f} KB"
            else:
                sz = f"{size} B"
            rel = f.relative_to(base)
            entries.append((str(rel), sz, f.suffix))

    _icons = {
        ".pt": "&#x1F9E0;",
        ".json": "&#x1F4CB;",
        ".yaml": "&#x2699;",
        ".md": "&#x1F4DD;",
        ".png": "&#x1F5BC;",
        ".mp4": "&#x1F3AC;",
        ".csv": "&#x1F4CA;",
        ".log": "&#x1F4C3;",
        ".pkl": "&#x1F4E6;",
        ".txt": "&#x1F4C4;",
    }

    items = ""
    prev_dir = ""
    for rel_path, sz, ext in entries:
        parts = rel_path.split("/")
        filename = parts[-1]
        dirname = "/".join(parts[:-1]) if len(parts) > 1 else ""

        if dirname != prev_dir:
            icon = "&#x1F4C1;"
            items += (
                f'<div style="margin-top:8px;padding:2px 0;'
                f'color:{_FG_SECONDARY};font-weight:bold">'
                f'{icon} {dirname}/</div>'
            )
            prev_dir = dirname

        file_icon = _icons.get(ext, "&#x1F4C4;")
        indent = "24px" if dirname else "0"
        items += (
            f'<div style="padding:2px 0 2px {indent};'
            f'font-family:monospace;font-size:12px">'
            f'{file_icon} <span style="color:{_FG_PRIMARY}">'
            f'{filename}</span>'
            f' <span style="color:{_FG_MUTED};font-size:11px">'
            f'({sz})</span></div>'
        )

    _html(f"""
    <div style="margin:10px 0">
      <div style="padding:8px 12px;background:{_BG_HEADER};color:white;
                  border-radius:4px 4px 0 0;font-size:14px;
                  font-weight:bold">
        Output Artifacts
      </div>
      <div style="padding:10px 14px;background:{_BG_SURFACE};
                  border:1px solid {_BORDER};border-radius:0 0 4px 4px;
                  max-height:400px;overflow-y:auto">
        <div style="color:{_FG_MUTED};font-size:11px;
                    margin-bottom:8px">{base.name}/</div>
        {items}
      </div>
    </div>""")
