"""``multi-scenario inspect-lero <run_dir>`` — summarise a LERO run.

F9.8 CLI hook. Reads ``output/lero/final_summary.json`` and the
evolution history; prints a human-readable summary so the user
doesn't have to ``jq`` through the trace files.

Usage:
    multi-scenario inspect-lero experiments/discovery/lero/er1_lero_s0__t/
"""

import json
from pathlib import Path

import typer

from multi_scenario.cli._app import app


@app.command("inspect-lero")
def inspect_lero(
    run_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="LERO run directory (the one containing output/lero/).",
    ),
) -> None:
    """Print a human-readable summary of a completed LERO run."""
    lero_root = run_dir / "output" / "lero"
    if not lero_root.is_dir():
        typer.echo(f"✗ no LERO output under {lero_root}", err=True)
        raise typer.Exit(code=1)

    summary_path = lero_root / "final_summary.json"
    history_path = lero_root / "evolution_history.json"

    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        typer.echo(f"=== LERO run summary: {summary['exp_id']}_s{summary['seed']} ===")
        typer.echo(f"  iterations completed:    {summary['n_iterations_completed']}")
        typer.echo(f"  candidates total:        {summary['n_candidates_total']}")
        typer.echo(f"  total cost (ledger USD): {summary['total_cost_usd']:.2f}")
        typer.echo(f"  best inner-loop verdict: {summary['best_candidate_verdict']}")
        best = summary["best_candidate_metrics"]
        for key in ("M1_success_rate", "M2_avg_return", "M6_coverage_progress"):
            val = best.get(key)
            if val is not None:
                typer.echo(f"  best inner {key:24s} {val:.3f}")
        typer.echo(f"  full training succeeded: {summary['full_training_succeeded']}")
        if summary.get("fallback_chain"):
            typer.echo("  fallback chain:")
            for entry in summary["fallback_chain"]:
                typer.echo(
                    f"    rank {entry['rank']} iter {entry['iteration']}/"
                    f"cand {entry['candidate_idx']}: {entry['outcome']}"
                    + (f" — {entry['error']}" if entry.get("error") else "")
                )
    else:
        typer.echo(f"  (no final_summary.json under {lero_root})")

    if history_path.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        typer.echo(f"  history records: {len(history)}")
    else:
        typer.echo("  (no evolution_history.json)")
