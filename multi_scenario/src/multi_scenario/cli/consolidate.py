"""``multi-scenario consolidate <exp_type_dir>`` — build runs.csv + runs.json (F5.2 + F5.3)."""

from pathlib import Path

import typer

from multi_scenario.adapters.storage.runs_csv import RunsCsvWriter
from multi_scenario.adapters.storage.runs_json import RunsJsonWriter

from ._app import app


@app.command()
def consolidate(
    exp_type_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
) -> None:
    """Build ``runs.csv`` + ``runs.json`` from all DONE runs (F5.2 + F5.3)."""
    csv_path = RunsCsvWriter().consolidate(exp_type_dir)
    json_path = RunsJsonWriter().consolidate(exp_type_dir)
    typer.echo(f"OK runs.csv -> {csv_path}")
    typer.echo(f"OK runs.json -> {json_path}")
