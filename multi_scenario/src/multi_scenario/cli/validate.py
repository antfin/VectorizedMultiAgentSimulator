"""``multi-scenario validate <yaml>`` — schema check (F5.1).

Pre-flight check before submitting OVH jobs or launching long local sweeps.
Errors are formatted as one ``<dotted.field.path>: <message>`` line per issue.
"""

from pathlib import Path

import typer
from pydantic import ValidationError

from multi_scenario.domain.models import ExperimentConfig

from ._app import app


@app.command()
def validate(
    yaml_path: Path = typer.Argument(..., exists=True, readable=True),
) -> None:
    """Validate a YAML against the ``ExperimentConfig`` schema; exit 1 on any error."""
    try:
        ExperimentConfig.from_yaml(yaml_path)
    except ValidationError as exc:
        typer.echo(f"✗ {yaml_path}: invalid", err=True)
        for err in exc.errors():
            path = ".".join(str(p) for p in err["loc"])
            typer.echo(f"  {path}: {err['msg']}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"OK {yaml_path}")
