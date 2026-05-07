"""multi_scenario CLI — ``run`` (drive a config end-to-end) + ``validate`` (lint a YAML).

``run`` wires:

- ``ExperimentConfig.from_yaml`` (F1.1) for parsing.
- ``LocalRunner`` (F2.6) with default ``ProvenanceWriter`` (F2.7).
- ``FileLogger`` (F2.7) pointed at ``<run_dir>/logs/run.log``.

The run folder is built at ``<storage.path>/<run_id>__<timestamp>`` using
``RunId`` (F1.3).

``validate`` (F5.1) parses the YAML, runs it through ``ExperimentConfig``
strict validation, and exits non-zero with one readable line per error
(``<dotted.field.path>: <message>``). Used as a pre-flight check before
expensive OVH submits or long local sweeps.
"""

from datetime import datetime, timezone
from pathlib import Path

import typer
from pydantic import ValidationError

from multi_scenario import __version__
from multi_scenario.adapters.logging.file_logger import FileLogger
from multi_scenario.adapters.runners.local import LocalRunner
from multi_scenario.domain.models import ExperimentConfig, RunId

app = typer.Typer(help="multi_scenario CLI")


@app.command()
def version() -> None:
    """Print the multi_scenario package version.

    Also forces typer's multi-command mode so ``run`` stays a real subcommand.
    """
    typer.echo(__version__)


@app.command()
def run(
    yaml_path: Path = typer.Argument(..., exists=True, readable=True),
) -> None:
    """Execute one experiment run from a YAML config file."""
    cfg = ExperimentConfig.from_yaml(yaml_path)

    run_id = RunId(exp_id=cfg.experiment.id, seed=cfg.experiment.seed)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    storage_root = (
        Path(cfg.runtime.storage.path) if cfg.runtime is not None else Path("experiments")
    )
    run_dir = storage_root / run_id.folder_name(timestamp)
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = FileLogger(run_dir / "logs" / "run.log")
    runner = LocalRunner(logger=logger)  # default ProvenanceWriter() from F2.7
    result = runner.run(cfg, run_dir=run_dir)
    typer.echo(f"DONE: {result.run_id} -> {run_dir}")


@app.command()
def validate(
    yaml_path: Path = typer.Argument(..., exists=True, readable=True),
) -> None:
    """Validate a YAML against the ``ExperimentConfig`` schema; exit 1 on any error.

    Pre-flight check before submitting OVH jobs or launching long local sweeps.
    Errors are formatted as one ``<dotted.field.path>: <message>`` line per issue.
    """
    try:
        ExperimentConfig.from_yaml(yaml_path)
    except ValidationError as exc:
        typer.echo(f"✗ {yaml_path}: invalid", err=True)
        for err in exc.errors():
            path = ".".join(str(p) for p in err["loc"])
            typer.echo(f"  {path}: {err['msg']}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"OK {yaml_path}")


def main() -> None:
    """Entry point used by the ``multi-scenario`` console script."""
    app()
