"""multi_scenario CLI — single ``run`` command that drives a config end-to-end.

Wires:

- ``ExperimentConfig.from_yaml`` (F1.1) for parsing.
- ``LocalRunner`` (F2.6) with default ``ProvenanceWriter`` (F2.7).
- ``FileLogger`` (F2.7) pointed at ``<run_dir>/logs/run.log``.

The run folder is built at ``<storage.path>/<run_id>__<timestamp>`` using
``RunId`` (F1.3). Returns the produced run folder path.
"""

from datetime import datetime, timezone
from pathlib import Path

import typer

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


def main() -> None:
    """Entry point used by the ``multi-scenario`` console script."""
    app()
