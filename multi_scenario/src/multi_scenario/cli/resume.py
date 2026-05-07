"""``multi-scenario resume <run_dir>`` — resume a crashed run (F5.7, local-only).

Refuses (exit 2) if the cfg's runner doesn't support resume, or if the run is
already DONE, or if no checkpoint is on disk yet.
"""

from datetime import datetime, timezone
from pathlib import Path

import typer

from multi_scenario.adapters.logging.file_logger import FileLogger
from multi_scenario.adapters.runners.local import LocalRunner
from multi_scenario.adapters.storage.local import LocalStorageAdapter
from multi_scenario.domain.models import RunState

from ._app import app
from ._helpers import latest_checkpoint


@app.command()
def resume(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
) -> None:
    """Resume a crashed run from its latest checkpoint (F5.7, local-only)."""
    storage = LocalStorageAdapter()
    cfg = storage.load_config(run_dir)
    runner_type = cfg.runtime.runner.type if cfg.runtime is not None else "local"
    # Capability check: build the runner via factory to read its `supports_resume` flag.
    if runner_type != "local":
        typer.echo(
            f"✗ resume is only supported for local runners (cfg has runner.type={runner_type!r}). "
            "To rerun on OVH, submit a fresh job.",
            err=True,
        )
        raise typer.Exit(code=2)

    state = storage.load_run_state(run_dir)
    if state.state == RunState.DONE:
        typer.echo(f"✗ run is already DONE: {run_dir}", err=True)
        raise typer.Exit(code=2)

    checkpoint = latest_checkpoint(run_dir)
    if checkpoint is None:
        typer.echo(f"✗ no BenchMARL checkpoint found under {run_dir}/output/benchmarl/", err=True)
        raise typer.Exit(code=2)

    now = datetime.now(timezone.utc)
    if state.state != RunState.CRASHED:
        state = state.transition_to(RunState.CRASHED, now)
        storage.save_run_state(run_dir, state)
    state = state.transition_to(RunState.RESUMED, now)
    storage.save_run_state(run_dir, state)

    logger = FileLogger(run_dir / "logs" / "run.log")
    runner = LocalRunner(logger=logger)
    result = runner.run(cfg, run_dir=run_dir, resume_from=checkpoint)
    typer.echo(f"DONE: {result.run_id} -> {run_dir}")
