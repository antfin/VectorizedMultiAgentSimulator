"""``multi-scenario regenerate-videos <run_dir>`` — rerecord before/after MP4s (F6.6).

Thin Typer wrapper around :func:`multi_scenario.application.regenerate_videos.regenerate_videos`
so the same logic can be invoked from ``OvhRunner.run()`` after a job's results
sync back to the local machine.
"""

from pathlib import Path

import typer

from multi_scenario.application.regenerate_videos import (
    VideoRegenerationError,
    regenerate_videos,
)

from ._app import app


@app.command(name="regenerate-videos")
def regenerate_videos_cmd(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
) -> None:
    """Regenerate before/after MP4s for a completed run (F6.6)."""
    typer.echo(f"recording before_training.mp4 + after_training.mp4 for {run_dir}…")
    try:
        videos_dir = regenerate_videos(run_dir)
    except VideoRegenerationError as exc:
        typer.echo(f"✗ {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(f"OK regenerated videos → {videos_dir}")
