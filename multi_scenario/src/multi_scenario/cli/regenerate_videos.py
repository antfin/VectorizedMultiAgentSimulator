"""``multi-scenario regenerate-videos <run_dir>`` — rerecord before/after MP4s (F6.6).

Reads cfg + checkpoint from ``<run_dir>``, rebuilds the experiment with the
original seed (BEFORE → random-init policy), reloads the saved checkpoint
(AFTER → trained policy), records both videos to ``<run_dir>/output/videos/``,
and refreshes ``output/report.json`` so the video links resolve.

Use this after pulling an OVH-trained run back to a local machine where Pyglet
rendering works (the OVH job's video recording would have failed fail-soft per
F6.6 Part 1).
"""

from pathlib import Path

import typer

from multi_scenario.adapters.storage.local import LocalStorageAdapter
from multi_scenario.application.factories import make_algorithm

from ._app import app
from ._helpers import latest_checkpoint


# This command body wires cfg/checkpoint/storage/videos_dir/adapter/two
# Experiment objects/result/run_state/report — naturally many locals; the
# linear flow is the docstring and extracting helpers would fracture it.
# pylint: disable=too-many-locals
@app.command(name="regenerate-videos")
def regenerate_videos(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
) -> None:
    """Regenerate before/after MP4s for a completed run (F6.6)."""
    config_path = run_dir / "input" / "config.json"
    if not config_path.is_file():
        typer.echo(f"✗ no input/config.json under {run_dir}", err=True)
        raise typer.Exit(code=2)
    checkpoint = latest_checkpoint(run_dir)
    if checkpoint is None:
        typer.echo(f"✗ no BenchMARL checkpoint under {run_dir}/output/benchmarl/", err=True)
        raise typer.Exit(code=2)

    storage = LocalStorageAdapter()
    cfg = storage.load_config(run_dir)
    videos_dir = run_dir / "output" / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Local imports to keep BenchMARL/torch off the cli module top-level so
    # `multi-scenario --help` stays fast.
    # pylint: disable=import-outside-toplevel
    import tempfile

    from benchmarl.experiment import Experiment

    from multi_scenario.adapters.algorithms.benchmarl_base import BenchmarlBaseAdapter
    from multi_scenario.adapters.storage.report_builder import ReportBuilder
    from multi_scenario.adapters.video.recorder import VideoRecorder

    # BEFORE: build a fresh experiment with the original seed via the same
    # algorithm adapter (so the policy architecture matches). Use a temp dir
    # so BenchMARL's native output doesn't pollute the real run_dir's
    # benchmarl/ subtree.
    adapter = make_algorithm(cfg.algorithm.type)
    if not isinstance(adapter, BenchmarlBaseAdapter):
        typer.echo(
            f"✗ regenerate-videos only supports BenchMARL-backed algorithms; "
            f"cfg has algorithm.type={cfg.algorithm.type!r}",
            err=True,
        )
        raise typer.Exit(code=2)

    typer.echo("recording before_training.mp4 (random-init policy)...")
    with tempfile.TemporaryDirectory() as td:
        fresh_experiment = adapter.build_experiment(cfg, run_dir=Path(td))
        VideoRecorder().record(
            test_env=fresh_experiment.test_env,
            policy=fresh_experiment.policy,
            max_steps=fresh_experiment.max_steps,
            output_path=videos_dir / "before_training.mp4",
        )

    # AFTER: reload the trained checkpoint → trained policy.
    typer.echo("recording after_training.mp4 (trained policy from checkpoint)...")
    trained_experiment = Experiment.reload_from_file(str(checkpoint))
    VideoRecorder().record(
        test_env=trained_experiment.test_env,
        policy=trained_experiment.policy,
        max_steps=trained_experiment.max_steps,
        output_path=videos_dir / "after_training.mp4",
    )

    # Refresh report.json so the video links populate.
    typer.echo("refreshing report.json...")
    result = storage.load_result(run_dir)
    run_state = storage.load_run_state(run_dir)
    report = ReportBuilder().build(run_dir, result, run_state)
    storage.save_report(run_dir, report)
    typer.echo(f"OK regenerated videos → {videos_dir}")
