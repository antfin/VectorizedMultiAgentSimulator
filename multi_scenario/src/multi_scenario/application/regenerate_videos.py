"""Re-record before/after training MP4s for a completed run.

Extracted from the ``multi-scenario regenerate-videos`` CLI command so the
same logic can be invoked programmatically — notably by ``OvhRunner.run()``,
which calls this on the local machine after pulling OVH results back (the
in-container Pyglet renderer fails fail-soft, so videos are produced
post-hoc on the dev host that submitted the job).
"""

import tempfile
from pathlib import Path

from multi_scenario.adapters.storage.local import LocalStorageAdapter
from multi_scenario.application.factories import make_algorithm


class VideoRegenerationError(RuntimeError):
    """Raised when ``regenerate_videos`` cannot proceed (missing checkpoint, bad cfg, …)."""


def latest_checkpoint(run_dir: Path) -> Path | None:
    """Locate the most-recent ``*.pt`` under ``run_dir/output/benchmarl/.../checkpoints/``."""
    bm_root = run_dir / "output" / "benchmarl"
    if not bm_root.is_dir():
        return None
    pts = list(bm_root.rglob("checkpoints/*.pt"))
    if not pts:
        return None
    return max(pts, key=lambda p: p.stat().st_mtime)


# Both branches build experiments + record + refresh — ~14 locals, well within
# what's readable for a top-down scripted function. Splitting it would obscure
# the linear flow.
# pylint: disable-next=too-many-locals
def regenerate_videos(run_dir: Path) -> Path:
    """Record ``before_training.mp4`` + ``after_training.mp4`` under ``run_dir/output/videos/``.

    Returns the videos directory on success; raises :class:`VideoRegenerationError`
    on missing config / missing checkpoint / non-BenchMARL algorithm. Does NOT
    swallow renderer failures — the caller decides what to do (the OVH path
    fails-soft via try/except, the CLI lets the traceback bubble up).
    """
    config_path = run_dir / "input" / "config.json"
    if not config_path.is_file():
        raise VideoRegenerationError(f"no input/config.json under {run_dir}")
    checkpoint = latest_checkpoint(run_dir)
    if checkpoint is None:
        raise VideoRegenerationError(
            f"no BenchMARL checkpoint under {run_dir}/output/benchmarl/"
        )

    storage = LocalStorageAdapter()
    cfg = storage.load_config(run_dir)
    videos_dir = run_dir / "output" / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Local imports to keep BenchMARL/torch off the module top-level so callers
    # paying for "is regen needed?" decisions don't take the import hit.
    # pylint: disable=import-outside-toplevel
    from benchmarl.experiment import Experiment

    from multi_scenario.adapters.algorithms.benchmarl_base import BenchmarlBaseAdapter
    from multi_scenario.adapters.storage.report_builder import ReportBuilder
    from multi_scenario.adapters.video.recorder import VideoRecorder

    adapter = make_algorithm(cfg.algorithm.type)
    if not isinstance(adapter, BenchmarlBaseAdapter):
        raise VideoRegenerationError(
            "regenerate-videos only supports BenchMARL-backed algorithms; "
            f"cfg has algorithm.type={cfg.algorithm.type!r}"
        )

    # BEFORE: random-init policy via a fresh experiment built into a temp dir
    # so BenchMARL's native scratch space doesn't pollute the real run.
    with tempfile.TemporaryDirectory() as td:
        fresh = adapter.build_experiment(cfg, run_dir=Path(td))
        VideoRecorder().record(
            test_env=fresh.test_env,
            policy=fresh.policy,
            max_steps=fresh.max_steps,
            output_path=videos_dir / "before_training.mp4",
        )

    # AFTER: reload the trained checkpoint → trained policy.
    trained = Experiment.reload_from_file(str(checkpoint))
    VideoRecorder().record(
        test_env=trained.test_env,
        policy=trained.policy,
        max_steps=trained.max_steps,
        output_path=videos_dir / "after_training.mp4",
    )

    # Refresh report.json so the video links populate.
    result = storage.load_result(run_dir)
    run_state = storage.load_run_state(run_dir)
    report = ReportBuilder().build(run_dir, result, run_state)
    storage.save_report(run_dir, report)
    return videos_dir
