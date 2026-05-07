"""ReportBuilder — assembles ``output/report.json`` by inspecting a run folder.

Pure filesystem inspection: locates BenchMARL's per-run subdir (one child of
``output/benchmarl/``), the latest policy checkpoint, and any opt-in artefacts
(videos / eval_episodes) that may or may not be present yet. Returns a
``RunReport`` ready for ``LocalStorageAdapter.save_report``.

Wired by ``LocalRunner`` *after* ``ExperimentService.run()`` returns so the
report's ``status`` reflects the on-disk run state. This keeps the ``Storage``
Protocol surface minimal — the report writer lives on the concrete adapter
only (per F1.9 design note).
"""

from pathlib import Path

from multi_scenario.domain.models import (
    BenchmarlLinks,
    ExperimentResult,
    ReportLinks,
    ReportVideos,
    RunReport,
    RunStateRecord,
)

# Headline metrics surfaced in the report's `summary` block. The full M1-M9
# bundle stays in `output/metrics.json`; the report just lifts the universal
# subset that's meaningful for any scenario.
_HEADLINE_METRICS = (
    "M1_success_rate",
    "M2_avg_return",
    "M3_steps",
    "M4_collisions",
)


class ReportBuilder:
    """Builds a ``RunReport`` from a run folder + result + run-state record."""

    # Pure inspector with one public method; pylint's defaults flag this.
    # pylint: disable=too-few-public-methods

    def build(
        self,
        run_dir: Path,
        result: ExperimentResult,
        run_state: RunStateRecord,
    ) -> RunReport:
        """Assemble the manifest by inspecting ``run_dir`` for artefacts."""
        started_at = run_state.transitions[0].ts
        finished_at = run_state.transitions[-1].ts
        duration_seconds = (finished_at - started_at).total_seconds()

        result_metrics = {m.name: m.value for m in result.metrics}
        summary = {name: result_metrics.get(name) for name in _HEADLINE_METRICS}

        bm_run = _find_bm_run(run_dir)
        links = ReportLinks(
            config=_required_rel(run_dir, run_dir / "input" / "config.json"),
            provenance=_required_rel(run_dir, run_dir / "input" / "provenance.json"),
            log=_optional_rel(run_dir, run_dir / "logs" / "run.log"),
            metrics=_required_rel(run_dir, run_dir / "output" / "metrics.json"),
            eval_episodes=_optional_rel(run_dir, run_dir / "output" / "eval_episodes.json"),
            videos=_videos(run_dir),
            policy=_latest_policy(run_dir, bm_run) if bm_run is not None else None,
            benchmarl=_benchmarl_block(run_dir, bm_run) if bm_run is not None else None,
        )
        return RunReport(
            status=run_state.state.value,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=duration_seconds,
            summary=summary,
            links=links,
        )


def _required_rel(run_dir: Path, target: Path) -> str:
    """Return ``target`` as a forward-slash relative path under ``run_dir``."""
    return target.relative_to(run_dir).as_posix()


def _optional_rel(run_dir: Path, target: Path) -> str | None:
    """Like ``_required_rel`` but returns None when ``target`` doesn't exist."""
    return target.relative_to(run_dir).as_posix() if target.exists() else None


def _videos(run_dir: Path) -> ReportVideos:
    """Wire up the F2.11 video paths if their files exist; otherwise both None."""
    videos_dir = run_dir / "output" / "videos"
    return ReportVideos(
        before_training=_optional_rel(run_dir, videos_dir / "before_training.mp4"),
        after_training=_optional_rel(run_dir, videos_dir / "after_training.mp4"),
    )


def _find_bm_run(run_dir: Path) -> Path | None:
    """Locate the *inner* BenchMARL run dir (where scalars/checkpoints live).

    BenchMARL writes a nested layout: ``output/benchmarl/<bm_run>/<bm_run>/
    {scalars,texts,videos,checkpoints}``. We point the report at the **inner**
    dir so ``scalars[i]`` paths stay clean (``scalars/train_loss.csv`` rather
    than ``<bm_run>/scalars/train_loss.csv``). Returns None if no benchmarl
    output present.
    """
    bm_root = run_dir / "output" / "benchmarl"
    if not bm_root.is_dir():
        return None
    # Find the shallowest dir that contains a `scalars/` subfolder — that's
    # the BenchMARL run root regardless of nesting depth.
    candidates = [s.parent for s in bm_root.rglob("scalars") if s.is_dir() and s.parent.is_dir()]
    if not candidates:
        return None
    return min(candidates, key=lambda p: len(p.parts))


def _benchmarl_block(run_dir: Path, bm_run: Path) -> BenchmarlLinks:
    """Build the report's ``benchmarl`` block: dir + sorted list of scalar CSVs.

    ``dir`` is relative to ``run_dir`` and points at the BenchMARL run root.
    ``scalars[i]`` are relative to ``dir`` (so a consumer resolves a CSV via
    ``run_dir / dir / scalars[i]``) — typically just ``scalars/<name>.csv``.
    """
    scalars_dir = bm_run / "scalars"
    files = sorted(scalars_dir.glob("*.csv")) if scalars_dir.is_dir() else []
    return BenchmarlLinks(
        dir=_required_rel(run_dir, bm_run),
        scalars=[f.relative_to(bm_run).as_posix() for f in files],
    )


def _latest_policy(run_dir: Path, bm_run: Path) -> str | None:
    """Return the relative path to the most recent ``*.pt`` under any ``checkpoints/``."""
    pts = list(bm_run.rglob("checkpoints/*.pt"))
    if not pts:
        return None
    latest = max(pts, key=lambda p: p.stat().st_mtime)
    return _required_rel(run_dir, latest)
