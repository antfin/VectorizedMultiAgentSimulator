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

        links = ReportLinks(
            config=_required_rel(run_dir, run_dir / "input" / "config.json"),
            provenance=_required_rel(run_dir, run_dir / "input" / "provenance.json"),
            log=_optional_rel(run_dir, run_dir / "logs" / "run.log"),
            metrics=_required_rel(run_dir, run_dir / "output" / "metrics.json"),
            eval_episodes=_optional_rel(run_dir, run_dir / "output" / "eval_episodes.json"),
            videos=_videos(run_dir),
            **_benchmarl_links(run_dir),
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


def _benchmarl_links(run_dir: Path) -> dict[str, str | None]:
    """Locate BenchMARL's per-run subdir + scalars dir + latest policy checkpoint.

    BenchMARL writes a nested layout: ``output/benchmarl/<bm_run>/<bm_run>/
    {scalars,texts,videos,checkpoints}``. The outer dir holds ``config.pkl``;
    actual artefacts live one level deeper. We recurse to find them so the
    inner-name quirk doesn't leak into report consumers.
    """
    bm_root = run_dir / "output" / "benchmarl"
    if not bm_root.is_dir():
        return {"benchmarl_dir": None, "benchmarl_scalars": None, "policy": None}
    children = [p for p in bm_root.iterdir() if p.is_dir()]
    if not children:
        return {"benchmarl_dir": None, "benchmarl_scalars": None, "policy": None}
    # BenchMARL produces exactly one per-run subdir; pick the only / newest.
    bm_run = max(children, key=lambda p: p.stat().st_mtime)
    return {
        "benchmarl_dir": _required_rel(run_dir, bm_run),
        "benchmarl_scalars": _find_descendant_dir(run_dir, bm_run, "scalars"),
        "policy": _latest_policy(run_dir, bm_run),
    }


def _find_descendant_dir(run_dir: Path, root: Path, name: str) -> str | None:
    """Return the relative path of the first ``name``-named directory under ``root``."""
    matches = [p for p in root.rglob(name) if p.is_dir()]
    if not matches:
        return None
    # Shallowest match wins — keeps the link stable when BenchMARL adds nesting.
    chosen = min(matches, key=lambda p: len(p.parts))
    return _required_rel(run_dir, chosen)


def _latest_policy(run_dir: Path, bm_run: Path) -> str | None:
    """Return the relative path to the most recent ``*.pt`` under any ``checkpoints/``."""
    pts = list(bm_run.rglob("checkpoints/*.pt"))
    if not pts:
        return None
    latest = max(pts, key=lambda p: p.stat().st_mtime)
    return _required_rel(run_dir, latest)
