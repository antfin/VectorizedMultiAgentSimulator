"""Probe a run folder for every artefact the F7.3 detail page might want.

Pure logic — no Streamlit imports — so it's testable in isolation against
synthetic run dirs. Required artefacts (config + result) raise on miss;
optional ones (benchmarl scalars, videos, log file) become ``None`` /
empty so the page can render best-effort even on incomplete runs.
"""

from dataclasses import dataclass
from pathlib import Path

from multi_scenario.adapters.storage.local import LocalStorageAdapter
from multi_scenario.domain.models import (
    ExperimentConfig,
    ExperimentResult,
    RunStateRecord,
)


@dataclass(frozen=True)
# Detail bundles every artefact the F7.3 page may render — one attribute per
# section. Splitting into sub-bundles would obscure the section-by-section
# rendering loop without saving any complexity.
# pylint: disable-next=too-many-instance-attributes
class RunDetail:
    """Bundle of every per-run artefact the F7.3 page needs.

    Attributes:
        run_dir: Top-level run folder (the one with ``input/``, ``output/``).
        cfg: Resolved ``ExperimentConfig``.
        result: Final ``ExperimentResult`` (M1–M9 metrics).
        run_state: ``RunStateRecord`` if present; ``None`` otherwise (legacy runs).
        benchmarl_dir: Inner BenchMARL run folder (where scalars live), or ``None``.
        scalar_csvs: Sorted list of ``*.csv`` files under ``benchmarl_dir/scalars/``.
        videos: ``{"before": Path, "after": Path}`` — only keys that exist on disk.
        log_path: Full path to ``output/logs/run.log`` or fallback, or ``None``.
        has_eval_episodes: ``True`` iff ``output/eval_episodes.json`` is present.
    """

    run_dir: Path
    cfg: ExperimentConfig
    result: ExperimentResult
    run_state: RunStateRecord | None
    benchmarl_dir: Path | None
    scalar_csvs: list[Path]
    videos: dict[str, Path]
    log_path: Path | None
    has_eval_episodes: bool


def load_run_detail(run_dir: Path) -> RunDetail | None:
    """Probe ``run_dir`` for every detail-page artefact.

    Returns ``None`` if ``run_dir`` doesn't exist, or required artefacts
    (``input/config.json`` + ``output/metrics.json``) are missing/corrupt.
    Optional artefacts return ``None``/empty when absent.
    """
    if not run_dir.is_dir():
        return None

    storage = LocalStorageAdapter()
    try:
        cfg = storage.load_config(run_dir)
        result = storage.load_result(run_dir)
    except (OSError, ValueError):
        return None

    try:
        run_state: RunStateRecord | None = storage.load_run_state(run_dir)
    except (OSError, ValueError):
        run_state = None

    benchmarl_dir = _find_benchmarl_dir(run_dir)
    scalar_csvs = _collect_scalar_csvs(benchmarl_dir) if benchmarl_dir else []
    videos = _collect_videos(run_dir)
    log_path = _find_log(run_dir)
    has_eval_episodes = (run_dir / "output" / "eval_episodes.json").is_file()

    return RunDetail(
        run_dir=run_dir,
        cfg=cfg,
        result=result,
        run_state=run_state,
        benchmarl_dir=benchmarl_dir,
        scalar_csvs=scalar_csvs,
        videos=videos,
        log_path=log_path,
        has_eval_episodes=has_eval_episodes,
    )


def _find_benchmarl_dir(run_dir: Path) -> Path | None:
    """Locate ``output/benchmarl/<inner>/`` — BenchMARL's per-run output folder."""
    bm_root = run_dir / "output" / "benchmarl"
    if not bm_root.is_dir():
        return None
    # BenchMARL nests one folder per experiment; pick the most recent if many.
    inner = sorted((p for p in bm_root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
    return inner[-1] if inner else None


def _collect_scalar_csvs(benchmarl_dir: Path) -> list[Path]:
    """Recursive walk for ``**/scalars/*.csv`` under ``benchmarl_dir``.

    BenchMARL nests its experiment folder TWICE (``benchmarl/<exp>/<exp>/scalars/``)
    so a one-level search misses everything; ``rglob`` keeps the loader
    insensitive to that quirk.
    """
    return sorted(benchmarl_dir.rglob("scalars/*.csv"))


def _collect_videos(run_dir: Path) -> dict[str, Path]:
    """Map ``label`` → MP4 path for any ``output/videos/<label>_training.mp4`` present."""
    videos_dir = run_dir / "output" / "videos"
    if not videos_dir.is_dir():
        return {}
    out: dict[str, Path] = {}
    for label in ("before", "after"):
        candidate = videos_dir / f"{label}_training.mp4"
        if candidate.is_file():
            out[label] = candidate
    return out


def _find_log(run_dir: Path) -> Path | None:
    """Probe canonical log paths.

    F2.7's ``LocalRunner`` wires ``FileLogger`` at ``<run_dir>/logs/run.log``
    (top-level ``logs/``, not nested under ``output/``). Older paths are kept
    as fallback so legacy run folders still light up.
    """
    candidates = (
        run_dir / "logs" / "run.log",
        run_dir / "output" / "logs" / "run.log",
        run_dir / "output" / "log.log",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None
