"""Run-end manifest — emitted as ``output/report.json`` per §3.5.2.

The report is a thin summary + link directory: status, timestamps, headline
metrics, and relative paths to every artefact the run produced. Streamlit's
run-detail page reads this and never globs the filesystem.

All link paths are relative to the run folder so the manifest stays portable
when a run is moved or archived. ``None`` means the artefact wasn't produced
(e.g. videos default-off on smoke runs; eval_episodes until F2.10.1).
"""

from datetime import datetime

from pydantic import BaseModel

from ._common import STRICT


class ReportVideos(BaseModel):
    """Optional rendered episode videos (F2.11)."""

    model_config = STRICT

    before_training: str | None = None
    after_training: str | None = None


class ReportLinks(BaseModel):
    """Relative-path links to every artefact the run produced.

    Required artefacts (always non-None on a successful run): ``config``,
    ``provenance``, ``log``, ``metrics``. Everything else is best-effort.
    """

    model_config = STRICT

    config: str
    provenance: str
    log: str | None
    metrics: str
    eval_episodes: str | None = None
    policy: str | None = None
    videos: ReportVideos = ReportVideos()
    benchmarl_dir: str | None = None
    benchmarl_scalars: str | None = None


class RunReport(BaseModel):
    """Top-level manifest persisted to ``output/report.json``."""

    model_config = STRICT

    status: str
    started_at: datetime
    finished_at: datetime
    duration_seconds: float
    summary: dict[str, float | None]
    links: ReportLinks
