"""Cross-run manifest — emitted as ``runs.json`` per §3.5.3.

Slim pointer-only manifest: scope summary, link to ``runs.csv``, per-metric
rankings, and a flat list of per-run ``report.json`` references. Consumers
(Streamlit) dereference via the ``report`` link instead of globbing per-run
file paths — no path duplication between manifest and the report it points to.
"""

from pydantic import BaseModel

from ._common import STRICT


class ManifestScope(BaseModel):
    """Aggregate counts/lists describing what's in this exp_type folder."""

    model_config = STRICT

    n_runs: int
    exp_ids: list[str]
    seeds: list[int]
    algorithms: list[str]


class RankingEntry(BaseModel):
    """One ranked run for a particular metric."""

    model_config = STRICT

    run_id: str
    value: float
    report: str | None  # relative path; None when the run has no report.json


class ManifestRunEntry(BaseModel):
    """One run in the flat ``runs[]`` list — pointer-only."""

    model_config = STRICT

    run_id: str
    report: str | None


class RunsManifest(BaseModel):
    """Top-level manifest persisted to ``runs.json``."""

    model_config = STRICT

    scope: ManifestScope
    csv: str
    rankings: dict[str, list[RankingEntry]]
    runs: list[ManifestRunEntry]
