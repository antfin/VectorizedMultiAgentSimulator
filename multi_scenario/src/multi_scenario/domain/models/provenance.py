"""Provenance models — what produced a run (hashes, git state, library versions)."""

from datetime import datetime

from pydantic import BaseModel

from ._common import STRICT


class LibraryVersions(BaseModel):
    """Version strings for the libraries that produced a run."""

    model_config = STRICT

    python: str
    torch: str
    vmas: str
    benchmarl: str
    multi_scenario: str


class Provenance(BaseModel):
    """Hashes, git state, library versions, and timestamps for one run.

    Persisted to ``input/provenance.json``. Streamlit uses this to flag stale
    results when the current code or config hash diverges from what produced
    the result.
    """

    model_config = STRICT

    config_hash: str
    code_hash: str
    hashed_source_files: list[str]
    git_sha: str
    git_dirty: bool
    created_at: datetime
    finished_at: datetime | None = None
    library_versions: LibraryVersions
