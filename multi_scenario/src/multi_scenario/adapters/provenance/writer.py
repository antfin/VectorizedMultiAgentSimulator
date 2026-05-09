"""ProvenanceWriter — builds Provenance for one run from cfg + git + library versions."""

import importlib.metadata
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from multi_scenario import __version__ as _MS_VERSION
from multi_scenario.domain.hashing import compute_code_hash, compute_config_hash
from multi_scenario.domain.models import ExperimentConfig, LibraryVersions, Provenance


class ProvenanceWriter:
    """Callable that builds a ``Provenance`` for one run.

    ``hashed_source_files`` is the curated list whose sha256 produces the
    ``code_hash`` field. Empty list → ``code_hash="sha256:empty"``. Pass an
    explicit list to track code-version drift in Streamlit's staleness check.
    """

    # Single callable; per-class disable is fine here.
    # pylint: disable=too-few-public-methods

    def __init__(
        self,
        hashed_source_files: Iterable[str | Path] = (),
        git_root: str | Path | None = None,
    ) -> None:
        self._hashed_source_files = [Path(p) for p in hashed_source_files]
        self._git_root = Path(git_root) if git_root is not None else None

    def __call__(self, cfg: ExperimentConfig) -> Provenance:
        """Build a Provenance instance reflecting the current process / git / cfg."""
        code_hash = (
            compute_code_hash(self._hashed_source_files)
            if self._hashed_source_files
            else "sha256:empty"
        )
        return Provenance(
            config_hash=compute_config_hash(cfg.model_dump(mode="json")),
            code_hash=code_hash,
            hashed_source_files=[str(p) for p in self._hashed_source_files],
            git_sha=self._git_sha(),
            git_dirty=self._git_dirty(),
            created_at=datetime.now(timezone.utc),
            library_versions=_library_versions(),
        )

    def _git_sha(self) -> str:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self._git_root,
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            return result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            return "unknown"

    def _git_dirty(self) -> bool:
        try:
            # `check=False` is intentional: a non-zero exit means "dirty",
            # which is the signal we want to capture (not raise on).
            result = subprocess.run(
                ["git", "diff-index", "--quiet", "HEAD"],
                cwd=self._git_root,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            return result.returncode != 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False


def _library_versions() -> LibraryVersions:
    return LibraryVersions(
        python=(
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        ),
        torch=_pkg_version("torch"),
        vmas=_pkg_version("vmas"),
        benchmarl=_pkg_version("benchmarl"),
        multi_scenario=_MS_VERSION,
    )


def _pkg_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
