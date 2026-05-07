"""CodeUploader — ship the local source tree to the OVH code bucket (F6.4).

Walks a curated set of include dirs / files under the repo root, applies an
fnmatch-based exclude list, and uploads each surviving file to S3 via
:class:`S3StorageAdapter.put_file`. Decoupled from job submission — the user
runs ``multi-scenario upload-code`` once per code change, then submits N jobs
that all reuse the already-uploaded code.

This is "rsync-style" only in the loose sense: it copies the include set
flat to S3 each run; per-file hash diffing is deferred until upload time
becomes a real bottleneck.
"""

import fnmatch
from pathlib import Path

from multi_scenario.adapters.storage.s3 import S3StorageAdapter

# Defaults tuned for multi_scenario's repo layout.
DEFAULT_INCLUDE_DIRS: tuple[str, ...] = (
    "src/multi_scenario",
    "experiments",
    "configs",
)
DEFAULT_INCLUDE_FILES: tuple[str, ...] = (
    "pyproject.toml",
    "README.md",
)
DEFAULT_EXCLUDE_PATTERNS: tuple[str, ...] = (
    "*/__pycache__/*",
    "*.pyc",
    "*.pyo",
    "*/.pytest_cache/*",
    "*/.ruff_cache/*",
    "*/.mypy_cache/*",
    "*.egg-info/*",
    # Don't ship results / videos / logs from prior local runs.
    "*/results/*",
    "*/output/*",
    "*/logs/*",
    # Skip per-run folders ("__" timestamp marker per §3.5.2).
    "experiments/*/*/*__*",
    "experiments/*/*/*__*/*",
    "*/.DS_Store",
)


class CodeUploader:
    """Uploads a curated subset of the repo to the S3 code bucket."""

    # Single public method by design; pylint default doesn't fit.
    # pylint: disable=too-few-public-methods,too-many-arguments,too-many-positional-arguments

    def __init__(self, s3: S3StorageAdapter) -> None:
        self._s3 = s3

    def upload(
        self,
        repo_root: Path,
        include_dirs: tuple[str, ...] = DEFAULT_INCLUDE_DIRS,
        include_files: tuple[str, ...] = DEFAULT_INCLUDE_FILES,
        exclude_patterns: tuple[str, ...] = DEFAULT_EXCLUDE_PATTERNS,
        dry_run: bool = False,
    ) -> list[Path]:
        """Upload curated files; return the list of relative paths uploaded.

        ``dry_run=True`` returns the list without putting any S3 objects.
        """
        repo_root = repo_root.resolve()
        targets: list[Path] = list(_collect_files(repo_root, include_dirs, include_files))
        kept = [p for p in targets if not _is_excluded(p, repo_root, exclude_patterns)]
        if dry_run:
            return [p.relative_to(repo_root) for p in kept]
        for path in kept:
            rel = path.relative_to(repo_root).as_posix()
            self._s3.put_file(rel, path.read_bytes())
        return [p.relative_to(repo_root) for p in kept]


def _collect_files(repo_root: Path, dirs: tuple[str, ...], files: tuple[str, ...]):
    """Yield every regular-file path under the include dirs + include files."""
    for d in dirs:
        sub = repo_root / d
        if not sub.is_dir():
            continue
        for path in sub.rglob("*"):
            if path.is_file():
                yield path
    for f in files:
        path = repo_root / f
        if path.is_file():
            yield path


def _is_excluded(path: Path, repo_root: Path, patterns: tuple[str, ...]) -> bool:
    """True when ``path``'s repo-relative form matches any fnmatch pattern."""
    rel = path.relative_to(repo_root).as_posix()
    return any(fnmatch.fnmatch(rel, pat) for pat in patterns)
