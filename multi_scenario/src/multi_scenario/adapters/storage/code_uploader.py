"""CodeUploader — ship the local source tree to the OVH code bucket (F6.4).

Walks a curated set of include dirs / files under the repo root, applies an
fnmatch-based exclude list, and uploads each surviving file to S3 via
:class:`S3StorageAdapter.put_file`. Decoupled from job submission — the user
runs ``multi-scenario upload-code`` once per code change, then submits N jobs
that all reuse the already-uploaded code.

A companion ``.code_hash`` blob is written alongside the uploaded files —
the Submit-page preflight probe reads it to detect "your local source has
drifted from what's in the bucket; re-run upload-code first" before
submitting an OVH job. The hash logic mirrors :func:`compute_code_hash`
from :mod:`multi_scenario.domain.hashing` so the value is reproducible.
"""

import fnmatch
from pathlib import Path

from multi_scenario.adapters.storage.s3 import S3StorageAdapter
from multi_scenario.domain.hashing import compute_code_hash

#: Fixed key under which the code-hash blob lives inside ``<bucket>/<prefix>/``.
CODE_HASH_KEY = ".code_hash"

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
        targets: list[Path] = list(
            _collect_files(repo_root, include_dirs, include_files)
        )
        kept = [p for p in targets if not _is_excluded(p, repo_root, exclude_patterns)]
        # Sort so the code-hash is deterministic regardless of filesystem walk order.
        kept = sorted(kept)
        if dry_run:
            return [p.relative_to(repo_root) for p in kept]
        for path in kept:
            rel = path.relative_to(repo_root).as_posix()
            self._s3.put_file(rel, path.read_bytes())
        # Write the companion .code_hash blob — Submit-page preflight reads
        # this and compares against the local source hash to detect drift.
        digest = compute_code_hash(kept)
        self._s3.put_file(CODE_HASH_KEY, digest.encode("utf-8"))
        return [p.relative_to(repo_root) for p in kept]


def compute_local_code_hash(
    repo_root: Path,
    include_dirs: tuple[str, ...] = DEFAULT_INCLUDE_DIRS,
    include_files: tuple[str, ...] = DEFAULT_INCLUDE_FILES,
    exclude_patterns: tuple[str, ...] = DEFAULT_EXCLUDE_PATTERNS,
) -> str:
    """Compute the same digest :class:`CodeUploader` writes — without touching S3.

    The Submit-page preflight probe uses this to compute the **local** hash
    so it can compare against the ``.code_hash`` blob ``CodeUploader``
    uploaded. Both must use the identical file set + ordering for the
    comparison to be meaningful.
    """
    repo_root = repo_root.resolve()
    targets = list(_collect_files(repo_root, include_dirs, include_files))
    kept = sorted(
        p for p in targets if not _is_excluded(p, repo_root, exclude_patterns)
    )
    return compute_code_hash(kept)


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
