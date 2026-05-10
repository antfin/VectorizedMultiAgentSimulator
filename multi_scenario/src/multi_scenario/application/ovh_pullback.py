"""Pull an OVH-completed run-folder back to the local filesystem.

After an OVH job reaches DONE, results live at
``s3://<bucket>/<run_dir.name>/...``. This module walks that prefix via the
``ovhai`` CLI (no AWS S3 keys; same auth path as ``upload-code``) and writes
the tree under the local ``run_dir`` so callers — Streamlit Run Detail,
``regenerate-videos``, eval-only re-runs — see the OVH run exactly as if it
had executed locally.

Stage 1 made the S3 prefix per-run (``<run_dir.name>``), so the local
destination and the S3 source share one identifier — no path translation.

Hex-clean: depends on ``adapters/runners/ovh_cli.OvhClient`` and
``domain/models.OvhJobConfig`` only — never on cli/ or frontend/.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from multi_scenario.adapters.runners.ovh_cli import OvhClient
from multi_scenario.domain.models import OvhJobConfig


class _Logger(Protocol):
    """Subset of the logger Protocol the pullback needs (info / warning)."""

    # pylint: disable=missing-function-docstring,too-few-public-methods
    def info(self, msg: str) -> None:
        ...

    def warning(self, msg: str) -> None:
        ...


@dataclass(frozen=True)
class PullbackResult:
    """Outcome of one pullback — what the caller surfaces to the user."""

    dest_dir: Path
    n_downloaded: int  # files newly fetched
    n_skipped: int  # files already present with matching size (idempotent)


def pullback_run_dir(
    *,
    ovh_cfg: OvhJobConfig,
    run_dir_name: str,
    dest_dir: Path,
    client: OvhClient | None = None,
    logger: _Logger | None = None,
) -> PullbackResult:
    """Materialise ``s3://<bucket_results>/<run_dir_name>/...`` under ``dest_dir``.

    Idempotent: a file already present with matching size is skipped, so
    re-running pullback (e.g. a manual refresh after auto-sync) is cheap.

    No-op-safe when the prefix is empty (returns zero counts) — callers can
    always call this after submission without first checking job state.

    Implementation note: lists then fetches per-object via the ``ovhai`` CLI.
    Per-call subprocess cost is ~100ms; a typical run-folder (~10 files,
    ~5MB) lands in ~1s. If runs grow large enough that per-file cost matters,
    add a recursive-download primitive to :class:`OvhClient`.
    """
    cli = client or OvhClient()
    log = logger or _NoopLogger()
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Trailing slash on the prefix matters: without it, ``run_dir_name="demo_s0"``
    # would also list ``demo_s0__20260510_..._something`` (a sibling re-run).
    prefix = f"{run_dir_name}/"
    objects = cli.bucket_list_objects(
        region=ovh_cfg.region,
        bucket=ovh_cfg.bucket_results,
        prefix=prefix,
    )
    if not objects:
        log.warning(
            f"pullback: no objects under "
            f"{ovh_cfg.bucket_results}@{ovh_cfg.region}/{prefix} — "
            "did the OVH job actually write results?"
        )
        return PullbackResult(dest_dir=dest_dir, n_downloaded=0, n_skipped=0)

    n_downloaded = 0
    n_skipped = 0
    for obj in objects:
        rel = obj.name[len(prefix) :]
        target = dest_dir / rel
        if target.is_file() and target.stat().st_size == obj.size_bytes:
            n_skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        body = cli.bucket_get_object(
            region=ovh_cfg.region, bucket=ovh_cfg.bucket_results, key=obj.name
        )
        target.write_bytes(body)
        n_downloaded += 1

    log.info(
        f"pullback: {n_downloaded} downloaded, {n_skipped} skipped "
        f"(already present) → {dest_dir}"
    )
    return PullbackResult(
        dest_dir=dest_dir, n_downloaded=n_downloaded, n_skipped=n_skipped
    )


class _NoopLogger:
    """Default logger when caller doesn't supply one."""

    # pylint: disable=missing-function-docstring,unused-argument
    def info(self, msg: str) -> None:
        return None

    def warning(self, msg: str) -> None:
        return None
