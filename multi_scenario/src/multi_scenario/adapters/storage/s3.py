"""S3StorageAdapter — boto3-backed implementation of the Storage port.

Inherits the eight Protocol methods (save/load × {config, provenance, result,
run_state}) from :class:`StorageAdapterBase`; only the I/O primitives
(``_write_text`` / ``_read_text``) plus S3-only off-Protocol helpers
(``sync_to_local``, ``sync_from_local``, ``put_file``) live here.

Mirrors the §3.5.2 layout under ``s3://<bucket>/<prefix>/<run_dir.name>/...``.
The boto3 client is built from the supplied ``S3StorageConfig`` (region +
optional ``endpoint_url`` for OVH Object Storage). Tests inject a moto-mocked
client via the ``client`` kwarg.
"""

from pathlib import Path
from typing import Any

import boto3

from multi_scenario.domain.models import S3StorageConfig

from ._base import StorageAdapterBase


class S3StorageAdapter(StorageAdapterBase):
    """S3-backed Storage Protocol implementation."""

    name = "s3"

    def __init__(self, config: S3StorageConfig, client: Any | None = None) -> None:
        self._config = config
        self._client = client or boto3.client(
            "s3",
            region_name=config.region,
            endpoint_url=config.endpoint_url,
        )

    # ── Off-Protocol sync helpers (used by OvhRunner) ──────────────────

    def sync_to_local(self, run_dir: Path, local_dir: Path) -> None:
        """Download every key under the run's prefix to ``local_dir``.

        Recreates the per-run folder tree locally — used by ``OvhRunner``
        after a successful OVH job to bring results back for downstream
        processing (eval-only, report regeneration, Streamlit).
        """
        # Trailing slash is required so the slice strips the run-prefix cleanly
        # (without it, leftover `rel` would start with `/` and `mkdir` would
        # treat it as an absolute path).
        prefix = self._key(run_dir, "") + "/"
        local_dir.mkdir(parents=True, exist_ok=True)
        for key in self._list(prefix):
            rel = key[len(prefix) :]
            target = local_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            obj = self._client.get_object(Bucket=self._config.bucket, Key=key)
            target.write_bytes(obj["Body"].read())

    def sync_from_local(self, local_dir: Path, run_dir: Path) -> None:
        """Upload every file under ``local_dir`` to the run's S3 key prefix."""
        for path in sorted(local_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(local_dir).as_posix()
            self._put(run_dir, rel, path.read_bytes())

    def put_file(self, key: str, body: bytes) -> None:
        """Flat upload at an explicit ``<prefix>/<key>`` (F6.4).

        Used by the code uploader: per-run-folder transforms don't apply, the
        caller decides the key relative to ``config.prefix``.
        """
        full_key = "/".join(p.strip("/") for p in (self._config.prefix, key) if p)
        self._client.put_object(Bucket=self._config.bucket, Key=full_key, Body=body)

    # ── I/O primitives (subclass hook for StorageAdapterBase) ──────────

    def _write_text(self, run_dir: Path, rel: str, body: str) -> None:
        self._put(run_dir, rel, body)

    def _read_text(self, run_dir: Path, rel: str) -> str:
        obj = self._client.get_object(
            Bucket=self._config.bucket, Key=self._key(run_dir, rel)
        )
        return obj["Body"].read().decode("utf-8")

    # ── internals ──────────────────────────────────────────────────────

    def _key(self, run_dir: Path, rel: str) -> str:
        """Build the canonical S3 key: ``<prefix>/<run_dir.name>/<rel>``."""
        parts = [self._config.prefix, run_dir.name]
        if rel:
            parts.append(rel)
        return "/".join(p.strip("/") for p in parts if p)

    def _put(self, run_dir: Path, rel: str, body: str | bytes) -> None:
        """Bytes-or-str put — used by ``_write_text`` and the bytes-mode sync helper."""
        if isinstance(body, str):
            body = body.encode("utf-8")
        self._client.put_object(
            Bucket=self._config.bucket, Key=self._key(run_dir, rel), Body=body
        )

    def _list(self, prefix: str) -> list[str]:
        """Page-aware list of keys under ``prefix``."""
        keys: list[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._config.bucket, Prefix=prefix):
            for obj in page.get("Contents", []) or []:
                keys.append(obj["Key"])
        return keys
