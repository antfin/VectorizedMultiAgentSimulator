"""S3StorageAdapter — boto3-backed implementation of the Storage port.

Mirrors the §3.5.2 layout under ``s3://<bucket>/<prefix>/<run_dir.name>/...``.
The 8 ``Storage`` Protocol methods write/read the four owned per-run JSON
artefacts (config / provenance / metrics / run_state); ``sync_to_local`` and
``sync_from_local`` are off-Protocol helpers used by ``OvhRunner`` (F6.2)
when transferring whole run folders between local fs and S3.

Construct the boto3 client from the supplied ``S3StorageConfig`` (region +
optional ``endpoint_url`` for OVH Object Storage). Tests inject a moto-mocked
client via the ``client`` kwarg.
"""

from pathlib import Path
from typing import Any

import boto3

from multi_scenario.domain.models import (
    ExperimentConfig,
    ExperimentResult,
    Provenance,
    RunStateRecord,
    S3StorageConfig,
)


class S3StorageAdapter:
    """S3-backed Storage Protocol implementation."""

    name = "s3"

    def __init__(self, config: S3StorageConfig, client: Any | None = None) -> None:
        self._config = config
        self._client = client or boto3.client(
            "s3",
            region_name=config.region,
            endpoint_url=config.endpoint_url,
        )

    # ── Storage Protocol surface ────────────────────────────────────────

    def save_config(self, run_dir: Path, config: ExperimentConfig) -> None:
        """Write ``input/config.json`` to S3 under the run's key prefix."""
        self._put(run_dir, "input/config.json", config.model_dump_json(indent=2))

    def save_provenance(self, run_dir: Path, provenance: Provenance) -> None:
        """Write ``input/provenance.json`` to S3."""
        self._put(run_dir, "input/provenance.json", provenance.model_dump_json(indent=2))

    def save_result(self, run_dir: Path, result: ExperimentResult) -> None:
        """Write ``output/metrics.json`` to S3."""
        self._put(run_dir, "output/metrics.json", result.model_dump_json(indent=2))

    def save_run_state(self, run_dir: Path, state: RunStateRecord) -> None:
        """Write ``run_state.json`` at the run's key root."""
        self._put(run_dir, "run_state.json", state.model_dump_json(indent=2))

    def load_config(self, run_dir: Path) -> ExperimentConfig:
        """Read back ``input/config.json`` as :class:`ExperimentConfig`."""
        return ExperimentConfig.model_validate_json(self._get(run_dir, "input/config.json"))

    def load_provenance(self, run_dir: Path) -> Provenance:
        """Read back ``input/provenance.json`` as :class:`Provenance`."""
        return Provenance.model_validate_json(self._get(run_dir, "input/provenance.json"))

    def load_result(self, run_dir: Path) -> ExperimentResult:
        """Read back ``output/metrics.json`` as :class:`ExperimentResult`."""
        return ExperimentResult.model_validate_json(self._get(run_dir, "output/metrics.json"))

    def load_run_state(self, run_dir: Path) -> RunStateRecord:
        """Read back ``run_state.json`` as :class:`RunStateRecord`."""
        return RunStateRecord.model_validate_json(self._get(run_dir, "run_state.json"))

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

    # ── internals ──────────────────────────────────────────────────────

    def _key(self, run_dir: Path, rel: str) -> str:
        """Build the canonical S3 key: ``<prefix>/<run_dir.name>/<rel>``."""
        parts = [self._config.prefix, run_dir.name]
        if rel:
            parts.append(rel)
        return "/".join(p.strip("/") for p in parts if p)

    def _put(self, run_dir: Path, rel: str, body: str | bytes) -> None:
        if isinstance(body, str):
            body = body.encode("utf-8")
        self._client.put_object(Bucket=self._config.bucket, Key=self._key(run_dir, rel), Body=body)

    def _get(self, run_dir: Path, rel: str) -> str:
        obj = self._client.get_object(Bucket=self._config.bucket, Key=self._key(run_dir, rel))
        return obj["Body"].read().decode("utf-8")

    def _list(self, prefix: str) -> list[str]:
        """Page-aware list of keys under ``prefix``."""
        keys: list[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._config.bucket, Prefix=prefix):
            for obj in page.get("Contents", []) or []:
                keys.append(obj["Key"])
        return keys
