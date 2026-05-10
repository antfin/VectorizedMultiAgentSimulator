"""``multi-scenario upload-code`` — push the local source tree to the OVH code bucket.

Two backends, picked by argument:

- **OVH CLI** (default; F7.7.A5) — uses ``ovhai bucket object upload`` via
  :class:`OvhClient`. Reads ``configs/ovh.yaml`` for region/bucket. Reuses
  ``ovhai login`` — no AWS-style S3 credentials needed. Matches the
  F7.7.A2 architecture (frontend/preflight already use OvhClient for
  bucket reads).
- **boto3 / S3** (legacy; explicit ``--s3-config`` flag) — uses the
  pre-existing :class:`S3StorageAdapter`. Requires AWS-style keys in
  ``~/.aws/credentials`` or env. Kept for back-compat with anyone whose
  setup already had this wired.

Either way, the file-walk and ``.code_hash`` companion-blob logic live in
:class:`CodeUploader` — only the per-file put callback differs.
"""

from pathlib import Path
from typing import Optional

import typer

from multi_scenario.adapters.storage.code_uploader import CodeUploader

from ._app import app


@app.command(name="upload-code")
def upload_code(
    ovh_config: Path = typer.Option(
        Path("configs/ovh.yaml"),
        "--ovh-config",
        help="OVH job config (region + bucket_code). Used by the default OVH-CLI path.",
    ),
    s3_config: Optional[Path] = typer.Option(
        None,
        "--s3-config",
        help=(
            "Legacy boto3 path: use this S3 config (needs AWS keys). "
            "When provided, overrides --ovh-config and uses S3StorageAdapter."
        ),
    ),
    repo_root: Path = typer.Option(
        Path.cwd(), "--repo-root", help="Repo root to upload from (defaults to CWD)."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print what would upload; no remote calls."
    ),
) -> None:
    """Upload local source tree to the OVH code bucket."""
    if s3_config is not None:
        uploader, target = _build_s3_uploader(s3_config)
    else:
        uploader, target = _build_ovh_uploader(ovh_config)
    files = uploader.upload(repo_root.resolve(), dry_run=dry_run)
    verb = "would upload" if dry_run else "uploaded"
    typer.echo(f"OK {verb} {len(files)} files → {target}")
    if dry_run:
        for rel in files:
            typer.echo(f"  {rel}")


def _build_ovh_uploader(ovh_config: Path) -> tuple[CodeUploader, str]:
    """OVH-CLI uploader from ``configs/ovh.yaml`` (F7.7.A5)."""
    # pylint: disable=import-outside-toplevel
    from multi_scenario.adapters.runners.ovh_cli import OvhClient
    from multi_scenario.domain.models import OvhJobConfig

    if not ovh_config.is_file():
        raise typer.BadParameter(
            f"OVH config not found at {ovh_config}. Either pass --ovh-config "
            f"<path> or create the file (see docs/ovh_setup.md)."
        )
    cfg = OvhJobConfig.from_yaml(ovh_config)
    client = OvhClient()
    client.ensure_available()  # fail-fast if ovhai missing
    uploader = CodeUploader.from_ovh_client(client, cfg.region, cfg.bucket_code)
    target = f"{cfg.bucket_code}@{cfg.region}/"
    return uploader, target


def _build_s3_uploader(s3_config: Path) -> tuple[CodeUploader, str]:
    """Legacy boto3 path. Requires AWS-style keys."""
    # pylint: disable=import-outside-toplevel
    from multi_scenario.adapters.storage.s3 import S3StorageAdapter
    from multi_scenario.domain.models import S3StorageConfig

    cfg = S3StorageConfig.from_yaml(s3_config)
    s3 = S3StorageAdapter(cfg)
    uploader = CodeUploader.from_s3_adapter(s3)
    target = f"s3://{cfg.bucket}/{cfg.prefix}/"
    return uploader, target
