"""``multi-scenario upload-code <s3.yaml>`` — upload local source tree to OVH (F6.4).

Walks the curated include set (``src/multi_scenario``, ``experiments``,
``configs``, plus ``pyproject.toml`` / ``README.md``) and uploads each
surviving file to ``s3://<bucket>/<prefix>/<rel-from-repo-root>``.
"""

from pathlib import Path

import typer

from multi_scenario.adapters.storage.code_uploader import CodeUploader
from multi_scenario.adapters.storage.s3 import S3StorageAdapter
from multi_scenario.domain.models import S3StorageConfig

from ._app import app


@app.command(name="upload-code")
def upload_code(
    s3_config: Path = typer.Argument(..., exists=True, readable=True),
    repo_root: Path = typer.Option(
        Path.cwd(), "--repo-root", help="Repo root to upload from (defaults to CWD)."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print what would upload; no S3 calls."
    ),
) -> None:
    """Upload local source tree to the OVH code bucket (F6.4)."""
    cfg = S3StorageConfig.from_yaml(s3_config)
    s3 = S3StorageAdapter(cfg)
    uploader = CodeUploader(s3)
    files = uploader.upload(repo_root.resolve(), dry_run=dry_run)
    verb = "would upload" if dry_run else "uploaded"
    typer.echo(f"OK {verb} {len(files)} files → s3://{cfg.bucket}/{cfg.prefix}/")
    if dry_run:
        for rel in files:
            typer.echo(f"  {rel}")
