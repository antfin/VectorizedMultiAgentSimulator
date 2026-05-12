"""``multi-scenario run <yaml>`` — execute one experiment.

Dispatches by ``cfg.runtime.runner.type`` (F7.7.A4):

- ``local`` → :class:`LocalRunner` runs training in this Python process.
- ``ovh``   → :class:`OvhRunner` shells out ``ovhai job run`` which boots
  an OVH container that internally runs ``LocalRunner``.

The CLI ``--runner`` flag overrides the YAML when set — useful for forcing
a YAML to run locally for debugging without editing the file.

Both branches delegate to :mod:`application.submission` so the Streamlit
Submit page and this CLI share identical orchestration (F7.7.A6 hex
compliance — the application layer owns the use case, callers only handle
caller-specific concerns: error wrapping + post-submission UX).

The run folder is built at ``<storage.path>/<run_id>__<timestamp>`` via
:func:`application.submission.build_run_dir`. For OVH runs, ``storage.path``
MUST be the in-container mount path (e.g. ``/workspace/results`` per
``OvhJobConfig.mount_results``).
"""

from pathlib import Path
from typing import Optional

import typer

from multi_scenario.adapters.logging.file_logger import FileLogger
from multi_scenario.adapters.runners.local import LocalRunner
from multi_scenario.application.config_validation import validate_known_types
from multi_scenario.application.submission import (
    build_run_dir,
    submit_to_local,
    submit_to_ovh,
)
from multi_scenario.domain.models import ExperimentConfig

from ._app import app


@app.command()
def run(
    yaml_path: Path = typer.Argument(..., exists=True, readable=True),
    runner_override: Optional[str] = typer.Option(
        None,
        "--runner",
        help=(
            "Override the YAML's ``runtime.runner.type``. Values: "
            "``local`` | ``ovh``. Useful for forcing a local debug run "
            "of an OVH-targeted YAML (or vice versa) without editing it."
        ),
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help=(
            "Emit a single JSON record as the final stdout line: "
            "``{run_id, job_id?, run_dir, s3_prefix?, dashboard_url?, runner}``. "
            "Use for scripted submissions / programmatic pipelines."
        ),
    ),
) -> None:
    """Execute one experiment run from a YAML config file."""
    cfg = ExperimentConfig.from_yaml(yaml_path)
    validate_known_types(cfg)

    runner_type = runner_override or (
        cfg.runtime.runner.type if cfg.runtime is not None else "local"
    )
    if runner_type not in ("local", "ovh"):
        raise typer.BadParameter(
            f"--runner must be 'local' or 'ovh', got {runner_type!r}"
        )

    if runner_type == "local":
        _dispatch_local(cfg, emit_json=json_output)
    else:
        _dispatch_ovh(cfg, yaml_path, emit_json=json_output)


def _dispatch_local(cfg: ExperimentConfig, *, emit_json: bool = False) -> None:
    """Local path: build run_dir (mkdir), surface cuda-availability errors, dispatch."""
    # pylint: disable=import-outside-toplevel
    import json as _json

    # Surface cuda-on-host-without-CUDA *before* mkdir — otherwise the user
    # hits an opaque "Read-only filesystem" error if the YAML's storage path
    # is a container mount (e.g. /workspace/results) that doesn't exist
    # locally.
    if cfg.training.device == "cuda":
        LocalRunner._assert_cuda_available()  # pylint: disable=protected-access
    _, run_dir = build_run_dir(cfg, mkdir=True)
    logger = FileLogger(run_dir / "logs" / "run.log")
    result = submit_to_local(cfg, run_dir=run_dir, logger=logger)
    if emit_json:
        # B: parseable line for scripted / chat-trigger workflows.
        typer.echo(
            _json.dumps(
                {
                    "runner": "local",
                    "run_id": str(result.run_id),
                    "run_dir": str(result.run_dir),
                }
            )
        )
    else:
        typer.echo(f"DONE: {result.run_id} -> {result.run_dir}")


def _dispatch_ovh(
    cfg: ExperimentConfig, yaml_path: Path, *, emit_json: bool = False
) -> None:
    """OVH path: load configs/ovh.yaml + resolve repo-relative YAML + submit.

    Phase 9 (#12): auto-uploads the local code tree when the bucket's
    ``.code_hash`` differs from the local hash. First Phase 3 attempt
    failed because the newly-created YAML wasn't in the bucket — this
    closes that gap so the user doesn't have to remember to
    ``upload-code`` after editing.
    """
    # pylint: disable=import-outside-toplevel
    from multi_scenario.domain.models import OvhJobConfig

    ovh_cfg_path = Path("configs/ovh.yaml")
    if not ovh_cfg_path.is_file():
        raise typer.BadParameter(
            f"runtime.runner.type=ovh requires {ovh_cfg_path} (missing). "
            "Create it with your OVH region/buckets/image — see docs/ovh_setup.md."
        )
    ovh_cfg = OvhJobConfig.from_yaml(ovh_cfg_path)

    # Resolve the YAML path relative to the repo root so it matches the
    # container's bucket-mounted path (``mount_code/<rel_path>``).
    repo_root = Path.cwd().resolve()
    try:
        yaml_in_repo = yaml_path.resolve().relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise typer.BadParameter(
            f"yaml_path {yaml_path} must live under the repo root "
            f"({repo_root}); OVH submission needs the repo-relative path."
        ) from exc

    # Auto-upload code when the local tree differs from the bucket.
    _maybe_upload_stale_code(ovh_cfg)

    _, run_dir = build_run_dir(cfg)
    submission = submit_to_ovh(
        cfg,
        ovh_cfg=ovh_cfg,
        yaml_path_in_repo=yaml_in_repo,
        run_dir=run_dir,
        logger=_StdoutLogger(),
    )
    if emit_json:
        # pylint: disable=import-outside-toplevel
        import json as _json

        typer.echo(
            _json.dumps(
                {
                    "runner": "ovh",
                    "run_id": str(submission.run_id),
                    "job_id": submission.job_id,
                    "run_dir": str(submission.run_dir),
                    "s3_prefix": submission.s3_prefix,
                    "dashboard_url": submission.dashboard_url,
                }
            )
        )
    else:
        typer.echo(f"SUBMITTED: {submission.run_id} -> job_id={submission.job_id}")
        typer.echo(f"  results: {submission.s3_prefix}")
        typer.echo(f"  dashboard: {submission.dashboard_url}")
        typer.echo(
            f"  pull back: multi-scenario sweep --follow --runner ovh {yaml_path}"
        )


def _maybe_upload_stale_code(ovh_cfg) -> None:
    """Compare the local code hash to the bucket's; upload if different.

    No-op when the bucket already has the current code (idempotent
    submits don't re-upload). When the bucket can't be queried (e.g.
    first ever submit, before anything was uploaded), assumes stale
    and uploads.

    Phase 9 fix: pre-Phase 9, the user had to run ``multi-scenario
    upload-code`` separately whenever a YAML or src file changed.
    First Phase 3 attempt failed because the new ``_lero_smoke.yaml``
    wasn't on the bucket — this closes the gap.
    """
    # pylint: disable=import-outside-toplevel
    from multi_scenario.adapters.runners.ovh_cli import OvhClient
    from multi_scenario.adapters.storage.code_uploader import (
        CODE_HASH_KEY,
        CodeUploader,
        compute_local_code_hash,
    )

    repo_root = Path.cwd().resolve()
    local_hash = compute_local_code_hash(repo_root)
    client = OvhClient()
    try:
        remote_hash_blob = client.bucket_get_object(
            ovh_cfg.region, ovh_cfg.bucket_code, CODE_HASH_KEY
        )
        remote_hash = remote_hash_blob.decode("utf-8").strip()
    except Exception:  # pylint: disable=broad-except
        # No .code_hash on the bucket → first-ever upload or transient
        # error. Upload defensively so the job doesn't see stale code.
        remote_hash = None

    if remote_hash == local_hash:
        typer.echo(f"code bucket up-to-date (hash={local_hash[:12]}…); skipping upload")
        return

    if remote_hash is None:
        typer.echo("no .code_hash on bucket — uploading code…")
    else:
        typer.echo(
            f"code hash drifted (local={local_hash[:12]}… remote={remote_hash[:12]}…); "
            "uploading…"
        )
    uploader = CodeUploader.from_ovh_client(client, ovh_cfg.region, ovh_cfg.bucket_code)
    files = uploader.upload(repo_root, dry_run=False)
    typer.echo(f"  uploaded {len(files)} files")


class _StdoutLogger:
    """Bare logger for CLI dispatch — pipes to stdout via ``typer.echo``."""

    # pylint: disable=missing-function-docstring,missing-class-docstring
    def info(self, msg: str) -> None:
        typer.echo(msg)

    def debug(self, msg: str) -> None:  # noqa: ARG002
        pass  # CLI doesn't surface debug

    def warning(self, msg: str) -> None:
        typer.echo(f"WARN: {msg}")

    def error(self, msg: str) -> None:
        typer.echo(f"ERROR: {msg}", err=True)
