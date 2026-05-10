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
        _dispatch_local(cfg)
    else:
        _dispatch_ovh(cfg, yaml_path)


def _dispatch_local(cfg: ExperimentConfig) -> None:
    """Local path: build run_dir (mkdir), surface cuda-availability errors, dispatch."""
    # Surface cuda-on-host-without-CUDA *before* mkdir — otherwise the user
    # hits an opaque "Read-only filesystem" error if the YAML's storage path
    # is a container mount (e.g. /workspace/results) that doesn't exist
    # locally.
    if cfg.training.device == "cuda":
        LocalRunner._assert_cuda_available()  # pylint: disable=protected-access
    _, run_dir = build_run_dir(cfg, mkdir=True)
    logger = FileLogger(run_dir / "logs" / "run.log")
    result = submit_to_local(cfg, run_dir=run_dir, logger=logger)
    typer.echo(f"DONE: {result.run_id} -> {result.run_dir}")


def _dispatch_ovh(cfg: ExperimentConfig, yaml_path: Path) -> None:
    """OVH path: load configs/ovh.yaml + resolve repo-relative YAML + submit."""
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

    _, run_dir = build_run_dir(cfg)
    submission = submit_to_ovh(
        cfg,
        ovh_cfg=ovh_cfg,
        yaml_path_in_repo=yaml_in_repo,
        run_dir=run_dir,
        logger=_StdoutLogger(),
    )
    typer.echo(f"SUBMITTED: {submission.run_id} -> job_id={submission.job_id}")
    typer.echo(f"  results: {submission.s3_prefix}")
    typer.echo(f"  dashboard: {submission.dashboard_url}")
    typer.echo(f"  pull back: multi-scenario sweep --follow --runner ovh {yaml_path}")


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
