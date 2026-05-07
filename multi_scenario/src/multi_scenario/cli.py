"""multi_scenario CLI — ``run``, ``validate``, ``consolidate``, ``sweep``.

``run`` wires:

- ``ExperimentConfig.from_yaml`` (F1.1) for parsing.
- ``LocalRunner`` (F2.6) with default ``ProvenanceWriter`` (F2.7).
- ``FileLogger`` (F2.7) pointed at ``<run_dir>/logs/run.log``.

The run folder is built at ``<storage.path>/<run_id>__<timestamp>`` using
``RunId`` (F1.3).

``validate`` (F5.1) parses the YAML, runs it through ``ExperimentConfig``
strict validation, and exits non-zero with one readable line per error
(``<dotted.field.path>: <message>``). Used as a pre-flight check before
expensive OVH submits or long local sweeps.

``sweep`` (F5.6) is CLI-level expansion over per-experiment YAMLs: ``input``
can be a single yaml, a directory, or a glob. With ``--seeds N1,N2,...`` each
matched yaml is cartesian-multiplied by the seed list. ``--dry-run`` previews;
otherwise each cell runs sequentially via ``LocalRunner``.
"""

import glob
from datetime import datetime, timezone
from pathlib import Path

import typer
from pydantic import ValidationError

from multi_scenario import __version__
from multi_scenario.adapters.logging.file_logger import FileLogger
from multi_scenario.adapters.runners.local import LocalRunner
from multi_scenario.adapters.storage.code_uploader import CodeUploader
from multi_scenario.adapters.storage.local import LocalStorageAdapter
from multi_scenario.adapters.storage.runs_csv import RunsCsvWriter
from multi_scenario.adapters.storage.runs_json import RunsJsonWriter
from multi_scenario.adapters.storage.s3 import S3StorageAdapter
from multi_scenario.application.factories import make_algorithm, make_scenario
from multi_scenario.domain.models import (
    EvalRunRecord,
    ExperimentConfig,
    RunId,
    RunState,
    S3StorageConfig,
)

app = typer.Typer(help="multi_scenario CLI")


@app.command()
def version() -> None:
    """Print the multi_scenario package version.

    Also forces typer's multi-command mode so ``run`` stays a real subcommand.
    """
    typer.echo(__version__)


@app.command()
def run(
    yaml_path: Path = typer.Argument(..., exists=True, readable=True),
) -> None:
    """Execute one experiment run from a YAML config file."""
    cfg = ExperimentConfig.from_yaml(yaml_path)

    run_id = RunId(exp_id=cfg.experiment.id, seed=cfg.experiment.seed)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    storage_root = (
        Path(cfg.runtime.storage.path) if cfg.runtime is not None else Path("experiments")
    )
    run_dir = storage_root / run_id.folder_name(timestamp)
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = FileLogger(run_dir / "logs" / "run.log")
    runner = LocalRunner(logger=logger)  # default ProvenanceWriter() from F2.7
    result = runner.run(cfg, run_dir=run_dir)
    typer.echo(f"DONE: {result.run_id} -> {run_dir}")


@app.command()
def validate(
    yaml_path: Path = typer.Argument(..., exists=True, readable=True),
) -> None:
    """Validate a YAML against the ``ExperimentConfig`` schema; exit 1 on any error.

    Pre-flight check before submitting OVH jobs or launching long local sweeps.
    Errors are formatted as one ``<dotted.field.path>: <message>`` line per issue.
    """
    try:
        ExperimentConfig.from_yaml(yaml_path)
    except ValidationError as exc:
        typer.echo(f"✗ {yaml_path}: invalid", err=True)
        for err in exc.errors():
            path = ".".join(str(p) for p in err["loc"])
            typer.echo(f"  {path}: {err['msg']}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"OK {yaml_path}")


@app.command()
def consolidate(
    exp_type_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
) -> None:
    """Build ``runs.csv`` + ``runs.json`` from all DONE runs (F5.2 + F5.3)."""
    csv_path = RunsCsvWriter().consolidate(exp_type_dir)
    json_path = RunsJsonWriter().consolidate(exp_type_dir)
    typer.echo(f"OK runs.csv -> {csv_path}")
    typer.echo(f"OK runs.json -> {json_path}")


# Sweep config has many orthogonal flags (input, seeds, dry-run, max-runs,
# seconds-per-run); typer flattens them onto one signature, hence the count.
# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
@app.command()
def sweep(
    input_path: str = typer.Argument(
        ..., help="Single yaml, a directory, or a glob pattern (e.g. 'configs/abs_*.yaml')."
    ),
    seeds: str = typer.Option(
        "",
        "--seeds",
        help=(
            "Comma-separated seed list, e.g. '0,1,2'. When set, each matched yaml "
            "is cartesian-multiplied by these seeds (the yaml's own experiment.seed "
            "is replaced per cell). When empty, each yaml's own seed is used."
        ),
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the expansion and exit; do not run anything."
    ),
    max_runs: int = typer.Option(
        100, "--max-runs", help="Refuse to launch if the expansion exceeds this cap."
    ),
    seconds_per_run: float = typer.Option(
        0.0,
        "--seconds-per-run",
        help="If set, print a wall-time estimate (cells × seconds_per_run).",
    ),
) -> None:
    """Run multiple experiments by globbing YAMLs and (optionally) sweeping seeds (F5.6)."""
    yaml_paths = _resolve_input(input_path)
    if not yaml_paths:
        typer.echo(f"✗ no yaml matched: {input_path}", err=True)
        raise typer.Exit(code=1)

    seed_list = _parse_seeds(seeds)
    cells = _expand_cells(yaml_paths, seed_list)

    if len(cells) > max_runs:
        typer.echo(
            f"✗ expansion has {len(cells)} cells, exceeds --max-runs={max_runs}.",
            err=True,
        )
        raise typer.Exit(code=2)

    typer.echo(f"sweep: {len(cells)} cell(s) from {len(yaml_paths)} yaml(s)")
    if seconds_per_run > 0:
        typer.echo(f"  estimated wall time: {_format_estimate(len(cells), seconds_per_run)}")
    for i, (cfg, run_dir) in enumerate(cells, start=1):
        typer.echo(f"  [{i}/{len(cells)}] {cfg.experiment.id}_s{cfg.experiment.seed} -> {run_dir}")
    if dry_run:
        return

    for i, (cfg, run_dir) in enumerate(cells, start=1):
        run_dir.mkdir(parents=True, exist_ok=True)
        logger = FileLogger(run_dir / "logs" / "run.log")
        runner = LocalRunner(logger=logger)
        result = runner.run(cfg, run_dir=run_dir)
        typer.echo(f"  [{i}/{len(cells)}] DONE {result.run_id}")


def _resolve_input(input_path: str) -> list[Path]:
    """Resolve sweep input to a sorted list of yaml paths.

    Resolution rules: existing file → [file]; existing directory → ``<dir>/*.yaml``;
    otherwise treat as a glob pattern via :func:`glob.glob`. Filters to ``.yaml``
    extension only.
    """
    p = Path(input_path)
    if p.is_file():
        return [p] if p.suffix == ".yaml" else []
    if p.is_dir():
        return sorted(p.glob("*.yaml"))
    matches = [Path(m) for m in glob.glob(input_path, recursive=True)]
    return sorted(m for m in matches if m.is_file() and m.suffix == ".yaml")


def _parse_seeds(seeds: str) -> list[int]:
    """Parse a comma-separated seed string into a list of ints; empty input → []."""
    seeds = seeds.strip()
    if not seeds:
        return []
    return [int(s.strip()) for s in seeds.split(",") if s.strip()]


def _expand_cells(
    yaml_paths: list[Path], seed_list: list[int]
) -> list[tuple[ExperimentConfig, Path]]:
    """Build the list of (config, run_dir) cells from yaml paths × seeds.

    When ``seed_list`` is empty, each yaml's own ``experiment.seed`` is used (one
    cell per yaml). Otherwise cartesian: each yaml × each seed, with the yaml's
    seed *replaced* per cell.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    cells: list[tuple[ExperimentConfig, Path]] = []
    for yaml_path in yaml_paths:
        base_cfg = ExperimentConfig.from_yaml(yaml_path)
        seeds_for_yaml = seed_list if seed_list else [base_cfg.experiment.seed]
        for seed in seeds_for_yaml:
            cfg = base_cfg.model_copy(deep=True)
            cfg.experiment.seed = seed
            run_id = RunId(exp_id=cfg.experiment.id, seed=seed)
            storage_root = (
                Path(cfg.runtime.storage.path) if cfg.runtime is not None else Path("experiments")
            )
            run_dir = storage_root / run_id.folder_name(timestamp)
            cells.append((cfg, run_dir))
    return cells


def _format_estimate(n_cells: int, seconds_per_run: float) -> str:
    """Format a cells × seconds wall-time estimate (returns a human-readable string)."""
    total_sec = n_cells * seconds_per_run
    if total_sec < 60:
        return f"{n_cells} × {seconds_per_run:.0f}s = {total_sec:.0f}s"
    minutes, secs = divmod(int(total_sec), 60)
    return f"{n_cells} × {seconds_per_run:.0f}s = {total_sec:.0f}s ({minutes}m{secs:02d}s)"


@app.command()
def resume(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
) -> None:
    """Resume a crashed run from its latest checkpoint (F5.7, local-only).

    Refuses (exit 2) if the cfg's runner doesn't support resume, or if the run
    is already DONE, or if no checkpoint is on disk yet.
    """
    storage = LocalStorageAdapter()
    cfg = storage.load_config(run_dir)
    runner_type = cfg.runtime.runner.type if cfg.runtime is not None else "local"
    # Capability check: build the runner via factory to read its `supports_resume` flag.
    if runner_type != "local":
        typer.echo(
            f"✗ resume is only supported for local runners (cfg has runner.type={runner_type!r}). "
            "To rerun on OVH, submit a fresh job.",
            err=True,
        )
        raise typer.Exit(code=2)

    state = storage.load_run_state(run_dir)
    if state.state == RunState.DONE:
        typer.echo(f"✗ run is already DONE: {run_dir}", err=True)
        raise typer.Exit(code=2)

    checkpoint = _latest_checkpoint(run_dir)
    if checkpoint is None:
        typer.echo(f"✗ no BenchMARL checkpoint found under {run_dir}/output/benchmarl/", err=True)
        raise typer.Exit(code=2)

    now = datetime.now(timezone.utc)
    if state.state != RunState.CRASHED:
        state = state.transition_to(RunState.CRASHED, now)
        storage.save_run_state(run_dir, state)
    state = state.transition_to(RunState.RESUMED, now)
    storage.save_run_state(run_dir, state)

    logger = FileLogger(run_dir / "logs" / "run.log")
    runner = LocalRunner(logger=logger)
    result = runner.run(cfg, run_dir=run_dir, resume_from=checkpoint)
    typer.echo(f"DONE: {result.run_id} -> {run_dir}")


def _latest_checkpoint(run_dir: Path) -> Path | None:
    """Locate the most-recent ``*.pt`` under ``run_dir/output/benchmarl/.../checkpoints/``."""
    bm_root = run_dir / "output" / "benchmarl"
    if not bm_root.is_dir():
        return None
    pts = list(bm_root.rglob("checkpoints/*.pt"))
    if not pts:
        return None
    return max(pts, key=lambda p: p.stat().st_mtime)


@app.command(name="eval")
def eval_only(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    episodes: int = typer.Option(
        0, "--episodes", help="Override cfg.evaluation.episodes (0 = keep original)."
    ),
    name: str = typer.Option(
        "", "--name", help="Output filename tag; default = 'eval_<UTC_timestamp>'."
    ),
) -> None:
    """Re-evaluate a trained policy without retraining (F5.8, local-only).

    Loads the latest BenchMARL checkpoint under ``<run_dir>``, rebuilds the
    experiment, runs eval (with optional ``--episodes`` override), and writes
    ``<run_dir>/output/eval_runs/<TAG>.json``. Multiple eval runs coexist as
    separate files; default tag is timestamped.
    """
    config_path = run_dir / "input" / "config.json"
    if not config_path.is_file():
        typer.echo(f"✗ no input/config.json under {run_dir}", err=True)
        raise typer.Exit(code=2)

    storage = LocalStorageAdapter()
    cfg = storage.load_config(run_dir)
    if episodes > 0:
        cfg.evaluation.episodes = episodes

    checkpoint = _latest_checkpoint(run_dir)
    if checkpoint is None:
        typer.echo(f"✗ no BenchMARL checkpoint under {run_dir}/output/benchmarl/", err=True)
        raise typer.Exit(code=2)

    # Reconstruct the experiment from the checkpoint, run eval through the
    # algorithm adapter (reusing F2.4.3 aggregation), score via the metrics bundle.
    algorithm = make_algorithm(cfg.algorithm.type)
    scenario = make_scenario(cfg.scenario.type)
    # Local import to avoid pulling BenchMARL into the cli module top-level.
    # pylint: disable=import-outside-toplevel
    from benchmarl.experiment import Experiment

    from multi_scenario.adapters.metrics.common import CommonMetricsBundle

    experiment = Experiment.reload_from_file(str(checkpoint))
    rollout = algorithm.evaluate(experiment, env=None, cfg=cfg, run_dir=run_dir)
    metric_dict = CommonMetricsBundle().compute(rollout, scenario)

    eval_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    eval_id = name or f"eval_{eval_timestamp}"
    run_id = RunId(exp_id=cfg.experiment.id, seed=cfg.experiment.seed)
    record = EvalRunRecord(
        eval_id=eval_id,
        run_id=str(run_id),
        scenario=cfg.scenario.type,
        algorithm=cfg.algorithm.type,
        seed=cfg.experiment.seed,
        eval_timestamp=eval_timestamp,
        n_eval_episodes=cfg.evaluation.episodes,
        metrics=metric_dict,
        policy_checkpoint=checkpoint.relative_to(run_dir).as_posix(),
    )
    storage.save_eval_run(run_dir, record)
    typer.echo(f"OK eval -> {run_dir / 'output' / 'eval_runs' / (eval_id + '.json')}")


@app.command(name="upload-code")
def upload_code(
    s3_config: Path = typer.Argument(..., exists=True, readable=True),
    repo_root: Path = typer.Option(
        Path.cwd(), "--repo-root", help="Repo root to upload from (defaults to CWD)."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print what would upload; no S3 calls."),
) -> None:
    """Upload local source tree to the OVH code bucket (F6.4).

    Walks the curated include set (``src/multi_scenario``, ``experiments``,
    ``configs``, plus ``pyproject.toml`` / ``README.md``) and uploads each
    surviving file to ``s3://<bucket>/<prefix>/<rel-from-repo-root>``.
    """
    cfg = S3StorageConfig.from_yaml(s3_config)
    s3 = S3StorageAdapter(cfg)
    uploader = CodeUploader(s3)
    files = uploader.upload(repo_root.resolve(), dry_run=dry_run)
    verb = "would upload" if dry_run else "uploaded"
    typer.echo(f"OK {verb} {len(files)} files → s3://{cfg.bucket}/{cfg.prefix}/")
    if dry_run:
        for rel in files:
            typer.echo(f"  {rel}")


def main() -> None:
    """Entry point used by the ``multi-scenario`` console script."""
    app()
