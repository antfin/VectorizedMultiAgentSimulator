"""``multi-scenario sweep <input>`` — run a YAML batch optionally × seeds (F5.6 + F6.7).

CLI-level expansion: ``input`` can be a single yaml, a directory, or a glob.
With ``--seeds N1,N2,...`` each matched yaml is cartesian-multiplied by the
seed list. ``--dry-run`` previews; otherwise each cell runs sequentially via
``LocalRunner`` (``--runner local``) or fan-outs as parallel OVH jobs
(``--runner ovh``, F6.7).

Per-cell S3 prefix isolation comes from ``OvhRunner._build_submit_args``,
which uses ``<run_id>`` (= ``<exp_id>_s<seed>``, distinct per cell). This is
what prevents the rendezvous_comm 2026-04-16 collision bug where parallel
jobs FINALIZING into the same prefix overwrote each other.
"""

import glob
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer

from multi_scenario.adapters.logging.file_logger import FileLogger
from multi_scenario.adapters.runners.local import LocalRunner
from multi_scenario.domain.models import ExperimentConfig, RunId

from ._app import app


# Sweep command has many orthogonal flags (input, seeds, dry-run, max-runs,
# seconds-per-run); typer flattens them onto one signature, hence the count.
# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
@app.command()
def sweep(
    input_path: str = typer.Argument(
        ...,
        help="Single yaml, a directory, or a glob pattern (e.g. 'configs/abs_*.yaml').",
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
    runner_kind: str = typer.Option(
        "local",
        "--runner",
        help="'local' (sequential, default) or 'ovh' (F6.7: parallel OVH submission).",
    ),
    ovh_config: Path = typer.Option(
        None,
        "--ovh-config",
        help="Path to configs/ovh.yaml (required when --runner ovh).",
    ),
    follow: bool = typer.Option(
        False,
        "--follow",
        help="With --runner ovh: poll all jobs until terminal; print per-cell DONE/FAILED.",
    ),
    max_parallel: int = typer.Option(
        0,
        "--max-parallel",
        help="With --runner ovh: cap concurrent OVH jobs (0 = unlimited).",
    ),
    confirm_yes: bool = typer.Option(
        False,
        "--yes",
        help="With --runner ovh: bypass cost-cap confirmation prompt.",
    ),
    repo_root: Path = typer.Option(
        Path.cwd(),
        "--repo-root",
        help="With --runner ovh: repo root used to resolve yaml paths relative to bucket_code.",
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
    """Run multiple experiments by globbing YAMLs and (optionally) sweeping seeds (F5.6 + F6.7)."""
    yaml_paths = _resolve_input(input_path)
    if not yaml_paths:
        typer.echo(f"✗ no yaml matched: {input_path}", err=True)
        raise typer.Exit(code=1)

    seed_list = _parse_seeds(seeds)
    cells = _expand_cells(yaml_paths, seed_list, runner_kind=runner_kind)

    if len(cells) > max_runs:
        typer.echo(
            f"✗ expansion has {len(cells)} cells, exceeds --max-runs={max_runs}.",
            err=True,
        )
        raise typer.Exit(code=2)

    typer.echo(
        f"sweep: {len(cells)} cell(s) from {len(yaml_paths)} yaml(s) (runner={runner_kind})"
    )
    if seconds_per_run > 0:
        typer.echo(
            f"  estimated wall time: {_format_estimate(len(cells), seconds_per_run)}"
        )
    for i, (cfg, run_dir, _yaml) in enumerate(cells, start=1):
        typer.echo(
            f"  [{i}/{len(cells)}] {cfg.experiment.id}_s{cfg.experiment.seed} -> {run_dir}"
        )

    if runner_kind == "ovh":
        _sweep_run_ovh(
            cells=cells,
            ovh_config_path=ovh_config,
            dry_run=dry_run,
            follow=follow,
            max_parallel=max_parallel,
            confirm_yes=confirm_yes,
            seconds_per_run=seconds_per_run,
            repo_root=repo_root.resolve(),
        )
        return

    if runner_kind != "local":
        typer.echo(
            f"✗ unknown --runner {runner_kind!r}; expected 'local' or 'ovh'.", err=True
        )
        raise typer.Exit(code=2)

    if dry_run:
        return

    for i, (cfg, run_dir, _yaml) in enumerate(cells, start=1):
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
    yaml_paths: list[Path],
    seed_list: list[int],
    runner_kind: str = "local",
) -> list[tuple[ExperimentConfig, Path, Path]]:
    """Build the list of (config, run_dir, yaml_path) cells from yaml paths × seeds.

    When ``seed_list`` is empty, each yaml's own ``experiment.seed`` is used (one
    cell per yaml). Otherwise cartesian: each yaml × each seed, with the yaml's
    seed *replaced* per cell. ``yaml_path`` is the source yaml file — used by
    the OVH runner as ``yaml_path_in_repo`` (F6.7).

    For ``runner_kind == "ovh"``, ``run_dir`` resolution doesn't matter for the
    submission (OvhRunner derives its own per-cell S3 prefix from ``run_id``);
    we still build a placeholder for consistency.
    """
    del runner_kind  # currently same expansion either way; reserved for future divergence.
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    cells: list[tuple[ExperimentConfig, Path, Path]] = []
    for yaml_path in yaml_paths:
        base_cfg = ExperimentConfig.from_yaml(yaml_path)
        seeds_for_yaml = seed_list if seed_list else [base_cfg.experiment.seed]
        for seed in seeds_for_yaml:
            cfg = base_cfg.model_copy(deep=True)
            cfg.experiment.seed = seed
            run_id = RunId(exp_id=cfg.experiment.id, seed=seed)
            storage_root = (
                Path(cfg.runtime.storage.path)
                if cfg.runtime is not None
                else Path("experiments")
            )
            run_dir = storage_root / run_id.folder_name(timestamp)
            cells.append((cfg, run_dir, yaml_path))
    return cells


def _sweep_run_ovh(
    cells: list[tuple[ExperimentConfig, Path, Path]],
    ovh_config_path: Path | None,
    dry_run: bool,
    follow: bool,
    max_parallel: int,
    confirm_yes: bool,
    seconds_per_run: float,
    repo_root: Path,
) -> None:
    """OVH path of ``multi-scenario sweep`` (F6.7) — submit each cell, optionally poll."""
    # F6.7's CLI extension folds many orthogonal flags through a single helper;
    # bundling them into a config object would obscure the docstring's contract.
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,import-outside-toplevel
    from multi_scenario.adapters.runners.ovh import OvhRunner
    from multi_scenario.adapters.runners.ovh_cli import OvhClient, OvhCliError
    from multi_scenario.adapters.secrets.fernet import FernetSecretsAdapter
    from multi_scenario.domain.models import OvhJobConfig

    if ovh_config_path is None:
        typer.echo("✗ --ovh-config required when --runner ovh.", err=True)
        raise typer.Exit(code=2)
    # Friendly error if `ovhai` is missing — surfaces install instructions
    # instead of a bare FileNotFoundError traceback (F6.7.1).
    try:
        OvhClient().ensure_available()
    except OvhCliError as exc:
        typer.echo(f"✗ {exc}", err=True)
        raise typer.Exit(code=2) from exc
    ovh_cfg = OvhJobConfig.from_yaml(ovh_config_path)

    cost = (
        ovh_cfg.estimate_cost_eur(seconds_per_run / 3600.0)
        if seconds_per_run > 0
        else None
    )
    if cost is not None:
        total = cost * len(cells)
        typer.echo(
            f"  estimated cost: {len(cells)} cells × {cost:.3f} EUR ≈ {total:.2f} EUR"
        )
        if total > ovh_cfg.cost_cap_eur and not confirm_yes:
            typer.echo(
                f"✗ estimated cost {total:.2f} EUR exceeds cap {ovh_cfg.cost_cap_eur:.2f} EUR. "
                "Pass --yes to proceed.",
                err=True,
            )
            raise typer.Exit(code=2)

    if dry_run:
        return

    secrets = FernetSecretsAdapter()
    job_ids: list[tuple[str, ExperimentConfig, Path]] = []
    for i, (cfg, run_dir, yaml_path) in enumerate(cells, start=1):
        # If the user submits at parallel cap, wait for room before next submit.
        if max_parallel > 0:
            _wait_for_capacity(job_ids, max_parallel, OvhClient())
        runner = OvhRunner(
            ovh_config=ovh_cfg,
            client=OvhClient(),
            secrets=secrets,
            logger=_StdoutLogger(),
            yaml_path_in_repo=str(yaml_path.resolve().relative_to(repo_root)),
        )
        job_id = runner.submit(cfg, run_dir)
        run_id = f"{cfg.experiment.id}_s{cfg.experiment.seed}"
        typer.echo(f"  [{i}/{len(cells)}] submitted {run_id} → job_id={job_id}")
        job_ids.append((job_id, cfg, run_dir))

    if not follow:
        return
    _follow_ovh_jobs(job_ids, OvhClient(), ovh_cfg=ovh_cfg)


def _wait_for_capacity(
    in_flight: list[tuple[str, ExperimentConfig, Path]],
    max_parallel: int,
    client: Any,
) -> None:
    """Block until fewer than ``max_parallel`` jobs are non-terminal."""
    while True:
        active = [j for j in in_flight if not _job_is_terminal(client, j[0])]
        if len(active) < max_parallel:
            return
        time.sleep(5)


def _follow_ovh_jobs(
    job_ids: list[tuple[str, ExperimentConfig, Path]],
    client: Any,
    *,
    ovh_cfg: Any = None,
) -> None:
    """Poll ``ovhai job get`` until every submitted job reaches a terminal state.

    When ``ovh_cfg`` is supplied, each DONE job triggers a Stage 2 pullback —
    the OVH-side run-folder is materialised at the same local path local runs
    use, so post-sweep tooling (Streamlit, regenerate-videos, eval) sees OVH
    cells indistinguishably from local cells.
    """
    # pylint: disable=import-outside-toplevel
    from multi_scenario.application.ovh_pullback import pullback_run_dir
    from multi_scenario.application.regenerate_videos import (
        latest_checkpoint,
        regenerate_videos,
    )

    pending = list(job_ids)
    while pending:
        still_running = []
        for job_id, cfg, run_dir in pending:
            info = client.get(job_id)
            if info.is_terminal:
                run_id = f"{cfg.experiment.id}_s{cfg.experiment.seed}"
                marker = "DONE" if info.state.upper() == "DONE" else f"{info.state}"
                typer.echo(f"  {marker} {run_id} → {run_dir}")
                if info.state.upper() == "DONE" and ovh_cfg is not None:
                    pulled_ok = False
                    try:
                        result = pullback_run_dir(
                            ovh_cfg=ovh_cfg,
                            run_dir_name=run_dir.name,
                            dest_dir=run_dir,
                            client=client,
                        )
                        typer.echo(
                            f"    pulled {result.n_downloaded} files "
                            f"({result.n_skipped} skipped) → {run_dir}"
                        )
                        pulled_ok = True
                    # pylint: disable=broad-except
                    except Exception as exc:
                        # Don't stall the whole sweep on a single pullback hiccup;
                        # surface the failure and let the user retry manually.
                        typer.echo(
                            f"    ✗ pullback failed for {run_id}: {exc}",
                            err=True,
                        )
                    # F8.2.D: if results landed and the run-folder has no
                    # videos yet, regenerate them locally — same behaviour as
                    # Streamlit Refresh. Best-effort: a regen failure here
                    # doesn't stall the sweep (results are already on disk;
                    # user can retry via the Run Detail button).
                    # F8.2.H: skip silently when there's no checkpoint to
                    # load from (smoke runs disable checkpoints by design).
                    if (
                        pulled_ok
                        and not _videos_present(run_dir)
                        and latest_checkpoint(run_dir) is not None
                    ):
                        try:
                            regenerate_videos(run_dir)
                            typer.echo(
                                f"    regenerated videos → {run_dir}/output/videos"
                            )
                        # pylint: disable=broad-except
                        except Exception as exc:
                            typer.echo(
                                f"    ✗ video regen failed for {run_id}: {exc}",
                                err=True,
                            )
            else:
                still_running.append((job_id, cfg, run_dir))
        pending = still_running
        if pending:
            time.sleep(15)


def _videos_present(run_dir: Path) -> bool:
    """True iff ``run_dir/output/videos/`` has ≥1 mp4 — used by F8.2.D regen gate."""
    videos_dir = run_dir / "output" / "videos"
    if not videos_dir.is_dir():
        return False
    return any(p.suffix.lower() == ".mp4" for p in videos_dir.iterdir())


def _job_is_terminal(client: Any, job_id: str) -> bool:
    """True when ``ovhai job get <job_id>`` reports a terminal state."""
    return client.get(job_id).is_terminal


class _StdoutLogger:
    """Minimal ``Logger`` impl for the sweep CLI (no file binding required)."""

    # Minimal protocol-fake-style class; pylint's defaults flag every method.
    # pylint: disable=missing-function-docstring,unused-argument

    def info(self, msg: str) -> None:
        typer.echo(msg)

    def debug(self, msg: str) -> None:
        return None

    def warning(self, msg: str) -> None:
        typer.echo(msg, err=True)

    def error(self, msg: str) -> None:
        typer.echo(msg, err=True)


def _format_estimate(n_cells: int, seconds_per_run: float) -> str:
    """Format a cells × seconds wall-time estimate (returns a human-readable string)."""
    total_sec = n_cells * seconds_per_run
    if total_sec < 60:
        return f"{n_cells} × {seconds_per_run:.0f}s = {total_sec:.0f}s"
    minutes, secs = divmod(int(total_sec), 60)
    return (
        f"{n_cells} × {seconds_per_run:.0f}s = {total_sec:.0f}s ({minutes}m{secs:02d}s)"
    )
