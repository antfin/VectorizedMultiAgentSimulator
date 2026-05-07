"""``multi-scenario eval <run_dir>`` — re-evaluate a trained policy (F5.8, local-only).

Loads the latest BenchMARL checkpoint under ``<run_dir>``, rebuilds the
experiment, runs eval (with optional ``--episodes`` override), and writes
``<run_dir>/output/eval_runs/<TAG>.json``. Multiple eval runs coexist as
separate files; default tag is timestamped.

Module is named ``eval_run`` to avoid shadowing the ``eval`` builtin; the
subcommand name on the CLI stays ``eval`` via ``@app.command(name="eval")``.
"""

from datetime import datetime, timezone
from pathlib import Path

import typer

from multi_scenario.adapters.storage.local import LocalStorageAdapter
from multi_scenario.application.factories import make_algorithm, make_scenario
from multi_scenario.domain.models import EvalRunRecord, RunId

from ._app import app
from ._helpers import latest_checkpoint


# This command body wires together cfg/checkpoint/algorithm/scenario/experiment/
# rollout/metrics/record — naturally many locals; extracting helpers would
# fracture a 60-line linear flow without simplifying anything.
# pylint: disable=too-many-locals
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
    """Re-evaluate a trained policy without retraining (F5.8, local-only)."""
    config_path = run_dir / "input" / "config.json"
    if not config_path.is_file():
        typer.echo(f"✗ no input/config.json under {run_dir}", err=True)
        raise typer.Exit(code=2)

    storage = LocalStorageAdapter()
    cfg = storage.load_config(run_dir)
    if episodes > 0:
        cfg.evaluation.episodes = episodes

    checkpoint = latest_checkpoint(run_dir)
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
