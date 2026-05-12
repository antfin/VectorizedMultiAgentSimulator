"""``multi-scenario eval <run_dir>`` — re-evaluate a trained policy (F5.8, local-only).

Loads the latest BenchMARL checkpoint under ``<run_dir>``, rebuilds the
experiment, runs eval (with optional ``--episodes`` override), and writes
``<run_dir>/output/eval_runs/<TAG>.json``. Multiple eval runs coexist as
separate files; default tag is timestamped.

LERO-aware (Phase 7): when ``cfg.lero is not None``, this CLI loads the
winning candidate's ``obs_source`` / ``reward_source`` from
``output/lero/evolution_history.json`` + ``final_summary.json``, rebuilds
the patched Discovery scenario class, and monkey-patches the BenchMARL
task factory so the policy is evaluated against the same observation /
reward pipeline it was trained with. Without this, the reload would
use the bare Discovery scenario and either crash on tensor-shape
mismatch (LERO-enhanced obs) or silently report meaningless metrics.

Module is named ``eval_run`` to avoid shadowing the ``eval`` builtin; the
subcommand name on the CLI stays ``eval`` via ``@app.command(name="eval")``.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer

from multi_scenario.adapters.storage.local import LocalStorageAdapter
from multi_scenario.application.factories import make_algorithm, make_scenario
from multi_scenario.domain.models import EvalRunRecord, ExperimentConfig, RunId

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
        typer.echo(
            f"✗ no BenchMARL checkpoint under {run_dir}/output/benchmarl/", err=True
        )
        raise typer.Exit(code=2)

    # Reconstruct the experiment from the checkpoint, run eval through the
    # algorithm adapter (reusing F2.4.3 aggregation), score via the metrics bundle.
    algorithm = make_algorithm(cfg.algorithm.type)
    scenario = make_scenario(cfg.scenario.type)
    # Local import to avoid pulling BenchMARL into the cli module top-level.
    # pylint: disable=import-outside-toplevel
    from benchmarl.experiment import Experiment

    from multi_scenario.adapters.metrics.common import CommonMetricsBundle

    # Phase 7 — LERO awareness. ``experiment_patch`` overrides ensure
    # the saved-by-container paths (``/workspace/results``) get
    # re-pointed at the local layout, and CUDA-saved tensors land on
    # CPU. The factory monkey-patch (when cfg.lero is set) rebuilds
    # the patched scenario class from the winning candidate's code so
    # the policy is eval'd against the correct obs/reward pipeline.
    experiment_patch = _build_experiment_patch(run_dir, cfg)
    if cfg.lero is not None:
        _install_patched_factory_for_lero_reload(run_dir, cfg)
    experiment = Experiment.reload_from_file(
        str(checkpoint), experiment_patch=experiment_patch
    )
    rollout = algorithm.evaluate(experiment, env=None, cfg=cfg, run_dir=run_dir)
    metric_dict = CommonMetricsBundle().compute(rollout, scenario)

    eval_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
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


def _build_experiment_patch(run_dir: Path, cfg: ExperimentConfig) -> dict[str, Any]:
    """Construct the ``experiment_patch`` for BenchMARL's reload_from_file.

    Three overrides keep the reload portable across hosts:

    - ``save_folder``: points at the local experiment dir (the pickled
      cfg's ``save_folder`` is the container-side ``/workspace/results``;
      keeping it would make BenchMARL mkdir a path that doesn't exist
      on the user's laptop).
    - ``sampling_device`` / ``train_device`` / ``buffer_device``: forced
      to ``cpu`` when the user runs on a CPU-only host (CUDA-saved
      tensors otherwise raise on torch.load).
    - ``restore_map_location``: ``cpu`` for the same reason.
    - ``loggers``: empty list — the eval pass shouldn't write any logs
      that would clobber the training-time scalars under the same dir.
    """
    # The benchmarl experiment dir is the parent of the checkpoint's
    # parent. e.g. ``<run_dir>/output/benchmarl/<exp>/`` is the
    # ``save_folder`` BenchMARL needs to find ``config.pkl`` under and
    # write its working files into.
    benchmarl_dirs = list((run_dir / "output" / "benchmarl").iterdir())
    save_folder = (
        str(benchmarl_dirs[0].parent.resolve()) if benchmarl_dirs else str(run_dir)
    )
    return {
        "save_folder": save_folder,
        "sampling_device": cfg.training.device,
        "train_device": cfg.training.device,
        "buffer_device": cfg.training.device,
        "restore_map_location": cfg.training.device,
        "loggers": [],
    }


def _install_patched_factory_for_lero_reload(
    run_dir: Path, cfg: ExperimentConfig
) -> None:
    """Monkey-patch ScenarioEnvFunFactory.__setstate__ so reload uses the patched class.

    The factory's pickled state SHOULD now carry ``scenario_class`` +
    ``config`` (Phase 2 fix), but legacy LERO runs (Phase 5a) have a
    dummy-state pickle. To re-eval those without re-training, we install
    a setstate that reconstructs the factory from on-disk metadata:

    1. Load the winning candidate's ``obs_source`` / ``reward_source``
       from ``output/lero/final_summary.json`` (fallback chain's
       success entry).
    2. Build the patched class via ``make_patched_discovery_class``.
    3. Install a ``__setstate__`` that injects this class + the cfg's
       scenario params into any factory the unpickler resurrects.

    No-op for new runs (Phase 2 pickle persists state correctly), but
    cheap to install and protects against legacy artefacts.
    """
    # pylint: disable=import-outside-toplevel
    from multi_scenario.adapters.lero.scenario_env_fun_factory import (
        ScenarioEnvFunFactory,
    )
    from multi_scenario.adapters.scenarios.patched_discovery import (
        make_patched_discovery_class,
    )

    summary_path = run_dir / "output" / "lero" / "final_summary.json"
    history_path = run_dir / "output" / "lero" / "evolution_history.json"
    if not (summary_path.is_file() and history_path.is_file()):
        return  # no LERO trace files — let the default factory state apply
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    winner = next(
        (e for e in summary.get("fallback_chain", []) if e.get("outcome") == "success"),
        None,
    )
    if winner is None:
        return  # nothing trained → no patched class to inject
    history = json.loads(history_path.read_text(encoding="utf-8"))
    winning_entry = next(
        (
            e
            for e in history
            if e["candidate"]["iteration"] == winner["iteration"]
            and e["candidate"]["candidate_idx"] == winner["candidate_idx"]
        ),
        None,
    )
    if winning_entry is None:
        return
    code = winning_entry["candidate"]["code"]
    patched_cls = make_patched_discovery_class(
        reward_source=code.get("reward_source"),
        obs_source=code.get("obs_source"),
        reward_mode="legacy",
        obs_state_mode="local",
        bonus_scale=0.0,
        reward_clip=(cfg.lero.reward_clip if cfg.lero is not None else 50.0),
        whitelist_strict=(cfg.lero.whitelist_strict if cfg.lero is not None else True),
    )
    factory_config = {
        k: v for k, v in cfg.scenario.params.items() if k not in ("obs_lidar_agents",)
    }

    def _restore_setstate(self, state):  # noqa: ARG001
        self.scenario_class = patched_cls
        self.config = factory_config

    ScenarioEnvFunFactory.__setstate__ = _restore_setstate  # type: ignore[assignment]
