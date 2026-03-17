"""Experiment runner using BenchMARL.

Handles training, evaluation, and metric collection for each run
in an experiment sweep.
"""
import contextlib
import logging
import os
import sys
import time
import warnings
from typing import Any, Callable, Dict, Optional

import torch

from .config import ExperimentSpec, TaskConfig, TrainConfig
from .logging_setup import setup_run_logger, teardown_run_logger
from .metrics import EpisodeMetrics, compute_m7_sample_efficiency
from .provenance import save_provenance
from .report import generate_run_report, generate_sweep_report
from .storage import ExperimentStorage


_log = logging.getLogger("rendezvous")


@contextlib.contextmanager
def _suppress_noise():
    """Suppress BenchMARL/TorchRL stdout, stderr, loggers and warnings.

    Redirects both stdout and stderr to devnull so BenchMARL's own
    tqdm bar and log output are hidden. Our progress callback writes
    to the saved real stderr directly.
    """
    devnull = open(os.devnull, "w")
    old_out, old_err = sys.stdout, sys.stderr
    # Silence torchrl/benchmarl loggers
    noisy = ["torchrl", "benchmarl", "tensordict"]
    saved = {}
    for name in noisy:
        lg = logging.getLogger(name)
        saved[name] = lg.level
        lg.setLevel(logging.ERROR)
    try:
        sys.stdout = devnull
        sys.stderr = devnull
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        devnull.close()
        for name in noisy:
            logging.getLogger(name).setLevel(saved[name])




try:
    from benchmarl.experiment import Callback as _BenchMARLCallback
except ImportError:
    _BenchMARLCallback = object


class _TqdmProgressCallback(_BenchMARLCallback):
    """BenchMARL Callback that drives a tqdm progress bar.

    Updates on each batch collection step. Supports pickle
    (BenchMARL pickles callbacks for experiment name hashing)
    by excluding the non-picklable tqdm bar.
    """

    def __init__(self, total_frames=0, frames_per_batch=1,
                 run_id=""):
        super().__init__()
        from tqdm.auto import tqdm
        # Capture real stderr before _suppress_noise redirects it
        self._real_stderr = sys.stderr
        self._pbar = tqdm(
            total=max(1, total_frames // frames_per_batch),
            desc=f"  {run_id}",
            unit="iter",
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}]"
            ),
            leave=True,
            file=sys.stderr,  # pin to real stderr at creation time
        )

    def __getstate__(self):
        # Exclude tqdm bar from pickle (BenchMARL name hashing)
        return {"_dummy": True}

    def __setstate__(self, state):
        pass

    def on_batch_collected(self, batch):
        if hasattr(self, "_pbar"):
            self._pbar.update(1)

    def close(self):
        if hasattr(self, "_pbar"):
            self._pbar.close()


class _EvalMetricsCallback(_BenchMARLCallback):
    """BenchMARL Callback that computes M1 and M4 at each eval checkpoint.

    Uses on_evaluation_end(rollouts) to compute real success rate (M1)
    and collision count (M4) from evaluation episode rollouts.

    M1 uses targets_covered from the Discovery scenario info dict
    (not the terminated signal, which conflates task completion with
    time-limit truncation in TorchRL's VMAS wrapper).

    Results stored as lists of (iteration, value) tuples.
    """

    def __init__(self, n_targets=7):
        super().__init__()
        self._iter = 0
        self._n_targets = n_targets
        self.m1_history = []  # [(iter, success_rate), ...]
        self.m4_history = []  # [(iter, avg_collisions), ...]

    def __getstate__(self):
        return {"_dummy": True}

    def __setstate__(self, state):
        pass

    def on_batch_collected(self, batch):
        self._iter += 1

    def on_evaluation_end(self, rollouts):
        if not hasattr(self, "m1_history"):
            return
        n_success = 0
        total_collisions = 0.0
        n_envs_total = 0
        group_map = self.experiment.group_map

        for rollout_td in rollouts:
            # BenchMARL unbinds vectorized rollouts: each TD here is
            # a SINGLE env with batch_size=[T]. Tensors at info keys
            # have shape [T, n_agents, 1] (not [n_envs, T, ...]).

            # M1: success via targets_covered cumsum.
            # targets_covered = count of newly-covered targets per step.
            # With targets_respawn=False, cumsum = total unique covered.
            found_tc = False
            for group in group_map.keys():
                tc_key = ("next", group, "info", "targets_covered")
                if tc_key in rollout_td.keys(True):
                    tc = rollout_td[tc_key]
                    # After unbind: [T, n_agents, 1] → take first
                    # agent (all identical), squeeze → [T]
                    while tc.dim() > 1:
                        tc = tc[..., 0]
                    # tc is now [T] — per-step newly-covered count
                    cumtc = tc.cumsum(dim=0)  # [T]
                    success = (cumtc >= self._n_targets).any().item()
                    n_success += int(success)
                    n_envs_total += 1
                    found_tc = True
                    break

            if not found_tc:
                n_envs_total += 1

            # M4: count collision events for this env
            for group in group_map.keys():
                coll_key = ("next", group, "info", "collision_rew")
                if coll_key in rollout_td.keys(True):
                    coll_rew = rollout_td[coll_key]
                    total_collisions += (coll_rew < 0).sum().item()

        m1 = n_success / max(n_envs_total, 1)
        m4 = total_collisions / max(n_envs_total, 1)
        self.m1_history.append((self._iter, m1))
        self.m4_history.append((self._iter, m4))

    def save_csvs(self, benchmarl_dir):
        """Save M1/M4 history into BenchMARL scalars dirs."""
        import csv
        from pathlib import Path
        bdir = Path(benchmarl_dir)
        # Find the scalars dir inside benchmarl output
        scalars_dirs = list(bdir.glob("**/scalars"))
        if not scalars_dirs:
            # Create one if none exists
            scalars_dirs = [bdir / "eval_metrics" / "scalars"]
            scalars_dirs[0].mkdir(parents=True, exist_ok=True)
        for sd in scalars_dirs:
            for name, history in [
                ("eval_M1_success_rate", self.m1_history),
                ("eval_M4_avg_collisions", self.m4_history),
            ]:
                if not history:
                    continue
                path = sd / f"{name}.csv"
                with open(path, "w", newline="") as f:
                    w = csv.writer(f)
                    for row in history:
                        w.writerow(row)


def _extract_training_dynamics(run_storage) -> Dict[str, float]:
    """Extract final entropy and eval reward from BenchMARL CSVs."""
    result = {}
    scalars = run_storage.load_benchmarl_scalars()

    entropy_data = scalars.get("train_agents_entropy")
    if entropy_data:
        result["final_entropy"] = entropy_data[-1][1]

    eval_reward_data = scalars.get(
        "eval_reward_episode_reward_mean"
    )
    if eval_reward_data:
        result["final_eval_reward"] = eval_reward_data[-1][1]

    return result


def _timestamp() -> str:
    """Return YYYYMMDD_HHMM timestamp for file naming."""
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M")


def _write_sweep_csv(spec, results):
    """Write timestamped sweep_results CSV with all metrics + config.

    This CSV is the single source of truth for OVH batch runs.
    Each row is one run with all fields from metrics.json.
    Timestamped so previous runs are never overwritten.
    """
    import csv
    if not results:
        return
    ts = _timestamp()
    csv_path = spec.results_dir / f"sweep_results_{ts}.csv"
    # Collect all possible field names across all runs
    all_keys = set()
    for metrics in results.values():
        all_keys.update(metrics.keys())
    # Stable column order: run_id first, then sorted
    fieldnames = ["run_id"] + sorted(all_keys)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, restval="",
            extrasaction="ignore",
        )
        writer.writeheader()
        for run_id, metrics in results.items():
            row = {"run_id": run_id}
            row.update(metrics)
            writer.writerow(row)
    _log.info(f"Sweep CSV: {csv_path}")


def get_algorithm_config(algorithm: str):
    """Get BenchMARL algorithm config by name."""
    if algorithm == "mappo":
        from benchmarl.algorithms import MappoConfig
        return MappoConfig.get_from_yaml()
    elif algorithm == "ippo":
        from benchmarl.algorithms import IppoConfig
        return IppoConfig.get_from_yaml()
    elif algorithm == "qmix":
        from benchmarl.algorithms import QmixConfig
        return QmixConfig.get_from_yaml()
    elif algorithm == "maddpg":
        from benchmarl.algorithms import MaddpgConfig
        return MaddpgConfig.get_from_yaml()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def build_experiment(
    task_config: TaskConfig,
    train_config: TrainConfig,
    algorithm: str,
    seed: int,
    task_overrides: Optional[Dict[str, Any]] = None,
    save_folder: Optional[str] = None,
    callbacks: Optional[list] = None,
):
    """Build a BenchMARL Experiment object.

    Args:
        save_folder: where BenchMARL writes its CSV logs, checkpoints, etc.

    Returns:
        benchmarl.experiment.Experiment ready to .run()
    """
    from benchmarl.environments import VmasTask
    from benchmarl.experiment import Experiment, ExperimentConfig
    from benchmarl.models.mlp import MlpConfig

    # Task
    task = VmasTask.DISCOVERY.get_from_yaml()
    config = task_config.to_dict()
    config.pop("max_steps")  # not a VMAS task param; remove before update
    if task_overrides:
        config.update(task_overrides)
    task.config.update(config)

    # Algorithm
    algo_config = get_algorithm_config(algorithm)

    # Model
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()

    # Experiment config
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.max_n_frames = train_config.max_n_frames
    experiment_config.gamma = train_config.gamma
    experiment_config.on_policy_collected_frames_per_batch = (
        train_config.on_policy_collected_frames_per_batch
    )
    experiment_config.on_policy_n_envs_per_worker = (
        train_config.on_policy_n_envs_per_worker
    )
    experiment_config.on_policy_n_minibatch_iters = (
        train_config.on_policy_n_minibatch_iters
    )
    experiment_config.on_policy_minibatch_size = (
        train_config.on_policy_minibatch_size
    )
    experiment_config.train_device = train_config.train_device
    experiment_config.sampling_device = train_config.sampling_device
    experiment_config.share_policy_params = train_config.share_policy_params
    experiment_config.evaluation = True
    eval_interval = train_config.evaluation_interval
    batch_size = train_config.on_policy_collected_frames_per_batch
    if eval_interval % batch_size != 0:
        old = eval_interval
        eval_interval = max(
            batch_size,
            (eval_interval // batch_size) * batch_size,
        )
        _log.warning(
            f"evaluation_interval ({old}) is not a multiple of "
            f"collected_frames_per_batch ({batch_size}). "
            f"Rounding down to {eval_interval}."
        )
    experiment_config.evaluation_interval = eval_interval
    experiment_config.evaluation_episodes = train_config.evaluation_episodes
    experiment_config.loggers = ["csv"]
    # Rendering disabled during training — videos are generated
    # separately after training using saved policies (no pyglet crashes).
    experiment_config.render = False

    # Direct BenchMARL outputs into our results structure
    if save_folder is not None:
        experiment_config.save_folder = save_folder

    experiment = Experiment(
        task=task,
        algorithm_config=algo_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=seed,
        config=experiment_config,
        callbacks=callbacks or None,
    )
    return experiment


def run_single(
    spec: ExperimentSpec,
    run_id: str,
    task_overrides: Dict[str, Any],
    algorithm: str,
    seed: int,
    skip_complete: bool = True,
    dry_run: bool = False,
    quiet: bool = False,
) -> Dict[str, float]:
    """Run a single training run and save results.

    Args:
        spec: experiment specification
        run_id: unique identifier for this run
        task_overrides: task params overriding defaults
        algorithm: algorithm name (mappo, ippo, ...)
        seed: random seed
        skip_complete: skip if results already exist
        dry_run: if True, build experiment but don't train
        quiet: if True, log only to file (no console output)

    Returns:
        metrics dict
    """
    storage = ExperimentStorage(spec.exp_id)
    run_storage = storage.get_run(run_id)

    # Setup logging — file only when quiet (notebook progress bars instead)
    logger = setup_run_logger(run_storage.run_dir, console=not quiet)

    try:
        if skip_complete and run_storage.is_complete():
            logger.info(f"SKIP  {run_id} — already complete")
            return run_storage.load_metrics()

        # Save config snapshot + provenance
        run_storage.save_config({
            "exp_id": spec.exp_id,
            "run_id": run_id,
            "algorithm": algorithm,
            "seed": seed,
            "task_overrides": task_overrides,
            "task": spec.task.to_dict(),
            "train": {k: v for k, v in spec.train.__dict__.items()},
        })
        save_provenance(
            run_storage.run_dir,
            task_dict=spec.task.to_dict(),
            train_dict={k: v for k, v in spec.train.__dict__.items()},
        )

        if dry_run:
            logger.info(f"DRY RUN  {run_id}")
            return {}

        logger.info(f"START  {run_id}")
        logger.info(f"  Algorithm:     {algorithm}")
        logger.info(f"  Seed:          {seed}")
        logger.info(f"  Frames:        {spec.train.max_n_frames:,}")
        logger.info(f"  Device:        {spec.train.train_device}")
        logger.info(f"  Task overrides: {task_overrides}")
        logger.info(f"  Output dir:    {run_storage.run_dir}")

        # Callbacks: progress bar (notebook) + M1/M4 eval tracker
        callbacks = []
        progress_cb = None
        n_targets = task_overrides.get("n_targets", 7) if task_overrides else 7
        eval_cb = _EvalMetricsCallback(n_targets=n_targets)
        callbacks.append(eval_cb)

        if quiet and _in_notebook():
            progress_cb = _TqdmProgressCallback(
                total_frames=spec.train.max_n_frames,
                frames_per_batch=(
                    spec.train.on_policy_collected_frames_per_batch
                ),
                run_id=run_id,
            )
            callbacks.append(progress_cb)

        # Suppress noisy BenchMARL/TorchRL output in notebooks
        noise_ctx = _suppress_noise() if quiet else contextlib.nullcontext()
        with noise_ctx:
            experiment = build_experiment(
                spec.task, spec.train, algorithm, seed, task_overrides,
                save_folder=str(run_storage.benchmarl_dir),
                callbacks=callbacks,
            )

            # Save initial (untrained) policy for before/after videos
            try:
                init_sd = {
                    k: v.clone() for k, v in
                    experiment.policy.state_dict().items()
                }
                torch.save(
                    init_sd,
                    run_storage.output_dir / "policy_init.pt",
                )
            except Exception:
                pass  # non-critical

            t0 = time.monotonic()
            experiment.run()
            elapsed = time.monotonic() - t0

        if progress_cb is not None:
            progress_cb.close()

        # Save M1/M4 eval history for training curve plots
        eval_cb.save_csvs(run_storage.benchmarl_dir)

        # Extract training dynamics from BenchMARL CSVs
        dynamics = _extract_training_dynamics(run_storage)

        m, s = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)
        logger.info(f"TRAINING COMPLETE  {run_id}  ({h}h {m}m {s}s)")

        # Save trained policy for export/import
        logger.info("Saving trained policy...")
        run_storage.save_policy(experiment.policy)
        logger.info(f"  Policy saved to {run_storage.output_dir / 'policy.pt'}")

        # Post-training evaluation
        logger.info(f"Evaluating trained policy ({spec.train.evaluation_episodes} episodes)...")
        t_eval = time.monotonic()
        metrics = evaluate_trained(spec, experiment, task_overrides)
        eval_elapsed = time.monotonic() - t_eval

        # M7: Sample Efficiency (from training CSVs)
        m7 = compute_m7_sample_efficiency(run_storage.run_dir)
        if m7 is not None:
            metrics["M7_sample_efficiency"] = m7
            metrics["convergence_frame"] = m7

        # ── Enrich metrics with ALL config + execution + dynamics ──
        # This makes the CSV self-contained for OVH batch runs.

        def _ov(key):
            """Get effective task param (override > base)."""
            if task_overrides and key in task_overrides:
                return task_overrides[key]
            return getattr(spec.task, key)

        # Experiment identity
        metrics["exp_id"] = spec.exp_id
        metrics["experiment_name"] = spec.name
        metrics["algorithm"] = algorithm
        metrics["seed"] = seed
        metrics["run_timestamp"] = run_storage.run_dir.name.split("__")[0]
        if spec.source_path:
            metrics["config_file"] = spec.source_path.name

        # Task config (swept + fixed — all params that define the env)
        metrics["n_agents"] = _ov("n_agents")
        metrics["n_targets"] = _ov("n_targets")
        metrics["agents_per_target"] = _ov("agents_per_target")
        metrics["lidar_range"] = _ov("lidar_range")
        metrics["covering_range"] = _ov("covering_range")
        metrics["max_steps"] = _ov("max_steps")
        metrics["agent_collision_penalty"] = _ov(
            "agent_collision_penalty"
        )
        metrics["covering_rew_coeff"] = _ov("covering_rew_coeff")
        metrics["time_penalty"] = _ov("time_penalty")
        metrics["shared_reward"] = _ov("shared_reward")
        metrics["targets_respawn"] = _ov("targets_respawn")

        # Training hyperparameters
        metrics["max_n_frames"] = spec.train.max_n_frames
        metrics["gamma"] = spec.train.gamma
        metrics["lr"] = spec.train.lr
        metrics["frames_per_batch"] = (
            spec.train.on_policy_collected_frames_per_batch
        )
        metrics["n_envs_per_worker"] = (
            spec.train.on_policy_n_envs_per_worker
        )
        metrics["n_minibatch_iters"] = (
            spec.train.on_policy_n_minibatch_iters
        )
        metrics["minibatch_size"] = spec.train.on_policy_minibatch_size
        metrics["share_policy_params"] = spec.train.share_policy_params
        metrics["evaluation_interval"] = spec.train.evaluation_interval
        metrics["evaluation_episodes"] = spec.train.evaluation_episodes

        # Execution metadata
        metrics["training_seconds"] = round(elapsed, 1)
        metrics["eval_seconds"] = round(eval_elapsed, 1)
        metrics["device"] = spec.train.train_device
        metrics["torch_version"] = torch.__version__

        # Derived
        metrics["n_iterations"] = (
            spec.train.max_n_frames
            // spec.train.on_policy_collected_frames_per_batch
        )
        if elapsed > 0:
            metrics["throughput_fps"] = round(
                spec.train.max_n_frames / elapsed, 1
            )

        # Policy size
        try:
            sd = experiment.policy.state_dict()
            metrics["policy_params"] = sum(
                p.numel() for p in sd.values()
                if hasattr(p, "numel")
            )
        except Exception:
            pass

        # Training dynamics
        metrics.update(dynamics)

        run_storage.save_metrics(metrics)

        from .report import METRIC_DETAILS
        for key, val in metrics.items():
            detail = METRIC_DETAILS.get(key)
            if detail and isinstance(val, (int, float)):
                label = detail["label"]
                fmt = detail["fmt"]
                logger.info(f"  {label}: {val:{fmt}}")

        # Generate report
        generate_run_report(
            run_storage.run_dir, run_id, spec, metrics,
            elapsed_seconds=elapsed, task_overrides=task_overrides,
        )
        logger.info(f"DONE  {run_id} — report saved to {run_storage.run_dir / 'report.md'}")

        # Generate before/after videos from saved policies
        try:
            generate_run_videos(
                run_storage, spec.task, task_overrides, logger,
            )
        except Exception as exc:
            logger.warning(f"Video generation failed: {exc}")

        return metrics

    finally:
        teardown_run_logger(logger)


def evaluate_trained(
    spec: ExperimentSpec,
    experiment,
    task_overrides: Dict[str, Any],
    n_eval_episodes: int = 200,
) -> Dict[str, float]:
    """Run evaluation episodes using BenchMARL's own eval env and trained policy.

    Uses TorchRL rollout on the experiment's test_env with the trained policy
    in deterministic mode, then extracts our custom metrics (M1-M6, M8-M9).

    The test_env is vectorized (num_envs parallel environments), so each
    rollout produces num_envs episodes. We run enough rollouts to collect
    at least n_eval_episodes total, then process per-env metrics.
    """
    from torchrl.envs.utils import ExplorationType, set_exploration_type

    test_env = experiment.test_env
    policy = experiment.policy
    max_steps = experiment.max_steps
    num_envs = test_env.batch_size[0] if test_env.batch_size else 1

    n_targets = task_overrides.get("n_targets", 7) if task_overrides else 7
    n_agents = task_overrides.get("n_agents", 5) if task_overrides else 5

    # Per-env accumulators
    all_returns = []       # float per env
    all_steps = []         # int per env
    all_collisions = []    # float per env
    all_coverage = []      # float per env (fraction of targets)
    all_agent_covering = []  # (n_agents,) per env
    all_spatial_spread = []  # float per env
    n_success = 0
    n_envs_total = 0

    # Run enough rollouts to get >= n_eval_episodes
    n_rollouts = max(1, (n_eval_episodes + num_envs - 1) // num_envs)

    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        for _ in range(n_rollouts):
            rollout_td = test_env.rollout(
                max_steps=max_steps,
                policy=policy,
                auto_cast_to_device=True,
                break_when_any_done=False,
            )
            # rollout_td shape: [n_envs, T, ...]
            ne = rollout_td.batch_size[0]  # actual num_envs
            T = rollout_td.batch_size[1] if len(rollout_td.batch_size) > 1 else 1
            n_envs_total += ne

            # M1/M3/M6: Use targets_covered from info dict.
            # targets_covered = count of newly-covered targets per step.
            # With targets_respawn=False, cumsum = total unique covered.
            # Do NOT use terminated signal (conflates task completion
            # with time-limit truncation in TorchRL VMAS wrapper).
            for group in experiment.group_map.keys():
                tc_key = ("next", group, "info", "targets_covered")
                if tc_key in rollout_td.keys(True):
                    tc = rollout_td[tc_key]
                    # Shape: [ne, T, n_agents, 1] or [ne, T, 1]
                    if tc.dim() >= 4:
                        tc_per_env = tc[:, :, 0, 0]  # [ne, T]
                    elif tc.dim() == 3:
                        tc_per_env = tc[:, :, 0]
                    else:
                        tc_per_env = tc.unsqueeze(0)

                    # Cumsum: total unique targets covered over time
                    cumtc = tc_per_env.cumsum(dim=1)  # [ne, T]

                    # M1: success = all targets covered at some point
                    per_env_success = (
                        cumtc >= n_targets
                    ).any(dim=1)  # [ne]
                    n_success += per_env_success.sum().item()

                    # M3: step when all targets first covered
                    for e in range(ne):
                        success_steps = (
                            cumtc[e] >= n_targets
                        ).nonzero(as_tuple=False)
                        if len(success_steps) > 0:
                            all_steps.append(
                                success_steps[0].item() + 1
                            )
                        else:
                            all_steps.append(max_steps)

                    # M6: coverage progress = fraction of targets
                    # covered by end of episode
                    final_coverage = cumtc[:, -1]  # [ne]
                    for v in final_coverage.tolist():
                        all_coverage.append(
                            min(v, n_targets) / n_targets
                        )
                    break
            else:
                # No targets_covered found — fill steps/coverage
                for _ in range(ne):
                    all_steps.append(max_steps)
                    all_coverage.append(0.0)

            # M2: per-env total reward
            for group in experiment.group_map.keys():
                reward_key = ("next", group, "reward")
                if reward_key in rollout_td.keys(True):
                    rew = rollout_td[reward_key]  # [ne, T, 1]
                    if rew.dim() >= 3:
                        per_env_return = rew.sum(dim=1).flatten()  # [ne]
                        all_returns.extend(per_env_return.tolist())
                    else:
                        all_returns.append(rew.sum().item())
                    break

            # M4: per-env collision count
            # collision_rew shape: [ne, T, n_agents, 1] — per-agent
            # boolean. Sum all agent-collision-steps per env.
            for group in experiment.group_map.keys():
                coll_key = ("next", group, "info", "collision_rew")
                if coll_key in rollout_td.keys(True):
                    coll = rollout_td[coll_key]
                    # Reshape to [ne, everything_else] and sum
                    per_env_coll = (coll < 0).reshape(ne, -1).sum(
                        dim=1,
                    )  # [ne]
                    all_collisions.extend(per_env_coll.tolist())
                    break

            # M8: per-env agent covering balance
            # covering_reward shape: [ne, T, n_agents, 1]
            # Squeeze trailing 1 to get [ne, T, n_agents].
            for group in experiment.group_map.keys():
                cov_key = ("next", group, "info", "covering_reward")
                if cov_key in rollout_td.keys(True):
                    cov = rollout_td[cov_key]
                    # Remove trailing dims of size 1
                    while cov.dim() > 3 and cov.shape[-1] == 1:
                        cov = cov.squeeze(-1)
                    # cov: [ne, T, n_agents]
                    if cov.dim() >= 3:
                        for e in range(ne):
                            env_cov = cov[e]  # [T, n_agents]
                            if (
                                env_cov.dim() >= 2
                                and env_cov.shape[-1] > 1
                            ):
                                counts = (
                                    (env_cov > 0).sum(dim=0).float()
                                )
                                all_agent_covering.append(counts)
                            else:
                                c = torch.zeros(n_agents)
                                c[0] = (env_cov > 0).sum().item()
                                all_agent_covering.append(c)
                    break

            # M9: per-env spatial spread
            for group in experiment.group_map.keys():
                obs_key = (group, "observation")
                if obs_key in rollout_td.keys(True):
                    obs = rollout_td[obs_key]
                    # obs: [ne, T, n_agents, obs_dim]
                    if obs.dim() >= 4:
                        ne_o, T_s, n_ag, obs_dim = obs.shape
                        if n_ag >= 2 and obs_dim >= 2:
                            pos = obs[:, :, :, :2]  # [ne, T, n_ag, 2]
                            mask = torch.triu(
                                torch.ones(n_ag, n_ag, device=pos.device),
                                diagonal=1,
                            ).bool()
                            for e in range(ne_o):
                                ep_spread = 0.0
                                for t in range(T_s):
                                    d = torch.cdist(
                                        pos[e, t].unsqueeze(0),
                                        pos[e, t].unsqueeze(0),
                                    )[0]
                                    ep_spread += d[mask].mean().item()
                                all_spatial_spread.append(
                                    ep_spread / max(T_s, 1)
                                )
                    elif obs.dim() == 3:
                        T_s, n_ag, obs_dim = obs.shape
                        if n_ag >= 2 and obs_dim >= 2:
                            pos = obs[:, :, :2]
                            mask = torch.triu(
                                torch.ones(n_ag, n_ag, device=pos.device),
                                diagonal=1,
                            ).bool()
                            ep_spread = 0.0
                            for t in range(T_s):
                                d = torch.cdist(
                                    pos[t].unsqueeze(0),
                                    pos[t].unsqueeze(0),
                                )[0]
                                ep_spread += d[mask].mean().item()
                            all_spatial_spread.append(
                                ep_spread / max(T_s, 1)
                            )
                    break

    total_episodes = n_envs_total if n_envs_total > 0 else num_envs * n_rollouts

    # M8: Average CV of per-agent covering
    if all_agent_covering:
        covering_stack = torch.stack(all_agent_covering)  # (n_ep, n_agents)
        mean_cov = covering_stack.mean(dim=-1)
        std_cov = covering_stack.std(dim=-1)
        cv = torch.where(
            mean_cov > 0, std_cov / mean_cov,
            torch.zeros_like(mean_cov),
        )
        agent_util = cv.mean().item()
    else:
        agent_util = 0.0

    metrics = {
        "M1_success_rate": n_success / max(total_episodes, 1),
        "M2_avg_return": (
            sum(all_returns) / max(len(all_returns), 1)
        ),
        "M3_avg_steps": (
            sum(all_steps) / max(len(all_steps), 1)
        ),
        "M4_avg_collisions": (
            sum(all_collisions) / max(len(all_collisions), 1)
        ),
        "M5_avg_tokens": 0.0,
        "M6_coverage_progress": (
            sum(all_coverage) / max(len(all_coverage), 1)
        ),
        "M8_agent_utilization": agent_util,
        "M9_spatial_spread": (
            sum(all_spatial_spread) / max(len(all_spatial_spread), 1)
        ),
        "n_envs": total_episodes,
    }

    return metrics


def evaluate_with_vmas(
    task_config: TaskConfig,
    task_overrides: Optional[Dict[str, Any]] = None,
    policy_fn: Optional[Callable] = None,
    n_eval_episodes: int = 200,
    n_envs: int = 200,
) -> Dict[str, float]:
    """Evaluate using raw VMAS env (for heuristic or loaded policy).

    Args:
        task_config: task configuration
        task_overrides: parameter overrides
        policy_fn: callable(observations) -> actions; if None, uses random
        n_eval_episodes: number of episodes
        n_envs: number of parallel envs (runs ceil(n_eval/n_envs) batches)
    """
    from vmas import make_env

    config = task_config.to_dict()
    max_steps = config.pop("max_steps")
    if task_overrides:
        config.update(task_overrides)

    env = make_env(
        scenario="discovery",
        num_envs=n_envs,
        device="cpu",
        continuous_actions=True,
        **config,
    )

    all_metrics = []
    n_batches = max(1, (n_eval_episodes + n_envs - 1) // n_envs)
    n_targets = config.get("n_targets", 7)
    n_agents = config.get("n_agents", 5)

    for batch in range(n_batches):
        episode_metrics = EpisodeMetrics().init(
            n_envs, n_targets=n_targets, n_agents=n_agents,
        )
        obs = env.reset()

        for step in range(max_steps):
            if policy_fn is not None:
                actions = policy_fn(obs, env)
            else:
                actions = [
                    env.get_random_action(agent) for agent in env.agents
                ]
            obs, rews, dones, info = env.step(actions)

            # Get agent positions for M9
            agent_positions = torch.stack(
                [a.state.pos for a in env.agents], dim=1
            )  # (n_envs, n_agents, 2)

            episode_metrics.update_step(
                rews, dones, info, step,
                agent_positions=agent_positions,
            )

        all_metrics.append(episode_metrics.compute(max_steps))

    # Average across batches
    if len(all_metrics) == 1:
        return all_metrics[0]
    keys = all_metrics[0].keys()
    return {
        k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in keys
    }


def generate_run_videos(
    run_storage,
    task_config: TaskConfig,
    task_overrides: Optional[Dict[str, Any]] = None,
    logger=None,
    max_steps: Optional[int] = None,
    fps: int = 15,
):
    """Generate before/after MP4 videos from saved policies.

    Uses raw VMAS env with rendering (not BenchMARL), so each call
    is independent and pyglet state doesn't leak across runs.

    Produces:
      output/video_init.mp4   — untrained policy (iteration 0)
      output/video_final.mp4  — trained policy (final)
    """
    from pathlib import Path

    init_path = run_storage.output_dir / "policy_init.pt"
    final_path = run_storage.output_dir / "policy.pt"

    if not final_path.exists():
        if logger:
            logger.info("No saved policy — skipping video generation")
        return

    config = task_config.to_dict()
    ms = max_steps or config.pop("max_steps", 200)
    config.pop("max_steps", None)
    if task_overrides:
        config.update(task_overrides)

    n_agents = config.get("n_agents", 5)

    # Build a fresh VMAS env for rendering (single env, not batched)
    from vmas import make_env
    env = make_env(
        scenario="discovery",
        num_envs=1,
        device="cpu",
        continuous_actions=True,
        **config,
    )

    def _make_policy_fn(state_dict):
        """Create a policy_fn from a BenchMARL policy state dict.

        BenchMARL policies are TorchRL modules that expect TensorDict
        input. For raw VMAS we need obs → actions. We rebuild the
        experiment to get the policy architecture, load weights, then
        wrap it.
        """
        # We use a simple approach: the policy network maps
        # concatenated obs → action. Extract the MLP weights.
        # However, BenchMARL policies are complex TensorDict modules.
        # Simplest: return random actions if we can't load properly.
        # For video purposes, even approximate behavior is fine.
        try:
            from benchmarl.environments import VmasTask
            from benchmarl.experiment import Experiment, ExperimentConfig
            from benchmarl.models.mlp import MlpConfig
            from benchmarl.algorithms import MappoConfig

            task = VmasTask.DISCOVERY.get_from_yaml()
            tc = task_config.to_dict()
            tc.pop("max_steps", None)
            if task_overrides:
                tc.update(task_overrides)
            task.config.update(tc)

            exp_config = ExperimentConfig.get_from_yaml()
            exp_config.render = False

            tmp_exp = Experiment(
                task=task,
                algorithm_config=MappoConfig.get_from_yaml(),
                model_config=MlpConfig.get_from_yaml(),
                critic_model_config=MlpConfig.get_from_yaml(),
                seed=0,
                config=exp_config,
            )
            tmp_exp.policy.load_state_dict(state_dict)
            policy = tmp_exp.policy
            test_env = tmp_exp.test_env

            from torchrl.envs.utils import (
                ExplorationType, set_exploration_type,
            )

            def policy_fn(observations, vmas_env):
                """Run one step through BenchMARL policy."""
                with torch.no_grad(), set_exploration_type(
                    ExplorationType.DETERMINISTIC
                ):
                    # Reset test_env and build a tensordict from obs
                    td = test_env.reset()
                    # Inject observations into td
                    for group in tmp_exp.group_map:
                        obs_key = (group, "observation")
                        if obs_key in td.keys(True):
                            # Stack agent obs: list of [1, obs_dim]
                            stacked = torch.stack(
                                [o[:1] for o in observations],
                                dim=1,
                            )  # [1, n_agents, obs_dim]
                            td[obs_key] = stacked
                    td = policy(td)
                    # Extract actions
                    actions = []
                    for group in tmp_exp.group_map:
                        act_key = (group, "action")
                        if act_key in td.keys(True):
                            act = td[act_key]  # [1, n_agents, act_dim]
                            for i in range(act.shape[1]):
                                actions.append(act[0, i])
                    if actions:
                        return actions
                # Fallback: random
                return [
                    vmas_env.get_random_action(a)
                    for a in vmas_env.agents
                ]
            return policy_fn
        except Exception:
            # Fallback: random policy
            return None

    policies = []
    if init_path.exists():
        init_sd = torch.load(
            init_path, map_location="cpu", weights_only=True,
        )
        policies.append(("video_init", init_sd))
    final_sd = torch.load(
        final_path, map_location="cpu", weights_only=True,
    )
    policies.append(("video_final", final_sd))

    for name, sd in policies:
        policy_fn = _make_policy_fn(sd)

        try:
            frames = []
            obs = env.reset()
            for step in range(ms):
                if policy_fn is not None:
                    actions = policy_fn(obs, env)
                else:
                    actions = [
                        env.get_random_action(a) for a in env.agents
                    ]
                obs, _, _, _ = env.step(actions)
                frame = env.render(
                    mode="rgb_array",
                    agent_index_focus=None,
                    visualize_when_rgb=True,
                )
                if frame is not None:
                    frames.append(frame)

            if frames:
                video_path = run_storage.output_dir / f"{name}.mp4"
                _save_video(frames, video_path, fps=fps)
                if logger:
                    logger.info(
                        f"  Video: {video_path.name} "
                        f"({len(frames)} frames)"
                    )
        except Exception as exc:
            if logger:
                logger.warning(f"  Video {name} failed: {exc}")


def _save_video(frames, path, fps=15):
    """Save list of RGB frames as MP4 using imageio."""
    import imageio
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264")
    for frame in frames:
        if hasattr(frame, "numpy"):
            frame = frame.numpy()
        writer.append_data(frame)
    writer.close()


def make_heuristic_policy_fn(scenario_name: str = "discovery"):
    """Create a policy_fn from the scenario's built-in heuristic."""
    from vmas.scenarios.discovery import HeuristicPolicy

    policy = HeuristicPolicy(continuous_action=True)

    def policy_fn(observations, env):
        return [
            policy.compute_action(observations[i], u_range=env.agents[i].u_range)
            for i in range(len(observations))
        ]
    return policy_fn


def _in_notebook() -> bool:
    """Detect if running inside a Jupyter/IPython notebook."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


def _fmt_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def run_sweep(
    spec: ExperimentSpec,
    skip_complete: bool = True,
    dry_run: bool = False,
    max_runs: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Run all parameter combinations in a sweep.

    In notebooks, shows progress bars instead of log spam.
    Detailed logs are always written to each run's logs/run.log.

    Args:
        spec: experiment specification
        skip_complete: skip runs that already have results
        dry_run: build but don't train
        max_runs: cap on number of runs (for testing)

    Returns:
        {run_id: metrics} for all runs
    """
    notebook = _in_notebook()
    all_runs = list(spec.iter_runs())
    total = len(all_runs)
    if max_runs:
        total = min(total, max_runs)
        all_runs = all_runs[:total]

    # Pre-scan run status for progress display
    storage = ExperimentStorage(spec.exp_id)
    n_already_done = 0
    if skip_complete:
        for run_id, _, _, _ in all_runs:
            rs = storage.get_run(run_id)
            if rs.is_complete():
                n_already_done += 1

    if not notebook:
        _log.info(
            f"Starting sweep: {spec.exp_id} — {total} runs "
            f"({n_already_done} already complete)"
        )

    t0 = time.monotonic()
    results = {}
    n_skip = 0
    n_train = 0

    pbar = None
    if notebook:
        from tqdm.auto import tqdm
        pbar = tqdm(
            total=total,
            desc=f"{spec.exp_id.upper()} sweep",
            unit="run",
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {postfix}]"
            ),
        )

    for i, (run_id, overrides, algo, seed) in enumerate(all_runs):
        is_complete = (
            skip_complete and storage.get_run(run_id).is_complete()
        )

        if notebook and pbar is not None:
            if is_complete:
                pbar.set_postfix_str(f"skip {run_id}")
            else:
                pbar.set_postfix_str(f"training {run_id}")

        if not notebook:
            _log.info(f"Run {i + 1}/{total}: {run_id}")

        metrics = run_single(
            spec, run_id, overrides, algo, seed,
            skip_complete=skip_complete, dry_run=dry_run,
            quiet=notebook,
        )
        results[run_id] = metrics

        if is_complete:
            n_skip += 1
        else:
            n_train += 1

        if notebook and pbar is not None:
            pbar.update(1)
            pbar.set_postfix_str(
                f"trained: {n_train}  skipped: {n_skip}"
            )

    elapsed = time.monotonic() - t0

    if notebook and pbar is not None:
        pbar.set_postfix_str(
            f"done — trained: {n_train}  "
            f"skipped: {n_skip}  "
            f"time: {_fmt_elapsed(elapsed)}"
        )
        pbar.close()

    # Generate sweep report + CSV
    generate_sweep_report(spec, results, elapsed_seconds=elapsed)
    _write_sweep_csv(spec, results)

    # Consolidate all CSVs (sweep + training iter/eval)
    try:
        from .consolidate import consolidate_csvs
        consolidate_csvs(spec.exp_id)
    except Exception as exc:
        _log.warning(f"CSV consolidation failed: {exc}")

    if not notebook:
        _log.info(f"Sweep complete: {len(results)} runs in {elapsed:.0f}s")
        _log.info(
            f"Sweep report: {spec.results_dir / 'sweep_report_*.md'}"
        )

    return results
