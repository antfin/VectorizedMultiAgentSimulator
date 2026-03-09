"""Experiment runner using BenchMARL.

Handles training, evaluation, and metric collection for each run
in an experiment sweep.
"""
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

from .config import ExperimentSpec, TaskConfig, TrainConfig
from .logging_setup import setup_run_logger, teardown_run_logger
from .metrics import EpisodeMetrics
from .report import generate_run_report, generate_sweep_report
from .storage import ExperimentStorage, RunStorage


_log = logging.getLogger("rendezvous")


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
    max_steps = config.pop("max_steps")
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
    experiment_config.evaluation_interval = train_config.evaluation_interval
    experiment_config.evaluation_episodes = train_config.evaluation_episodes
    experiment_config.loggers = ["csv"]

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

    Returns:
        metrics dict
    """
    storage = ExperimentStorage(spec.exp_id)
    run_storage = storage.get_run(run_id)

    # Setup logging to file + console
    logger = setup_run_logger(run_storage.run_dir)

    try:
        if skip_complete and run_storage.is_complete():
            logger.info(f"SKIP  {run_id} — already complete")
            return run_storage.load_metrics()

        # Save config snapshot
        run_storage.save_config({
            "exp_id": spec.exp_id,
            "run_id": run_id,
            "algorithm": algorithm,
            "seed": seed,
            "task_overrides": task_overrides,
            "task": spec.task.to_dict(),
            "train": {k: v for k, v in spec.train.__dict__.items()},
        })

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

        experiment = build_experiment(
            spec.task, spec.train, algorithm, seed, task_overrides,
            save_folder=str(run_storage.benchmarl_dir),
        )

        t0 = time.monotonic()
        experiment.run()
        elapsed = time.monotonic() - t0

        m, s = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)
        logger.info(f"TRAINING COMPLETE  {run_id}  ({h}h {m}m {s}s)")

        # Save trained policy for export/import
        logger.info("Saving trained policy...")
        run_storage.save_policy(experiment.policy)
        logger.info(f"  Policy saved to {run_storage.output_dir / 'policy.pt'}")

        # Post-training evaluation
        logger.info(f"Evaluating trained policy ({spec.train.evaluation_episodes} episodes)...")
        metrics = evaluate_trained(spec, experiment, task_overrides)
        run_storage.save_metrics(metrics)

        for key, val in metrics.items():
            if key != "n_envs":
                logger.info(f"  {key}: {val:.4f}")

        # Generate date-prefixed report
        report = generate_run_report(
            run_storage.run_dir, run_id, spec, metrics,
            elapsed_seconds=elapsed, task_overrides=task_overrides,
        )
        logger.info(f"DONE  {run_id} — report saved to {run_storage.run_dir / 'report.txt'}")

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
    in deterministic mode, then extracts our custom metrics.
    """
    from torchrl.envs.utils import ExplorationType, set_exploration_type

    test_env = experiment.test_env
    policy = experiment.policy
    max_steps = experiment.max_steps

    all_returns = []
    n_done = 0
    total_episodes = 0

    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        for _ in range(n_eval_episodes):
            rollout_td = test_env.rollout(
                max_steps=max_steps,
                policy=policy,
                auto_cast_to_device=True,
                break_when_any_done=True,
            )
            total_episodes += 1

            # Extract rewards: sum across agents and steps
            episode_return = 0.0
            for group in experiment.group_map.keys():
                reward_key = ("next", group, "reward")
                if reward_key in rollout_td.keys(True):
                    episode_return += rollout_td[reward_key].sum().item()
            all_returns.append(episode_return)

            # Check if episode completed (done fired)
            done_key = ("next", "done")
            if done_key in rollout_td.keys(True):
                if rollout_td[done_key].any().item():
                    n_done += 1

    metrics = {
        "M1_success_rate": n_done / max(total_episodes, 1),
        "M1b_avg_targets_covered_per_step": 0.0,
        "M2_avg_return": sum(all_returns) / max(len(all_returns), 1),
        "M3_avg_steps": max_steps,
        "M4_avg_collisions": 0.0,
        "M5_avg_tokens": 0.0,
        "n_envs": n_eval_episodes,
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

    for batch in range(n_batches):
        episode_metrics = EpisodeMetrics().init(n_envs)
        obs = env.reset()

        for step in range(max_steps):
            if policy_fn is not None:
                actions = policy_fn(obs, env)
            else:
                actions = [
                    env.get_random_action(agent) for agent in env.agents
                ]
            obs, rews, dones, info = env.step(actions)
            episode_metrics.update_step(rews, dones, info, step)

        all_metrics.append(episode_metrics.compute(max_steps))

    # Average across batches
    if len(all_metrics) == 1:
        return all_metrics[0]
    keys = all_metrics[0].keys()
    return {
        k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in keys
    }


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


def run_sweep(
    spec: ExperimentSpec,
    skip_complete: bool = True,
    dry_run: bool = False,
    max_runs: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Run all parameter combinations in a sweep.

    Args:
        spec: experiment specification
        skip_complete: skip runs that already have results
        dry_run: build but don't train
        max_runs: cap on number of runs (for testing)

    Returns:
        {run_id: metrics} for all runs
    """
    total = sum(1 for _ in spec.iter_runs())
    if max_runs:
        total = min(total, max_runs)
    _log.info(f"Starting sweep: {spec.exp_id} — {total} runs")

    t0 = time.monotonic()
    results = {}
    for i, (run_id, overrides, algo, seed) in enumerate(spec.iter_runs()):
        if max_runs and i >= max_runs:
            break
        _log.info(f"Run {i + 1}/{total}: {run_id}")
        metrics = run_single(
            spec, run_id, overrides, algo, seed,
            skip_complete=skip_complete, dry_run=dry_run,
        )
        results[run_id] = metrics

    elapsed = time.monotonic() - t0

    # Generate sweep report
    report = generate_sweep_report(spec, results, elapsed_seconds=elapsed)
    _log.info(f"Sweep complete: {len(results)} runs in {elapsed:.0f}s")
    _log.info(f"Sweep report: {spec.results_dir / 'sweep_report.txt'}")

    return results
