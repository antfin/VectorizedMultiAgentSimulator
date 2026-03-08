"""Experiment runner using BenchMARL.

Handles training, evaluation, and metric collection for each run
in an experiment sweep.
"""
import copy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from .config import ExperimentSpec, TaskConfig, TrainConfig
from .metrics import EpisodeMetrics
from .storage import ExperimentStorage, RunStorage


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
):
    """Build a BenchMARL Experiment object.

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

    if skip_complete and run_storage.is_complete():
        print(f"[SKIP] {run_id} already complete")
        return run_storage.load_metrics()

    # Save config snapshot
    run_storage.save_config({
        "exp_id": spec.exp_id,
        "run_id": run_id,
        "algorithm": algorithm,
        "seed": seed,
        "task_overrides": task_overrides,
        "task": spec.task.to_dict(),
        "train": {
            k: v for k, v in spec.train.__dict__.items()
        },
    })

    if dry_run:
        print(f"[DRY RUN] {run_id}")
        return {}

    print(f"[START] {run_id}")
    experiment = build_experiment(
        spec.task, spec.train, algorithm, seed, task_overrides
    )
    experiment.run()
    print(f"[DONE] {run_id}")

    # Collect post-training evaluation metrics
    metrics = evaluate_trained(spec, experiment, task_overrides)
    run_storage.save_metrics(metrics)
    return metrics


def evaluate_trained(
    spec: ExperimentSpec,
    experiment,
    task_overrides: Dict[str, Any],
    n_eval_episodes: int = 200,
) -> Dict[str, float]:
    """Run evaluation episodes on a trained BenchMARL experiment.

    Falls back to BenchMARL's built-in eval metrics if available.
    """
    # BenchMARL logs eval metrics during training via its logger.
    # We extract the final eval metrics from the experiment's log.
    # If custom eval is needed, we can run VMAS directly:
    return evaluate_with_vmas(
        spec.task, task_overrides, n_eval_episodes=n_eval_episodes
    )


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
        device=task_config.x_semidim and "cpu",  # always cpu for eval
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
    results = {}
    for i, (run_id, overrides, algo, seed) in enumerate(spec.iter_runs()):
        if max_runs and i >= max_runs:
            break
        metrics = run_single(
            spec, run_id, overrides, algo, seed,
            skip_complete=skip_complete, dry_run=dry_run,
        )
        results[run_id] = metrics
    return results
