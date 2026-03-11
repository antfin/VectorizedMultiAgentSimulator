"""Experiment runner using BenchMARL.

Handles training, evaluation, and metric collection for each run
in an experiment sweep.
"""
import logging
import time
from typing import Any, Callable, Dict, Optional

import torch

from .config import ExperimentSpec, TaskConfig, TrainConfig
from .logging_setup import setup_run_logger, teardown_run_logger
from .metrics import EpisodeMetrics, compute_m7_sample_efficiency
from .provenance import save_provenance
from .report import generate_run_report, generate_sweep_report
from .storage import ExperimentStorage


_log = logging.getLogger("rendezvous")

# Track whether pyglet rendering is safe. First run in a process works;
# subsequent runs crash because pyglet's display state goes stale.
_render_available = True


def _make_progress_callback(total_frames, frames_per_batch, run_id):
    """Create a BenchMARL Callback that drives a tqdm progress bar."""
    from benchmarl.experiment import Callback
    from tqdm.auto import tqdm

    class _TqdmProgress(Callback):
        def __init__(self):
            super().__init__()
            self.total_iters = max(1, total_frames // frames_per_batch)
            self.pbar = tqdm(
                total=self.total_iters,
                desc=f"  {run_id}",
                unit="iter",
                bar_format=(
                    "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}]"
                ),
                leave=True,
            )

        def on_batch_collected(self, batch):
            self.pbar.update(1)

        def close(self):
            self.pbar.close()

    return _TqdmProgress()


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
    experiment_config.evaluation_interval = train_config.evaluation_interval
    experiment_config.evaluation_episodes = train_config.evaluation_episodes
    experiment_config.loggers = ["csv"]
    # Pyglet rendering works for the first run in a process but crashes
    # on subsequent runs (stale display state). Enable only when safe.
    experiment_config.render = _render_available

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
        if spec.source_path:
            save_provenance(run_storage.run_dir, spec.source_path)

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

        # Create iteration progress bar in notebooks
        progress_cb = None
        if quiet and _in_notebook():
            progress_cb = _make_progress_callback(
                total_frames=spec.train.max_n_frames,
                frames_per_batch=(
                    spec.train.on_policy_collected_frames_per_batch
                ),
                run_id=run_id,
            )

        experiment = build_experiment(
            spec.task, spec.train, algorithm, seed, task_overrides,
            save_folder=str(run_storage.benchmarl_dir),
            callbacks=[progress_cb] if progress_cb else None,
        )

        global _render_available
        t0 = time.monotonic()
        experiment.run()
        elapsed = time.monotonic() - t0
        # Disable rendering for subsequent runs — pyglet display goes stale
        _render_available = False

        if progress_cb is not None:
            progress_cb.close()

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

        # M7: Sample Efficiency (from training CSVs)
        m7 = compute_m7_sample_efficiency(run_storage.run_dir)
        if m7 is not None:
            metrics["M7_sample_efficiency"] = m7

        run_storage.save_metrics(metrics)

        from .report import METRIC_DETAILS
        for key, val in metrics.items():
            if key != "n_envs":
                detail = METRIC_DETAILS.get(key)
                label = detail["label"] if detail else key
                fmt = detail["fmt"] if detail else ".4f"
                logger.info(f"  {label}: {val:{fmt}}")

        # Generate report
        generate_run_report(
            run_storage.run_dir, run_id, spec, metrics,
            elapsed_seconds=elapsed, task_overrides=task_overrides,
        )
        logger.info(f"DONE  {run_id} — report saved to {run_storage.run_dir / 'report.md'}")

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
    """
    from torchrl.envs.utils import ExplorationType, set_exploration_type

    test_env = experiment.test_env
    policy = experiment.policy
    max_steps = experiment.max_steps

    n_targets = task_overrides.get("n_targets", 7) if task_overrides else 7
    n_agents = task_overrides.get("n_agents", 5) if task_overrides else 5

    all_returns = []
    all_steps = []
    all_collisions = []
    all_targets_covered = []
    all_agent_covering = []  # list of (n_agents,) tensors
    all_spatial_spread = []
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

            ep_steps = rollout_td.batch_size[0]
            all_steps.append(ep_steps)

            # M2: Extract rewards
            episode_return = 0.0
            for group in experiment.group_map.keys():
                reward_key = ("next", group, "reward")
                if reward_key in rollout_td.keys(True):
                    episode_return += rollout_td[reward_key].sum().item()
            all_returns.append(episode_return)

            # M1: Check if episode completed
            done_key = ("next", "done")
            episode_done = False
            if done_key in rollout_td.keys(True):
                if rollout_td[done_key].any().item():
                    n_done += 1
                    episode_done = True
            if not episode_done:
                all_steps[-1] = max_steps

            # M4: Collisions
            ep_collisions = 0.0
            for group in experiment.group_map.keys():
                collision_key = ("next", group, "info", "collision_rew")
                if collision_key in rollout_td.keys(True):
                    coll_rew = rollout_td[collision_key]
                    ep_collisions += (coll_rew < 0).sum().item()
            all_collisions.append(ep_collisions)

            # M6: Coverage progress
            ep_targets = 0.0
            for group in experiment.group_map.keys():
                tc_key = ("next", group, "info", "targets_covered")
                if tc_key in rollout_td.keys(True):
                    tc = rollout_td[tc_key]
                    # Sum across time steps; divide by n_agents if stacked
                    ep_targets = tc.sum().item()
                    if tc.dim() > 1 and tc.shape[-1] > 1:
                        ep_targets /= tc.shape[-1]
                    break
            all_targets_covered.append(min(ep_targets, n_targets))

            # M8: Per-agent covering counts
            ep_agent_covering = torch.zeros(n_agents)
            for group in experiment.group_map.keys():
                cov_key = ("next", group, "info", "covering_reward")
                if cov_key in rollout_td.keys(True):
                    cov_rew = rollout_td[cov_key]  # (T, n_agents) or (T,)
                    if cov_rew.dim() >= 2:
                        for i in range(min(cov_rew.shape[-1], n_agents)):
                            ep_agent_covering[i] = (
                                cov_rew[..., i] > 0
                            ).sum().item()
                    else:
                        ep_agent_covering[0] = (cov_rew > 0).sum().item()
                    break
            all_agent_covering.append(ep_agent_covering)

            # M9: Spatial spread from observations
            # Discovery obs layout: [pos_x, pos_y, vel_x, vel_y, lidar...]
            ep_spread = 0.0
            spread_steps = 0
            for group in experiment.group_map.keys():
                obs_key = (group, "observation")
                if obs_key in rollout_td.keys(True):
                    obs = rollout_td[obs_key]
                    # obs shape: (T, n_agents, obs_dim) with shared policy
                    # or (T, obs_dim) without. Need exactly 3 dims.
                    if obs.dim() == 2:
                        # Single agent or stacked — can't compute pairwise
                        break
                    if obs.dim() >= 3:
                        # Take last 3 dims as (T, n_agents, obs_dim)
                        # Flatten any leading batch dims
                        while obs.dim() > 3:
                            obs = obs.reshape(-1, *obs.shape[-2:])
                        T_steps, n_ag, obs_dim = obs.shape
                        if n_ag >= 2 and obs_dim >= 2:
                            positions = obs[:, :, :2]  # (T, n_agents, 2)
                            # Vectorized pairwise distances
                            dists = torch.cdist(positions, positions)  # (T, n_ag, n_ag)
                            mask = torch.triu(
                                torch.ones(n_ag, n_ag, device=obs.device), diagonal=1
                            ).bool()
                            for t in range(T_steps):
                                d = dists[t]  # (n_ag, n_ag)
                                if mask.sum() > 0:
                                    ep_spread += d[mask].mean().item()
                                    spread_steps += 1
                    break
            if spread_steps > 0:
                all_spatial_spread.append(ep_spread / spread_steps)
            else:
                all_spatial_spread.append(0.0)

    # Aggregate
    avg_coverage = (
        sum(t / n_targets for t in all_targets_covered)
        / max(len(all_targets_covered), 1)
    )

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
        "M1_success_rate": n_done / max(total_episodes, 1),
        "M2_avg_return": sum(all_returns) / max(len(all_returns), 1),
        "M3_avg_steps": sum(all_steps) / max(len(all_steps), 1),
        "M4_avg_collisions": sum(all_collisions) / max(len(all_collisions), 1),
        "M5_avg_tokens": 0.0,
        "M6_coverage_progress": avg_coverage,
        "M8_agent_utilization": agent_util,
        "M9_spatial_spread": (
            sum(all_spatial_spread) / max(len(all_spatial_spread), 1)
        ),
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

    # Generate sweep report
    generate_sweep_report(spec, results, elapsed_seconds=elapsed)

    if not notebook:
        _log.info(f"Sweep complete: {len(results)} runs in {elapsed:.0f}s")
        _log.info(
            f"Sweep report: {spec.results_dir / 'sweep_report.md'}"
        )

    return results
