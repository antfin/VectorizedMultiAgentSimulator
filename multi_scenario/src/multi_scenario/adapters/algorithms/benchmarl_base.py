"""Shared BenchMARL wiring for algorithm adapters.

Subclasses provide the algorithm-specific config via ``_algorithm_config``;
the base class handles task resolution, model config, BenchMARL
``ExperimentConfig`` translation, and the train/evaluate orchestration.

The ``env`` arg on ``train``/``evaluate`` is part of the ``Algorithm``
Protocol but unused here — BenchMARL builds its own env from the task
descriptor. Other (non-BenchMARL) algorithm adapters may use it.

The rollout shape returned by ``evaluate`` matches the contracts in F2.2 /
F2.3: ``episode_returns / episode_lengths / episode_collisions`` (universal)
plus ``targets_covered / n_targets`` for discovery. Aggregated from
BenchMARL's ``test_env.rollout()`` — pattern ported from
``rendezvous_comm/src/runner.py::evaluate_trained``.
"""

from typing import Any

import torch
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment
from benchmarl.experiment import ExperimentConfig as BenchmarlExperimentConfig
from benchmarl.models.mlp import MlpConfig
from torchrl.envs.utils import ExplorationType, set_exploration_type

from multi_scenario.domain.models import ExperimentConfig

_TASK_BY_TYPE: dict[str, VmasTask] = {
    "discovery": VmasTask.DISCOVERY,
    "navigation": VmasTask.NAVIGATION,
    "flocking": VmasTask.FLOCKING,
    "transport": VmasTask.TRANSPORT,
}


class BenchmarlBaseAdapter:
    """Shared scaffolding for BenchMARL-backed algorithm adapters."""

    # Subclasses override `name` and `_algorithm_config`.
    # pylint: disable=too-few-public-methods,unused-argument

    name: str = "benchmarl_base"

    def _algorithm_config(self, cfg: ExperimentConfig) -> Any:
        """Return the BenchMARL AlgorithmConfig (subclass responsibility)."""
        raise NotImplementedError

    def _task(self, cfg: ExperimentConfig) -> Any:
        scenario_type = cfg.scenario.type
        if scenario_type not in _TASK_BY_TYPE:
            raise ValueError(f"no VmasTask for scenario.type={scenario_type!r}")
        task = _TASK_BY_TYPE[scenario_type].get_from_yaml()
        # task.config is a plain dict; merge our params in (cfg overrides).
        task.config.update(cfg.scenario.params)
        return task

    def _model_config(self, cfg: ExperimentConfig) -> Any:
        return MlpConfig.get_from_yaml()

    def _experiment_config(
        self, cfg: ExperimentConfig, save_folder: str | None
    ) -> BenchmarlExperimentConfig:
        bm = BenchmarlExperimentConfig.get_from_yaml()
        # Bound smoke runs by iter count, not frame count.
        bm.max_n_iters = cfg.training.max_iters
        bm.max_n_frames = None
        bm.on_policy_n_envs_per_worker = cfg.training.num_envs
        bm.off_policy_n_envs_per_worker = cfg.training.num_envs
        bm.on_policy_collected_frames_per_batch = cfg.training.frames_per_batch
        bm.on_policy_minibatch_size = cfg.training.minibatch_size
        bm.on_policy_n_minibatch_iters = cfg.training.n_minibatch_iters
        bm.lr = cfg.training.lr
        bm.gamma = cfg.training.gamma
        bm.share_policy_params = cfg.training.share_policy_params
        bm.train_device = cfg.training.device
        bm.sampling_device = cfg.training.device
        bm.buffer_device = cfg.training.device
        # Eval cadence: cfg expresses it in iters; BenchMARL wants frames.
        bm.evaluation = True
        bm.evaluation_interval = cfg.evaluation.interval_iters * cfg.training.frames_per_batch
        bm.evaluation_episodes = cfg.evaluation.episodes
        bm.loggers = ["csv"]  # avoid wandb default; we don't need its setup
        bm.create_json = False
        bm.render = False  # avoid pyglet crashes in headless / OVH
        bm.checkpoint_interval = 0
        bm.checkpoint_at_end = False
        if save_folder is not None:
            bm.save_folder = save_folder
        return bm

    def train(self, env: Any, cfg: ExperimentConfig) -> Experiment:
        """Build the BenchMARL Experiment, run training, return the Experiment."""
        save_folder = cfg.runtime.storage.path if cfg.runtime is not None else None
        bm_cfg = self._experiment_config(cfg, save_folder=save_folder)
        experiment = Experiment(
            task=self._task(cfg),
            algorithm_config=self._algorithm_config(cfg),
            model_config=self._model_config(cfg),
            seed=cfg.experiment.seed,
            config=bm_cfg,
        )
        experiment.run()
        return experiment

    def evaluate(self, artifact: Experiment, env: Any, cfg: ExperimentConfig) -> dict[str, Any]:
        """Run eval episodes through the trained policy; aggregate to our rollout dict."""
        # The aggregation loop legitimately tracks several per-rollout
        # accumulators (returns / lengths / collisions / targets_covered);
        # extracting them into a separate state class would be more noise.
        # pylint: disable=too-many-locals
        experiment = artifact
        test_env = experiment.test_env
        policy = experiment.policy
        max_steps = experiment.max_steps
        num_envs = test_env.batch_size[0] if test_env.batch_size else 1
        n_episodes = cfg.evaluation.episodes
        n_rollouts = max(1, (n_episodes + num_envs - 1) // num_envs)

        all_returns: list[float] = []
        all_lengths: list[int] = []
        all_collisions: list[float] = []
        all_targets_covered: list[torch.Tensor] = []

        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            for _ in range(n_rollouts):
                rollout_td = test_env.rollout(
                    max_steps=max_steps,
                    policy=policy,
                    auto_cast_to_device=True,
                    break_when_any_done=False,
                )
                ne = rollout_td.batch_size[0]
                rollout_t = rollout_td.batch_size[1] if len(rollout_td.batch_size) > 1 else 1

                self._extract_returns(rollout_td, ne, experiment.group_map, all_returns)
                all_lengths.extend([rollout_t] * ne)
                self._extract_collisions(rollout_td, ne, experiment.group_map, all_collisions)

                if cfg.scenario.type == "discovery":
                    self._extract_targets_covered(
                        rollout_td, experiment.group_map, all_targets_covered
                    )

        rollout_dict: dict[str, Any] = {
            "episode_returns": torch.tensor(all_returns[:n_episodes], dtype=torch.float),
            "episode_lengths": torch.tensor(all_lengths[:n_episodes], dtype=torch.long),
            "episode_collisions": torch.tensor(all_collisions[:n_episodes], dtype=torch.float),
        }
        if cfg.scenario.type == "discovery":
            rollout_dict["n_targets"] = cfg.scenario.params.get(
                "n_targets",
                self._default_n_targets_from_task(),
            )
            if all_targets_covered:
                rollout_dict["targets_covered"] = torch.cat(all_targets_covered, dim=0)[:n_episodes]
        return rollout_dict

    @staticmethod
    def _extract_returns(rollout_td: Any, ne: int, group_map: Any, out: list[float]) -> None:
        for group in group_map:
            key = ("next", group, "reward")
            if key in rollout_td.keys(include_nested=True):
                rew = rollout_td[key]
                out.extend(rew.reshape(ne, -1).sum(dim=1).tolist())
                return

    @staticmethod
    def _extract_collisions(rollout_td: Any, ne: int, group_map: Any, out: list[float]) -> None:
        for group in group_map:
            key = ("next", group, "info", "collision_rew")
            if key in rollout_td.keys(include_nested=True):
                coll = rollout_td[key]
                per_env = (coll < 0).reshape(ne, -1).sum(dim=1).float()
                out.extend(per_env.tolist())
                return
        # collision_rew not exposed → record zeros so episode_collisions matches len.
        out.extend([0.0] * ne)

    @staticmethod
    def _extract_targets_covered(rollout_td: Any, group_map: Any, out: list[torch.Tensor]) -> None:
        for group in group_map:
            key = ("next", group, "info", "targets_covered")
            if key in rollout_td.keys(include_nested=True):
                tc = rollout_td[key]
                # Reshape into [ne, T]; target counts may carry per-agent dims.
                if tc.dim() >= 4:
                    tc_per_env = tc[:, :, 0, 0]
                elif tc.dim() == 3:
                    tc_per_env = tc[:, :, 0]
                else:
                    tc_per_env = tc.unsqueeze(0)
                # Cumsum: rendezvous_comm pattern — per-step *new* covered count
                # → cumulative total. NOT from `terminated` (project memory).
                out.append(tc_per_env.cumsum(dim=1))
                return

    def _default_n_targets_from_task(self) -> int:
        # Discovery scenario default; overridden by cfg in practice.
        return 7
