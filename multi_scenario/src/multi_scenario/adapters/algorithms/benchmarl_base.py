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

from pathlib import Path
from typing import Any

import torch
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment
from benchmarl.experiment import ExperimentConfig as BenchmarlExperimentConfig
from benchmarl.models.mlp import MlpConfig
from torchrl.envs.utils import ExplorationType, set_exploration_type

from multi_scenario.adapters.video.recorder import VideoRecorder
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

    def build_experiment(self, cfg: ExperimentConfig, run_dir: Path | None) -> Experiment:
        """Construct the BenchMARL ``Experiment`` without running it.

        Exposed so callers (and the F2.11 video recorder) can access the
        random-init ``policy`` and the ``test_env`` *before* training starts.
        ``train`` calls this then ``experiment.run()`` in sequence.
        """
        save_folder: str | None
        if run_dir is not None:
            save_folder_path = run_dir / "output" / "benchmarl"
            save_folder_path.mkdir(parents=True, exist_ok=True)
            save_folder = str(save_folder_path)
        else:
            save_folder = None
        bm_cfg = self._experiment_config(cfg, save_folder=save_folder)
        return Experiment(
            task=self._task(cfg),
            algorithm_config=self._algorithm_config(cfg),
            model_config=self._model_config(cfg),
            seed=cfg.experiment.seed,
            config=bm_cfg,
        )

    def train(self, env: Any, cfg: ExperimentConfig, run_dir: Path | None = None) -> Experiment:
        """Build the BenchMARL Experiment, run training, return the Experiment.

        BenchMARL's native scalars / checkpoints land at ``run_dir/output/benchmarl/``
        when run_dir is supplied (per the §3.5.2 layout). Falls back to the
        BenchMARL default (cwd) when run_dir is None.

        F2.11: when ``runtime.runner.params.record_video`` is true (default off
        for ``*_smoke`` runs, on otherwise) and ``run_dir`` is set, records
        ``before_training.mp4`` (random-init policy) and ``after_training.mp4``
        (trained policy) under ``run_dir/output/videos/``.
        """
        experiment = self.build_experiment(cfg, run_dir)

        record = _should_record_video(cfg, run_dir)
        videos_dir = run_dir / "output" / "videos" if run_dir is not None else None
        if record and videos_dir is not None:
            VideoRecorder().record(
                test_env=experiment.test_env,
                policy=experiment.policy,
                max_steps=experiment.max_steps,
                output_path=videos_dir / "before_training.mp4",
            )

        experiment.run()

        if record and videos_dir is not None:
            VideoRecorder().record(
                test_env=experiment.test_env,
                policy=experiment.policy,
                max_steps=experiment.max_steps,
                output_path=videos_dir / "after_training.mp4",
            )

        return experiment

    def evaluate(
        self,
        artifact: Experiment,
        env: Any,
        cfg: ExperimentConfig,
        run_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Run eval episodes through the trained policy; aggregate to our rollout dict."""
        # The aggregation loop legitimately tracks several per-rollout
        # accumulators (returns / lengths / collisions / targets_covered);
        # extracting them into a separate state class would be more noise.
        # `run_dir` is part of the protocol but unused here — BenchMARL's eval
        # writes through the same save_folder set in `train`.
        # pylint: disable=too-many-locals
        del run_dir
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


def _should_record_video(cfg: ExperimentConfig, run_dir: Path | None) -> bool:
    """F2.11 gating: explicit ``record_video`` flag, defaulting to off-for-smoke.

    No ``run_dir`` → no place to write → False. Otherwise read
    ``cfg.runtime.runner.params.record_video``; missing flag falls back to
    the smoke-suffix heuristic (``*_smoke`` exp_ids default off).
    """
    if run_dir is None or cfg.runtime is None:
        return False
    default = not cfg.experiment.id.endswith("_smoke")
    return bool(cfg.runtime.runner.params.get("record_video", default))
