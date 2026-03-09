"""Experiment configuration loading and management."""
import copy
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


RENDEZVOUS_ROOT = Path(__file__).parent.parent
CONFIGS_DIR = RENDEZVOUS_ROOT / "configs"
RESULTS_DIR = RENDEZVOUS_ROOT / "results"
CHECKPOINTS_DIR = RENDEZVOUS_ROOT / "checkpoints"

# ── Profiles ───────────────────────────────────────────────────────
# "fast" runs 1 config with 100 training iterations (~15 min on CPU).
# "complete" uses the YAML as-is (full sweep).

PROFILES = {
    "fast": {
        "train": {
            "max_n_frames": 6_000_000,
            "on_policy_n_envs_per_worker": 60,
            "evaluation_interval": 600_000,
            "evaluation_episodes": 50,
        },
        "sweep": {
            "seeds": [0],
            "n_agents": [4],
            "lidar_range": [0.35],
            "algorithms": ["mappo"],
        },
    },
    "complete": {},  # no overrides
}

# ── Short names for run IDs ───────────────────────────────────────
_SHORT = {
    "n_agents": "n",
    "n_targets": "t",
    "agents_per_target": "k",
    "lidar_range": "l",
}


@dataclass
class TaskConfig:
    """Discovery scenario parameters."""

    n_agents: int = 5
    n_targets: int = 7
    agents_per_target: int = 2
    lidar_range: float = 0.35
    covering_range: float = 0.25
    use_agent_lidar: bool = False
    n_lidar_rays_entities: int = 15
    n_lidar_rays_agents: int = 12
    targets_respawn: bool = False
    shared_reward: bool = False
    agent_collision_penalty: float = -0.1
    covering_rew_coeff: float = 1.0
    time_penalty: float = -0.01
    x_semidim: float = 1.0
    y_semidim: float = 1.0
    min_dist_between_entities: float = 0.2
    max_steps: int = 200

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class TrainConfig:
    """BenchMARL training parameters."""

    algorithm: str = "mappo"  # mappo, ippo, qmix, maddpg
    max_n_frames: int = 10_000_000
    gamma: float = 0.99
    on_policy_collected_frames_per_batch: int = 60_000
    on_policy_n_envs_per_worker: int = 600
    on_policy_n_minibatch_iters: int = 45
    on_policy_minibatch_size: int = 4096
    lr: float = 5e-5
    share_policy_params: bool = True
    evaluation_interval: int = 120_000
    evaluation_episodes: int = 200
    train_device: str = "cpu"
    sampling_device: str = "cpu"


@dataclass
class SweepConfig:
    """Defines parameter sweeps for an experiment."""

    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    n_agents: List[int] = field(default_factory=lambda: [4])
    n_targets: List[int] = field(default_factory=lambda: [7])
    agents_per_target: List[int] = field(default_factory=lambda: [2])
    lidar_range: List[float] = field(default_factory=lambda: [0.35])
    algorithms: List[str] = field(default_factory=lambda: ["mappo"])


@dataclass
class ExperimentSpec:
    """Full specification for one experiment (e.g., ER1)."""

    exp_id: str
    name: str
    description: str
    task: TaskConfig = field(default_factory=TaskConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)

    @property
    def results_dir(self) -> Path:
        return RESULTS_DIR / self.exp_id.lower()

    @property
    def checkpoints_dir(self) -> Path:
        return CHECKPOINTS_DIR / self.exp_id.lower()

    def ensure_dirs(self):
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def iter_runs(self):
        """Yield (run_id, task_overrides, algorithm, seed) for each sweep combo.

        Run IDs use short param names: n=agents, t=targets, k=agents_per_target, l=lidar.
        Example: er1_mappo_n4_t7_k2_l035_s0
        """
        sweep = self.sweep
        param_names = [
            "n_agents", "n_targets", "agents_per_target", "lidar_range"
        ]
        param_values = [
            sweep.n_agents, sweep.n_targets,
            sweep.agents_per_target, sweep.lidar_range,
        ]
        for algo in sweep.algorithms:
            for combo in itertools.product(*param_values):
                overrides = dict(zip(param_names, combo))
                for seed in sweep.seeds:
                    parts = []
                    for k, v in overrides.items():
                        short = _SHORT.get(k, k)
                        # Format lidar_range without dot: 0.35 → 035
                        if k == "lidar_range":
                            v_str = str(v).replace(".", "")
                        else:
                            v_str = str(v)
                        parts.append(f"{short}{v_str}")
                    run_id = f"{self.exp_id}_{algo}_{'_'.join(parts)}_s{seed}"
                    yield run_id, overrides, algo, seed


def load_experiment(
    yaml_path: str,
    profile: str = "complete",
) -> ExperimentSpec:
    """Load an experiment spec from a YAML config file.

    Args:
        yaml_path: path to the experiment YAML
        profile: "fast" for quick validation (~15 min), "complete" for full sweep
    """
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    # Apply profile overrides
    overrides = PROFILES.get(profile, {})
    if overrides:
        for section in ("task", "train", "sweep"):
            if section in overrides:
                raw.setdefault(section, {}).update(overrides[section])

    task = TaskConfig(**raw.get("task", {}))
    train = TrainConfig(**raw.get("train", {}))
    sweep = SweepConfig(**raw.get("sweep", {}))

    return ExperimentSpec(
        exp_id=raw["exp_id"],
        name=raw["name"],
        description=raw.get("description", ""),
        task=task,
        train=train,
        sweep=sweep,
    )
