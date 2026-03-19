"""Experiment configuration loading and management."""
import itertools
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


RENDEZVOUS_ROOT = Path(__file__).parent.parent
CONFIGS_DIR = RENDEZVOUS_ROOT / "configs"
RESULTS_DIR = Path(
    os.environ.get("RESULTS_DIR", str(RENDEZVOUS_ROOT / "results"))
)
CHECKPOINTS_DIR = Path(
    os.environ.get("CHECKPOINTS_DIR", str(RENDEZVOUS_ROOT / "checkpoints"))
)

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

    # Algorithm-level overrides (None = use BenchMARL defaults)
    entropy_coef: Optional[float] = None
    lmbda: Optional[float] = None
    clip_epsilon: Optional[float] = None

    # Model overrides (None = use BenchMARL defaults: [256,256] Tanh)
    hidden_layers: Optional[List[int]] = None
    activation: Optional[str] = None


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
    source_path: Optional[Path] = None

    @property
    def results_dir(self) -> Path:
        return RESULTS_DIR / self.exp_id.lower()

    @property
    def checkpoints_dir(self) -> Path:
        return CHECKPOINTS_DIR / self.exp_id.lower()

    def ensure_dirs(self):
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def config_tag(self) -> str:
        """Generate a descriptive filename tag from sweep dimensions.

        Examples:
            single_mappo_n4_l035        (1 run)
            sweep_mappo-ippo_n2-6_l025-045  (120 runs)
        """
        sweep = self.sweep
        total = sum(1 for _ in self.iter_runs())
        prefix = "single" if total == 1 else "sweep"

        # Algorithms
        algos = "-".join(sweep.algorithms)

        # Only include dimensions with variation or non-default values
        parts = [prefix, algos]

        def _range_str(vals, fmt=str):
            if len(vals) == 1:
                return fmt(vals[0])
            return f"{fmt(vals[0])}-{fmt(vals[-1])}"

        def _lidar_fmt(v):
            return str(v).replace(".", "")

        parts.append(f"n{_range_str(sweep.n_agents)}")
        parts.append(f"l{_range_str(sweep.lidar_range, _lidar_fmt)}")

        return "_".join(parts)

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
                        if k == "lidar_range":
                            v_str = str(v).replace(".", "")
                        else:
                            v_str = str(v)
                        parts.append(f"{short}{v_str}")
                    run_id = f"{self.exp_id}_{algo}_{'_'.join(parts)}_s{seed}"
                    yield run_id, overrides, algo, seed


def load_experiment(yaml_path) -> ExperimentSpec:
    """Load an experiment spec from a YAML config file.

    Config layout:
        configs/<exp_id>/<config_tag>.yaml

    Each YAML contains all parameters directly.
    """
    yaml_path = Path(yaml_path)
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

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
        source_path=yaml_path.resolve(),
    )


def find_configs(exp_id: str) -> List[Tuple[Path, ExperimentSpec]]:
    """Find all config YAMLs for an experiment.

    Returns [(yaml_path, spec), ...] sorted by filename.
    """
    config_dir = CONFIGS_DIR / exp_id.lower()
    if not config_dir.exists():
        return []
    results = []
    for yaml_file in sorted(config_dir.glob("*.yaml")):
        try:
            spec = load_experiment(yaml_file)
            results.append((yaml_file, spec))
        except Exception:
            continue
    return results
