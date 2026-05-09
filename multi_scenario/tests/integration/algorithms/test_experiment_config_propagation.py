"""F2.4.1 tests: cfg.training knobs propagate into BenchMARL ExperimentConfig."""

# Tests poke at the protected `_experiment_config` / `_algorithm_config` so we
# can verify the cfg-to-BenchMARL translation without doing a full training run.
# pylint: disable=protected-access

from multi_scenario.adapters.algorithms.mappo import MappoAdapter
from multi_scenario.domain.models import ExperimentConfig


def _cfg() -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "experiment": {"id": "propagation_test", "seed": 0},
            "scenario": {"type": "discovery", "params": {"n_targets": 3}},
            "algorithm": {"type": "mappo", "params": {"lmbda": 0.97}},
            "training": {
                "max_iters": 5,
                "num_envs": 8,
                "device": "cpu",
                "lr": 1e-4,
                "gamma": 0.95,
                "frames_per_batch": 1234,
                "minibatch_size": 256,
                "n_minibatch_iters": 12,
                "share_policy_params": False,
            },
            "evaluation": {"interval_iters": 3, "episodes": 7},
        }
    )


def test_training_knobs_flow_through_to_benchmarl():
    """All cfg.training fields land on the BenchMARL ExperimentConfig."""
    bm = MappoAdapter()._experiment_config(_cfg(), save_folder=None)  # noqa: SLF001
    assert bm.max_n_iters == 5
    assert bm.lr == 1e-4
    assert bm.gamma == 0.95
    assert bm.on_policy_collected_frames_per_batch == 1234
    assert bm.on_policy_minibatch_size == 256
    assert bm.on_policy_n_minibatch_iters == 12
    assert bm.share_policy_params is False
    assert bm.on_policy_n_envs_per_worker == 8
    assert bm.train_device == "cpu"
    assert bm.render is False


def test_evaluation_interval_in_frames_not_iters():
    """eval cadence: cfg expresses iters; BenchMARL receives frames."""
    bm = MappoAdapter()._experiment_config(_cfg(), save_folder=None)  # noqa: SLF001
    # interval_iters=3 × frames_per_batch=1234 = 3702 frames
    assert bm.evaluation_interval == 3702
    assert bm.evaluation_episodes == 7


def test_save_folder_is_propagated():
    """Explicit save_folder lands on the BenchMARL config."""
    bm = MappoAdapter()._experiment_config(
        _cfg(), save_folder="/tmp/runs"
    )  # noqa: SLF001
    assert bm.save_folder == "/tmp/runs"


def test_mappo_algorithm_params_route_to_mappo_config():
    """`lmbda` (algorithm-specific) lands on MappoConfig, not ExperimentConfig."""
    algo = MappoAdapter()._algorithm_config(_cfg())  # noqa: SLF001
    assert algo.lmbda == 0.97
