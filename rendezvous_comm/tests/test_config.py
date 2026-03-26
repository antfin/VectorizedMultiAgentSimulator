"""Tests for the config module — TaskConfig, TrainConfig, SweepConfig,
ExperimentSpec, load_experiment, find_configs."""
from pathlib import Path

import pytest
import yaml

from src.config import (
    ExperimentSpec,
    SweepConfig,
    TaskConfig,
    TrainConfig,
    find_configs,
    load_experiment,
)


# ── Helpers ───────────────────────────────────────────────────────


def _minimal_yaml_data(**overrides):
    """Return a minimal valid YAML dict for ExperimentSpec."""
    base = {
        "exp_id": "er1",
        "name": "No-Comm Control",
        "description": "Baseline without communication",
    }
    base.update(overrides)
    return base


def _write_yaml(path, data):
    """Write *data* as YAML to *path* and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


# ── TaskConfig ────────────────────────────────────────────────────


class TestTaskConfig:
    """Verify TaskConfig defaults and to_dict round-trip."""

    def test_default_values(self):
        tc = TaskConfig()
        assert tc.n_agents == 5
        assert tc.n_targets == 7
        assert tc.agents_per_target == 2
        assert tc.lidar_range == 0.35
        assert tc.covering_range == 0.25
        assert tc.use_agent_lidar is False
        assert tc.n_lidar_rays_entities == 15
        assert tc.n_lidar_rays_agents == 12
        assert tc.targets_respawn is False
        assert tc.shared_reward is False
        assert tc.agent_collision_penalty == -0.1
        assert tc.covering_rew_coeff == 1.0
        assert tc.time_penalty == -0.01
        assert tc.x_semidim == 1.0
        assert tc.y_semidim == 1.0
        assert tc.min_dist_between_entities == 0.2
        assert tc.max_steps == 200

    def test_to_dict_returns_all_fields(self):
        tc = TaskConfig()
        d = tc.to_dict()
        assert isinstance(d, dict)
        # Every dataclass field should be present
        expected_keys = {
            "n_agents", "n_targets", "agents_per_target", "lidar_range",
            "covering_range", "use_agent_lidar", "n_lidar_rays_entities",
            "n_lidar_rays_agents", "targets_respawn", "shared_reward",
            "agent_collision_penalty", "covering_rew_coeff", "time_penalty",
            "x_semidim", "y_semidim", "min_dist_between_entities",
            "max_steps", "dim_c", "comm_proximity", "dict_obs",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values_match_instance(self):
        tc = TaskConfig(n_agents=3, lidar_range=0.5)
        d = tc.to_dict()
        assert d["n_agents"] == 3
        assert d["lidar_range"] == 0.5


# ── TrainConfig ───────────────────────────────────────────────────


class TestTrainConfig:
    """Verify TrainConfig defaults."""

    def test_default_values(self):
        tr = TrainConfig()
        assert tr.algorithm == "mappo"
        assert tr.max_n_frames == 10_000_000
        assert tr.gamma == 0.99
        assert tr.on_policy_collected_frames_per_batch == 60_000
        assert tr.on_policy_n_envs_per_worker == 600
        assert tr.on_policy_n_minibatch_iters == 45
        assert tr.on_policy_minibatch_size == 4096
        assert tr.lr == 5e-5
        assert tr.share_policy_params is True
        assert tr.evaluation_interval == 120_000
        assert tr.evaluation_episodes == 200
        assert tr.train_device == "cpu"
        assert tr.sampling_device == "cpu"


# ── SweepConfig ───────────────────────────────────────────────────


class TestSweepConfig:
    """Verify SweepConfig defaults."""

    def test_default_values(self):
        sw = SweepConfig()
        assert sw.seeds == [0, 1, 2, 3, 4]
        assert sw.algorithms == ["mappo"]
        assert sw.n_agents == [4]
        assert sw.n_targets == [7]
        assert sw.agents_per_target == [2]
        assert sw.lidar_range == [0.35]


# ── ExperimentSpec ────────────────────────────────────────────────


class TestExperimentSpecProperties:
    """Test results_dir, checkpoints_dir, ensure_dirs."""

    def test_results_dir(self):
        spec = ExperimentSpec(exp_id="ER1", name="Test", description="")
        # results_dir is results/er1/runs/
        assert spec.results_dir.name == "runs"
        assert spec.results_dir.parent.name == "er1"

    def test_checkpoints_dir(self):
        spec = ExperimentSpec(exp_id="ER2", name="Test", description="")
        assert spec.checkpoints_dir.name == "er2"
        assert spec.checkpoints_dir.parent.name == "checkpoints"

    def test_ensure_dirs_creates_directories(self, tmp_path, monkeypatch):
        import src.config as config_mod

        monkeypatch.setattr(config_mod, "RESULTS_DIR", tmp_path / "results")
        monkeypatch.setattr(
            config_mod, "CHECKPOINTS_DIR", tmp_path / "checkpoints"
        )

        spec = ExperimentSpec(exp_id="er1", name="T", description="")
        spec.ensure_dirs()

        assert (tmp_path / "results" / "er1").is_dir()
        assert (tmp_path / "checkpoints" / "er1").is_dir()


class TestExperimentSpecConfigTag:
    """Test config_tag() generation for single and sweep cases."""

    def test_single_run_tag(self):
        spec = ExperimentSpec(
            exp_id="er1", name="T", description="",
            sweep=SweepConfig(
                seeds=[0], algorithms=["mappo"],
                n_agents=[4], lidar_range=[0.35],
            ),
        )
        tag = spec.config_tag()
        assert tag.startswith("single_")
        assert "mappo" in tag
        assert "n4" in tag
        assert "l035" in tag

    def test_sweep_prefix_for_multiple_runs(self):
        spec = ExperimentSpec(
            exp_id="er1", name="T", description="",
            sweep=SweepConfig(
                seeds=[0, 1], algorithms=["mappo"],
                n_agents=[4], lidar_range=[0.35],
            ),
        )
        tag = spec.config_tag()
        assert tag.startswith("sweep_")

    def test_multiple_algorithms(self):
        spec = ExperimentSpec(
            exp_id="er1", name="T", description="",
            sweep=SweepConfig(
                seeds=[0], algorithms=["mappo", "ippo"],
                n_agents=[4], lidar_range=[0.35],
            ),
        )
        tag = spec.config_tag()
        assert tag.startswith("sweep_")
        assert "mappo-ippo" in tag

    def test_range_formatting(self):
        spec = ExperimentSpec(
            exp_id="er1", name="T", description="",
            sweep=SweepConfig(
                seeds=[0], algorithms=["mappo"],
                n_agents=[2, 4, 6], lidar_range=[0.25, 0.45],
            ),
        )
        tag = spec.config_tag()
        assert "n2-6" in tag
        assert "l025-045" in tag


class TestExperimentSpecIterRuns:
    """Test iter_runs() yields correct run_id / overrides combos."""

    def test_single_run_yields_one(self):
        spec = ExperimentSpec(
            exp_id="er1", name="T", description="",
            sweep=SweepConfig(
                seeds=[0], algorithms=["mappo"],
                n_agents=[4], n_targets=[7],
                agents_per_target=[2], lidar_range=[0.35],
            ),
        )
        runs = list(spec.iter_runs())
        assert len(runs) == 1

    def test_correct_number_of_combinations(self):
        spec = ExperimentSpec(
            exp_id="er1", name="T", description="",
            sweep=SweepConfig(
                seeds=[0, 1], algorithms=["mappo", "ippo"],
                n_agents=[4, 6], n_targets=[7],
                agents_per_target=[2], lidar_range=[0.25, 0.35],
            ),
        )
        runs = list(spec.iter_runs())
        # 2 algos * 2 n_agents * 1 n_targets * 1 k * 2 lidar * 2 seeds = 16
        assert len(runs) == 16

    def test_run_id_format(self):
        spec = ExperimentSpec(
            exp_id="er1", name="T", description="",
            sweep=SweepConfig(
                seeds=[0], algorithms=["mappo"],
                n_agents=[4], n_targets=[7],
                agents_per_target=[2], lidar_range=[0.35],
            ),
        )
        run_id, overrides, algo, seed = list(spec.iter_runs())[0]
        assert run_id == "er1_mappo_n4_t7_k2_l035_s0"

    def test_yield_tuple_structure(self):
        spec = ExperimentSpec(
            exp_id="er1", name="T", description="",
            sweep=SweepConfig(
                seeds=[42], algorithms=["ippo"],
                n_agents=[6], n_targets=[3],
                agents_per_target=[1], lidar_range=[0.5],
            ),
        )
        run_id, overrides, algo, seed = list(spec.iter_runs())[0]
        assert algo == "ippo"
        assert seed == 42
        assert overrides == {
            "n_agents": 6,
            "n_targets": 3,
            "agents_per_target": 1,
            "lidar_range": 0.5,
        }

    def test_overrides_keys(self):
        spec = ExperimentSpec(
            exp_id="er1", name="T", description="",
            sweep=SweepConfig(
                seeds=[0], algorithms=["mappo"],
                n_agents=[4], n_targets=[7],
                agents_per_target=[2], lidar_range=[0.35],
            ),
        )
        _, overrides, _, _ = list(spec.iter_runs())[0]
        assert set(overrides.keys()) == {
            "n_agents", "n_targets", "agents_per_target", "lidar_range",
        }


# ── load_experiment ───────────────────────────────────────────────


class TestLoadExperiment:
    """Test loading ExperimentSpec from YAML files."""

    def test_load_minimal_yaml(self, tmp_path):
        data = _minimal_yaml_data()
        path = _write_yaml(tmp_path / "cfg.yaml", data)
        spec = load_experiment(path)

        assert spec.exp_id == "er1"
        assert spec.name == "No-Comm Control"
        assert spec.description == "Baseline without communication"

    def test_source_path_is_resolved(self, tmp_path):
        data = _minimal_yaml_data()
        path = _write_yaml(tmp_path / "cfg.yaml", data)
        spec = load_experiment(path)

        assert spec.source_path is not None
        assert spec.source_path == path.resolve()
        assert spec.source_path.is_absolute()

    def test_missing_optional_fields_get_defaults(self, tmp_path):
        # No task/train/sweep sections — should get default dataclasses
        data = _minimal_yaml_data()
        path = _write_yaml(tmp_path / "cfg.yaml", data)
        spec = load_experiment(path)

        assert spec.task.n_agents == 5  # TaskConfig default
        assert spec.train.algorithm == "mappo"  # TrainConfig default
        assert spec.sweep.seeds == [0, 1, 2, 3, 4]  # SweepConfig default

    def test_load_with_task_overrides(self, tmp_path):
        data = _minimal_yaml_data(task={"n_agents": 4, "n_targets": 3})
        path = _write_yaml(tmp_path / "cfg.yaml", data)
        spec = load_experiment(path)

        assert spec.task.n_agents == 4
        assert spec.task.n_targets == 3
        # Other fields keep defaults
        assert spec.task.max_steps == 200

    def test_load_with_sweep_section(self, tmp_path):
        data = _minimal_yaml_data(
            sweep={
                "seeds": [0, 1],
                "algorithms": ["mappo", "ippo"],
                "n_agents": [4, 6],
            },
        )
        path = _write_yaml(tmp_path / "cfg.yaml", data)
        spec = load_experiment(path)

        assert spec.sweep.seeds == [0, 1]
        assert spec.sweep.algorithms == ["mappo", "ippo"]
        assert spec.sweep.n_agents == [4, 6]

    def test_load_with_train_overrides(self, tmp_path):
        data = _minimal_yaml_data(
            train={"algorithm": "qmix", "max_n_frames": 500_000},
        )
        path = _write_yaml(tmp_path / "cfg.yaml", data)
        spec = load_experiment(path)

        assert spec.train.algorithm == "qmix"
        assert spec.train.max_n_frames == 500_000

    def test_missing_description_defaults_to_empty(self, tmp_path):
        data = {"exp_id": "er1", "name": "T"}
        path = _write_yaml(tmp_path / "cfg.yaml", data)
        spec = load_experiment(path)

        assert spec.description == ""


# ── find_configs ──────────────────────────────────────────────────


class TestFindConfigs:
    """Test discovery of YAML configs for an experiment."""

    def test_nonexistent_exp_id_returns_empty(self):
        # Very unlikely to exist
        result = find_configs("nonexistent_experiment_xyz_999")
        assert result == []

    def test_returns_sorted_list(self, tmp_path, monkeypatch):
        import src.config as config_mod

        monkeypatch.setattr(config_mod, "CONFIGS_DIR", tmp_path)

        exp_dir = tmp_path / "er1"
        exp_dir.mkdir()

        # Write two valid configs with names that sort differently
        _write_yaml(
            exp_dir / "b_second.yaml",
            _minimal_yaml_data(),
        )
        _write_yaml(
            exp_dir / "a_first.yaml",
            _minimal_yaml_data(name="First"),
        )

        results = find_configs("er1")
        assert len(results) == 2
        # Sorted by filename
        assert results[0][0].name == "a_first.yaml"
        assert results[1][0].name == "b_second.yaml"
        # Each entry is (Path, ExperimentSpec)
        assert isinstance(results[0][1], ExperimentSpec)

    def test_skips_invalid_yaml(self, tmp_path, monkeypatch):
        import src.config as config_mod

        monkeypatch.setattr(config_mod, "CONFIGS_DIR", tmp_path)

        exp_dir = tmp_path / "er1"
        exp_dir.mkdir()

        # One valid, one invalid (missing required field exp_id)
        _write_yaml(exp_dir / "good.yaml", _minimal_yaml_data())
        _write_yaml(exp_dir / "bad.yaml", {"name": "No exp_id"})

        results = find_configs("er1")
        assert len(results) == 1
        assert results[0][0].name == "good.yaml"

    def test_case_insensitive_exp_id(self, tmp_path, monkeypatch):
        import src.config as config_mod

        monkeypatch.setattr(config_mod, "CONFIGS_DIR", tmp_path)

        exp_dir = tmp_path / "er1"
        exp_dir.mkdir()
        _write_yaml(exp_dir / "cfg.yaml", _minimal_yaml_data())

        results = find_configs("ER1")
        assert len(results) == 1
