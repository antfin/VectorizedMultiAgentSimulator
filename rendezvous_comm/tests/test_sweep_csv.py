"""Tests for sweep CSV export and enriched metrics."""
import csv
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.runner import _write_sweep_csv, _extract_training_dynamics
from src.storage import ExperimentStorage, RunStorage


# ── _write_sweep_csv ─────────────────────────────────────────────


class TestWriteSweepCsv:
    """Test _write_sweep_csv function."""

    def test_writes_csv_with_all_fields(self, tmp_path):
        spec = MagicMock()
        spec.results_dir = tmp_path
        results = {
            "run1": {
                "M1_success_rate": 0.8,
                "algorithm": "mappo",
                "seed": 0,
                "training_seconds": 120.5,
            },
            "run2": {
                "M1_success_rate": 0.9,
                "algorithm": "ippo",
                "seed": 1,
                "training_seconds": 130.0,
            },
        }
        _write_sweep_csv(spec, results)

        csv_path = tmp_path / "sweep_results.csv"
        assert csv_path.exists()
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["run_id"] == "run1"
        assert "M1_success_rate" in reader.fieldnames
        assert "algorithm" in reader.fieldnames
        assert "training_seconds" in reader.fieldnames

    def test_empty_results_no_file(self, tmp_path):
        spec = MagicMock()
        spec.results_dir = tmp_path
        _write_sweep_csv(spec, {})
        assert not (tmp_path / "sweep_results.csv").exists()

    def test_heterogeneous_keys(self, tmp_path):
        """Runs with different metric sets still produce valid CSV."""
        spec = MagicMock()
        spec.results_dir = tmp_path
        results = {
            "run1": {"M1_success_rate": 0.8, "extra_field": 42},
            "run2": {"M1_success_rate": 0.9},
        }
        _write_sweep_csv(spec, results)
        with open(tmp_path / "sweep_results.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert "extra_field" in reader.fieldnames
        # run2 should have empty string for missing extra_field
        assert rows[1]["extra_field"] == ""

    def test_run_id_is_first_column(self, tmp_path):
        spec = MagicMock()
        spec.results_dir = tmp_path
        results = {"r1": {"M1_success_rate": 0.5, "z_field": 1}}
        _write_sweep_csv(spec, results)
        with open(tmp_path / "sweep_results.csv") as f:
            reader = csv.DictReader(f)
            list(reader)  # consume
            assert reader.fieldnames[0] == "run_id"

    def test_csv_values_are_correct(self, tmp_path):
        spec = MagicMock()
        spec.results_dir = tmp_path
        results = {
            "er1_mappo_n4_t4_k1_l035_s0": {
                "M1_success_rate": 0.63,
                "n_agents": 4,
                "seed": 0,
                "training_seconds": 300.5,
            },
        }
        _write_sweep_csv(spec, results)
        with open(tmp_path / "sweep_results.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        row = rows[0]
        assert row["run_id"] == "er1_mappo_n4_t4_k1_l035_s0"
        assert float(row["M1_success_rate"]) == pytest.approx(0.63)
        assert int(row["n_agents"]) == 4
        assert int(row["seed"]) == 0
        assert float(row["training_seconds"]) == pytest.approx(300.5)

    def test_csv_contains_full_config_and_execution(self, tmp_path):
        """CSV should contain all fields for OVH self-contained runs."""
        spec = MagicMock()
        spec.results_dir = tmp_path
        # Simulate a fully enriched metrics dict (as produced by run_single)
        results = {
            "er1_mappo_n4_t4_k1_l035_s0": {
                # Metrics
                "M1_success_rate": 0.63,
                "M2_avg_return": 0.089,
                "M4_avg_collisions": 6.7,
                # Identity
                "exp_id": "er1",
                "experiment_name": "No-Comm Control",
                "algorithm": "mappo",
                "seed": 0,
                "config_file": "demo.yaml",
                "run_timestamp": "20260315_1121",
                # Task config
                "n_agents": 4,
                "n_targets": 4,
                "agents_per_target": 1,
                "lidar_range": 0.35,
                "covering_range": 0.25,
                "max_steps": 200,
                "agent_collision_penalty": -0.1,
                "time_penalty": -0.01,
                "shared_reward": False,
                "targets_respawn": False,
                # Train config
                "max_n_frames": 10000000,
                "gamma": 0.99,
                "lr": 5e-5,
                "frames_per_batch": 60000,
                "n_envs_per_worker": 60,
                "share_policy_params": True,
                "evaluation_interval": 960000,
                # Execution
                "training_seconds": 300.5,
                "eval_seconds": 12.3,
                "device": "cuda",
                "torch_version": "2.2.0",
                "n_iterations": 166,
                "throughput_fps": 33222.0,
                "policy_params": 71941,
                # Dynamics
                "final_entropy": 1.2,
                "final_eval_reward": 2.3,
            },
        }
        _write_sweep_csv(spec, results)
        with open(tmp_path / "sweep_results.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        row = rows[0]
        # Check a sample of all categories
        assert row["exp_id"] == "er1"
        assert row["experiment_name"] == "No-Comm Control"
        assert row["config_file"] == "demo.yaml"
        assert float(row["covering_range"]) == pytest.approx(0.25)
        assert float(row["gamma"]) == pytest.approx(0.99)
        assert float(row["lr"]) == pytest.approx(5e-5)
        assert int(row["n_iterations"]) == 166
        assert float(row["throughput_fps"]) == pytest.approx(33222.0)
        assert int(row["policy_params"]) == 71941
        assert float(row["final_entropy"]) == pytest.approx(1.2)
        assert row["device"] == "cuda"
        assert row["torch_version"] == "2.2.0"

    def test_sorted_columns(self, tmp_path):
        """Columns after run_id should be alphabetically sorted."""
        spec = MagicMock()
        spec.results_dir = tmp_path
        results = {
            "r1": {"z_metric": 1, "a_metric": 2, "m_metric": 3},
        }
        _write_sweep_csv(spec, results)
        with open(tmp_path / "sweep_results.csv") as f:
            reader = csv.DictReader(f)
            list(reader)
        cols = reader.fieldnames
        assert cols[0] == "run_id"
        assert cols[1:] == sorted(cols[1:])


# ── _extract_training_dynamics ───────────────────────────────────


class TestExtractTrainingDynamics:
    """Test _extract_training_dynamics reads BenchMARL CSVs."""

    def _make_run_with_scalars(self, tmp_path, scalars_data):
        """Create a RunStorage with fake BenchMARL CSV files."""
        rs = RunStorage(tmp_path / "run", "test_run")
        scalars_dir = rs.benchmarl_dir / "hash1" / "scalars"
        scalars_dir.mkdir(parents=True)
        for name, rows in scalars_data.items():
            csv_content = "\n".join(
                f"{step},{val}" for step, val in rows
            )
            (scalars_dir / f"{name}.csv").write_text(csv_content)
        return rs

    def test_extracts_entropy_and_reward(self, tmp_path):
        rs = self._make_run_with_scalars(tmp_path, {
            "train_agents_entropy": [(0, 2.5), (10, 1.8), (20, 1.2)],
            "eval_reward_episode_reward_mean": [
                (0, -5.0), (10, 0.5), (20, 2.3),
            ],
        })
        result = _extract_training_dynamics(rs)
        assert result["final_entropy"] == pytest.approx(1.2)
        assert result["final_eval_reward"] == pytest.approx(2.3)

    def test_missing_entropy(self, tmp_path):
        rs = self._make_run_with_scalars(tmp_path, {
            "eval_reward_episode_reward_mean": [(0, 1.0)],
        })
        result = _extract_training_dynamics(rs)
        assert "final_entropy" not in result
        assert result["final_eval_reward"] == pytest.approx(1.0)

    def test_empty_scalars(self, tmp_path):
        rs = RunStorage(tmp_path / "run", "test_run")
        result = _extract_training_dynamics(rs)
        assert result == {}


# ── to_dataframe flat field preference ───────────────────────────


class TestToDataFrameFlatFields:
    """to_dataframe() prefers flat fields over regex parsing."""

    def test_flat_fields_from_metrics(self, tmp_path):
        pd = pytest.importorskip("pandas")
        es = ExperimentStorage("er1", results_root=tmp_path)
        rs = es.get_run("er1_mappo_n4_t7_k2_l035_s0")
        rs.save_metrics({
            "M1_success_rate": 0.85,
            "n_agents": 4,
            "algorithm": "mappo",
            "seed": 0,
            "training_seconds": 120.5,
            "device": "cuda",
        })
        df = es.to_dataframe()
        assert df.iloc[0]["training_seconds"] == 120.5
        assert df.iloc[0]["device"] == "cuda"
        assert df.iloc[0]["n_agents"] == 4

    def test_fallback_to_regex_for_old_data(self, tmp_path):
        pd = pytest.importorskip("pandas")
        es = ExperimentStorage("er1", results_root=tmp_path)
        rs = es.get_run("er1_mappo_n4_t7_k2_l035_s0")
        # Old format: no flat config fields
        rs.save_metrics({"M1_success_rate": 0.85})
        df = es.to_dataframe()
        assert df.iloc[0]["algorithm"] == "mappo"
        assert df.iloc[0]["n_agents"] == 4

    def test_flat_fields_win_over_regex(self, tmp_path):
        """If metrics.json has n_agents=6, regex would parse n4 from run_id.
        Flat field should win."""
        pd = pytest.importorskip("pandas")
        es = ExperimentStorage("er1", results_root=tmp_path)
        # run_id says n4, but metrics says n6
        rs = es.get_run("er1_mappo_n4_t7_k2_l035_s0")
        rs.save_metrics({
            "M1_success_rate": 0.85,
            "n_agents": 6,  # overrides regex parse of n4
        })
        df = es.to_dataframe()
        assert df.iloc[0]["n_agents"] == 6

    def test_mixed_old_and_new_data(self, tmp_path):
        """Mix of old-format and new-format runs."""
        pd = pytest.importorskip("pandas")
        es = ExperimentStorage("er1", results_root=tmp_path)

        # Old format
        rs1 = es.get_run("er1_mappo_n4_t7_k2_l035_s0")
        rs1.save_metrics({"M1_success_rate": 0.5})

        # New format
        rs2 = es.get_run("er1_ippo_n6_t3_k1_l045_s1")
        rs2.save_metrics({
            "M1_success_rate": 0.9,
            "algorithm": "ippo",
            "n_agents": 6,
            "training_seconds": 200.0,
        })

        df = es.to_dataframe()
        assert len(df) == 2
        # Both should have algorithm
        algos = set(df["algorithm"])
        assert "mappo" in algos  # from regex
        assert "ippo" in algos   # from flat field
