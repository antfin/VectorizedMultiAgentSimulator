"""Integration tests verifying all artifacts land in the correct folders.

Simulates a complete run lifecycle (config save, logging, provenance,
metrics, report) and checks every file is in the expected location
within the results/<exp_id>/YYYYMMDD_HHMM__<run_id>/ tree.
"""
import json
import re
from pathlib import Path

import pytest
import yaml

from src.config import ExperimentSpec, SweepConfig, TaskConfig, TrainConfig
from src.logging_setup import setup_run_logger, teardown_run_logger
from src.provenance import load_provenance, save_provenance
from src.report import generate_run_report, generate_sweep_report
from src.storage import ExperimentStorage, RunStorage


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def spec(tmp_path):
    """Minimal ExperimentSpec with source_path pointing to a temp YAML."""
    yaml_path = tmp_path / "configs" / "er1" / "test.yaml"
    yaml_path.parent.mkdir(parents=True)
    yaml_path.write_text(yaml.dump({
        "exp_id": "er1",
        "name": "Test Experiment",
        "description": "Integration test",
        "task": {"n_agents": 4, "n_targets": 3},
        "train": {"algorithm": "mappo", "max_n_frames": 100_000},
        "sweep": {"seeds": [0], "algorithms": ["mappo"]},
    }))
    return ExperimentSpec(
        exp_id="er1",
        name="Test Experiment",
        description="Integration test",
        task=TaskConfig(n_agents=4, n_targets=3),
        train=TrainConfig(algorithm="mappo", max_n_frames=100_000),
        sweep=SweepConfig(seeds=[0], algorithms=["mappo"]),
        source_path=yaml_path,
    )


@pytest.fixture
def sample_metrics():
    return {
        "M1_success_rate": 0.75,
        "M2_avg_return": 12.5,
        "M3_avg_steps": 150.0,
        "M4_avg_collisions": 3.2,
        "M5_avg_tokens": 0.0,
        "M6_coverage_progress": 0.85,
        "M8_agent_utilization": 0.3,
        "M9_spatial_spread": 0.9,
        "n_envs": 200,
    }


# ── Run-level folder structure ───────────────────────────────────


class TestRunFolderStructure:
    """Verify all per-run artifacts land in the correct subdirectories."""

    def test_run_dir_has_timestamped_name(self, tmp_path, spec):
        storage = ExperimentStorage("er1", results_root=tmp_path)
        rs = storage.get_run("er1_mappo_n4_t3_k2_l035_s0")
        assert re.match(
            r"\d{8}_\d{4}__er1_mappo_n4_t3_k2_l035_s0$",
            rs.run_dir.name,
        )

    def test_run_dir_under_exp_id(self, tmp_path, spec):
        storage = ExperimentStorage("er1", results_root=tmp_path)
        rs = storage.get_run("er1_mappo_n4_t3_k2_l035_s0")
        assert rs.run_dir.parent.name == "er1"
        assert rs.run_dir.parent.parent == tmp_path

    def test_subdirectories_created(self, tmp_path):
        rs = RunStorage(tmp_path / "run1", "run1")
        assert (rs.run_dir / "input").is_dir()
        assert (rs.run_dir / "logs").is_dir()
        assert (rs.run_dir / "output").is_dir()
        assert (rs.run_dir / "output" / "benchmarl").is_dir()

    def test_config_in_input_dir(self, tmp_path):
        rs = RunStorage(tmp_path / "run1", "run1")
        rs.save_config({"algo": "mappo", "n_agents": 4})

        config_path = rs.run_dir / "input" / "config.yaml"
        assert config_path.exists()
        assert config_path.parent == rs.input_dir

    def test_provenance_in_input_dir(self, tmp_path, spec):
        rs = RunStorage(tmp_path / "run1", "run1")
        save_provenance(rs.run_dir, spec.source_path)

        prov_path = rs.run_dir / "input" / "provenance.json"
        assert prov_path.exists()
        prov = load_provenance(rs.run_dir)
        assert prov is not None
        assert prov.config_hash.startswith("sha256:")

    def test_log_file_in_logs_dir(self, tmp_path):
        rs = RunStorage(tmp_path / "run1", "run1")
        logger = setup_run_logger(rs.run_dir, name="test_folder_log")
        logger.info("Test message")
        teardown_run_logger(logger)

        log_path = rs.run_dir / "logs" / "run.log"
        assert log_path.exists()
        assert log_path.parent == rs.logs_dir
        assert "Test message" in log_path.read_text()

    def test_training_log_in_logs_dir(self, tmp_path):
        rs = RunStorage(tmp_path / "run1", "run1")
        rs.append_training_log({"step": 100, "loss": 0.5})

        csv_path = rs.run_dir / "logs" / "training_log.csv"
        assert csv_path.exists()
        assert csv_path.parent == rs.logs_dir

    def test_metrics_in_output_dir(self, tmp_path, sample_metrics):
        rs = RunStorage(tmp_path / "run1", "run1")
        rs.save_metrics(sample_metrics)

        metrics_path = rs.run_dir / "output" / "metrics.json"
        assert metrics_path.exists()
        assert metrics_path.parent == rs.output_dir

        loaded = json.loads(metrics_path.read_text())
        assert loaded["M1_success_rate"] == 0.75

    def test_eval_episodes_in_output_dir(self, tmp_path):
        rs = RunStorage(tmp_path / "run1", "run1")
        rs.save_eval_episodes([{"steps": 50}])

        path = rs.run_dir / "output" / "eval_episodes.json"
        assert path.exists()
        assert path.parent == rs.output_dir

    def test_policy_in_output_dir(self, tmp_path):
        rs = RunStorage(tmp_path / "run1", "run1")
        # Simulate policy save without torch
        policy_path = rs.output_dir / "policy.pt"
        policy_path.write_bytes(b"fake_weights")

        assert (rs.run_dir / "output" / "policy.pt").exists()
        assert rs.has_policy()

    def test_report_at_run_root(self, tmp_path, spec, sample_metrics):
        rs = RunStorage(tmp_path / "run1", "run1")
        generate_run_report(
            rs.run_dir, "run1", spec, sample_metrics, elapsed_seconds=60,
        )

        report_path = rs.run_dir / "report.md"
        assert report_path.exists()
        # NOT inside output/ or input/ — at run root
        assert report_path.parent == rs.run_dir

    def test_report_not_in_output_dir(self, tmp_path, spec, sample_metrics):
        rs = RunStorage(tmp_path / "run1", "run1")
        generate_run_report(
            rs.run_dir, "run1", spec, sample_metrics,
        )
        assert not (rs.output_dir / "report.md").exists()

    def test_benchmarl_dir_in_output(self, tmp_path):
        rs = RunStorage(tmp_path / "run1", "run1")
        assert rs.benchmarl_dir == rs.output_dir / "benchmarl"
        assert rs.benchmarl_dir.is_dir()


# ── Sweep-level folder structure ─────────────────────────────────


class TestSweepFolderStructure:
    """Verify sweep-level artifacts at results/<exp_id>/ root."""

    def test_sweep_report_at_results_root(self, tmp_path, spec, sample_metrics):
        results_dir = tmp_path / "er1"
        results_dir.mkdir()

        all_metrics = {"run_s0": sample_metrics}
        generate_sweep_report(
            spec, all_metrics, results_dir=results_dir,
        )

        report_path = results_dir / "sweep_report.md"
        assert report_path.exists()
        # At results root, not inside any run folder
        assert report_path.parent == results_dir

    def test_sweep_report_not_inside_run_dirs(self, tmp_path, spec, sample_metrics):
        storage = ExperimentStorage("er1", results_root=tmp_path)
        rs = storage.get_run("er1_mappo_n4_t3_k2_l035_s0")
        rs.save_metrics(sample_metrics)

        generate_sweep_report(
            spec, {"er1_mappo_n4_t3_k2_l035_s0": sample_metrics},
            results_dir=storage.results_dir,
        )

        # sweep_report.md at exp root
        assert (storage.results_dir / "sweep_report.md").exists()
        # NOT inside any run folder
        assert not (rs.run_dir / "sweep_report.md").exists()

    def test_multiple_runs_each_in_own_folder(self, tmp_path):
        storage = ExperimentStorage("er1", results_root=tmp_path)

        rs0 = storage.get_run("er1_mappo_n4_t3_k2_l035_s0")
        rs0.save_metrics({"M1": 0.8})

        rs1 = storage.get_run("er1_mappo_n4_t3_k2_l035_s1")
        rs1.save_metrics({"M1": 0.9})

        # Each run has its own folder
        assert rs0.run_dir != rs1.run_dir
        assert rs0.run_dir.parent == rs1.run_dir.parent  # both under er1/
        assert (rs0.output_dir / "metrics.json").exists()
        assert (rs1.output_dir / "metrics.json").exists()

    def test_exp_id_lowercased(self, tmp_path):
        storage = ExperimentStorage("ER1", results_root=tmp_path)
        assert storage.results_dir.name == "er1"


# ── Full lifecycle integration ───────────────────────────────────


class TestFullRunLifecycle:
    """Simulate a complete run and verify every artifact location."""

    def test_complete_run_produces_expected_tree(
        self, tmp_path, spec, sample_metrics,
    ):
        """Simulate all steps of run_single() and verify the folder tree."""
        storage = ExperimentStorage("er1", results_root=tmp_path)
        run_id = "er1_mappo_n4_t3_k2_l035_s0"
        rs = storage.get_run(run_id)

        # 1. Save config (like run_single does)
        rs.save_config({
            "exp_id": "er1",
            "run_id": run_id,
            "algorithm": "mappo",
            "seed": 0,
            "task": spec.task.to_dict(),
        })

        # 2. Save provenance
        save_provenance(rs.run_dir, spec.source_path)

        # 3. Setup logger
        logger = setup_run_logger(
            rs.run_dir, name="test_lifecycle_log",
        )
        logger.info("Training started")

        # 4. Simulate training log
        rs.append_training_log({"step": 100, "reward": 5.0})
        rs.append_training_log({"step": 200, "reward": 8.0})

        # 5. Save metrics
        rs.save_metrics(sample_metrics)

        # 6. Save eval episodes
        rs.save_eval_episodes([{"ep": 1, "done": True}])

        # 7. Simulate policy save
        (rs.output_dir / "policy.pt").write_bytes(b"weights")

        # 8. Generate report
        generate_run_report(
            rs.run_dir, run_id, spec, sample_metrics,
            elapsed_seconds=120,
        )

        # 9. Cleanup logger
        logger.info("Training complete")
        teardown_run_logger(logger)

        # ── Verify complete folder tree ──
        run_dir = rs.run_dir

        # input/
        assert (run_dir / "input" / "config.yaml").is_file()
        assert (run_dir / "input" / "provenance.json").is_file()

        # logs/
        assert (run_dir / "logs" / "run.log").is_file()
        assert (run_dir / "logs" / "training_log.csv").is_file()

        # output/
        assert (run_dir / "output" / "metrics.json").is_file()
        assert (run_dir / "output" / "eval_episodes.json").is_file()
        assert (run_dir / "output" / "policy.pt").is_file()
        assert (run_dir / "output" / "benchmarl").is_dir()

        # report at root
        assert (run_dir / "report.md").is_file()

        # Verify no stray files at wrong levels
        assert not (run_dir / "output" / "report.md").exists()
        assert not (run_dir / "input" / "metrics.json").exists()
        assert not (run_dir / "config.yaml").exists()
        assert not (run_dir / "metrics.json").exists()

    def test_complete_sweep_produces_expected_tree(
        self, tmp_path, spec, sample_metrics,
    ):
        """Simulate a 2-run sweep and verify sweep + run artifacts."""
        storage = ExperimentStorage("er1", results_root=tmp_path)

        run_ids = [
            "er1_mappo_n4_t3_k2_l035_s0",
            "er1_mappo_n4_t3_k2_l035_s1",
        ]
        all_metrics = {}

        for run_id in run_ids:
            rs = storage.get_run(run_id)
            rs.save_config({"run_id": run_id})
            rs.save_metrics(sample_metrics)
            generate_run_report(
                rs.run_dir, run_id, spec, sample_metrics,
            )
            all_metrics[run_id] = sample_metrics

        # Generate sweep report
        generate_sweep_report(
            spec, all_metrics, results_dir=storage.results_dir,
        )

        # ── Verify sweep-level artifacts ──
        assert (storage.results_dir / "sweep_report.md").is_file()

        # ── Verify each run has its own complete tree ──
        for run_id in run_ids:
            rs = storage.get_run(run_id)
            assert (rs.run_dir / "input" / "config.yaml").is_file()
            assert (rs.run_dir / "output" / "metrics.json").is_file()
            assert (rs.run_dir / "report.md").is_file()

        # ── Verify runs are separate folders ──
        run_dirs = storage.list_run_dirs()
        assert len(run_dirs) == 2
        assert run_dirs[0] != run_dirs[1]

    def test_is_complete_only_after_metrics_saved(self, tmp_path):
        """Run isn't complete until metrics.json exists."""
        storage = ExperimentStorage("er1", results_root=tmp_path)
        rs = storage.get_run("er1_mappo_n4_t3_k2_l035_s0")

        # Save everything except metrics
        rs.save_config({"test": True})
        (rs.output_dir / "policy.pt").write_bytes(b"weights")

        assert not rs.is_complete()
        assert storage.list_runs() == []

        # Now save metrics
        rs.save_metrics({"M1": 0.5})

        assert rs.is_complete()
        assert storage.list_runs() == ["er1_mappo_n4_t3_k2_l035_s0"]


# ── Cross-module path consistency ────────────────────────────────


class TestCrossModulePathConsistency:
    """Verify different modules agree on file locations."""

    def test_provenance_and_storage_share_input_dir(self, tmp_path, spec):
        rs = RunStorage(tmp_path / "run1", "run1")
        save_provenance(rs.run_dir, spec.source_path)

        # provenance.py writes to run_dir/input/provenance.json
        # storage.py defines input_dir as run_dir/input/
        prov_path = rs.run_dir / "input" / "provenance.json"
        assert prov_path.exists()
        assert prov_path.parent == rs.input_dir

    def test_logging_and_storage_share_logs_dir(self, tmp_path):
        rs = RunStorage(tmp_path / "run1", "run1")
        logger = setup_run_logger(
            rs.run_dir, name="test_path_consistency",
        )
        logger.info("hello")
        teardown_run_logger(logger)

        log_path = rs.run_dir / "logs" / "run.log"
        assert log_path.exists()
        assert log_path.parent == rs.logs_dir

    def test_report_uses_run_dir_root(self, tmp_path, spec, sample_metrics):
        rs = RunStorage(tmp_path / "run1", "run1")
        report_text = generate_run_report(
            rs.run_dir, "run1", spec, sample_metrics,
        )

        # report.py writes to run_dir/report.md
        report_path = rs.run_dir / "report.md"
        assert report_path.exists()
        assert report_path.read_text() == report_text

    def test_experiment_storage_finds_run_after_full_lifecycle(
        self, tmp_path, spec, sample_metrics,
    ):
        """ExperimentStorage.list_runs() finds runs created by RunStorage."""
        storage = ExperimentStorage("er1", results_root=tmp_path)
        run_id = "er1_mappo_n4_t3_k2_l035_s0"

        rs = storage.get_run(run_id)
        rs.save_config({"test": True})
        rs.save_metrics(sample_metrics)

        # load_all_metrics should find this run
        all_m = storage.load_all_metrics()
        assert run_id in all_m
        assert all_m[run_id]["M1_success_rate"] == 0.75
