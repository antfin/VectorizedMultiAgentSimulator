"""Tests for the storage module (RunStorage, ExperimentStorage, helpers)."""
import csv
import json
import re
from pathlib import Path

import pytest
import yaml

from src.storage import (
    ExperimentStorage,
    RunStorage,
    _extract_run_id,
    _make_folder_name,
    _parse_run_id,
)


# ── _extract_run_id ─────────────────────────────────────────────


class TestExtractRunId:
    """Strip timestamp prefix from folder names."""

    def test_with_timestamp(self):
        result = _extract_run_id("20260309_1430__er1_mappo_n4_t7_k2_l035_s0")
        assert result == "er1_mappo_n4_t7_k2_l035_s0"

    def test_without_timestamp(self):
        result = _extract_run_id("er1_mappo_n4_t7_k2_l035_s0")
        assert result == "er1_mappo_n4_t7_k2_l035_s0"

    def test_different_timestamp(self):
        result = _extract_run_id("20251231_2359__er2_ippo_n6_t3_k1_l045_s5")
        assert result == "er2_ippo_n6_t3_k1_l045_s5"

    def test_empty_after_timestamp(self):
        result = _extract_run_id("20260101_0000__x")
        assert result == "x"


# ── _make_folder_name ────────────────────────────────────────────


class TestMakeFolderName:
    """Timestamped folder names."""

    def test_format_matches_pattern(self):
        run_id = "er1_mappo_n4_t7_k2_l035_s0"
        name = _make_folder_name(run_id)
        assert re.match(r"\d{8}_\d{4}__er1_mappo_n4_t7_k2_l035_s0$", name)

    def test_contains_run_id(self):
        run_id = "er2_ippo_n6_t3_k1_l045_s1"
        name = _make_folder_name(run_id)
        assert name.endswith(f"__{run_id}")

    def test_extract_roundtrip(self):
        run_id = "er1_mappo_n4_t7_k2_l035_s0"
        name = _make_folder_name(run_id)
        assert _extract_run_id(name) == run_id


# ── _parse_run_id ────────────────────────────────────────────────


class TestParseRunId:
    """Parse structured params from run_id strings."""

    def test_full_parse(self):
        parsed = _parse_run_id("er1_mappo_n4_t7_k2_l035_s0")
        assert parsed["seed"] == 0
        assert parsed["algorithm"] == "mappo"
        assert parsed["exp_id"] == "er1"
        assert parsed["n_agents"] == 4
        assert parsed["n_targets"] == 7
        assert parsed["agents_per_target"] == 2
        assert parsed["lidar_range"] == pytest.approx(0.35)

    @pytest.mark.parametrize("algo", ["mappo", "ippo", "qmix", "maddpg"])
    def test_algorithms(self, algo):
        parsed = _parse_run_id(f"er1_{algo}_n4_t7_k2_l035_s0")
        assert parsed["algorithm"] == algo

    @pytest.mark.parametrize("exp_id", ["er1", "er2", "er3", "er4", "e1"])
    def test_exp_ids(self, exp_id):
        parsed = _parse_run_id(f"{exp_id}_mappo_n4_t7_k2_l035_s0")
        assert parsed["exp_id"] == exp_id

    def test_different_seed(self):
        parsed = _parse_run_id("er1_mappo_n4_t7_k2_l035_s42")
        assert parsed["seed"] == 42

    def test_different_lidar(self):
        parsed = _parse_run_id("er1_mappo_n4_t7_k2_l045_s0")
        assert parsed["lidar_range"] == pytest.approx(0.45)

    def test_unknown_format_partial(self):
        parsed = _parse_run_id("unknown_string")
        # Should return a dict (possibly empty), not crash
        assert isinstance(parsed, dict)
        assert "algorithm" not in parsed

    def test_partial_match(self):
        parsed = _parse_run_id("er1_mappo_n4_s3")
        assert parsed["exp_id"] == "er1"
        assert parsed["algorithm"] == "mappo"
        assert parsed["n_agents"] == 4
        assert parsed["seed"] == 3
        assert "n_targets" not in parsed


# ── RunStorage ───────────────────────────────────────────────────


class TestRunStorage:
    """RunStorage directory creation and data round-trips."""

    def test_init_creates_dirs(self, tmp_path):
        run_dir = tmp_path / "my_run"
        rs = RunStorage(run_dir, "test_run")
        assert rs.input_dir.is_dir()
        assert rs.logs_dir.is_dir()
        assert rs.output_dir.is_dir()
        assert rs.benchmarl_dir.is_dir()

    def test_save_and_load_config(self, tmp_path):
        rs = RunStorage(tmp_path / "run1", "run1")
        config = {"algorithm": "mappo", "n_agents": 4, "lr": 0.001}
        rs.save_config(config)

        saved = yaml.safe_load(open(rs.input_dir / "config.yaml"))
        assert saved["algorithm"] == "mappo"
        assert saved["n_agents"] == 4
        assert saved["lr"] == 0.001
        assert "_saved_at" in saved

    def test_save_and_load_metrics(self, tmp_path):
        rs = RunStorage(tmp_path / "run1", "run1")
        metrics = {"M1_success_rate": 0.85, "M2_avg_return": 12.5}
        rs.save_metrics(metrics)
        loaded = rs.load_metrics()
        assert loaded == metrics

    def test_load_metrics_no_file(self, tmp_path):
        rs = RunStorage(tmp_path / "run1", "run1")
        assert rs.load_metrics() is None

    def test_append_training_log_creates_csv(self, tmp_path):
        rs = RunStorage(tmp_path / "run1", "run1")
        rs.append_training_log({"epoch": 1, "loss": 0.5})
        rs.append_training_log({"epoch": 2, "loss": 0.3})

        path = rs.logs_dir / "training_log.csv"
        assert path.exists()
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["epoch"] == "1"
        assert rows[1]["loss"] == "0.3"

    def test_append_training_log_header_written_once(self, tmp_path):
        rs = RunStorage(tmp_path / "run1", "run1")
        rs.append_training_log({"a": 1, "b": 2})
        rs.append_training_log({"a": 3, "b": 4})
        rs.append_training_log({"a": 5, "b": 6})

        lines = (rs.logs_dir / "training_log.csv").read_text().strip().split("\n")
        # 1 header + 3 data rows
        assert len(lines) == 4

    def test_save_eval_episodes(self, tmp_path):
        rs = RunStorage(tmp_path / "run1", "run1")
        episodes = [{"steps": 50, "success": True}, {"steps": 80, "success": False}]
        rs.save_eval_episodes(episodes)

        path = rs.output_dir / "eval_episodes.json"
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert len(loaded) == 2
        assert loaded[0]["success"] is True

    def test_is_complete_false(self, tmp_path):
        rs = RunStorage(tmp_path / "run1", "run1")
        assert rs.is_complete() is False

    def test_is_complete_true(self, tmp_path):
        rs = RunStorage(tmp_path / "run1", "run1")
        rs.save_metrics({"M1": 1.0})
        assert rs.is_complete() is True

    def test_has_policy_false(self, tmp_path):
        rs = RunStorage(tmp_path / "run1", "run1")
        assert rs.has_policy() is False

    def test_has_policy_true(self, tmp_path):
        rs = RunStorage(tmp_path / "run1", "run1")
        # Create a fake policy file (skip torch)
        (rs.output_dir / "policy.pt").write_bytes(b"fake")
        assert rs.has_policy() is True

    def test_load_benchmarl_scalars(self, tmp_path):
        rs = RunStorage(tmp_path / "run1", "run1")

        # Create fake BenchMARL CSV structure: benchmarl/<hash>/scalars/<name>.csv
        scalars_dir = rs.benchmarl_dir / "abc123" / "scalars"
        scalars_dir.mkdir(parents=True)

        (scalars_dir / "reward.csv").write_text("0,1.5\n10,2.3\n20,3.1\n")
        (scalars_dir / "loss.csv").write_text("0,0.9\n10,0.5\n")

        result = rs.load_benchmarl_scalars()
        assert "reward" in result
        assert "loss" in result
        assert result["reward"] == [(0, 1.5), (10, 2.3), (20, 3.1)]
        assert result["loss"] == [(0, 0.9), (10, 0.5)]

    def test_load_benchmarl_scalars_empty(self, tmp_path):
        rs = RunStorage(tmp_path / "run1", "run1")
        assert rs.load_benchmarl_scalars() == {}

    def test_load_benchmarl_scalars_skips_bad_rows(self, tmp_path):
        rs = RunStorage(tmp_path / "run1", "run1")
        scalars_dir = rs.benchmarl_dir / "hash1" / "scalars"
        scalars_dir.mkdir(parents=True)
        # Header row and bad data should be skipped
        (scalars_dir / "metric.csv").write_text(
            "step,value\n0,1.0\nbad,data\n10,2.0\n"
        )
        result = rs.load_benchmarl_scalars()
        assert result["metric"] == [(0, 1.0), (10, 2.0)]


# ── ExperimentStorage ────────────────────────────────────────────


class TestExperimentStorage:
    """ExperimentStorage directory management and aggregation."""

    def test_init_creates_results_dir(self, tmp_path):
        es = ExperimentStorage("ER1", results_root=tmp_path)
        assert es.results_dir.is_dir()
        assert es.results_dir.name == "er1"

    def test_get_run_creates_new_folder(self, tmp_path):
        es = ExperimentStorage("er1", results_root=tmp_path)
        rs = es.get_run("er1_mappo_n4_t7_k2_l035_s0")
        assert rs.run_dir.exists()
        assert rs.run_id == "er1_mappo_n4_t7_k2_l035_s0"
        # Folder name should be timestamped
        assert re.match(r"\d{8}_\d{4}__", rs.run_dir.name)

    def test_get_run_returns_existing(self, tmp_path):
        es = ExperimentStorage("er1", results_root=tmp_path)
        run_id = "er1_mappo_n4_t7_k2_l035_s0"

        # Create first run and mark complete
        rs1 = es.get_run(run_id)
        folder1 = rs1.run_dir

        # Second get_run should find the existing folder
        rs2 = es.get_run(run_id)
        assert rs2.run_dir == folder1

    def test_list_runs_only_completed(self, tmp_path):
        es = ExperimentStorage("er1", results_root=tmp_path)

        # Create completed run
        rs1 = es.get_run("er1_mappo_n4_t7_k2_l035_s0")
        rs1.save_metrics({"M1": 0.9})

        # Create incomplete run
        es.get_run("er1_mappo_n4_t7_k2_l035_s1")

        runs = es.list_runs()
        assert runs == ["er1_mappo_n4_t7_k2_l035_s0"]

    def test_list_runs_empty(self, tmp_path):
        es = ExperimentStorage("er1", results_root=tmp_path)
        assert es.list_runs() == []

    def test_list_run_dirs_returns_paths(self, tmp_path):
        es = ExperimentStorage("er1", results_root=tmp_path)
        rs = es.get_run("er1_mappo_n4_t7_k2_l035_s0")
        rs.save_metrics({"M1": 0.9})

        dirs = es.list_run_dirs()
        assert len(dirs) == 1
        assert isinstance(dirs[0], Path)
        assert dirs[0] == rs.run_dir

    def test_load_all_metrics(self, tmp_path):
        es = ExperimentStorage("er1", results_root=tmp_path)

        # Two completed runs
        rs0 = es.get_run("er1_mappo_n4_t7_k2_l035_s0")
        rs0.save_metrics({"M1": 0.8, "M2": 10.0})
        rs1 = es.get_run("er1_mappo_n4_t7_k2_l035_s1")
        rs1.save_metrics({"M1": 0.9, "M2": 12.0})

        # One incomplete
        es.get_run("er1_mappo_n4_t7_k2_l035_s2")

        all_m = es.load_all_metrics()
        assert len(all_m) == 2
        assert all_m["er1_mappo_n4_t7_k2_l035_s0"]["M1"] == 0.8
        assert all_m["er1_mappo_n4_t7_k2_l035_s1"]["M1"] == 0.9

    def test_load_all_metrics_empty(self, tmp_path):
        es = ExperimentStorage("er1", results_root=tmp_path)
        assert es.load_all_metrics() == {}

    def test_to_dataframe(self, tmp_path):
        pd = pytest.importorskip("pandas")
        es = ExperimentStorage("er1", results_root=tmp_path)

        rs = es.get_run("er1_mappo_n4_t7_k2_l035_s0")
        rs.save_metrics({"M1": 0.85})

        df = es.to_dataframe()
        assert len(df) == 1
        assert "run_id" in df.columns
        assert "M1" in df.columns
        assert "algorithm" in df.columns
        assert df.iloc[0]["run_id"] == "er1_mappo_n4_t7_k2_l035_s0"
        assert df.iloc[0]["algorithm"] == "mappo"
        assert df.iloc[0]["n_agents"] == 4

    def test_to_dataframe_empty(self, tmp_path):
        pd = pytest.importorskip("pandas")
        es = ExperimentStorage("er1", results_root=tmp_path)
        df = es.to_dataframe()
        assert df.empty
