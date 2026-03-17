"""Tests for CSV consolidation and loading."""
import csv
import json
from pathlib import Path

import pytest

from src.consolidate import (
    consolidate_csvs,
    load_latest_csv,
    list_experiments_with_data,
    _build_sweep_row,
    _classify_scalars,
    _scalars_to_wide_df,
)
from src.storage import ExperimentStorage, RunStorage


# ── Helpers ──────────────────────────────────────────────────────────


def _make_run(es, run_id, metrics, config=None, scalars=None):
    """Create a completed run with metrics, optional config and scalars."""
    rs = es.get_run(run_id)
    rs.save_metrics(metrics)
    if config:
        rs.save_config(config)
    if scalars:
        scalars_dir = rs.benchmarl_dir / "hash1" / "scalars"
        scalars_dir.mkdir(parents=True, exist_ok=True)
        for name, rows in scalars.items():
            content = "\n".join(f"{s},{v}" for s, v in rows)
            (scalars_dir / f"{name}.csv").write_text(content)
    return rs


# ── consolidate_csvs ─────────────────────────────────────────────────


class TestConsolidateCsvs:

    def test_produces_three_csvs(self, tmp_path):
        import pandas as pd
        es = ExperimentStorage("er1", results_root=tmp_path)
        _make_run(es, "er1_mappo_n4_t3_k1_l035_s0",
                  {"M1_success_rate": 0.8},
                  scalars={
                      "train_agents_entropy": [(i, 1.0 + i) for i in range(50)],
                      "eval_reward_episode_reward_mean": [(0, -1.0), (15, 0.5)],
                  })

        result = consolidate_csvs("er1", results_root=tmp_path)

        assert "sweep" in result
        assert "iter" in result
        assert "eval" in result
        assert result["sweep"].exists()
        assert result["iter"].exists()
        assert result["eval"].exists()

    def test_sweep_csv_has_run_id_first(self, tmp_path):
        import pandas as pd
        es = ExperimentStorage("er1", results_root=tmp_path)
        _make_run(es, "run1", {"M1_success_rate": 0.5, "n_agents": 4})

        result = consolidate_csvs("er1", results_root=tmp_path)
        df = pd.read_csv(result["sweep"])
        assert df.columns[0] == "run_id"

    def test_sweep_csv_backfills_from_config(self, tmp_path):
        import pandas as pd
        es = ExperimentStorage("er1", results_root=tmp_path)
        _make_run(
            es, "er1_mappo_n6_t4_k1_l035_s0",
            {"M1_success_rate": 0.7},
            config={
                "task": {"n_agents": 4, "n_targets": 4},
                "task_overrides": {"n_agents": 6, "n_targets": 4},
                "algorithm": "mappo",
                "seed": 0,
            },
        )

        result = consolidate_csvs("er1", results_root=tmp_path)
        df = pd.read_csv(result["sweep"])
        # task_overrides should win over base task config
        assert df.iloc[0]["n_agents"] == 6

    def test_iter_csv_no_nans(self, tmp_path):
        import pandas as pd
        es = ExperimentStorage("er1", results_root=tmp_path)
        _make_run(es, "run1", {"M1_success_rate": 0.5}, scalars={
            "train_agents_entropy": [(i, float(i)) for i in range(50)],
            "train_agents_loss": [(i, float(i) * 0.1) for i in range(50)],
        })

        result = consolidate_csvs("er1", results_root=tmp_path)
        df = pd.read_csv(result["iter"])
        assert df.isna().sum().sum() == 0

    def test_eval_csv_aligns_steps(self, tmp_path):
        """Custom eval metrics (off-by-one steps) are aligned to native."""
        import pandas as pd
        es = ExperimentStorage("er1", results_root=tmp_path)
        # Native eval at steps 0, 15, 31
        # Custom M1 at steps 1, 16, 32 (off by one)
        _make_run(es, "run1", {"M1_success_rate": 0.5}, scalars={
            "eval_reward_episode_reward_mean": [
                (0, -1.0), (15, 0.5), (31, 1.0),
            ],
            "eval_M1_success_rate": [
                (1, 0.1), (16, 0.3), (32, 0.8),
            ],
        })

        result = consolidate_csvs("er1", results_root=tmp_path)
        df = pd.read_csv(result["eval"])
        assert df.isna().sum().sum() == 0
        assert len(df) == 3
        # Steps should be the native ones
        assert list(df["step"]) == [0, 15, 31]

    def test_empty_experiment_returns_empty(self, tmp_path):
        result = consolidate_csvs("er1", results_root=tmp_path)
        assert result == {}

    def test_multiple_runs_merged(self, tmp_path):
        import pandas as pd
        es = ExperimentStorage("er1", results_root=tmp_path)
        _make_run(es, "run1", {"M1_success_rate": 0.5}, scalars={
            "train_agents_entropy": [(i, float(i)) for i in range(30)],
        })
        _make_run(es, "run2", {"M1_success_rate": 0.9}, scalars={
            "train_agents_entropy": [(i, float(i) * 2) for i in range(30)],
        })

        result = consolidate_csvs("er1", results_root=tmp_path)
        sweep_df = pd.read_csv(result["sweep"])
        iter_df = pd.read_csv(result["iter"])
        assert len(sweep_df) == 2
        assert len(iter_df) == 60  # 30 rows x 2 runs
        assert set(iter_df["run_id"].unique()) == {"run1", "run2"}


# ── load_latest_csv ──────────────────────────────────────────────────


class TestLoadLatestCsv:

    def test_loads_most_recent(self, tmp_path):
        import pandas as pd
        # Write two files with different timestamps
        (tmp_path / "sweep_results_20260315_1000.csv").write_text(
            "run_id,M1\nold,0.1\n"
        )
        (tmp_path / "sweep_results_20260316_1000.csv").write_text(
            "run_id,M1\nnew,0.9\n"
        )

        df = load_latest_csv(tmp_path, "sweep_results")
        assert df.iloc[0]["run_id"] == "new"

    def test_returns_none_if_no_match(self, tmp_path):
        assert load_latest_csv(tmp_path, "sweep_results") is None

    def test_returns_none_if_dir_missing(self):
        assert load_latest_csv(Path("/nonexistent"), "sweep_results") is None


# ── list_experiments_with_data ───────────────────────────────────────


class TestListExperimentsWithData:

    def test_finds_experiment_with_csv(self, tmp_path):
        exp_dir = tmp_path / "er1"
        exp_dir.mkdir()
        (exp_dir / "sweep_results_20260316.csv").write_text("run_id\n")

        result = list_experiments_with_data(results_root=tmp_path)
        assert "er1" in result

    def test_finds_experiment_with_completed_run(self, tmp_path):
        es = ExperimentStorage("er1", results_root=tmp_path)
        rs = es.get_run("test_run")
        rs.save_metrics({"M1": 0.5})

        result = list_experiments_with_data(results_root=tmp_path)
        assert "er1" in result

    def test_ignores_empty_experiment(self, tmp_path):
        (tmp_path / "er2").mkdir()
        result = list_experiments_with_data(results_root=tmp_path)
        assert "er2" not in result

    def test_empty_results_root(self, tmp_path):
        result = list_experiments_with_data(results_root=tmp_path)
        assert result == []


# ── _classify_scalars ────────────────────────────────────────────────


class TestClassifyScalars:

    def test_splits_by_row_count(self):
        scalars = {
            "train_metric": [(i, float(i)) for i in range(50)],
            "eval_metric": [(0, 1.0), (10, 2.0)],
        }
        iter_sc, eval_sc = _classify_scalars(scalars)
        assert "train_metric" in iter_sc
        assert "eval_metric" in eval_sc

    def test_aligns_custom_eval_steps(self):
        scalars = {
            "eval_reward_episode_reward_mean": [(0, 1.0), (15, 2.0)],
            "eval_M1_success_rate": [(1, 0.5), (16, 0.8)],
        }
        _, eval_sc = _classify_scalars(scalars)
        # M1 steps should be shifted to match native
        m1_steps = [s for s, _ in eval_sc["eval_M1_success_rate"]]
        assert m1_steps == [0, 15]


# ── _scalars_to_wide_df ─────────────────────────────────────────────


class TestScalarsToWideDf:

    def test_creates_wide_format(self):
        import pandas as pd
        scalars = {
            "metric_a": [(0, 1.0), (1, 2.0)],
            "metric_b": [(0, 10.0), (1, 20.0)],
        }
        df = _scalars_to_wide_df("run1", scalars)
        assert len(df) == 2
        assert list(df.columns) == ["run_id", "step", "metric_a", "metric_b"]
        assert df.iloc[0]["metric_a"] == 1.0
        assert df.iloc[1]["metric_b"] == 20.0

    def test_empty_scalars_returns_empty(self):
        df = _scalars_to_wide_df("run1", {})
        assert df.empty

    def test_common_steps_only(self):
        """Only rows where all metrics have data are kept."""
        scalars = {
            "metric_a": [(0, 1.0), (1, 2.0), (2, 3.0)],
            "metric_b": [(0, 10.0), (2, 30.0)],
        }
        df = _scalars_to_wide_df("run1", scalars)
        # Only steps 0 and 2 are common
        assert len(df) == 2
        assert list(df["step"]) == [0, 2]
        assert df.isna().sum().sum() == 0


# ── _build_sweep_row ─────────────────────────────────────────────────


class TestBuildSweepRow:

    def test_basic_row(self, tmp_path):
        rs = RunStorage(tmp_path / "run", "test_run")
        rs.save_metrics({"M1_success_rate": 0.8, "n_agents": 4})

        row = _build_sweep_row(tmp_path / "run", "test_run")
        assert row["run_id"] == "test_run"
        assert row["M1_success_rate"] == 0.8

    def test_backfills_from_config_overrides(self, tmp_path):
        rs = RunStorage(tmp_path / "run", "er1_mappo_n6_t3_k1_l035_s0")
        rs.save_metrics({"M1_success_rate": 0.7})
        rs.save_config({
            "task": {"n_agents": 4, "n_targets": 7},
            "task_overrides": {"n_agents": 6, "n_targets": 3},
        })

        row = _build_sweep_row(tmp_path / "run", "er1_mappo_n6_t3_k1_l035_s0")
        assert row["n_agents"] == 6
        assert row["n_targets"] == 3

    def test_returns_none_if_no_metrics(self, tmp_path):
        (tmp_path / "run" / "output").mkdir(parents=True)
        row = _build_sweep_row(tmp_path / "run", "test_run")
        assert row is None


# ── _save_video ──────────────────────────────────────────────────────


class TestSaveVideo:

    def test_saves_mp4(self, tmp_path):
        import numpy as np
        from src.runner import _save_video

        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(10)]
        path = tmp_path / "test.mp4"
        _save_video(frames, path, fps=10)
        assert path.exists()
        assert path.stat().st_size > 0
