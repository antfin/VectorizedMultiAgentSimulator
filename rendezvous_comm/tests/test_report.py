"""Tests for the report module (generate_run_report, generate_sweep_report)."""
from pathlib import Path

import pytest

from src.config import ExperimentSpec, SweepConfig, TaskConfig, TrainConfig
from src.report import METRIC_DETAILS, generate_run_report, generate_sweep_report


# ── Helpers ───────────────────────────────────────────────────────


def _make_spec():
    """Build a minimal ExperimentSpec for testing."""
    return ExperimentSpec(
        exp_id="er1",
        name="Test",
        description="Test experiment",
        task=TaskConfig(),
        train=TrainConfig(),
        sweep=SweepConfig(seeds=[0]),
    )


def _sample_metrics():
    """Return a sample metrics dict covering all M keys."""
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


# ── METRIC_DETAILS ────────────────────────────────────────────────


class TestMetricDetails:
    """Verify METRIC_DETAILS structure and content."""

    EXPECTED_KEYS = {
        "M1_success_rate",
        "M2_avg_return",
        "M3_avg_steps",
        "M4_avg_collisions",
        "M5_avg_tokens",
        "M6_coverage_progress",
        "M7_sample_efficiency",
        "M8_agent_utilization",
        "M9_spatial_spread",
    }

    def test_contains_all_expected_keys(self):
        assert self.EXPECTED_KEYS == set(METRIC_DETAILS.keys())

    def test_each_entry_has_required_fields(self):
        for key, detail in METRIC_DETAILS.items():
            assert "label" in detail, f"{key} missing 'label'"
            assert "fmt" in detail, f"{key} missing 'fmt'"
            assert "description" in detail, f"{key} missing 'description'"

    def test_labels_start_with_m_prefix(self):
        for key, detail in METRIC_DETAILS.items():
            assert detail["label"].startswith("M"), (
                f"{key} label {detail['label']!r} does not start with 'M'"
            )


# ── generate_run_report ──────────────────────────────────────────


class TestGenerateRunReport:
    """Tests for single-run report generation."""

    def test_returns_nonempty_string(self, tmp_path):
        spec = _make_spec()
        result = generate_run_report(
            tmp_path, "run_0", spec, _sample_metrics()
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_writes_report_md(self, tmp_path):
        spec = _make_spec()
        generate_run_report(tmp_path, "run_0", spec, _sample_metrics())
        report_file = tmp_path / "report.md"
        assert report_file.exists()
        assert len(report_file.read_text()) > 0

    def test_contains_run_id(self, tmp_path):
        spec = _make_spec()
        result = generate_run_report(
            tmp_path, "er1_mappo_n4_s0", spec, _sample_metrics()
        )
        assert "er1_mappo_n4_s0" in result

    def test_contains_spec_name_and_exp_id(self, tmp_path):
        spec = _make_spec()
        result = generate_run_report(
            tmp_path, "run_0", spec, _sample_metrics()
        )
        assert "Test" in result
        assert "ER1" in result or "er1" in result

    def test_contains_formatted_metric_values(self, tmp_path):
        spec = _make_spec()
        result = generate_run_report(
            tmp_path, "run_0", spec, _sample_metrics()
        )
        # M1 formatted as .1% → "75.0%"
        assert "75.0%" in result
        # M2 formatted as .4f → "12.5000"
        assert "12.5000" in result
        # M6 formatted as .1% → "85.0%"
        assert "85.0%" in result

    def test_contains_task_configuration_section(self, tmp_path):
        spec = _make_spec()
        result = generate_run_report(
            tmp_path, "run_0", spec, _sample_metrics()
        )
        assert "Task Configuration" in result

    def test_contains_training_configuration_section(self, tmp_path):
        spec = _make_spec()
        result = generate_run_report(
            tmp_path, "run_0", spec, _sample_metrics()
        )
        assert "Training Configuration" in result

    def test_contains_evaluation_results_section(self, tmp_path):
        spec = _make_spec()
        result = generate_run_report(
            tmp_path, "run_0", spec, _sample_metrics()
        )
        assert "Evaluation Results" in result

    def test_with_elapsed_seconds_includes_wall_time(self, tmp_path):
        spec = _make_spec()
        result = generate_run_report(
            tmp_path, "run_0", spec, _sample_metrics(),
            elapsed_seconds=3661.0,
        )
        assert "Wall time" in result
        assert "1h 1m 1s" in result

    def test_without_elapsed_seconds_no_wall_time(self, tmp_path):
        spec = _make_spec()
        result = generate_run_report(
            tmp_path, "run_0", spec, _sample_metrics(),
            elapsed_seconds=0.0,
        )
        assert "not recorded" in result

    def test_with_task_overrides_marks_params(self, tmp_path):
        spec = _make_spec()
        overrides = {"n_agents": 6, "lidar_range": 0.45}
        result = generate_run_report(
            tmp_path, "run_0", spec, _sample_metrics(),
            task_overrides=overrides,
        )
        # Overridden params get a ⚙ marker
        assert "⚙" in result

    def test_written_file_matches_return_value(self, tmp_path):
        spec = _make_spec()
        result = generate_run_report(
            tmp_path, "run_0", spec, _sample_metrics()
        )
        report_file = tmp_path / "report.md"
        assert report_file.read_text() == result


# ── generate_sweep_report ────────────────────────────────────────


class TestGenerateSweepReport:
    """Tests for sweep-level report generation."""

    def _make_all_metrics(self):
        """Two runs with slightly different metrics."""
        return {
            "er1_mappo_n4_s0": _sample_metrics(),
            "er1_mappo_n4_s1": {
                "M1_success_rate": 0.80,
                "M2_avg_return": 14.0,
                "M3_avg_steps": 130.0,
                "M4_avg_collisions": 2.5,
                "M5_avg_tokens": 0.0,
                "M6_coverage_progress": 0.90,
                "M8_agent_utilization": 0.25,
                "M9_spatial_spread": 0.95,
                "n_envs": 200,
            },
        }

    def test_returns_nonempty_string(self, tmp_path):
        spec = _make_spec()
        result = generate_sweep_report(
            spec, self._make_all_metrics(), results_dir=tmp_path
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_writes_sweep_report_md(self, tmp_path):
        spec = _make_spec()
        generate_sweep_report(
            spec, self._make_all_metrics(), results_dir=tmp_path
        )
        report_file = tmp_path / "sweep_report.md"
        assert report_file.exists()
        assert len(report_file.read_text()) > 0

    def test_contains_sweep_header_with_run_count(self, tmp_path):
        spec = _make_spec()
        all_metrics = self._make_all_metrics()
        result = generate_sweep_report(
            spec, all_metrics, results_dir=tmp_path
        )
        assert "Sweep Report" in result
        assert str(len(all_metrics)) in result

    def test_contains_aggregate_results(self, tmp_path):
        spec = _make_spec()
        result = generate_sweep_report(
            spec, self._make_all_metrics(), results_dir=tmp_path
        )
        assert "Aggregate Results" in result
        # Should contain mean/std columns
        assert "Mean" in result
        assert "Std" in result

    def test_contains_best_and_worst_runs(self, tmp_path):
        spec = _make_spec()
        result = generate_sweep_report(
            spec, self._make_all_metrics(), results_dir=tmp_path
        )
        assert "Best & Worst" in result
        assert "best" in result.lower()
        assert "worst" in result.lower()

    def test_contains_per_run_results_table(self, tmp_path):
        spec = _make_spec()
        all_metrics = self._make_all_metrics()
        result = generate_sweep_report(
            spec, all_metrics, results_dir=tmp_path
        )
        assert "Per-Run Results" in result
        for run_id in all_metrics:
            assert run_id in result

    def test_contains_metric_glossary(self, tmp_path):
        spec = _make_spec()
        result = generate_sweep_report(
            spec, self._make_all_metrics(), results_dir=tmp_path
        )
        assert "Metric Glossary" in result

    def test_empty_all_metrics_shows_no_completed_runs(self, tmp_path):
        spec = _make_spec()
        result = generate_sweep_report(
            spec, {}, results_dir=tmp_path
        )
        assert "No completed runs" in result
        # Should still write the file
        assert (tmp_path / "sweep_report.md").exists()

    def test_empty_all_metrics_short_report(self, tmp_path):
        spec = _make_spec()
        result = generate_sweep_report(
            spec, {}, results_dir=tmp_path
        )
        # No aggregate or per-run sections
        assert "Aggregate Results" not in result
        assert "Per-Run Results" not in result

    def test_with_elapsed_seconds_shows_wall_time(self, tmp_path):
        spec = _make_spec()
        result = generate_sweep_report(
            spec, self._make_all_metrics(),
            elapsed_seconds=7200.0, results_dir=tmp_path,
        )
        assert "Wall time" in result
        assert "2h 0m 0s" in result

    def test_written_file_matches_return_value(self, tmp_path):
        spec = _make_spec()
        result = generate_sweep_report(
            spec, self._make_all_metrics(), results_dir=tmp_path
        )
        report_file = tmp_path / "sweep_report.md"
        assert report_file.read_text() == result
