"""Tests for the display module (text fallback paths)."""
import pytest

from src.display import (
    METRIC_INFO,
    display_config,
    display_metrics,
    display_sweep_summary,
    display_environment_info,
    scrollable,
    scrollable_md,
    display_metric_cards,
    display_verdict,
    display_baseline_comparison,
    display_results_dashboard,
    _in_notebook,
    _task_param_desc,
    _train_param_desc,
    _sweep_param_desc,
    _m1_color,
)
from src.config import ExperimentSpec, TaskConfig, TrainConfig, SweepConfig


# ── Helpers ───────────────────────────────────────────────────────

SAMPLE_METRICS = {
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


def _make_spec():
    """Build a minimal ExperimentSpec for testing."""
    return ExperimentSpec(
        exp_id="er1",
        name="Baseline",
        description="Test experiment",
        task=TaskConfig(n_agents=4, n_targets=7, lidar_range=0.35),
        train=TrainConfig(algorithm="mappo", max_n_frames=100_000),
        sweep=SweepConfig(
            seeds=[0], algorithms=["mappo"],
            n_agents=[4], lidar_range=[0.35],
        ),
    )


# ── METRIC_INFO ──────────────────────────────────────────────────


class TestMetricInfo:
    """Verify METRIC_INFO structure and completeness."""

    EXPECTED_KEYS = {
        "M1_success_rate", "M2_avg_return", "M3_avg_steps",
        "M4_avg_collisions", "M5_avg_tokens", "M6_coverage_progress",
        "M7_sample_efficiency", "M8_agent_utilization",
        "M9_spatial_spread", "n_envs",
    }

    def test_contains_all_expected_keys(self):
        assert set(METRIC_INFO.keys()) == self.EXPECTED_KEYS

    def test_each_value_is_4_tuple(self):
        for key, value in METRIC_INFO.items():
            assert len(value) == 4, f"{key} should be a 4-tuple"

    def test_tuple_fields_are_strings(self):
        for key, (mid, name, desc, fmt) in METRIC_INFO.items():
            assert isinstance(mid, str), f"{key} mid not str"
            assert isinstance(name, str), f"{key} name not str"
            assert isinstance(desc, str), f"{key} desc not str"
            assert isinstance(fmt, str), f"{key} fmt not str"


# ── _in_notebook ─────────────────────────────────────────────────


class TestInNotebook:
    """Verify notebook detection returns False in test env."""

    def test_returns_false_outside_notebook(self):
        assert _in_notebook() is False


# ── Param description helpers ────────────────────────────────────


class TestParamDescHelpers:
    """Verify parameter description lookup functions."""

    def test_task_param_desc_known_key(self):
        result = _task_param_desc("n_agents")
        assert result != ""
        assert "agent" in result.lower()

    def test_task_param_desc_unknown_key(self):
        assert _task_param_desc("nonexistent_key") == ""

    def test_train_param_desc_known_key(self):
        result = _train_param_desc("lr")
        assert result != ""

    def test_train_param_desc_unknown_key(self):
        assert _train_param_desc("nonexistent_key") == ""

    def test_sweep_param_desc_known_key(self):
        result = _sweep_param_desc("seeds")
        assert result != ""

    def test_sweep_param_desc_unknown_key(self):
        assert _sweep_param_desc("nonexistent_key") == ""


# ── _m1_color ────────────────────────────────────────────────────


class TestM1Color:
    """Verify color thresholds for M1 success rate."""

    def test_high_returns_green(self):
        result = _m1_color(0.9)
        assert result == "#27ae60"  # _GREEN

    def test_medium_returns_orange(self):
        result = _m1_color(0.6)
        assert result == "#e67e22"  # _ORANGE

    def test_low_returns_red(self):
        result = _m1_color(0.2)
        assert result == "#e74c3c"  # _RED

    def test_boundary_high(self):
        # Exactly 0.8 should be orange (> 0.8 needed for green)
        assert _m1_color(0.8) == "#e67e22"

    def test_boundary_low(self):
        # Exactly 0.4 should be red (> 0.4 needed for orange)
        assert _m1_color(0.4) == "#e74c3c"


# ── display_config ───────────────────────────────────────────────


class TestDisplayConfig:
    """Verify text fallback of display_config."""

    def test_does_not_raise(self, capsys):
        spec = _make_spec()
        display_config(spec)

    def test_prints_exp_id_and_name(self, capsys):
        spec = _make_spec()
        display_config(spec)
        out = capsys.readouterr().out
        assert "ER1" in out
        assert "Baseline" in out

    def test_prints_task_params(self, capsys):
        spec = _make_spec()
        display_config(spec)
        out = capsys.readouterr().out
        assert "TASK PARAMETERS" in out

    def test_prints_training_params(self, capsys):
        spec = _make_spec()
        display_config(spec)
        out = capsys.readouterr().out
        assert "TRAINING PARAMETERS" in out


# ── display_metrics ──────────────────────────────────────────────


class TestDisplayMetrics:
    """Verify text fallback of display_metrics."""

    def test_does_not_raise(self, capsys):
        display_metrics(SAMPLE_METRICS)

    def test_prints_metric_values(self, capsys):
        display_metrics(SAMPLE_METRICS)
        out = capsys.readouterr().out
        assert "Success Rate" in out
        assert "Avg Return" in out

    def test_custom_title(self, capsys):
        display_metrics(SAMPLE_METRICS, title="Custom Title")
        out = capsys.readouterr().out
        assert "Custom Title" in out


# ── display_sweep_summary ────────────────────────────────────────


class TestDisplaySweepSummary:
    """Verify display_sweep_summary text output."""

    def test_does_not_raise_with_valid_data(self, capsys):
        all_metrics = {"run_a": SAMPLE_METRICS, "run_b": SAMPLE_METRICS}
        display_sweep_summary(all_metrics)

    def test_empty_dict_prints_no_runs(self, capsys):
        display_sweep_summary({})
        out = capsys.readouterr().out
        assert "No completed runs" in out


# ── display_environment_info ─────────────────────────────────────


class TestDisplayEnvironmentInfo:
    """Verify text fallback of display_environment_info."""

    def test_does_not_raise(self, capsys):
        spec = _make_spec()
        display_environment_info(spec)

    def test_prints_field_dimensions(self, capsys):
        spec = _make_spec()
        display_environment_info(spec)
        out = capsys.readouterr().out
        assert "Field size" in out
        assert "ENVIRONMENT DIMENSIONS" in out


# ── scrollable ───────────────────────────────────────────────────


class TestScrollable:
    """Verify text fallback of scrollable."""

    def test_prints_text_content(self, capsys):
        scrollable("Hello world")
        out = capsys.readouterr().out
        assert "Hello world" in out

    def test_with_title(self, capsys):
        scrollable("content here", title="My Title")
        out = capsys.readouterr().out
        assert "My Title" in out
        assert "content here" in out


# ── scrollable_md ────────────────────────────────────────────────


class TestScrollableMd:
    """Verify text fallback of scrollable_md."""

    def test_prints_markdown_text(self, capsys):
        scrollable_md("# Heading\nSome text")
        out = capsys.readouterr().out
        assert "# Heading" in out
        assert "Some text" in out

    def test_with_title(self, capsys):
        scrollable_md("body", title="MD Title")
        out = capsys.readouterr().out
        assert "MD Title" in out


# ── display_metric_cards ─────────────────────────────────────────


class TestDisplayMetricCards:
    """Verify text fallback of display_metric_cards."""

    def test_prints_labels_and_values(self, capsys):
        display_metric_cards(SAMPLE_METRICS)
        out = capsys.readouterr().out
        assert "Success Rate" in out
        assert "Coverage" in out

    def test_with_title(self, capsys):
        display_metric_cards(SAMPLE_METRICS, title="KPI Cards")
        out = capsys.readouterr().out
        assert "KPI Cards" in out


# ── display_verdict ──────────────────────────────────────────────


class TestDisplayVerdict:
    """Verify verdict classification by success rate."""

    def test_high_success_too_easy(self, capsys):
        display_verdict(success_rate=0.60, avg_return=20.0)
        out = capsys.readouterr().out
        assert "TASK TOO EASY" in out

    def test_medium_success_floor_established(self, capsys):
        display_verdict(success_rate=0.30, avg_return=10.0)
        out = capsys.readouterr().out
        assert "FLOOR ESTABLISHED" in out

    def test_low_success_too_hard(self, capsys):
        display_verdict(success_rate=0.10, avg_return=2.0)
        out = capsys.readouterr().out
        assert "TASK TOO HARD" in out


# ── display_baseline_comparison ──────────────────────────────────


class TestDisplayBaselineComparison:
    """Verify text fallback of display_baseline_comparison."""

    def test_does_not_raise(self, capsys):
        display_baseline_comparison(
            heuristic_metrics=SAMPLE_METRICS,
            random_metrics=SAMPLE_METRICS,
        )

    def test_prints_both_baselines(self, capsys):
        display_baseline_comparison(
            heuristic_metrics=SAMPLE_METRICS,
            random_metrics=SAMPLE_METRICS,
        )
        out = capsys.readouterr().out
        assert "Heuristic" in out
        assert "Random" in out


# ── display_results_dashboard ────────────────────────────────────


class TestDisplayResultsDashboard:
    """Verify text fallback of display_results_dashboard."""

    def test_does_not_raise(self, capsys):
        display_results_dashboard(trained_metrics=SAMPLE_METRICS)

    def test_prints_metric_cards(self, capsys):
        display_results_dashboard(
            trained_metrics=SAMPLE_METRICS, run_id="test_run"
        )
        out = capsys.readouterr().out
        assert "test_run" in out
