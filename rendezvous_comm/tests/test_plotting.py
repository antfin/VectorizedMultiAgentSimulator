"""Tests for the plotting module."""
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for testing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.plotting import (
    METRIC_LABELS,
    COLORS,
    LABELS,
    set_style,
    plot_sweep_heatmap,
    plot_training_curves,
    plot_seed_variance,
    plot_baseline_comparison,
    plot_metric_radar,
    plot_success_vs_tokens,
    plot_training_dashboard,
    plot_baseline_comparison_bars,
    plot_baseline_grouped_bars,
    plot_results_comparison,
    plot_sweep_overview,
    save_figure,
)


# ── Fixture: close all figures after each test ───────────────────


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


# ── Test constants ───────────────────────────────────────────────


class TestConstants:
    """Verify module-level constants have expected keys."""

    def test_metric_labels_has_expected_keys(self):
        expected = {
            "M1_success_rate",
            "M2_avg_return",
            "M3_avg_steps",
            "M4_avg_collisions",
            "M5_avg_tokens",
            "M6_coverage_progress",
            "M8_agent_utilization",
            "M9_spatial_spread",
        }
        assert expected == set(METRIC_LABELS.keys())

    def test_colors_has_experiment_ids(self):
        expected = {"er1", "er2", "er3", "er4", "e1"}
        assert expected == set(COLORS.keys())

    def test_labels_has_matching_keys(self):
        assert set(LABELS.keys()) == set(COLORS.keys())


# ── Test set_style ───────────────────────────────────────────────


class TestSetStyle:
    """Verify set_style updates rcParams."""

    def test_does_not_raise(self):
        set_style()

    def test_updates_rcparams(self):
        set_style()
        assert plt.rcParams["figure.figsize"] == [10, 6]
        assert plt.rcParams["font.size"] == 12
        assert plt.rcParams["axes.grid"] is True
        assert plt.rcParams["grid.alpha"] == pytest.approx(0.3)


# ── Test plot_sweep_heatmap ──────────────────────────────────────


class TestPlotSweepHeatmap:
    """Verify plot_sweep_heatmap returns a Figure."""

    def test_returns_figure(self):
        df = pd.DataFrame({
            "n_agents": [4, 4, 6, 6],
            "n_targets": [3, 5, 3, 5],
            "M1_success_rate": [0.1, 0.2, 0.3, 0.4],
        })
        fig = plot_sweep_heatmap(df)
        assert isinstance(fig, plt.Figure)

    def test_custom_title(self):
        df = pd.DataFrame({
            "n_agents": [4, 4, 6, 6],
            "n_targets": [3, 5, 3, 5],
            "M1_success_rate": [0.1, 0.2, 0.3, 0.4],
        })
        fig = plot_sweep_heatmap(df, title="Custom Title")
        assert isinstance(fig, plt.Figure)


# ── Test plot_training_curves ────────────────────────────────────


class TestPlotTrainingCurves:
    """Verify plot_training_curves returns a Figure."""

    def test_returns_figure(self):
        training_logs = {
            "run1": pd.DataFrame({
                "step": [0, 100, 200],
                "episode_reward_mean": [0.1, 0.5, 0.8],
            }),
        }
        fig = plot_training_curves(training_logs)
        assert isinstance(fig, plt.Figure)

    def test_multiple_runs(self):
        training_logs = {
            "er1_run1": pd.DataFrame({
                "step": [0, 100, 200],
                "episode_reward_mean": [0.1, 0.5, 0.8],
            }),
            "er2_run2": pd.DataFrame({
                "step": [0, 100, 200],
                "episode_reward_mean": [0.2, 0.4, 0.9],
            }),
        }
        fig = plot_training_curves(training_logs)
        assert isinstance(fig, plt.Figure)


# ── Test plot_seed_variance ──────────────────────────────────────


class TestPlotSeedVariance:
    """Verify plot_seed_variance returns a Figure."""

    def test_returns_figure(self):
        df = pd.DataFrame({
            "algorithm": ["mappo", "mappo", "ippo", "ippo"],
            "M1_success_rate": [0.8, 0.85, 0.6, 0.65],
        })
        fig = plot_seed_variance(df)
        assert isinstance(fig, plt.Figure)


# ── Test plot_baseline_comparison ────────────────────────────────


class TestPlotBaselineComparison:
    """Verify plot_baseline_comparison returns a Figure."""

    def test_returns_figure(self):
        cross_df = pd.DataFrame({
            "experiment": ["er1", "er1", "er2", "er2"],
            "M1_success_rate": [0.5, 0.55, 0.7, 0.75],
        })
        fig = plot_baseline_comparison(cross_df)
        assert isinstance(fig, plt.Figure)


# ── Test plot_metric_radar ───────────────────────────────────────


class TestPlotMetricRadar:
    """Verify plot_metric_radar returns a Figure."""

    def test_returns_figure(self):
        data = {
            "er1": {
                "M1_success_rate": 0.5,
                "M2_avg_return": 10,
                "M3_avg_steps": 150,
                "M4_avg_collisions": 3,
            },
        }
        fig = plot_metric_radar(data)
        assert isinstance(fig, plt.Figure)

    def test_custom_metrics(self):
        data = {
            "er1": {
                "M1_success_rate": 0.5,
                "M6_coverage_progress": 0.8,
            },
        }
        fig = plot_metric_radar(
            data,
            metrics=["M1_success_rate", "M6_coverage_progress"],
        )
        assert isinstance(fig, plt.Figure)

    def test_multiple_experiments(self):
        data = {
            "er1": {
                "M1_success_rate": 0.5,
                "M2_avg_return": 10,
                "M3_avg_steps": 150,
                "M4_avg_collisions": 3,
            },
            "er2": {
                "M1_success_rate": 0.7,
                "M2_avg_return": 15,
                "M3_avg_steps": 100,
                "M4_avg_collisions": 1,
            },
        }
        fig = plot_metric_radar(data)
        assert isinstance(fig, plt.Figure)


# ── Test plot_success_vs_tokens ──────────────────────────────────


class TestPlotSuccessVsTokens:
    """Verify plot_success_vs_tokens returns a Figure."""

    def test_returns_figure(self):
        data = {
            "er1": {"M1_success_rate": 0.5, "M5_avg_tokens": 0},
            "er2": {"M1_success_rate": 0.7, "M5_avg_tokens": 50},
        }
        fig = plot_success_vs_tokens(data)
        assert isinstance(fig, plt.Figure)


# ── Test plot_training_dashboard ─────────────────────────────────


class TestPlotTrainingDashboard:
    """Verify plot_training_dashboard returns a Figure."""

    def test_returns_figure(self):
        scalars = {
            "eval_reward_episode_reward_mean": [
                (0, 1.0), (100, 5.0), (200, 8.0),
            ],
            "collection_agents_info_targets_covered": [
                (0, 0.0), (100, 1.0), (200, 2.0),
            ],
            "collection_agents_info_covering_reward": [
                (0, 0.0), (100, 0.5), (200, 1.0),
            ],
            "collection_agents_info_collision_rew": [
                (0, -0.5), (100, -0.3), (200, -0.1),
            ],
            "eval_reward_episode_len_mean": [
                (0, 100.0), (100, 80.0), (200, 60.0),
            ],
            "train_agents_entropy": [
                (0, 2.0), (100, 1.5), (200, 1.0),
            ],
        }
        fig = plot_training_dashboard(scalars)
        assert isinstance(fig, plt.Figure)

    def test_with_heuristic_reward(self):
        scalars = {
            "eval_reward_episode_reward_mean": [
                (0, 1.0), (100, 5.0),
            ],
        }
        fig = plot_training_dashboard(
            scalars, heuristic_reward=3.5,
        )
        assert isinstance(fig, plt.Figure)

    def test_empty_scalars(self):
        fig = plot_training_dashboard({})
        assert isinstance(fig, plt.Figure)


# ── Test plot_baseline_comparison_bars ────────────────────────────


class TestPlotBaselineComparisonBars:
    """Verify plot_baseline_comparison_bars returns a Figure."""

    def test_returns_figure(self):
        metrics_dict = {
            "Random": {"M1_success_rate": 0.1},
            "Trained": {"M1_success_rate": 0.7},
            "Heuristic": {"M1_success_rate": 0.4},
        }
        fig = plot_baseline_comparison_bars(metrics_dict)
        assert isinstance(fig, plt.Figure)

    def test_non_rate_metric(self):
        metrics_dict = {
            "Random": {"M2_avg_return": -5.0},
            "Trained": {"M2_avg_return": 12.0},
        }
        fig = plot_baseline_comparison_bars(
            metrics_dict, metric_key="M2_avg_return",
        )
        assert isinstance(fig, plt.Figure)


# ── Test plot_baseline_grouped_bars ──────────────────────────────


class TestPlotBaselineGroupedBars:
    """Verify plot_baseline_grouped_bars returns a Figure."""

    def test_returns_figure(self):
        heuristic = {
            "M1_success_rate": 0.4,
            "M6_coverage_progress": 0.7,
            "M2_avg_return": 5.0,
            "M9_spatial_spread": 1.2,
        }
        random = {
            "M1_success_rate": 0.1,
            "M6_coverage_progress": 0.3,
            "M2_avg_return": -2.0,
            "M9_spatial_spread": 2.5,
        }
        fig = plot_baseline_grouped_bars(heuristic, random)
        assert isinstance(fig, plt.Figure)


# ── Test plot_results_comparison ─────────────────────────────────


class TestPlotResultsComparison:
    """Verify plot_results_comparison returns a Figure."""

    def test_returns_figure(self):
        comparison = {
            "Random": {
                "M1_success_rate": 0.1,
                "M2_avg_return": -2.0,
                "M6_coverage_progress": 0.3,
            },
            "Heuristic": {
                "M1_success_rate": 0.4,
                "M2_avg_return": 5.0,
                "M6_coverage_progress": 0.7,
            },
            "Trained (MAPPO)": {
                "M1_success_rate": 0.8,
                "M2_avg_return": 12.0,
                "M6_coverage_progress": 0.9,
            },
        }
        fig = plot_results_comparison(comparison)
        assert isinstance(fig, plt.Figure)

    def test_custom_colors(self):
        comparison = {
            "A": {"M1_success_rate": 0.5, "M2_avg_return": 1.0,
                   "M6_coverage_progress": 0.6},
        }
        fig = plot_results_comparison(
            comparison, colors={"A": "#ff0000"},
        )
        assert isinstance(fig, plt.Figure)


# ── Test plot_sweep_overview ─────────────────────────────────────


class TestPlotSweepOverview:
    """Verify plot_sweep_overview returns a Figure."""

    def test_returns_figure_with_n_agents(self):
        df = pd.DataFrame({
            "M1_success_rate": [0.3, 0.5, 0.7, 0.9],
            "M3_avg_steps": [200, 150, 100, 80],
            "M4_avg_collisions": [5, 3, 2, 1],
            "M6_coverage_progress": [0.4, 0.6, 0.8, 0.95],
            "M8_agent_utilization": [0.5, 0.3, 0.2, 0.1],
            "M9_spatial_spread": [2.0, 1.5, 1.2, 0.8],
            "n_agents": [2, 4, 6, 8],
        })
        fig = plot_sweep_overview(df)
        assert isinstance(fig, plt.Figure)

    def test_returns_figure_without_n_agents(self):
        df = pd.DataFrame({
            "M1_success_rate": [0.3, 0.5, 0.7],
            "M3_avg_steps": [200, 150, 100],
            "M4_avg_collisions": [5, 3, 2],
            "M6_coverage_progress": [0.4, 0.6, 0.8],
            "M8_agent_utilization": [0.5, 0.3, 0.2],
            "M9_spatial_spread": [2.0, 1.5, 1.2],
        })
        fig = plot_sweep_overview(df)
        assert isinstance(fig, plt.Figure)


# ── Test save_figure ─────────────────────────────────────────────


class TestSaveFigure:
    """Verify save_figure writes a file to disk."""

    def test_saves_png(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        out = tmp_path / "test_plot.png"
        save_figure(fig, str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_creates_parent_dirs(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        out = tmp_path / "subdir" / "nested" / "plot.png"
        save_figure(fig, str(out))
        assert out.exists()
