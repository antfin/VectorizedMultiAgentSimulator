"""Integration tests for evaluate_with_vmas using real VMAS env."""
import pytest
import torch

from src.config import TaskConfig
from src.runner import evaluate_with_vmas


EXPECTED_KEYS = {
    "M1_success_rate", "M2_avg_return", "M3_avg_steps",
    "M4_avg_collisions", "M5_avg_tokens", "M6_coverage_progress",
    "M8_agent_utilization", "M9_spatial_spread", "n_envs",
}


class TestEvaluateWithVmas:
    """Test evaluate_with_vmas returns correct metric keys and types."""

    @pytest.fixture
    def task_config(self):
        return TaskConfig(
            n_agents=3, n_targets=3, agents_per_target=1,
            covering_range=0.25, lidar_range=0.35,
            targets_respawn=False, max_steps=20,
        )

    def test_random_policy_returns_all_keys(self, task_config):
        metrics = evaluate_with_vmas(
            task_config, policy_fn=None,
            n_eval_episodes=4, n_envs=4,
        )
        assert set(metrics.keys()) == EXPECTED_KEYS

    def test_random_policy_metric_ranges(self, task_config):
        metrics = evaluate_with_vmas(
            task_config, policy_fn=None,
            n_eval_episodes=4, n_envs=4,
        )
        assert 0.0 <= metrics["M1_success_rate"] <= 1.0
        assert 0.0 <= metrics["M6_coverage_progress"] <= 1.0
        assert metrics["M3_avg_steps"] <= 20.0
        assert metrics["M4_avg_collisions"] >= 0.0
        assert metrics["M5_avg_tokens"] == 0.0
        assert metrics["M8_agent_utilization"] >= 0.0
        assert metrics["M9_spatial_spread"] >= 0.0
        assert metrics["n_envs"] == 4

    def test_with_task_overrides(self, task_config):
        metrics = evaluate_with_vmas(
            task_config,
            task_overrides={"n_agents": 2, "n_targets": 2},
            policy_fn=None,
            n_eval_episodes=4, n_envs=4,
        )
        assert set(metrics.keys()) == EXPECTED_KEYS

    def test_heuristic_policy(self, task_config):
        from src.runner import make_heuristic_policy_fn

        heuristic_fn = make_heuristic_policy_fn()
        metrics = evaluate_with_vmas(
            task_config, policy_fn=heuristic_fn,
            n_eval_episodes=4, n_envs=4,
        )
        assert set(metrics.keys()) == EXPECTED_KEYS
        # Heuristic should generally do better than random
        assert metrics["M9_spatial_spread"] > 0


class TestEvaluateMultiBatch:
    """Test that multi-batch evaluation averages correctly."""

    def test_more_episodes_than_envs(self):
        task_config = TaskConfig(
            n_agents=2, n_targets=2, agents_per_target=1,
            covering_range=0.25, lidar_range=0.35,
            targets_respawn=False, max_steps=10,
        )
        metrics = evaluate_with_vmas(
            task_config, policy_fn=None,
            n_eval_episodes=8, n_envs=4,
        )
        assert set(metrics.keys()) == EXPECTED_KEYS
        # n_envs in result is per-batch, averaged → should be 4
        assert metrics["n_envs"] == 4
