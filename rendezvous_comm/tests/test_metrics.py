"""Tests for the metrics module (M1-M9)."""
import csv
import tempfile
from pathlib import Path

import pytest
import torch

from src.metrics import (
    EpisodeMetrics,
    compute_ate,
    compute_budget_frontier,
    compute_delta_return_per_msg,
    compute_m7_sample_efficiency,
    compute_transfer_score,
)


# ── Helpers ───────────────────────────────────────────────────────


def _make_info(n_envs, covering_reward=0.0, collision_rew=0.0, targets_covered=0):
    """Build a single agent's info dict with the given scalar values."""
    return {
        "covering_reward": torch.full((n_envs,), covering_reward),
        "collision_rew": torch.full((n_envs,), collision_rew),
        "targets_covered": torch.full((n_envs,), targets_covered),
    }


def _make_agent_info_list(n_agents, n_envs, covering_rewards=None, collision_rew=0.0, targets_covered=0):
    """Build a list of per-agent info dicts."""
    if covering_rewards is None:
        covering_rewards = [0.0] * n_agents
    return [
        {
            "covering_reward": torch.full((n_envs,), covering_rewards[i]),
            "collision_rew": torch.full((n_envs,), collision_rew),
            "targets_covered": torch.full((n_envs,), targets_covered),
        }
        for i in range(n_agents)
    ]


# ── Test metric key names ────────────────────────────────────────


class TestMetricKeys:
    """Verify output keys match the M1-M9 scheme."""

    def test_compute_returns_correct_keys(self):
        m = EpisodeMetrics().init(4, n_targets=3, n_agents=2)
        result = m.compute(max_steps=100)
        expected = {
            "M1_success_rate", "M2_avg_return", "M3_avg_steps",
            "M4_avg_collisions", "M5_avg_tokens", "M6_coverage_progress",
            "M8_agent_utilization", "M9_spatial_spread", "n_envs",
        }
        assert set(result.keys()) == expected

    def test_no_old_m1b_key(self):
        m = EpisodeMetrics().init(4, n_targets=3, n_agents=2)
        result = m.compute(max_steps=100)
        assert "M1b_coverage_progress" not in result
        assert "M1b_avg_targets_covered_per_step" not in result


# ── Test M1: Success Rate ────────────────────────────────────────


class TestM1SuccessRate:
    def test_all_done(self):
        n_envs = 10
        m = EpisodeMetrics().init(n_envs, n_targets=3, n_agents=2)
        dones = torch.ones(n_envs, dtype=torch.bool)
        rewards = [torch.zeros(n_envs), torch.zeros(n_envs)]
        info = _make_agent_info_list(2, n_envs)
        m.update_step(rewards, dones, info, step=5)
        result = m.compute(max_steps=100)
        assert result["M1_success_rate"] == pytest.approx(1.0)

    def test_none_done(self):
        n_envs = 10
        m = EpisodeMetrics().init(n_envs, n_targets=3, n_agents=2)
        dones = torch.zeros(n_envs, dtype=torch.bool)
        rewards = [torch.zeros(n_envs), torch.zeros(n_envs)]
        info = _make_agent_info_list(2, n_envs)
        m.update_step(rewards, dones, info, step=5)
        result = m.compute(max_steps=100)
        assert result["M1_success_rate"] == pytest.approx(0.0)

    def test_half_done(self):
        n_envs = 10
        m = EpisodeMetrics().init(n_envs, n_targets=3, n_agents=2)
        dones = torch.zeros(n_envs, dtype=torch.bool)
        dones[:5] = True
        rewards = [torch.zeros(n_envs), torch.zeros(n_envs)]
        info = _make_agent_info_list(2, n_envs)
        m.update_step(rewards, dones, info, step=5)
        result = m.compute(max_steps=100)
        assert result["M1_success_rate"] == pytest.approx(0.5)


# ── Test M2: Average Return ──────────────────────────────────────


class TestM2AvgReturn:
    def test_cumulative_reward(self):
        n_envs = 4
        m = EpisodeMetrics().init(n_envs, n_targets=3, n_agents=2)
        dones = torch.zeros(n_envs, dtype=torch.bool)
        info = _make_agent_info_list(2, n_envs)

        # 3 steps, each agent gets 1.0 reward per step
        for step in range(3):
            rewards = [torch.ones(n_envs), torch.ones(n_envs)]
            m.update_step(rewards, dones, info, step)

        result = m.compute(max_steps=100)
        # 3 steps * 2 agents * 1.0 = 6.0 per env
        assert result["M2_avg_return"] == pytest.approx(6.0)


# ── Test M3: Steps to Completion ─────────────────────────────────


class TestM3Steps:
    def test_completed_episodes(self):
        n_envs = 4
        m = EpisodeMetrics().init(n_envs, n_targets=3, n_agents=2)
        info = _make_agent_info_list(2, n_envs)
        rewards = [torch.zeros(n_envs), torch.zeros(n_envs)]

        # Run 10 steps, done at step 5 for all envs
        for step in range(10):
            dones = torch.zeros(n_envs, dtype=torch.bool)
            if step == 5:
                dones[:] = True
            m.update_step(rewards, dones, info, step)

        result = m.compute(max_steps=200)
        assert result["M3_avg_steps"] == pytest.approx(5.0)

    def test_incomplete_episodes_use_max_steps(self):
        n_envs = 4
        m = EpisodeMetrics().init(n_envs, n_targets=3, n_agents=2)
        info = _make_agent_info_list(2, n_envs)
        rewards = [torch.zeros(n_envs), torch.zeros(n_envs)]

        for step in range(10):
            dones = torch.zeros(n_envs, dtype=torch.bool)
            m.update_step(rewards, dones, info, step)

        result = m.compute(max_steps=200)
        assert result["M3_avg_steps"] == pytest.approx(200.0)

    def test_mixed_completion(self):
        n_envs = 4
        m = EpisodeMetrics().init(n_envs, n_targets=3, n_agents=2)
        info = _make_agent_info_list(2, n_envs)
        rewards = [torch.zeros(n_envs), torch.zeros(n_envs)]

        for step in range(10):
            dones = torch.zeros(n_envs, dtype=torch.bool)
            if step == 3:
                dones[:2] = True  # first 2 envs done at step 3
            m.update_step(rewards, dones, info, step)

        result = m.compute(max_steps=200)
        # 2 envs at step 3, 2 envs at 200 → avg = (3+3+200+200)/4 = 101.5
        assert result["M3_avg_steps"] == pytest.approx(101.5)


# ── Test M4: Collisions ──────────────────────────────────────────


class TestM4Collisions:
    def test_no_collisions(self):
        n_envs = 4
        m = EpisodeMetrics().init(n_envs, n_targets=3, n_agents=2)
        dones = torch.zeros(n_envs, dtype=torch.bool)
        rewards = [torch.zeros(n_envs), torch.zeros(n_envs)]
        info = _make_agent_info_list(2, n_envs, collision_rew=0.0)
        m.update_step(rewards, dones, info, step=0)
        result = m.compute(max_steps=100)
        assert result["M4_avg_collisions"] == pytest.approx(0.0)

    def test_collisions_counted(self):
        n_envs = 4
        m = EpisodeMetrics().init(n_envs, n_targets=3, n_agents=2)
        dones = torch.zeros(n_envs, dtype=torch.bool)
        rewards = [torch.zeros(n_envs), torch.zeros(n_envs)]
        # Both agents collide in every env
        info = _make_agent_info_list(2, n_envs, collision_rew=-0.1)
        m.update_step(rewards, dones, info, step=0)
        result = m.compute(max_steps=100)
        # 2 agents * 4 envs collisions / 4 envs = 2.0
        assert result["M4_avg_collisions"] == pytest.approx(2.0)


# ── Test M6: Coverage Progress ───────────────────────────────────


class TestM6CoverageProgress:
    def test_full_coverage(self):
        n_envs = 4
        n_targets = 7
        m = EpisodeMetrics().init(n_envs, n_targets=n_targets, n_agents=2)
        dones = torch.zeros(n_envs, dtype=torch.bool)
        rewards = [torch.zeros(n_envs), torch.zeros(n_envs)]

        # Each step covers 1 target; after 7 steps all covered
        for step in range(n_targets):
            info = _make_agent_info_list(2, n_envs, targets_covered=1)
            m.update_step(rewards, dones, info, step)

        result = m.compute(max_steps=200)
        assert result["M6_coverage_progress"] == pytest.approx(1.0)

    def test_partial_coverage(self):
        n_envs = 4
        n_targets = 10
        m = EpisodeMetrics().init(n_envs, n_targets=n_targets, n_agents=2)
        dones = torch.zeros(n_envs, dtype=torch.bool)
        rewards = [torch.zeros(n_envs), torch.zeros(n_envs)]

        # Cover 5 targets across 5 steps
        for step in range(5):
            info = _make_agent_info_list(2, n_envs, targets_covered=1)
            m.update_step(rewards, dones, info, step)

        result = m.compute(max_steps=200)
        assert result["M6_coverage_progress"] == pytest.approx(0.5)

    def test_zero_coverage(self):
        n_envs = 4
        m = EpisodeMetrics().init(n_envs, n_targets=7, n_agents=2)
        dones = torch.zeros(n_envs, dtype=torch.bool)
        rewards = [torch.zeros(n_envs), torch.zeros(n_envs)]
        info = _make_agent_info_list(2, n_envs, targets_covered=0)
        m.update_step(rewards, dones, info, step=0)
        result = m.compute(max_steps=200)
        assert result["M6_coverage_progress"] == pytest.approx(0.0)


# ── Test M7: Sample Efficiency ───────────────────────────────────


class TestM7SampleEfficiency:
    def _make_csv_dir(self, tmp_path):
        """Create a mock BenchMARL output structure."""
        scalars = tmp_path / "output" / "benchmarl" / "run_abc" / "run_abc" / "scalars"
        scalars.mkdir(parents=True)
        return scalars

    def test_basic(self, tmp_path):
        scalars = self._make_csv_dir(tmp_path)

        # Write eval reward CSV
        with open(scalars / "eval_reward_episode_reward_mean.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "value"])
            w.writerows([(0, 1.0), (10, 5.0), (20, 8.0), (30, 10.0)])

        # Write frames CSV
        with open(scalars / "counters_total_frames.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "value"])
            w.writerows([(0, 0), (10, 60000), (20, 120000), (30, 180000)])

        result = compute_m7_sample_efficiency(tmp_path, threshold_fraction=0.80)
        # 80% of 10.0 = 8.0, first reached at step 20 → 120000 frames
        assert result == pytest.approx(120000.0)

    def test_never_reaches_threshold(self, tmp_path):
        scalars = self._make_csv_dir(tmp_path)

        with open(scalars / "eval_reward_episode_reward_mean.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "value"])
            w.writerows([(0, -10.0), (10, -5.0)])

        with open(scalars / "counters_total_frames.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "value"])
            w.writerows([(0, 0), (10, 60000)])

        result = compute_m7_sample_efficiency(tmp_path)
        # Final reward is negative → returns None
        assert result is None

    def test_missing_dir(self, tmp_path):
        result = compute_m7_sample_efficiency(tmp_path)
        assert result is None

    def test_immediate_threshold(self, tmp_path):
        scalars = self._make_csv_dir(tmp_path)

        with open(scalars / "eval_reward_episode_reward_mean.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "value"])
            w.writerows([(0, 10.0), (5, 10.0)])

        with open(scalars / "counters_total_frames.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "value"])
            w.writerows([(0, 0), (5, 30000)])

        result = compute_m7_sample_efficiency(tmp_path, threshold_fraction=0.80)
        # 80% of 10 = 8, already at step 0 (value=10) → 0 frames
        assert result == pytest.approx(0.0)


# ── Test M8: Agent Utilization ───────────────────────────────────


class TestM8AgentUtilization:
    def test_balanced_covering(self):
        """All agents cover equally → CV should be 0."""
        n_envs = 4
        n_agents = 3
        m = EpisodeMetrics().init(n_envs, n_targets=7, n_agents=n_agents)
        dones = torch.zeros(n_envs, dtype=torch.bool)
        rewards = [torch.zeros(n_envs)] * n_agents

        # All agents get equal covering reward
        info = _make_agent_info_list(
            n_agents, n_envs,
            covering_rewards=[1.0, 1.0, 1.0],
        )
        for step in range(10):
            m.update_step(rewards, dones, info, step)

        result = m.compute(max_steps=200)
        assert result["M8_agent_utilization"] == pytest.approx(0.0, abs=1e-6)

    def test_unbalanced_covering(self):
        """One agent does all covering → high CV."""
        n_envs = 4
        n_agents = 3
        m = EpisodeMetrics().init(n_envs, n_targets=7, n_agents=n_agents)
        dones = torch.zeros(n_envs, dtype=torch.bool)
        rewards = [torch.zeros(n_envs)] * n_agents

        # Only agent 0 covers targets
        info = _make_agent_info_list(
            n_agents, n_envs,
            covering_rewards=[1.0, 0.0, 0.0],
        )
        for step in range(10):
            m.update_step(rewards, dones, info, step)

        result = m.compute(max_steps=200)
        # Agent counts: [10, 0, 0]. mean=3.33, std=5.77 → CV ≈ 1.73
        assert result["M8_agent_utilization"] > 1.0

    def test_no_covering(self):
        """No covering at all → CV should be 0."""
        n_envs = 4
        m = EpisodeMetrics().init(n_envs, n_targets=7, n_agents=3)
        dones = torch.zeros(n_envs, dtype=torch.bool)
        rewards = [torch.zeros(n_envs)] * 3
        info = _make_agent_info_list(3, n_envs, covering_rewards=[0.0, 0.0, 0.0])
        m.update_step(rewards, dones, info, step=0)
        result = m.compute(max_steps=200)
        assert result["M8_agent_utilization"] == pytest.approx(0.0)


# ── Test M9: Spatial Spread ──────────────────────────────────────


class TestM9SpatialSpread:
    def test_spread_agents(self):
        """Agents far apart → high spatial spread."""
        n_envs = 2
        n_agents = 3
        m = EpisodeMetrics().init(n_envs, n_targets=7, n_agents=n_agents)
        dones = torch.zeros(n_envs, dtype=torch.bool)
        rewards = [torch.zeros(n_envs)] * n_agents
        info = _make_agent_info_list(n_agents, n_envs)

        # Agents at (-1,0), (0,0), (1,0) — spread 1.0 apart
        positions = torch.tensor([
            [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
            [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
        ])
        m.update_step(rewards, dones, info, step=0, agent_positions=positions)
        result = m.compute(max_steps=200)
        # Pairwise distances: (1, 1, 2) → mean = 4/3 ≈ 1.333
        assert result["M9_spatial_spread"] == pytest.approx(4.0 / 3.0, abs=0.01)

    def test_clumped_agents(self):
        """Agents all at same position → spread ≈ 0."""
        n_envs = 2
        n_agents = 3
        m = EpisodeMetrics().init(n_envs, n_targets=7, n_agents=n_agents)
        dones = torch.zeros(n_envs, dtype=torch.bool)
        rewards = [torch.zeros(n_envs)] * n_agents
        info = _make_agent_info_list(n_agents, n_envs)

        positions = torch.zeros(n_envs, n_agents, 2)
        m.update_step(rewards, dones, info, step=0, agent_positions=positions)
        result = m.compute(max_steps=200)
        assert result["M9_spatial_spread"] == pytest.approx(0.0, abs=1e-6)

    def test_spread_larger_than_clumped(self):
        """Verify spread agents produce higher M9 than clumped."""
        n_envs = 4
        n_agents = 4

        # Spread case
        m_spread = EpisodeMetrics().init(n_envs, n_targets=7, n_agents=n_agents)
        positions_spread = torch.tensor([
            [[-1, -1], [1, -1], [-1, 1], [1, 1]],
        ], dtype=torch.float).expand(n_envs, -1, -1)
        dones = torch.zeros(n_envs, dtype=torch.bool)
        rewards = [torch.zeros(n_envs)] * n_agents
        info = _make_agent_info_list(n_agents, n_envs)
        m_spread.update_step(rewards, dones, info, step=0, agent_positions=positions_spread)
        r_spread = m_spread.compute(max_steps=200)

        # Clumped case
        m_clump = EpisodeMetrics().init(n_envs, n_targets=7, n_agents=n_agents)
        positions_clump = torch.zeros(n_envs, n_agents, 2)
        positions_clump += torch.randn_like(positions_clump) * 0.01
        m_clump.update_step(rewards, dones, info, step=0, agent_positions=positions_clump)
        r_clump = m_clump.compute(max_steps=200)

        assert r_spread["M9_spatial_spread"] > r_clump["M9_spatial_spread"]

    def test_no_positions_gives_zero(self):
        """Without agent positions, M9 should be 0."""
        n_envs = 4
        m = EpisodeMetrics().init(n_envs, n_targets=7, n_agents=3)
        dones = torch.zeros(n_envs, dtype=torch.bool)
        rewards = [torch.zeros(n_envs)] * 3
        info = _make_agent_info_list(3, n_envs)
        m.update_step(rewards, dones, info, step=0)  # No agent_positions
        result = m.compute(max_steps=200)
        assert result["M9_spatial_spread"] == pytest.approx(0.0)


# ── Test cross-experiment utilities ──────────────────────────────


class TestCrossExperimentUtils:
    def test_budget_frontier(self):
        results = {
            0.25: {"M1_success_rate": 0.3},
            0.50: {"M1_success_rate": 0.6},
            1.00: {"M1_success_rate": 0.9},
        }
        frontier = compute_budget_frontier(results)
        assert frontier == {0.25: 0.3, 0.50: 0.6, 1.00: 0.9}

    def test_transfer_score(self):
        source = {"M1_success_rate": 0.8}
        target = {"M1_success_rate": 0.6}
        assert compute_transfer_score(source, target) == pytest.approx(0.75)

    def test_transfer_score_zero_source(self):
        assert compute_transfer_score(
            {"M1_success_rate": 0.0},
            {"M1_success_rate": 0.5},
        ) == 0.0

    def test_ate(self):
        baseline = {"M1_success_rate": 0.8, "M2_avg_return": 50.0}
        ablated = {"M1_success_rate": 0.3, "M2_avg_return": 20.0}
        result = compute_ate(baseline, ablated)
        assert result["ate_success"] == pytest.approx(0.5)
        assert result["ate_return"] == pytest.approx(30.0)

    def test_delta_return_per_msg(self):
        with_comm = {"M2_avg_return": 60.0, "M5_avg_tokens": 100.0}
        no_comm = {"M2_avg_return": 40.0}
        assert compute_delta_return_per_msg(with_comm, no_comm) == pytest.approx(0.2)

    def test_delta_return_zero_tokens(self):
        assert compute_delta_return_per_msg(
            {"M2_avg_return": 60.0, "M5_avg_tokens": 0.0},
            {"M2_avg_return": 40.0},
        ) == 0.0


# ── Test multi-step episode ──────────────────────────────────────


class TestMultiStepEpisode:
    """Integration test simulating a realistic multi-step episode."""

    def test_full_episode(self):
        n_envs = 8
        n_agents = 4
        n_targets = 5
        max_steps = 50

        m = EpisodeMetrics().init(
            n_envs, n_targets=n_targets, n_agents=n_agents,
        )

        for step in range(max_steps):
            rewards = [torch.randn(n_envs) * 0.5 for _ in range(n_agents)]

            # Some envs complete at step 20
            dones = torch.zeros(n_envs, dtype=torch.bool)
            if step == 20:
                dones[:4] = True

            # Covering: 1 target per step for first 5 steps
            tc = 1 if step < n_targets else 0
            cov_rewards = [1.0 if step < n_targets else 0.0] * n_agents
            info = _make_agent_info_list(
                n_agents, n_envs,
                covering_rewards=cov_rewards,
                collision_rew=-0.1 if step % 5 == 0 else 0.0,
                targets_covered=tc,
            )

            # Agent positions moving outward over time
            scale = min(step / 20.0, 1.0)
            positions = torch.tensor([
                [-1, -1], [1, -1], [-1, 1], [1, 1]
            ], dtype=torch.float) * scale
            positions = positions.unsqueeze(0).expand(n_envs, -1, -1)

            m.update_step(rewards, dones, info, step, agent_positions=positions)

        result = m.compute(max_steps=max_steps)

        # Verify all metrics are present and reasonable
        assert 0.0 <= result["M1_success_rate"] <= 1.0
        assert result["M1_success_rate"] == pytest.approx(0.5)  # 4/8 envs done
        assert result["M3_avg_steps"] < max_steps  # Some completed early
        assert result["M4_avg_collisions"] > 0  # We injected collisions
        assert result["M6_coverage_progress"] == pytest.approx(1.0)  # 5/5 covered
        assert result["M8_agent_utilization"] == pytest.approx(0.0, abs=1e-6)  # All equal
        assert result["M9_spatial_spread"] > 0  # Agents were spread out
        assert result["n_envs"] == n_envs
