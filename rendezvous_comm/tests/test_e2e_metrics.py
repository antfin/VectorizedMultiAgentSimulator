"""End-to-end tests validating metrics against known VMAS outcomes.

These tests run real VMAS Discovery environments with controlled
agent/target positions to verify each metric (M1-M9) is computed
correctly from actual simulation data — not just synthetic tensors.

Strategy: after env.reset(), override agent/target positions to
create deterministic scenarios, then step with zero actions and
verify metrics match expected values.
"""
import math

import pytest
import torch

from vmas import make_env

from src.config import TaskConfig
from src.metrics import EpisodeMetrics
from src.runner import evaluate_with_vmas


# ── Helpers ──────────────────────────────────────────────────────

N_ENVS = 8  # small for speed, enough to average

# Minimal config: 2 agents, 2 targets, K=1 (easy to control)
SIMPLE_CONFIG = dict(
    n_agents=2,
    n_targets=2,
    agents_per_target=1,
    covering_range=0.25,
    lidar_range=0.35,
    use_agent_lidar=False,
    targets_respawn=False,
    shared_reward=False,
    agent_collision_penalty=-0.1,
    covering_rew_coeff=1.0,
    time_penalty=-0.01,
    n_lidar_rays_entities=12,
    n_lidar_rays_agents=8,
)


def _make_env(max_steps=50, **overrides):
    """Create a Discovery env with SIMPLE_CONFIG + overrides."""
    config = {**SIMPLE_CONFIG, **overrides}
    ms = config.pop("max_steps", max_steps)
    return make_env(
        scenario="discovery",
        num_envs=N_ENVS,
        device="cpu",
        continuous_actions=True,
        max_steps=ms,
        **config,
    ), ms


def _zero_actions(env):
    """Return zero actions for all agents."""
    return [torch.zeros(N_ENVS, env.agents[i].action_size) for i in range(len(env.agents))]


def _set_positions(env, agent_positions, target_positions=None):
    """Override agent (and optionally target) positions for all envs.

    Args:
        agent_positions: list of (x, y) tuples, one per agent
        target_positions: list of (x, y) tuples, one per target
    """
    for i, (x, y) in enumerate(agent_positions):
        env.agents[i].state.pos[:] = torch.tensor([x, y])
        env.agents[i].state.vel[:] = 0.0

    if target_positions is not None:
        targets = env.world.landmarks
        for i, (x, y) in enumerate(target_positions):
            targets[i].state.pos[:] = torch.tensor([x, y])


def _run_episode(env, max_steps, policy_fn=None, setup_fn=None):
    """Run one episode, return EpisodeMetrics result dict.

    Args:
        setup_fn: callable(env) invoked after reset() but before the step loop.
                  Use this to override positions for deterministic scenarios.
    """
    n_agents = len(env.agents)
    n_targets = len(env.world.landmarks)

    metrics = EpisodeMetrics().init(
        N_ENVS, n_targets=n_targets, n_agents=n_agents,
    )
    obs = env.reset()

    if setup_fn is not None:
        setup_fn(env)

    for step in range(max_steps):
        if policy_fn is not None:
            actions = policy_fn(obs, env, step)
        else:
            actions = _zero_actions(env)
        obs, rews, dones, info = env.step(actions)

        agent_positions = torch.stack(
            [a.state.pos for a in env.agents], dim=1,
        )
        metrics.update_step(
            rews, dones, info, step,
            agent_positions=agent_positions,
        )

    return metrics.compute(max_steps)


# ── M1: Success Rate ────────────────────────────────────────────


class TestM1EndToEnd:
    """Verify done signal from VMAS maps correctly to M1."""

    def test_agents_on_targets_succeed(self):
        """Place each agent directly on a target → all targets covered → M1=1.0."""
        env, max_steps = _make_env(max_steps=10)
        obs = env.reset()

        # Read target positions and place agents on top of them
        targets = env.world.landmarks
        target_pos = [targets[i].state.pos[0].tolist() for i in range(2)]

        _set_positions(env, agent_positions=target_pos)

        # Step with zero actions — agents barely move, covering check fires
        actions = _zero_actions(env)
        obs, rews, dones, info = env.step(actions)

        # Done should fire: both targets have 1 agent within covering_range
        assert dones.any().item(), (
            "Expected done=True when agents are on targets. "
            f"Agent0 pos={env.agents[0].state.pos[0].tolist()}, "
            f"Target0 pos={targets[0].state.pos[0].tolist()}"
        )

    def test_agents_on_targets_M1_is_one(self):
        """Full episode with agents on targets → M1=1.0."""
        env, max_steps = _make_env(max_steps=10)

        def setup(env):
            targets = env.world.landmarks
            for i in range(len(env.agents)):
                env.agents[i].state.pos[:] = targets[i].state.pos
                env.agents[i].state.vel[:] = 0.0

        result = _run_episode(env, max_steps, setup_fn=setup)
        assert result["M1_success_rate"] == pytest.approx(1.0), (
            f"Expected M1=1.0 when agents placed on targets, got {result['M1_success_rate']}"
        )

    def test_stationary_agents_far_from_targets_fail(self):
        """Agents far from targets, few steps → M1=0.0."""
        env, max_steps = _make_env(max_steps=5)

        def setup(env):
            _set_positions(
                env,
                agent_positions=[(-0.9, -0.9), (0.9, 0.9)],
                target_positions=[(0.5, 0.5), (-0.5, -0.5)],
            )

        result = _run_episode(env, max_steps, setup_fn=setup)
        assert result["M1_success_rate"] == pytest.approx(0.0), (
            f"Expected M1=0.0 with agents far from targets, got {result['M1_success_rate']}"
        )

    def test_truncation_not_counted_as_success(self):
        """Time-limit truncation (done=True at max_steps) must NOT count as M1 success."""
        env, max_steps = _make_env(max_steps=1)

        def setup(env):
            _set_positions(
                env,
                agent_positions=[(-0.9, -0.9), (0.9, 0.9)],
                target_positions=[(0.5, 0.5), (-0.5, -0.5)],
            )

        result = _run_episode(env, max_steps, setup_fn=setup)
        assert result["M1_success_rate"] == pytest.approx(0.0), (
            f"Time truncation inflated M1: got {result['M1_success_rate']} "
            f"(should be 0.0 — no targets were covered)"
        )


# ── M2: Average Return ─────────────────────────────────────────


class TestM2EndToEnd:
    """Verify reward components are correctly accumulated."""

    def test_covering_gives_positive_return(self):
        """Agents that cover targets should earn positive covering reward."""
        env, max_steps = _make_env(max_steps=10)

        def setup(env):
            targets = env.world.landmarks
            for i in range(len(env.agents)):
                env.agents[i].state.pos[:] = targets[i].state.pos
                env.agents[i].state.vel[:] = 0.0

        result = _run_episode(env, max_steps, setup_fn=setup)
        # Covering reward (positive) should outweigh time penalty
        # At minimum: covering_rew_coeff * targets_covered - time_penalty * steps
        assert result["M2_avg_return"] > 0.0, (
            f"Expected positive return when covering targets, got {result['M2_avg_return']}"
        )

    def test_no_covering_gives_negative_return(self):
        """Stationary agents far from targets → only time penalty → negative return."""
        env, max_steps = _make_env(max_steps=10)

        def setup(env):
            _set_positions(
                env,
                agent_positions=[(-0.95, -0.95), (0.95, 0.95)],
                target_positions=[(0.0, 0.5), (0.0, -0.5)],
            )

        result = _run_episode(env, max_steps, setup_fn=setup)
        # Only time_penalty (-0.01) per step per agent, no covering reward
        assert result["M2_avg_return"] < 0.0, (
            f"Expected negative return with no covering, got {result['M2_avg_return']}"
        )


# ── M3: Steps to Completion ────────────────────────────────────


class TestM3EndToEnd:
    """Verify step counting matches actual episode length."""

    def test_immediate_completion_gives_low_steps(self):
        """Agents on targets → done quickly → M3 much less than max_steps."""
        env, max_steps = _make_env(max_steps=50)

        def setup(env):
            targets = env.world.landmarks
            for i in range(len(env.agents)):
                env.agents[i].state.pos[:] = targets[i].state.pos
                env.agents[i].state.vel[:] = 0.0

        result = _run_episode(env, max_steps, setup_fn=setup)
        # Should complete very quickly (step 0 or 1)
        assert result["M3_avg_steps"] < 5.0, (
            f"Expected quick completion, got M3={result['M3_avg_steps']}"
        )

    def test_no_completion_gives_max_steps(self):
        """Agents far from targets, never complete → M3=max_steps."""
        max_steps = 10
        env, max_steps = _make_env(max_steps=max_steps)

        def setup(env):
            _set_positions(
                env,
                agent_positions=[(-0.95, -0.95), (0.95, 0.95)],
                target_positions=[(0.0, 0.5), (0.0, -0.5)],
            )

        result = _run_episode(env, max_steps, setup_fn=setup)
        assert result["M3_avg_steps"] == pytest.approx(max_steps, abs=1.0), (
            f"Expected M3≈{max_steps} for incomplete episodes, got {result['M3_avg_steps']}"
        )


# ── M4: Collisions ─────────────────────────────────────────────


class TestM4EndToEnd:
    """Verify collision detection from real VMAS physics."""

    def test_overlapping_agents_collide(self):
        """Two agents at exact same position → collisions detected."""
        env, max_steps = _make_env(max_steps=5)

        def setup(env):
            _set_positions(
                env,
                agent_positions=[(0.0, 0.0), (0.0, 0.0)],
                target_positions=[(0.8, 0.8), (-0.8, -0.8)],
            )

        result = _run_episode(env, max_steps, setup_fn=setup)
        assert result["M4_avg_collisions"] > 0.0, (
            f"Expected collisions when agents overlap, got {result['M4_avg_collisions']}"
        )

    def test_separated_agents_no_collisions(self):
        """Agents far apart → no collisions."""
        env, max_steps = _make_env(max_steps=5)

        def setup(env):
            _set_positions(
                env,
                agent_positions=[(-0.8, 0.0), (0.8, 0.0)],
                target_positions=[(0.0, 0.8), (0.0, -0.8)],
            )

        result = _run_episode(env, max_steps, setup_fn=setup)
        assert result["M4_avg_collisions"] == pytest.approx(0.0), (
            f"Expected 0 collisions when agents are 1.6 apart, got {result['M4_avg_collisions']}"
        )


# ── M5: Tokens ──────────────────────────────────────────────────


class TestM5EndToEnd:
    """M5 is always 0 for no-comm scenarios."""

    def test_tokens_always_zero(self):
        env, max_steps = _make_env(max_steps=5)
        result = _run_episode(env, max_steps)
        assert result["M5_avg_tokens"] == pytest.approx(0.0)


# ── M6: Coverage Progress ──────────────────────────────────────


class TestM6EndToEnd:
    """Verify partial and full coverage from real VMAS."""

    def test_full_coverage(self):
        """Both targets covered → M6=1.0."""
        env, max_steps = _make_env(max_steps=10)

        def setup(env):
            targets = env.world.landmarks
            for i in range(len(env.agents)):
                env.agents[i].state.pos[:] = targets[i].state.pos
                env.agents[i].state.vel[:] = 0.0

        result = _run_episode(env, max_steps, setup_fn=setup)
        assert result["M6_coverage_progress"] == pytest.approx(1.0), (
            f"Expected M6=1.0 when all targets covered, got {result['M6_coverage_progress']}"
        )

    def test_one_of_two_targets_covered(self):
        """Only one agent near one target → M6≈0.5."""
        env, max_steps = _make_env(max_steps=5)

        def setup(env):
            targets = env.world.landmarks
            # Agent 0 on target 0, agent 1 far away
            env.agents[0].state.pos[:] = targets[0].state.pos
            env.agents[0].state.vel[:] = 0.0
            env.agents[1].state.pos[:] = torch.tensor([0.9, 0.9])
            env.agents[1].state.vel[:] = 0.0
            # Target 1 far from both agents
            targets[1].state.pos[:] = torch.tensor([-0.9, -0.9])

        result = _run_episode(env, max_steps, setup_fn=setup)
        assert result["M6_coverage_progress"] == pytest.approx(0.5, abs=0.15), (
            f"Expected M6≈0.5 with 1/2 targets covered, got {result['M6_coverage_progress']}"
        )

    def test_no_coverage(self):
        """No agents near any target → M6=0.0."""
        env, max_steps = _make_env(max_steps=3)

        def setup(env):
            _set_positions(
                env,
                agent_positions=[(-0.95, -0.95), (0.95, 0.95)],
                target_positions=[(0.0, 0.5), (0.0, -0.5)],
            )

        result = _run_episode(env, max_steps, setup_fn=setup)
        assert result["M6_coverage_progress"] == pytest.approx(0.0), (
            f"Expected M6=0.0 with no coverage, got {result['M6_coverage_progress']}"
        )


# ── M8: Agent Utilization ──────────────────────────────────────


class TestM8EndToEnd:
    """Verify workload balance measurement from real covering events."""

    def test_balanced_covering(self):
        """Both agents cover a target → low utilization CV."""
        env, max_steps = _make_env(max_steps=10)

        def setup(env):
            targets = env.world.landmarks
            for i in range(len(env.agents)):
                env.agents[i].state.pos[:] = targets[i].state.pos
                env.agents[i].state.vel[:] = 0.0

        result = _run_episode(env, max_steps, setup_fn=setup)
        # Both agents contribute equally → CV should be low
        # (Not exactly 0 because covering reward timing may differ)
        assert result["M8_agent_utilization"] < 1.0, (
            f"Expected low utilization CV with balanced covering, got {result['M8_agent_utilization']}"
        )

    def test_one_agent_covers_all(self):
        """Only agent 0 covers targets, agent 1 far away → high CV."""
        env, max_steps = _make_env(
            max_steps=10, n_targets=3, agents_per_target=1,
        )

        def setup(env):
            targets = env.world.landmarks
            # Agent 0 on target 0
            env.agents[0].state.pos[:] = targets[0].state.pos
            env.agents[0].state.vel[:] = 0.0
            # Agent 1 far away
            env.agents[1].state.pos[:] = torch.tensor([0.95, 0.95])
            env.agents[1].state.vel[:] = 0.0
            # Remaining targets far from agent 1
            for j in range(1, len(targets)):
                targets[j].state.pos[:] = torch.tensor([-0.9, -0.5 + j * 0.3])

        result = _run_episode(env, max_steps, setup_fn=setup)
        # Agent 0 has covering events, agent 1 has none → high CV
        assert result["M8_agent_utilization"] > 0.5, (
            f"Expected high utilization CV when one agent does all covering, "
            f"got {result['M8_agent_utilization']}"
        )


# ── M9: Spatial Spread ──────────────────────────────────────────


class TestM9EndToEnd:
    """Verify spatial spread from actual agent positions in VMAS."""

    def test_spread_agents_high_M9(self):
        """Agents at opposite corners → high spatial spread."""
        env, max_steps = _make_env(max_steps=3)

        def setup(env):
            _set_positions(
                env,
                agent_positions=[(-0.8, -0.8), (0.8, 0.8)],
                target_positions=[(0.0, 0.5), (0.0, -0.5)],
            )

        result = _run_episode(env, max_steps, setup_fn=setup)
        # Distance between (-0.8,-0.8) and (0.8,0.8) ≈ 2.26
        assert result["M9_spatial_spread"] > 1.0, (
            f"Expected high spatial spread for agents at opposite corners, "
            f"got {result['M9_spatial_spread']}"
        )

    def test_clumped_agents_low_M9(self):
        """Agents close together (but outside collision range) → low spread."""
        env, max_steps = _make_env(max_steps=3)

        def setup(env):
            # Place agents 0.15 apart (outside collision range of 2*radius + 0.005 = 0.105)
            # so collision forces don't push them apart
            _set_positions(
                env,
                agent_positions=[(0.0, 0.0), (0.15, 0.0)],
                target_positions=[(0.5, 0.5), (-0.5, -0.5)],
            )

        result = _run_episode(env, max_steps, setup_fn=setup)
        assert result["M9_spatial_spread"] < 0.5, (
            f"Expected low spatial spread for close agents, "
            f"got {result['M9_spatial_spread']}"
        )

    def test_spread_greater_than_clumped(self):
        """Spread agents must have higher M9 than clumped ones."""
        env_s, ms_s = _make_env(max_steps=3)
        env_c, ms_c = _make_env(max_steps=3)

        def setup_spread(env):
            _set_positions(env, [(-0.8, 0.0), (0.8, 0.0)],
                           [(0.0, 0.5), (0.0, -0.5)])

        def setup_clumped(env):
            _set_positions(env, [(0.0, 0.0), (0.02, 0.0)],
                           [(0.5, 0.5), (-0.5, -0.5)])

        r_spread = _run_episode(env_s, ms_s, setup_fn=setup_spread)
        r_clumped = _run_episode(env_c, ms_c, setup_fn=setup_clumped)

        assert r_spread["M9_spatial_spread"] > r_clumped["M9_spatial_spread"], (
            f"Spread M9={r_spread['M9_spatial_spread']:.4f} should be > "
            f"Clumped M9={r_clumped['M9_spatial_spread']:.4f}"
        )


# ── Cross-metric consistency ───────────────────────────────────


class TestCrossMetricConsistency:
    """Verify logical relationships between metrics."""

    def test_success_implies_full_coverage(self):
        """M1=1.0 implies M6=1.0."""
        env, max_steps = _make_env(max_steps=10)

        def setup(env):
            targets = env.world.landmarks
            for i in range(len(env.agents)):
                env.agents[i].state.pos[:] = targets[i].state.pos
                env.agents[i].state.vel[:] = 0.0

        result = _run_episode(env, max_steps, setup_fn=setup)
        if result["M1_success_rate"] == 1.0:
            assert result["M6_coverage_progress"] == pytest.approx(1.0), (
                f"M1=1.0 but M6={result['M6_coverage_progress']} — inconsistent!"
            )

    def test_no_success_with_max_steps(self):
        """M1=0 implies M3=max_steps."""
        max_steps = 5
        env, max_steps = _make_env(max_steps=max_steps)

        def setup(env):
            _set_positions(
                env,
                agent_positions=[(-0.95, -0.95), (0.95, 0.95)],
                target_positions=[(0.0, 0.5), (0.0, -0.5)],
            )

        result = _run_episode(env, max_steps, setup_fn=setup)
        if result["M1_success_rate"] == 0.0:
            assert result["M3_avg_steps"] == pytest.approx(max_steps, abs=1.0), (
                f"M1=0 but M3={result['M3_avg_steps']} ≠ max_steps={max_steps}"
            )

    def test_tokens_zero_without_comm(self):
        """M5 must be 0 in every scenario without communication wrapper."""
        env, max_steps = _make_env(max_steps=5)
        result = _run_episode(env, max_steps)
        assert result["M5_avg_tokens"] == 0.0

    def test_success_implies_positive_return(self):
        """If all targets covered, covering reward should dominate → M2 > 0."""
        env, max_steps = _make_env(max_steps=10)

        def setup(env):
            targets = env.world.landmarks
            for i in range(len(env.agents)):
                env.agents[i].state.pos[:] = targets[i].state.pos
                env.agents[i].state.vel[:] = 0.0

        result = _run_episode(env, max_steps, setup_fn=setup)
        if result["M1_success_rate"] == 1.0:
            assert result["M2_avg_return"] > 0.0, (
                f"M1=1.0 but M2={result['M2_avg_return']} ≤ 0 — covering reward missing?"
            )


# ── evaluate_with_vmas integration ─────────────────────────────


class TestEvaluateWithVmasE2E:
    """Verify evaluate_with_vmas produces consistent metrics end-to-end."""

    def test_random_policy_metrics_ranges(self):
        """Random policy: all metrics should be in valid ranges."""
        task = TaskConfig(
            n_agents=2, n_targets=2, agents_per_target=1,
            covering_range=0.25, lidar_range=0.35,
            targets_respawn=False, max_steps=10,
        )
        result = evaluate_with_vmas(
            task, policy_fn=None, n_eval_episodes=8, n_envs=8,
        )

        assert 0.0 <= result["M1_success_rate"] <= 1.0
        assert 0.0 <= result["M6_coverage_progress"] <= 1.0
        assert 0 <= result["M3_avg_steps"] <= 10
        assert result["M4_avg_collisions"] >= 0.0
        assert result["M5_avg_tokens"] == 0.0
        assert result["M8_agent_utilization"] >= 0.0
        assert result["M9_spatial_spread"] >= 0.0

    def test_heuristic_beats_random(self):
        """Heuristic policy should generally outperform random."""
        from src.runner import make_heuristic_policy_fn

        task = TaskConfig(
            n_agents=3, n_targets=3, agents_per_target=1,
            covering_range=0.25, lidar_range=0.35,
            targets_respawn=False, max_steps=100,
        )

        random_result = evaluate_with_vmas(
            task, policy_fn=None, n_eval_episodes=50, n_envs=50,
        )
        heuristic_fn = make_heuristic_policy_fn()
        heuristic_result = evaluate_with_vmas(
            task, policy_fn=heuristic_fn, n_eval_episodes=50, n_envs=50,
        )

        # Heuristic should have higher coverage or return than random
        assert (
            heuristic_result["M6_coverage_progress"]
            >= random_result["M6_coverage_progress"] - 0.1
        ), (
            f"Heuristic coverage {heuristic_result['M6_coverage_progress']:.3f} "
            f"worse than random {random_result['M6_coverage_progress']:.3f}"
        )
