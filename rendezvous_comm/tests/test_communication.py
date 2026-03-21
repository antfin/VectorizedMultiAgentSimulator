"""Tests for ER2 emergent communication (dim_c) in Discovery scenario."""
import pytest
import torch
from vmas import make_env


# ── Scenario creation ──

def _make_env(dim_c=0, comm_proximity=True, n_agents=4, n_targets=4,
              use_agent_lidar=True, num_envs=3, **kwargs):
    return make_env(
        scenario="discovery", num_envs=num_envs, device="cpu",
        continuous_actions=True, n_agents=n_agents, n_targets=n_targets,
        agents_per_target=2, lidar_range=0.35, covering_range=0.25,
        use_agent_lidar=use_agent_lidar, dim_c=dim_c,
        comm_proximity=comm_proximity, **kwargs,
    )


class TestBackwardCompatibility:
    """dim_c=0 must behave identically to original Discovery."""

    def test_no_comm_obs_shape(self):
        env = _make_env(dim_c=0, use_agent_lidar=False)
        obs = env.reset()
        # pos(2) + vel(2) + target_lidar(15) = 19
        assert obs[0].shape[-1] == 19

    def test_no_comm_with_agent_lidar_obs_shape(self):
        env = _make_env(dim_c=0, use_agent_lidar=True)
        obs = env.reset()
        # pos(2) + vel(2) + target_lidar(15) + agent_lidar(12) = 31
        assert obs[0].shape[-1] == 31

    def test_no_comm_action_size(self):
        env = _make_env(dim_c=0)
        assert env.get_agent_action_size(env.agents[0]) == 2

    def test_agents_silent_when_no_comm(self):
        env = _make_env(dim_c=0)
        for agent in env.agents:
            assert agent.silent is True

    def test_no_comm_tokens_in_info(self):
        env = _make_env(dim_c=0)
        obs = env.reset()
        actions = [torch.zeros(3, 2) for _ in env.agents]
        _, _, _, info = env.step(actions)
        assert "comm_tokens" not in info[0]


class TestCommunicationEnabled:
    """dim_c > 0 enables message passing."""

    def test_obs_shape_with_comm(self):
        env = _make_env(dim_c=8, use_agent_lidar=True)
        obs = env.reset()
        # 31 (base + agent_lidar) + 3*8 (msgs from 3 other agents) = 55
        assert obs[0].shape[-1] == 55

    def test_obs_shape_comm_without_agent_lidar(self):
        env = _make_env(dim_c=8, use_agent_lidar=False)
        obs = env.reset()
        # 19 (base) + 3*8 (msgs) = 43
        assert obs[0].shape[-1] == 43

    def test_action_size_with_comm(self):
        env = _make_env(dim_c=8)
        # 2 (physical) + 8 (comm) = 10
        assert env.get_agent_action_size(env.agents[0]) == 10

    def test_agents_not_silent(self):
        env = _make_env(dim_c=8)
        for agent in env.agents:
            assert agent.silent is False

    def test_comm_tokens_in_info(self):
        env = _make_env(dim_c=8)
        obs = env.reset()
        actions = [torch.rand(3, 10) for _ in env.agents]
        _, _, _, info = env.step(actions)
        assert "comm_tokens" in info[0]
        assert (info[0]["comm_tokens"] == 8.0).all()

    def test_messages_propagate(self):
        """Messages sent at step t are visible at step t+1."""
        env = _make_env(dim_c=4, n_agents=2, comm_proximity=False)
        env.reset()
        # Send known message from agent 0
        action_size = env.get_agent_action_size(env.agents[0])
        a0 = torch.zeros(3, action_size)
        a0[:, 2:] = 0.7  # comm part = 0.7 for all dims
        a1 = torch.zeros(3, action_size)
        a1[:, 2:] = 0.3
        obs, _, _, _ = env.step([a0, a1])
        # Agent 1's obs should contain agent 0's message (0.7)
        # obs layout: pos(2) + vel(2) + target_lidar(15) + agent_lidar(12) + msgs(4)
        msg_start = 31  # with agent lidar
        msg_from_agent0 = obs[1][:, msg_start:msg_start + 4]
        assert torch.allclose(msg_from_agent0, torch.full_like(msg_from_agent0, 0.7),
                              atol=0.01)

    def test_dim_c_varies(self):
        """Different dim_c values produce correct shapes."""
        for dc in [2, 4, 8, 16]:
            env = _make_env(dim_c=dc, use_agent_lidar=True, n_agents=3)
            obs = env.reset()
            # 31 (base+agent_lidar) + 2*dc (msgs from 2 other agents)
            expected = 31 + 2 * dc
            assert obs[0].shape[-1] == expected, f"dim_c={dc}: got {obs[0].shape[-1]}, expected {expected}"


class TestProximityGating:
    """comm_proximity=True masks messages from distant agents."""

    def test_proximity_masks_distant(self):
        """Agents far apart should receive zero messages."""
        env = _make_env(dim_c=4, n_agents=2, comm_proximity=True,
                        num_envs=1)
        env.reset()
        # Move agents far apart (beyond lidar_range=0.35)
        env.agents[0].set_pos(torch.tensor([[0.8, 0.0]]), batch_index=None)
        env.agents[1].set_pos(torch.tensor([[-0.8, 0.0]]), batch_index=None)
        # Distance = 1.6, well beyond 0.35
        action_size = env.get_agent_action_size(env.agents[0])
        a0 = torch.zeros(1, action_size)
        a0[:, 2:] = 0.9  # send loud message
        a1 = torch.zeros(1, action_size)
        obs, _, _, _ = env.step([a0, a1])
        # Agent 1 should see zeros (agent 0 is out of range)
        msg_start = 31
        msg_from_agent0 = obs[1][:, msg_start:msg_start + 4]
        assert torch.allclose(msg_from_agent0, torch.zeros_like(msg_from_agent0),
                              atol=0.01)

    def test_proximity_allows_close(self):
        """Agents close together should receive messages."""
        env = _make_env(dim_c=4, n_agents=2, comm_proximity=True,
                        num_envs=1)
        env.reset()
        # Move agents close together (within lidar_range=0.35)
        env.agents[0].set_pos(torch.tensor([[0.0, 0.0]]), batch_index=None)
        env.agents[1].set_pos(torch.tensor([[0.1, 0.0]]), batch_index=None)
        # Distance = 0.1, within 0.35
        action_size = env.get_agent_action_size(env.agents[0])
        a0 = torch.zeros(1, action_size)
        a0[:, 2:] = 0.9
        a1 = torch.zeros(1, action_size)
        obs, _, _, _ = env.step([a0, a1])
        msg_start = 31
        msg_from_agent0 = obs[1][:, msg_start:msg_start + 4]
        assert torch.allclose(msg_from_agent0, torch.full_like(msg_from_agent0, 0.9),
                              atol=0.01)

    def test_no_proximity_always_receives(self):
        """comm_proximity=False: messages arrive regardless of distance."""
        env = _make_env(dim_c=4, n_agents=2, comm_proximity=False,
                        num_envs=1)
        env.reset()
        # Far apart
        env.agents[0].set_pos(torch.tensor([[0.8, 0.0]]), batch_index=None)
        env.agents[1].set_pos(torch.tensor([[-0.8, 0.0]]), batch_index=None)
        action_size = env.get_agent_action_size(env.agents[0])
        a0 = torch.zeros(1, action_size)
        a0[:, 2:] = 0.9
        a1 = torch.zeros(1, action_size)
        obs, _, _, _ = env.step([a0, a1])
        msg_start = 31
        msg_from_agent0 = obs[1][:, msg_start:msg_start + 4]
        assert torch.allclose(msg_from_agent0, torch.full_like(msg_from_agent0, 0.9),
                              atol=0.01)


class TestRewardUnchanged:
    """Communication should not alter the reward structure."""

    def test_reward_shape(self):
        env = _make_env(dim_c=8)
        env.reset()
        actions = [torch.rand(3, 10) for _ in env.agents]
        _, rew, _, _ = env.step(actions)
        assert len(rew) == 4  # n_agents
        assert rew[0].shape == (3,)  # batch_dim

    def test_done_still_works(self):
        env = _make_env(dim_c=8)
        env.reset()
        actions = [torch.rand(3, 10) for _ in env.agents]
        _, _, done, _ = env.step(actions)
        assert done.shape == (3,)


class TestScaling:
    """Communication scales with agent count."""

    def test_3_agents(self):
        env = _make_env(dim_c=4, n_agents=3)
        obs = env.reset()
        # 31 + 2*4 = 39 (msgs from 2 others)
        assert obs[0].shape[-1] == 39

    def test_6_agents(self):
        env = _make_env(dim_c=4, n_agents=6)
        obs = env.reset()
        # 31 + 5*4 = 51 (msgs from 5 others)
        assert obs[0].shape[-1] == 51
