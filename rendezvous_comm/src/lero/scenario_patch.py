"""Monkey-patch Discovery scenario with LLM-generated functions.

Two key design decisions:

1. REWARD = original + LLM bonus (not replacement)
   The LLM generates a bonus added to the original reward. This prevents
   reward hacking — the base task signal (covering reward) is always present.

2. TWO-TIER scenario_state (CTDE)
   - Reward gets FULL global state (centralized training)
   - Observation enhancement gets LOCAL sensor state only (decentralized execution)
   This ensures agents don't get oracle information at execution time.
"""

import logging
from typing import Any, Callable, Dict, Optional

import torch

_log = logging.getLogger("rendezvous.lero")

# Namespace available to LLM-generated code
_EXEC_NAMESPACE = {
    "torch": torch,
    "math": __import__("math"),
    "F": torch.nn.functional,
}


def _build_reward_state(scenario, agent, agent_idx: int) -> Dict[str, Any]:
    """Build FULL scenario_state for reward function (centralized training).

    The reward function can see everything — this is standard CTDE
    (Centralized Training, Decentralized Execution).
    """
    agents_pos = getattr(scenario, "agents_pos", None)
    if agents_pos is None:
        agents_pos = torch.stack(
            [a.state.pos for a in scenario.world.agents], dim=1,
        )
    targets_pos = getattr(scenario, "targets_pos", None)
    if targets_pos is None:
        targets_pos = torch.stack(
            [t.state.pos for t in scenario._targets], dim=1,
        )
    agents_targets_dists = getattr(scenario, "agents_targets_dists", None)
    if agents_targets_dists is None:
        agents_targets_dists = torch.cdist(agents_pos, targets_pos)
    agents_per_target = getattr(scenario, "agents_per_target", None)
    if agents_per_target is None:
        agents_per_target = torch.sum(
            (agents_targets_dists < scenario._covering_range).int(), dim=1,
        )
    covered_targets = getattr(scenario, "covered_targets", None)
    if covered_targets is None:
        covered_targets = agents_per_target >= scenario._agents_per_target

    batch_dim = scenario.world.batch_dim
    device = scenario.world.device

    state = {
        "agents_pos": agents_pos,
        "targets_pos": targets_pos,
        "agents_targets_dists": agents_targets_dists,
        "covered_targets": covered_targets,
        "agents_per_target": agents_per_target,
        "all_time_covered": getattr(
            scenario, "all_time_covered_targets",
            torch.zeros_like(covered_targets),
        ),
        "agent_pos": agent.state.pos,
        "agent_vel": agent.state.vel,
        "agent_idx": agent_idx,
        "n_agents": len(scenario.world.agents),
        "n_targets": scenario.n_targets,
        "covering_range": scenario._covering_range,
        "agents_per_target_required": scenario._agents_per_target,
        "collision_penalty": scenario.agent_collision_penalty,
        "collision_rew": getattr(agent, "collision_rew", torch.zeros(
            batch_dim, device=device,
        )),
        "time_penalty": scenario.time_penalty,
    }
    dim_c = getattr(scenario, "dim_c", 0)
    if dim_c > 0:
        messages = []
        for other in scenario.world.agents:
            if other is not agent and other.state.c is not None:
                messages.append(other.state.c)
        if messages:
            state["messages"] = torch.stack(messages, dim=1)
        else:
            state["messages"] = torch.zeros(
                batch_dim, len(scenario.world.agents) - 1,
                dim_c, device=device,
            )
    return state


def _build_obs_state(scenario, agent, agent_idx: int) -> Dict[str, Any]:
    """Build LOCAL sensor state for observation enhancement (decentralized).

    Only includes information derivable from the agent's own sensors.
    No global state — agents can't see other agents' positions, which
    targets are covered, or how many agents are at each target.
    """
    batch_dim = scenario.world.batch_dim
    device = scenario.world.device

    state = {
        "agent_pos": agent.state.pos,
        "agent_vel": agent.state.vel,
        "agent_idx": agent_idx,
        "n_agents": len(scenario.world.agents),
        "n_targets": scenario.n_targets,
        "covering_range": scenario._covering_range,
        "agents_per_target_required": scenario._agents_per_target,
        # LiDAR sensor readings (local, what the agent actually sees)
        "lidar_targets": agent.sensors[0].measure(),
    }
    if getattr(scenario, "use_agent_lidar", False) and len(agent.sensors) > 1:
        state["lidar_agents"] = agent.sensors[1].measure()

    # Communication messages (if enabled — these ARE part of the obs)
    dim_c = getattr(scenario, "dim_c", 0)
    if dim_c > 0:
        messages = []
        comm_proximity = getattr(scenario, "comm_proximity", False)
        comms_range = getattr(scenario, "_comms_range", None)
        for other in scenario.world.agents:
            if other is not agent and other.state.c is not None:
                msg = other.state.c
                if comm_proximity and comms_range is not None:
                    dist = torch.linalg.vector_norm(
                        agent.state.pos - other.state.pos,
                        dim=-1, keepdim=True,
                    )
                    in_range = (dist <= comms_range).float()
                    msg = msg * in_range
                messages.append(msg)
        if messages:
            state["messages"] = torch.stack(messages, dim=1)

    return state


def _compile_function(source: str, func_name: str) -> Callable:
    """Compile LLM-generated source into a callable function."""
    namespace = dict(_EXEC_NAMESPACE)
    exec(source, namespace)  # noqa: S102
    if func_name not in namespace:
        raise ValueError(
            f"Function '{func_name}' not found after exec. "
            f"Available: {[k for k in namespace if not k.startswith('_')]}"
        )
    return namespace[func_name]


def make_patched_scenario_class(
    reward_source: Optional[str] = None,
    obs_source: Optional[str] = None,
    reward_mode: str = "replace",
    obs_state_mode: str = "global",
    bonus_scale: float = 0.5,
):
    """Create a Discovery Scenario subclass with LLM-generated methods.

    reward_mode:
      "replace" (paper): R = compute_reward(full_state)
        LLM designs the entire reward. Original reward is NOT used.
      "bonus": R = R_original + bonus_scale * tanh(compute_reward_bonus(state))
        LLM designs a bonus added to the original reward.

    obs_state_mode:
      "global" (paper): enhance_observation gets full state (all positions, coverage)
      "local" (CTDE): enhance_observation gets only own sensors
    """
    from vmas.scenarios.discovery import Scenario as OrigScenario

    reward_fn = None
    obs_fn = None
    if reward_source:
        # Accept both names — paper uses compute_reward, bonus mode uses compute_reward_bonus
        if "def compute_reward_bonus" in reward_source:
            reward_fn = _compile_function(reward_source, "compute_reward_bonus")
        else:
            reward_fn = _compile_function(reward_source, "compute_reward")
    if obs_source:
        obs_fn = _compile_function(obs_source, "enhance_observation")

    class PatchedDiscoveryScenario(OrigScenario):
        """Discovery scenario with LLM reward bonus + obs enhancement."""

        def info(self, agent):
            # Always return per-agent covering_reward (not the shared total)
            # so metrics.M8 (agent_utilization CV) can measure per-agent
            # contribution variance. Discovery's upstream info() returns
            # self.shared_covering_rew for every agent when shared_reward=True,
            # which makes M8 structurally zero.
            base = super().info(agent)
            base["covering_reward"] = agent.covering_reward
            return base

        if reward_fn is not None:
            _rmode = reward_mode
            _bs = bonus_scale

            if _rmode == "replace":
                def reward(self, agent):
                    # Run original reward for side effects (computes
                    # shared state, collision penalties, target respawning)
                    _ = super().reward(agent)

                    # Return LLM-generated reward ONLY
                    agent_idx = self.world.agents.index(agent)
                    state = _build_reward_state(self, agent, agent_idx)
                    return reward_fn(state)
            else:
                def reward(self, agent, _bonus_scale=_bs):
                    original_reward = super().reward(agent)
                    agent_idx = self.world.agents.index(agent)
                    state = _build_reward_state(self, agent, agent_idx)
                    raw_bonus = reward_fn(state)
                    bonus = _bonus_scale * torch.tanh(raw_bonus)
                    return original_reward + bonus

        if obs_fn is not None:
            # obs_state_mode is captured via closure of make_patched_scenario_class.
            # Do NOT redefine it as a class attribute — class-body names aren't
            # visible inside method bodies (Python scoping rule).
            def observation(self, agent, _mode=obs_state_mode):
                base_obs = super().observation(agent)
                agent_idx = self.world.agents.index(agent)
                if _mode == "global":
                    state = _build_reward_state(self, agent, agent_idx)
                else:
                    state = _build_obs_state(self, agent, agent_idx)
                extra = obs_fn(state)
                if isinstance(base_obs, dict):
                    base_obs["observation"] = torch.cat(
                        [base_obs["observation"], extra], dim=-1,
                    )
                    return base_obs
                return torch.cat([base_obs, extra], dim=-1)

    _log.info(
        "Created PatchedDiscoveryScenario (reward=%s, obs=%s)",
        reward_source is not None, obs_source is not None,
    )
    return PatchedDiscoveryScenario
