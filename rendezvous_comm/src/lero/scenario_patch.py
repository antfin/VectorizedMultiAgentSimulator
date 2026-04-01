"""Monkey-patch Discovery scenario with LLM-generated functions.

Rather than subclassing (which requires modifying BenchMARL's task
registry), we patch the scenario instance after BenchMARL creates it.
"""

import logging
import types
from typing import Any, Callable, Dict, Optional, Tuple

import torch

_log = logging.getLogger("rendezvous.lero")

# Namespace available to LLM-generated code
_EXEC_NAMESPACE = {
    "torch": torch,
    "math": __import__("math"),
    "F": torch.nn.functional,
}


def _build_scenario_state(scenario, agent, agent_idx: int) -> Dict[str, Any]:
    """Build the scenario_state dict passed to LLM-generated functions."""
    state = {
        "agents_pos": scenario.agents_pos,
        "targets_pos": scenario.targets_pos,
        "agents_targets_dists": scenario.agents_targets_dists,
        "covered_targets": scenario.covered_targets,
        "agents_per_target": scenario.agents_per_target,
        "all_time_covered": scenario.all_time_covered_targets,
        "agent_pos": agent.state.pos,
        "agent_vel": agent.state.vel,
        "agent_idx": agent_idx,
        "n_agents": len(scenario.world.agents),
        "n_targets": scenario.n_targets,
        "covering_range": scenario._covering_range,
        "agents_per_target_required": scenario._agents_per_target,
        "collision_penalty": scenario.agent_collision_penalty,
        "collision_rew": agent.collision_rew,
        "time_penalty": scenario.time_penalty,
    }
    # Communication state (when dim_c > 0)
    if scenario.dim_c > 0:
        messages = []
        for other in scenario.world.agents:
            if other is not agent and other.state.c is not None:
                messages.append(other.state.c)
        if messages:
            state["messages"] = torch.stack(messages, dim=1)
        else:
            state["messages"] = torch.zeros(
                scenario.world.batch_dim, len(scenario.world.agents) - 1,
                scenario.dim_c, device=scenario.world.device,
            )
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


def patch_scenario(
    scenario,
    reward_source: Optional[str] = None,
    obs_source: Optional[str] = None,
) -> Dict[str, Any]:
    """Patch a Discovery scenario instance with LLM-generated functions.

    Args:
        scenario: Discovery scenario instance (already created by BenchMARL)
        reward_source: Python source for compute_reward(scenario_state)
        obs_source: Python source for enhance_observation(scenario_state)

    Returns:
        Dict with original methods for later restoration via unpatch_scenario.
    """
    originals = {
        "reward": scenario.reward,
        "observation": scenario.observation,
    }

    if reward_source is not None:
        reward_fn = _compile_function(reward_source, "compute_reward")
        original_reward = scenario.reward

        def patched_reward(agent, _scenario=scenario, _fn=reward_fn,
                           _orig=original_reward):
            # The original reward() must be called first to compute
            # shared state (agents_pos, targets_pos, covered_targets, etc.)
            # and handle collision penalties / target respawning.
            is_first = agent == _scenario.world.agents[0]
            is_last = agent == _scenario.world.agents[-1]

            if is_first:
                # Compute shared state (same as original reward preamble)
                _scenario.time_rew = torch.full(
                    (_scenario.world.batch_dim,),
                    _scenario.time_penalty,
                    device=_scenario.world.device,
                )
                _scenario.agents_pos = torch.stack(
                    [a.state.pos for a in _scenario.world.agents], dim=1
                )
                _scenario.targets_pos = torch.stack(
                    [t.state.pos for t in _scenario._targets], dim=1
                )
                _scenario.agents_targets_dists = torch.cdist(
                    _scenario.agents_pos, _scenario.targets_pos,
                )
                _scenario.agents_per_target = torch.sum(
                    (_scenario.agents_targets_dists
                     < _scenario._covering_range).int(),
                    dim=1,
                )
                _scenario.covered_targets = (
                    _scenario.agents_per_target
                    >= _scenario._agents_per_target
                )
                _scenario.shared_covering_rew[:] = 0
                for a in _scenario.world.agents:
                    _scenario.shared_covering_rew += (
                        _scenario.agent_reward(a)
                    )
                _scenario.shared_covering_rew[
                    _scenario.shared_covering_rew != 0
                ] /= 2

            # Collision penalty (always computed from original logic)
            agent.collision_rew[:] = 0
            for a in _scenario.world.agents:
                if a != agent:
                    agent.collision_rew[
                        _scenario.world.get_distance(a, agent)
                        < _scenario.min_collision_distance
                    ] += _scenario.agent_collision_penalty

            # Target respawning / moving covered targets out (last agent)
            if is_last:
                if _scenario.targets_respawn:
                    # Use original respawn logic
                    occupied_agents = [_scenario.agents_pos]
                    for i, target in enumerate(_scenario._targets):
                        occupied_targets = [
                            o.state.pos.unsqueeze(1)
                            for o in _scenario._targets
                            if o is not target
                        ]
                        occupied = torch.cat(
                            occupied_agents + occupied_targets, dim=1,
                        )
                        from vmas.simulator.utils import ScenarioUtils
                        pos = ScenarioUtils.find_random_pos_for_entity(
                            occupied, env_index=None,
                            world=_scenario.world,
                            min_dist_between_entities=(
                                _scenario._min_dist_between_entities
                            ),
                            x_bounds=(
                                -_scenario.world.x_semidim,
                                _scenario.world.x_semidim,
                            ),
                            y_bounds=(
                                -_scenario.world.y_semidim,
                                _scenario.world.y_semidim,
                            ),
                        )
                        target.state.pos[
                            _scenario.covered_targets[:, i]
                        ] = pos[
                            _scenario.covered_targets[:, i]
                        ].squeeze(1)
                else:
                    _scenario.all_time_covered_targets += (
                        _scenario.covered_targets
                    )
                    for i, target in enumerate(_scenario._targets):
                        target.state.pos[
                            _scenario.covered_targets[:, i]
                        ] = _scenario.get_outside_pos(None)[
                            _scenario.covered_targets[:, i]
                        ]

            # Call the LLM-generated reward function
            agent_idx = _scenario.world.agents.index(agent)
            state = _build_scenario_state(_scenario, agent, agent_idx)
            return _fn(state)

        scenario.reward = patched_reward
        _log.info("Patched scenario.reward with LLM-generated function")

    if obs_source is not None:
        obs_fn = _compile_function(obs_source, "enhance_observation")
        original_obs = scenario.observation

        def patched_observation(agent, _scenario=scenario,
                                _fn=obs_fn, _orig=original_obs):
            base_obs = _orig(agent)
            agent_idx = _scenario.world.agents.index(agent)
            state = _build_scenario_state(_scenario, agent, agent_idx)
            extra = _fn(state)

            if isinstance(base_obs, dict):
                # Dict observation mode (GNN)
                base_obs["observation"] = torch.cat(
                    [base_obs["observation"], extra], dim=-1,
                )
                return base_obs
            return torch.cat([base_obs, extra], dim=-1)

        scenario.observation = patched_observation
        _log.info("Patched scenario.observation with LLM enhancement")

    return originals


def unpatch_scenario(scenario, originals: Dict[str, Any]):
    """Restore original scenario methods."""
    for name, method in originals.items():
        setattr(scenario, name, method)
    _log.info("Restored original scenario methods")
