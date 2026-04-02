"""Monkey-patch Discovery scenario with LLM-generated functions.

KEY DESIGN: We patch at the Scenario CLASS level, not instance level.
This is necessary because BenchMARL probes the env to determine the
observation space size when building the experiment. If we patch after
build_experiment(), the MLP input dimension won't match the patched
observation size.

Usage:
    originals = patch_scenario_class(reward_source, obs_source)
    experiment = build_experiment(...)  # sees patched obs size
    # ... train ...
    unpatch_scenario_class(originals)
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


def _build_scenario_state(scenario, agent, agent_idx: int) -> Dict[str, Any]:
    """Build the scenario_state dict passed to LLM-generated functions.

    Some fields (agents_pos, targets_pos, etc.) are only computed during
    reward(). When observation() is called during env.reset() or before
    the first reward(), we compute them on-the-fly.
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
        "all_time_covered": scenario.all_time_covered_targets,
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
    if scenario.dim_c > 0:
        messages = []
        for other in scenario.world.agents:
            if other is not agent and other.state.c is not None:
                messages.append(other.state.c)
        if messages:
            state["messages"] = torch.stack(messages, dim=1)
        else:
            state["messages"] = torch.zeros(
                batch_dim, len(scenario.world.agents) - 1,
                scenario.dim_c, device=device,
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


def patch_scenario_class(
    reward_source: Optional[str] = None,
    obs_source: Optional[str] = None,
) -> Dict[str, Any]:
    """Patch the Discovery Scenario CLASS with LLM-generated functions.

    Must be called BEFORE build_experiment() so BenchMARL sees the
    correct observation size when probing the env.

    Returns:
        Dict with original class methods for restoration.
    """
    from vmas.scenarios.discovery import Scenario

    originals = {
        "reward": Scenario.reward,
        "observation": Scenario.observation,
    }

    if reward_source is not None:
        reward_fn = _compile_function(reward_source, "compute_reward")
        _original_reward = Scenario.reward

        def patched_reward(self, agent, _fn=reward_fn,
                           _orig=_original_reward):
            is_first = agent == self.world.agents[0]
            is_last = agent == self.world.agents[-1]

            if is_first:
                self.time_rew = torch.full(
                    (self.world.batch_dim,),
                    self.time_penalty,
                    device=self.world.device,
                )
                self.agents_pos = torch.stack(
                    [a.state.pos for a in self.world.agents], dim=1,
                )
                self.targets_pos = torch.stack(
                    [t.state.pos for t in self._targets], dim=1,
                )
                self.agents_targets_dists = torch.cdist(
                    self.agents_pos, self.targets_pos,
                )
                self.agents_per_target = torch.sum(
                    (self.agents_targets_dists
                     < self._covering_range).int(),
                    dim=1,
                )
                self.covered_targets = (
                    self.agents_per_target >= self._agents_per_target
                )
                self.shared_covering_rew[:] = 0
                for a in self.world.agents:
                    self.shared_covering_rew += self.agent_reward(a)
                self.shared_covering_rew[
                    self.shared_covering_rew != 0
                ] /= 2

            # Collision penalty
            agent.collision_rew[:] = 0
            for a in self.world.agents:
                if a != agent:
                    agent.collision_rew[
                        self.world.get_distance(a, agent)
                        < self.min_collision_distance
                    ] += self.agent_collision_penalty

            # Target handling (last agent)
            if is_last:
                if self.targets_respawn:
                    from vmas.simulator.utils import ScenarioUtils
                    occupied_agents = [self.agents_pos]
                    for i, target in enumerate(self._targets):
                        occupied_targets = [
                            o.state.pos.unsqueeze(1)
                            for o in self._targets
                            if o is not target
                        ]
                        occupied = torch.cat(
                            occupied_agents + occupied_targets, dim=1,
                        )
                        pos = ScenarioUtils.find_random_pos_for_entity(
                            occupied, env_index=None,
                            world=self.world,
                            min_dist_between_entities=(
                                self._min_dist_between_entities
                            ),
                            x_bounds=(
                                -self.world.x_semidim,
                                self.world.x_semidim,
                            ),
                            y_bounds=(
                                -self.world.y_semidim,
                                self.world.y_semidim,
                            ),
                        )
                        target.state.pos[
                            self.covered_targets[:, i]
                        ] = pos[
                            self.covered_targets[:, i]
                        ].squeeze(1)
                else:
                    self.all_time_covered_targets += self.covered_targets
                    for i, target in enumerate(self._targets):
                        target.state.pos[
                            self.covered_targets[:, i]
                        ] = self.get_outside_pos(None)[
                            self.covered_targets[:, i]
                        ]

            agent_idx = self.world.agents.index(agent)
            state = _build_scenario_state(self, agent, agent_idx)
            return _fn(state)

        Scenario.reward = patched_reward
        _log.info("Patched Scenario.reward (class-level)")

    if obs_source is not None:
        obs_fn = _compile_function(obs_source, "enhance_observation")
        _original_obs = Scenario.observation

        def patched_observation(self, agent, _fn=obs_fn,
                                _orig=_original_obs):
            base_obs = _orig(self, agent)
            agent_idx = self.world.agents.index(agent)
            state = _build_scenario_state(self, agent, agent_idx)
            extra = _fn(state)

            if isinstance(base_obs, dict):
                base_obs["observation"] = torch.cat(
                    [base_obs["observation"], extra], dim=-1,
                )
                return base_obs
            return torch.cat([base_obs, extra], dim=-1)

        Scenario.observation = patched_observation
        _log.info("Patched Scenario.observation (class-level)")

    return originals


def unpatch_scenario_class(originals: Dict[str, Any]):
    """Restore original Scenario class methods."""
    from vmas.scenarios.discovery import Scenario

    for name, method in originals.items():
        setattr(Scenario, name, method)
    _log.info("Restored original Scenario class methods")


# ── Keep old names as aliases for backward compat in tests ───────
patch_scenario = patch_scenario_class
unpatch_scenario = unpatch_scenario_class
