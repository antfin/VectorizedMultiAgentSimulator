"""Monkey-patch Discovery scenario with LLM-generated functions.

VMAS and TorchRL's VmasEnv accept a BaseScenario INSTANCE (not just
a string name). We create a patched Scenario subclass with the
LLM-generated reward/observation, then pass it to BenchMARL's env
factory. This way the patched methods are visible when TorchRL
computes the observation spec during env.__init__.
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
    """Build the scenario_state dict passed to LLM-generated functions."""
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


def make_patched_scenario_class(
    reward_source: Optional[str] = None,
    obs_source: Optional[str] = None,
):
    """Create a Discovery Scenario subclass with LLM-generated methods.

    Returns a NEW class (not instance) that can be instantiated by
    VMAS/TorchRL. The subclass overrides reward() and/or observation()
    with the LLM-generated functions.

    This is the correct approach because:
    - VMAS dynamically loads scenarios, so class-level patching on the
      imported Scenario class doesn't propagate.
    - TorchRL computes observation_spec during env.__init__, so
      instance-level patching after creation is too late.
    - A subclass carries the patched methods from the start.
    """
    from vmas.scenarios.discovery import Scenario as OrigScenario

    reward_fn = None
    obs_fn = None
    if reward_source:
        reward_fn = _compile_function(reward_source, "compute_reward")
    if obs_source:
        obs_fn = _compile_function(obs_source, "enhance_observation")

    class PatchedDiscoveryScenario(OrigScenario):
        """Discovery scenario with LLM-generated reward/observation."""

        if reward_fn is not None:
            def reward(self, agent):
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
                                occupied_agents + occupied_targets,
                                dim=1,
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
                        self.all_time_covered_targets += (
                            self.covered_targets
                        )
                        for i, target in enumerate(self._targets):
                            target.state.pos[
                                self.covered_targets[:, i]
                            ] = self.get_outside_pos(None)[
                                self.covered_targets[:, i]
                            ]

                agent_idx = self.world.agents.index(agent)
                state = _build_scenario_state(self, agent, agent_idx)
                return reward_fn(state)

        if obs_fn is not None:
            def observation(self, agent):
                base_obs = super().observation(agent)
                agent_idx = self.world.agents.index(agent)
                state = _build_scenario_state(self, agent, agent_idx)
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
