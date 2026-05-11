"""F9.5 — torch-using helpers for the patched scenario class.

Split from :mod:`adapters.scenarios.patched_discovery` so the state
builders and reward sanitiser are independently testable without
spinning up a full VMAS world. The fairness / whitelist enforcement
lives in :mod:`multi_scenario.domain.lero.whitelist` (pure-Python) —
this module strictly handles torch-tensor construction + cleanup.

Imports a leaf VMAS scenario only inside :func:`make_patched_discovery_class`
(lazy) so importing this module doesn't transitively load all of VMAS.
"""

from typing import Any, Callable

import torch
import torch.nn.functional as F  # noqa: N812 — Torch convention

from multi_scenario.domain.lero import AllowedKeysDict, FairnessViolation


#: Globals available to LLM-generated functions when they're ``exec``'d.
#: Locked to torch + math + nn.functional + numpy aliases — matches the
#: ALLOWED_IMPORTS whitelist enforced by :func:`validate_function`. Any
#: additional name would let the LLM bypass codegen validation by
#: lookups against this dict, so keep it minimal.
EXEC_NAMESPACE: dict[str, Any] = {
    "torch": torch,
    "math": __import__("math"),
    "F": F,
}


def build_reward_state(scenario: Any, agent: Any, agent_idx: int) -> dict[str, Any]:
    """FULL global scenario state for centralised-training reward functions.

    Reward functions see everything (CTDE: Centralised Training,
    Decentralised Execution). ``compute_reward`` always gets this; only
    ``enhance_observation`` is gated by the observation mode.
    """
    agents_pos = getattr(scenario, "agents_pos", None)
    if agents_pos is None:
        agents_pos = torch.stack([a.state.pos for a in scenario.world.agents], dim=1)
    targets_pos = getattr(scenario, "targets_pos", None)
    if targets_pos is None:
        # pylint: disable=protected-access
        targets_pos = torch.stack([t.state.pos for t in scenario._targets], dim=1)
    agents_targets_dists = getattr(scenario, "agents_targets_dists", None)
    if agents_targets_dists is None:
        agents_targets_dists = torch.cdist(agents_pos, targets_pos)
    agents_per_target = getattr(scenario, "agents_per_target", None)
    if agents_per_target is None:
        # pylint: disable=protected-access
        agents_per_target = torch.sum(
            (agents_targets_dists < scenario._covering_range).int(), dim=1
        )
    covered_targets = getattr(scenario, "covered_targets", None)
    if covered_targets is None:
        # pylint: disable=protected-access
        covered_targets = agents_per_target >= scenario._agents_per_target

    batch_dim = scenario.world.batch_dim
    device = scenario.world.device

    state: dict[str, Any] = {
        "agents_pos": agents_pos,
        "targets_pos": targets_pos,
        "agents_targets_dists": agents_targets_dists,
        "covered_targets": covered_targets,
        "agents_per_target": agents_per_target,
        "all_time_covered": getattr(
            scenario,
            "all_time_covered_targets",
            torch.zeros_like(covered_targets),
        ),
        "agent_pos": agent.state.pos,
        "agent_vel": agent.state.vel,
        "agent_idx": agent_idx,
        "n_agents": torch.tensor(len(scenario.world.agents), device=device).long(),
        "n_targets": torch.tensor(scenario.n_targets, device=device).long(),
        # pylint: disable=protected-access
        "covering_range": torch.tensor(scenario._covering_range, device=device).float(),
        # pylint: disable=protected-access
        "agents_per_target_required": torch.tensor(
            scenario._agents_per_target, device=device
        ).float(),
        "collision_penalty": torch.tensor(
            scenario.agent_collision_penalty, device=device
        ).float(),
        "collision_rew": getattr(
            agent, "collision_rew", torch.zeros(batch_dim, device=device)
        ),
        "time_penalty": torch.tensor(scenario.time_penalty, device=device).float(),
    }
    dim_c = getattr(scenario, "dim_c", 0)
    if dim_c > 0:
        messages = [
            other.state.c
            for other in scenario.world.agents
            if other is not agent and other.state.c is not None
        ]
        if messages:
            state["messages"] = torch.stack(messages, dim=1)
    return state


def build_obs_state(scenario: Any, agent: Any, agent_idx: int) -> dict[str, Any]:
    """LOCAL sensor state for the decentralised-execution observation enhancer.

    Includes only what an agent could observe through its own sensors:
    own pos/vel/idx, LiDAR rays, incoming comm messages. Static scenario
    constants (n_agents, n_targets, covering_range, …) are included
    because they're identical for every agent — knowing them is not an
    oracle leak. The CTDE-fairness check in :class:`AllowedKeysDict`
    enforces this contract at runtime.
    """
    device = scenario.world.device

    state: dict[str, Any] = {
        "agent_pos": agent.state.pos,
        "agent_vel": agent.state.vel,
        "agent_idx": agent_idx,
        "n_agents": torch.tensor(len(scenario.world.agents), device=device).long(),
        "n_targets": torch.tensor(scenario.n_targets, device=device).long(),
        # pylint: disable=protected-access
        "covering_range": torch.tensor(scenario._covering_range, device=device).float(),
        # pylint: disable=protected-access
        "agents_per_target_required": torch.tensor(
            scenario._agents_per_target, device=device
        ).float(),
        "lidar_targets": agent.sensors[0].measure(),
    }
    if getattr(scenario, "use_agent_lidar", False) and len(agent.sensors) > 1:
        state["lidar_agents"] = agent.sensors[1].measure()

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
                        agent.state.pos - other.state.pos, dim=-1, keepdim=True
                    )
                    msg = msg * (dist <= comms_range).float()
                messages.append(msg)
        if messages:
            state["messages"] = torch.stack(messages, dim=1)
    return state


def compile_llm_function(source: str, func_name: str) -> Callable:
    """``exec`` LLM-validated source and pluck the requested function.

    Called only AFTER :func:`validate_function` has AST-validated the
    source — this is the trust boundary. Validation pinned in
    :mod:`multi_scenario.domain.lero.codegen`.
    """
    namespace = dict(EXEC_NAMESPACE)
    # The whole point: bring an LLM-validated function into scope.
    # pylint: disable=exec-used
    exec(source, namespace)  # noqa: S102
    if func_name not in namespace:
        raise ValueError(
            f"function {func_name!r} not found after exec; "
            f"available: {[k for k in namespace if not k.startswith('_')]}"
        )
    return namespace[func_name]


def maybe_wrap_obs_state(
    state: dict[str, Any],
    *,
    mode: str,
    whitelist_strict: bool,
) -> Any:
    """In ``local`` + strict mode, wrap ``state`` in :class:`AllowedKeysDict`.

    Pure dispatch — extracted so the wrap decision is unit-testable
    without a VMAS scenario.
    """
    if mode != "local" or not whitelist_strict:
        return state
    return AllowedKeysDict(state)


def sanitize_reward(r: torch.Tensor, *, clip: float | None) -> torch.Tensor:
    """Two-stage defensive cleanup of LLM-generated reward output.

    1. ``nan_to_num``: replace NaN/inf with 0 (LLM divisions by zero,
       ``log(0)`` on degenerate states).
    2. ``clamp`` to ``[-clip, +clip]`` when ``clip`` is set, otherwise
       skipped. Default ``50.0`` — rendezvous_comm observed PPO NaN
       crashes 70-90% into 10M-frame training when chosen reward had
       |M2| > 100; clipping stops the gradient explosion at the source.
    """
    r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    if clip is not None and clip > 0:
        r = torch.clamp(r, -clip, clip)
    return r


# Re-export for the patched-discovery factory + tests.
__all__ = [
    "EXEC_NAMESPACE",
    "AllowedKeysDict",  # re-exported from domain.lero for caller convenience
    "FairnessViolation",
    "build_obs_state",
    "build_reward_state",
    "compile_llm_function",
    "maybe_wrap_obs_state",
    "sanitize_reward",
]
