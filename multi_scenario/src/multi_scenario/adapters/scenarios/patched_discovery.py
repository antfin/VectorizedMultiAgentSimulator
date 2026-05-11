"""F9.5 — :func:`make_patched_discovery_class` factory.

Builds a subclass of ``vmas.scenarios.discovery.Scenario`` with the
LERO-generated ``compute_reward`` / ``enhance_observation`` functions
spliced in. Returns the **class** (not an instance) because BenchMARL's
env factory needs the class to introspect observation specs at init
time — handing it an instance breaks the TorchRL env builder.

Three modes the factory threads through to the patched class:

- ``reward_mode``: ``"replace"`` (paper default) → R = compute_reward(state)
  with sanitisation / clipping. ``"bonus"`` → R = R_original +
  bonus_scale * tanh(compute_reward_bonus(state)).
- ``obs_state_mode``: ``"global"`` (paper) → the enhance_observation
  function sees the full reward-side state. ``"local"`` (CTDE-fair) →
  only local sensor state (own pos/vel/idx, LiDAR, comms).
- ``whitelist_strict``: when True AND ``obs_state_mode="local"``, wraps
  the state in :class:`AllowedKeysDict`; forbidden lookups raise
  :class:`FairnessViolation`. Off in ``global`` mode (the LLM has
  access by design).

Two regression-blocker comments worth reading before editing the patched
class body:

1. **Closure-bug avoidance** (project memory 2026-04-16): the patched
   methods use *function default arguments* to thread ``mode`` /
   ``clip`` / ``bonus_scale`` into the method body — class-body
   assignments (``_obs_mode = ...``) are not visible inside method
   bodies (Python scoping rule). Tests pin this; don't refactor to
   class attributes.
2. **Per-agent ``info()`` override** (rendezvous_comm §3.3): upstream
   Discovery returns ``self.shared_covering_rew`` for every agent when
   ``shared_reward=True``, which makes M8 (agent-utilization CV)
   structurally zero. The patched class overrides ``info()`` to return
   ``agent.covering_reward`` per-agent so M8 has signal.
"""

import logging

from multi_scenario.adapters.scenarios._lero_patch_helpers import (
    build_obs_state,
    build_reward_state,
    compile_llm_function,
    maybe_wrap_obs_state,
    sanitize_reward,
)


_log = logging.getLogger(__name__)


# The factory legitimately takes 7 kwargs (reward + obs source, two
# modes, bonus scale, clip, strict flag) because each is independently
# tunable. Bundling into a config object would obscure the public API.
# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def make_patched_discovery_class(
    *,
    reward_source: str | None = None,
    obs_source: str | None = None,
    reward_mode: str = "replace",
    obs_state_mode: str = "global",
    bonus_scale: float = 0.5,
    reward_clip: float | None = 50.0,
    whitelist_strict: bool = False,
) -> type:
    """Construct a ``PatchedDiscoveryScenario`` subclass with LLM methods spliced in.

    Args:
        reward_source: validated source for ``compute_reward`` or
            ``compute_reward_bonus``. ``None`` → keep the original
            Discovery reward (no LLM reward).
        obs_source: validated source for ``enhance_observation``.
            ``None`` → no observation enhancement.
        reward_mode: ``"replace"`` | ``"bonus"`` (see module doc).
        obs_state_mode: ``"global"`` | ``"local"``.
        bonus_scale: outer scale on ``tanh(compute_reward_bonus(state))``.
            Only relevant when ``reward_mode == "bonus"``.
        reward_clip: ±N clamp on the (sanitised) reward tensor. ``None``
            disables clipping. Default ``50.0`` matches the rendezvous_comm
            crash-prevention default — set lower for early experiments
            with very unstable LLM reward designs.
        whitelist_strict: when ``True`` AND ``obs_state_mode="local"``,
            wraps the obs state in :class:`AllowedKeysDict` to enforce
            CTDE fairness at runtime.

    Returns:
        A ``type`` subclass of ``vmas.scenarios.discovery.Scenario``
        ready to be handed to BenchMARL's env factory.
    """
    # pylint: disable=import-outside-toplevel
    from vmas.scenarios.discovery import Scenario as OrigScenario

    reward_fn = None
    obs_fn = None
    if reward_source:
        # Replace mode → compute_reward; bonus mode → compute_reward_bonus.
        # The codegen extractor accepts either name; we dispatch here.
        if "def compute_reward_bonus" in reward_source:
            reward_fn = compile_llm_function(reward_source, "compute_reward_bonus")
        else:
            reward_fn = compile_llm_function(reward_source, "compute_reward")
    if obs_source:
        obs_fn = compile_llm_function(obs_source, "enhance_observation")

    class PatchedDiscoveryScenario(OrigScenario):
        """Discovery + LLM-generated reward / observation enhancement."""

        def info(self, agent):
            # rendezvous_comm §3.3 workaround: upstream Discovery returns
            # ``self.shared_covering_rew`` for every agent when
            # shared_reward=True, which makes M8 structurally zero.
            # Override to return per-agent covering_reward so M8 has signal.
            base = super().info(agent)
            base["covering_reward"] = agent.covering_reward
            return base

        if reward_fn is not None:

            if reward_mode == "replace":

                def reward(self, agent, _rc=reward_clip):
                    # Run original reward for its side effects (shared
                    # state, collision penalties, target respawn) but
                    # discard its return value — full replacement.
                    _ = super().reward(agent)
                    agent_idx = self.world.agents.index(agent)
                    state = build_reward_state(self, agent, agent_idx)
                    # Captured ``reward_fn`` from the enclosing factory's
                    # closure — Python scoping makes the kwargs-default
                    # the cleanest way to thread per-instance config in
                    # (see the closure-bug regression test).
                    return sanitize_reward(reward_fn(state), clip=_rc)

            else:  # bonus

                def reward(self, agent, _bs=bonus_scale, _rc=reward_clip):
                    original = super().reward(agent)
                    agent_idx = self.world.agents.index(agent)
                    state = build_reward_state(self, agent, agent_idx)
                    # pylint: disable=import-outside-toplevel
                    import torch

                    raw_bonus = reward_fn(state)
                    bonus = sanitize_reward(_bs * torch.tanh(raw_bonus), clip=_rc)
                    return original + bonus

        if obs_fn is not None:

            def observation(
                self,
                agent,
                _mode=obs_state_mode,
                _strict=whitelist_strict,
            ):
                # pylint: disable=import-outside-toplevel
                import torch

                base_obs = super().observation(agent)
                agent_idx = self.world.agents.index(agent)
                if _mode == "global":
                    state = build_reward_state(self, agent, agent_idx)
                else:
                    state = build_obs_state(self, agent, agent_idx)
                    state = maybe_wrap_obs_state(
                        state, mode=_mode, whitelist_strict=_strict
                    )
                extra = obs_fn(state)
                if isinstance(base_obs, dict):
                    base_obs["observation"] = torch.cat(
                        [base_obs["observation"], extra], dim=-1
                    )
                    return base_obs
                return torch.cat([base_obs, extra], dim=-1)

    _log.info(
        "PatchedDiscoveryScenario built (reward=%s, obs=%s, reward_mode=%s, "
        "obs_state_mode=%s, whitelist_strict=%s)",
        reward_source is not None,
        obs_source is not None,
        reward_mode,
        obs_state_mode,
        whitelist_strict,
    )
    return PatchedDiscoveryScenario
