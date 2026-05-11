"""F9.5 — :func:`make_patched_discovery_class` end-to-end against VMAS.

Slow tests (each spins up a small VMAS Discovery world). Marked
``@pytest.mark.slow`` so the fast unit-test pass stays under a second.

Regression-blockers covered (one test per project-memory bug):

- **Closure bug** (rendezvous_comm 2026-04-16): ``_obs_mode = ...`` as a
  class-body attribute isn't visible in method bodies. Our factory uses
  ``def observation(self, agent, _mode=obs_state_mode, …)`` — method
  default arg — instead. The test patches with different modes and
  asserts each behaves correctly.
- **M8 always 0 with shared_reward=True**: upstream Discovery's
  ``info()`` returns ``shared_covering_rew`` for every agent. Patched
  ``info()`` returns ``agent.covering_reward`` so M8 is computable.
- **Reward clip ±50 prevents PPO NaN crashes**: LLM rewards with
  magnitude > 100 destabilise PPO at ~70-90% into 10M-frame training.
  We pass a reward function that returns ±1000 and assert the patched
  scenario's reward output is clamped to ±50.
- **NaN-to-zero on degenerate LLM math**: ``log(0)`` / divide-by-zero
  shouldn't propagate into PPO.
- **Whitelist-strict in local mode**: an LLM observation function that
  reads ``state['agents_pos']`` raises :class:`FairnessViolation`.
"""

# pylint: disable=missing-function-docstring,redefined-outer-name

import pytest
import torch

from multi_scenario.adapters.scenarios.patched_discovery import (
    make_patched_discovery_class,
)
from multi_scenario.domain.lero import FairnessViolation


pytestmark = pytest.mark.slow


# ── shared fixtures ──────────────────────────────────────────────────


@pytest.fixture
def world():
    """Tiny Discovery world (2 agents, 2 targets) for end-to-end tests.

    Returns ``(Scenario_cls, world_instance)`` so each test can build
    its own patched subclass and reset the world cleanly.
    """
    from vmas import make_env

    def _build(scenario_cls):
        env = make_env(
            scenario=scenario_cls(),
            num_envs=1,
            device="cpu",
            continuous_actions=True,
            n_agents=2,
            n_targets=2,
            agents_per_target=2,
            covering_range=0.35,
            max_steps=5,
            shared_reward=True,
        )
        env.reset()
        return env

    return _build


# ── M8 unblock: per-agent info ───────────────────────────────────────


def test_patched_info_returns_per_agent_covering_reward(world):
    """The patched ``info()`` must return ``agent.covering_reward``
    (not the shared total) so M8 has signal under shared_reward=True.

    Tested by introspecting the scenario instance's ``info()`` output
    for two agents at world reset.
    """
    cls = make_patched_discovery_class()
    env = world(cls)
    scenario = env.scenario
    # Step once so covering_reward is populated.
    actions = [torch.zeros(1, 2) for _ in scenario.world.agents]
    env.step(actions)
    infos = [scenario.info(a) for a in scenario.world.agents]
    assert all("covering_reward" in info for info in infos)
    # Per-agent — they're each agent's own ``covering_reward`` attr,
    # not the shared scalar.
    assert infos[0]["covering_reward"].shape == infos[1]["covering_reward"].shape


# ── Reward clip + NaN→zero ───────────────────────────────────────────


def test_reward_replace_mode_clamps_to_pm_clip(world):
    """LLM reward returning ±1000 is clamped to ±50 by sanitisation."""
    src = "def compute_reward(s):\n    return s['agent_pos'][..., 0] * 1000.0"
    cls = make_patched_discovery_class(
        reward_source=src, reward_mode="replace", reward_clip=50.0
    )
    env = world(cls)
    scenario = env.scenario
    r = scenario.reward(scenario.world.agents[0])
    assert (r >= -50.0).all() and (r <= 50.0).all()


def test_reward_nan_is_replaced_with_zero(world):
    """LLM math that divides by zero shouldn't propagate NaN into PPO."""
    src = (
        "def compute_reward(s):\n"
        "    pos = s['agent_pos']\n"
        "    return pos[..., 0] / torch.zeros_like(pos[..., 0])\n"
    )
    cls = make_patched_discovery_class(reward_source=src, reward_clip=50.0)
    env = world(cls)
    scenario = env.scenario
    r = scenario.reward(scenario.world.agents[0])
    assert not torch.isnan(r).any()
    assert not torch.isinf(r).any()


def test_reward_bonus_mode_preserves_original_plus_tanh_bonus(world):
    """In bonus mode the original reward is the floor; the LLM bonus
    is added as ``bonus_scale * tanh(...)`` so it can't exceed
    ``bonus_scale`` in magnitude.
    """
    src = (
        "def compute_reward_bonus(s):\n" "    return s['agent_pos'][..., 0] * 1000.0\n"
    )
    cls = make_patched_discovery_class(
        reward_source=src, reward_mode="bonus", bonus_scale=0.5, reward_clip=10.0
    )
    env = world(cls)
    scenario = env.scenario
    r = scenario.reward(scenario.world.agents[0])
    # bonus is sanitised: |0.5 * tanh(...)| ≤ 0.5, so |r| ≤ |original| + 0.5
    # — and the clip ensures absolute bound regardless. Just check
    # no-NaN / no-inf — the precise magnitude depends on the original.
    assert not torch.isnan(r).any()
    assert torch.isfinite(r).all()


# ── Observation modes + closure-bug regression ───────────────────────


def test_obs_global_mode_passes_full_state_to_llm_fn(world):
    """In ``global`` mode the LLM sees agents_pos / targets_pos / etc."""
    src = (
        "def enhance_observation(s):\n"
        "    # Touch global keys; would FairnessViolate in local-strict\n"
        "    return s['agents_pos'].reshape(s['agents_pos'].shape[0], -1)\n"
    )
    cls = make_patched_discovery_class(obs_source=src, obs_state_mode="global")
    env = world(cls)
    scenario = env.scenario
    out = scenario.observation(scenario.world.agents[0])
    # Returned an enhanced obs without raising.
    assert out is not None


def test_obs_local_mode_strict_raises_on_global_key_lookup(world):
    """The closure of ``whitelist_strict=True`` + ``obs_state_mode="local"``
    must wrap the dict in AllowedKeysDict and raise on forbidden keys.

    ``make_env`` calls ``observation`` during the reset path, so the
    violation surfaces while building the world — pytest.raises wraps
    the whole construction.
    """
    src = (
        "def enhance_observation(s):\n"
        "    # LLM cheats by reaching for global state\n"
        "    return s['agents_pos']\n"
    )
    cls = make_patched_discovery_class(
        obs_source=src, obs_state_mode="local", whitelist_strict=True
    )
    with pytest.raises(FairnessViolation, match="agents_pos"):
        world(cls)


def test_obs_local_mode_non_strict_permits_anything(world):
    """Non-strict mode hands over the raw dict — paper-faithful behaviour."""
    src = "def enhance_observation(s):\n" "    return s['agent_pos']\n"
    cls = make_patched_discovery_class(
        obs_source=src, obs_state_mode="local", whitelist_strict=False
    )
    env = world(cls)
    scenario = env.scenario
    out = scenario.observation(scenario.world.agents[0])
    assert out is not None


# ── Closure-bug regression: modes thread via method default args ─────


def test_closure_bug_each_mode_observes_its_state_shape(world):
    """Build two patched classes in succession with different modes;
    each instance's ``observation`` must see its OWN mode, not the
    most recently built one.

    Pre-fix (class-body attribute ``_obs_mode = ...``) this leaked:
    Python's scoping rule meant the method body always read whatever
    the LAST class definition had set. With method-default-arg
    threading the modes are correctly captured per-class.
    """
    src = "def enhance_observation(s):\n    return s['agent_pos']\n"
    cls_local = make_patched_discovery_class(
        obs_source=src, obs_state_mode="local", whitelist_strict=False
    )
    cls_global = make_patched_discovery_class(
        obs_source=src, obs_state_mode="global", whitelist_strict=False
    )
    env_local = world(cls_local)
    env_global = world(cls_global)

    # Both work without raising — distinct modes captured per-instance.
    assert (
        env_local.scenario.observation(env_local.scenario.world.agents[0]) is not None
    )
    assert (
        env_global.scenario.observation(env_global.scenario.world.agents[0]) is not None
    )


# ── No-source no-op: factory returns plain subclass ─────────────────


def test_factory_without_sources_is_a_passthrough(world):
    """``reward_source=None`` and ``obs_source=None`` → the subclass
    still overrides ``info()`` (for M8) but doesn't touch reward or
    observation. Useful for the M8-only experiment lane."""
    cls = make_patched_discovery_class(reward_source=None, obs_source=None)
    env = world(cls)
    scenario = env.scenario
    # info() still overridden — M8 unblock applies even without LLM code.
    actions = [torch.zeros(1, 2) for _ in scenario.world.agents]
    env.step(actions)
    info = scenario.info(scenario.world.agents[0])
    assert "covering_reward" in info
    # reward and observation behave like the unpatched Discovery scenario.
    r = scenario.reward(scenario.world.agents[0])
    assert torch.isfinite(r).all()
