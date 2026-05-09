"""VMAS navigation scenario adapter — implements the Scenario port.

M1: binary "all agents reached their goals during the episode" via the
universal ``episode_terminated`` rollout key (F4.1 base extension). Mirrors
discovery's all-or-nothing success semantics.

M6 (coverage_progress) inherits the base ``None`` stub; sharpening it
(e.g. per-agent on-goal fraction at episode end) needs per-agent
position/goal extraction into the rollout dict — out of scope for F4.1.
"""

from typing import Any

from multi_scenario.adapters.scenarios.base import VmasScenarioBase


class VmasNavigationAdapter(VmasScenarioBase):
    """Scenario adapter wrapping ``vmas.make_env(scenario='navigation', ...)``."""

    name = "navigation"

    def default_params(self) -> dict[str, Any]:
        """Baseline navigation params — small env, partial-observation goals."""
        return {
            "n_agents": 4,
            "agents_with_same_goal": 1,
            "observe_all_goals": False,
            "shared_rew": True,
            "max_steps": 100,
        }

    def success_predicate(self, rollout: Any) -> Any:
        """M1 = "env terminated naturally" = all agents reached their goals."""
        return self._terminated_based_success(rollout)
