"""VMAS transport scenario adapter — implements the Scenario port.

M1: binary "all packages delivered to their goals during the episode" via
the universal ``episode_terminated`` rollout key (transport's ``done()``
fires when all packages are on goal). Same template as navigation.

M6 inherits the base ``None`` stub; sharpening it (e.g. fraction of
packages on goal at episode end) needs per-package position/goal
extraction into the rollout dict — out of scope for F4.3.
"""

from typing import Any

from multi_scenario.adapters.scenarios.base import VmasScenarioBase


class VmasTransportAdapter(VmasScenarioBase):
    """Scenario adapter wrapping ``vmas.make_env(scenario='transport', ...)``."""

    name = "transport"

    def default_params(self) -> dict[str, Any]:
        """Baseline transport params — single heavy package needs cooperative push."""
        return {
            "n_agents": 4,
            "n_packages": 1,
            "package_mass": 50,
        }

    def success_predicate(self, rollout: Any) -> Any:
        """M1 = "env terminated naturally" = all packages delivered to goals."""
        return self._terminated_based_success(rollout)
