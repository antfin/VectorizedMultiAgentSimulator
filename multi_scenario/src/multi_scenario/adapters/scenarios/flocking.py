"""VMAS flocking scenario adapter — implements the Scenario port.

Flocking is continuous-control optimisation toward ``desired_distance`` +
matched velocity; there is no natural binary success metric. M1 inherits
the base ``None`` stub — M2 / M4 carry the evaluation weight. A sharper
M1 (e.g. "fraction of timesteps in flocking-acceptable state") would need
per-step per-agent position/velocity extraction; deferred until needed.
"""

from typing import Any

from multi_scenario.adapters.scenarios.base import VmasScenarioBase


class VmasFlockingAdapter(VmasScenarioBase):
    """Scenario adapter wrapping ``vmas.make_env(scenario='flocking', ...)``."""

    name = "flocking"

    def default_params(self) -> dict[str, Any]:
        """Baseline flocking params — small env with default desired distance."""
        return {
            "n_agents": 4,
            "collision_reward": -0.1,
            "dist_shaping_factor": 1,
        }
