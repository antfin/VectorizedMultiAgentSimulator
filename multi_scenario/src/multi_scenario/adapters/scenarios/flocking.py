"""VMAS flocking scenario adapter — implements the Scenario port.

Flocking is continuous-control optimisation toward ``desired_distance`` +
matched velocity; there is no natural binary success metric. M1 returns
None — M2 (avg_return) and M4 (collisions) carry the evaluation weight.

A sharper M1 (e.g. "fraction of timesteps in flocking-acceptable state")
would require per-step per-agent position/velocity extraction into the
rollout dict; deferred until needed.
"""

from typing import Any

import vmas

from multi_scenario.domain.models import ScenarioSection


class VmasFlockingAdapter:
    """Scenario adapter wrapping ``vmas.make_env(scenario='flocking', ...)``."""

    # All three predicate methods return None until/unless a sharper metric
    # lands; the unused args are required by the Scenario Protocol.
    # pylint: disable=unused-argument

    name = "flocking"

    def make_env(self, cfg: ScenarioSection, num_envs: int, seed: int) -> Any:
        """Build the VMAS flocking env; cfg.params override default_params."""
        params = {**self.default_params(), **cfg.params}
        return vmas.make_env(
            scenario="flocking",
            num_envs=num_envs,
            seed=seed,
            **params,
        )

    def default_params(self) -> dict[str, Any]:
        """Baseline flocking params — small env with default desired distance."""
        return {
            "n_agents": 4,
            "collision_reward": -0.1,
            "dist_shaping_factor": 1,
        }

    def has_comm(self) -> bool:
        """Flocking has no comm channel; M5 is N/A."""
        return False

    def success_predicate(self, rollout: Any) -> Any:
        """M1: flocking has no natural binary success metric — returns None."""
        return None

    def coverage_progress(self, rollout: Any) -> Any | None:
        """M6: not meaningful for flocking — returns None."""
        return None

    def utilization_predicate(self, state: Any) -> Any:
        """M8: per-(env, agent) utilization — stub, lands in a later feature."""
        return None
