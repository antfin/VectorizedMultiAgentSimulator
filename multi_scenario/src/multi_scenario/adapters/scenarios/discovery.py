"""VMAS discovery scenario adapter — implements the Scenario port.

The four DI metric primitives (``success_predicate``, ``coverage_progress``,
``utilization_predicate``) are stubbed as ``None``-returners here; F2.3
fills them in once the metrics bundle lands. ``has_comm`` defaults False
(no comm channel); flip via ``cfg.params['dim_c']`` if/when comm scenarios
are added.
"""

from typing import Any

import vmas

from multi_scenario.domain.models import ScenarioSection


class VmasDiscoveryAdapter:
    """Scenario adapter wrapping ``vmas.make_env(scenario='discovery', ...)``."""

    # The three predicate methods stub-return None until F2.3 wires them up;
    # the unused args are required by the Scenario Protocol. Drop this disable
    # when F2.3 implements them.
    # pylint: disable=unused-argument

    name = "discovery"

    def make_env(self, cfg: ScenarioSection, num_envs: int, seed: int) -> Any:
        """Build the VMAS discovery env; cfg.params override default_params."""
        params = {**self.default_params(), **cfg.params}
        return vmas.make_env(
            scenario="discovery",
            num_envs=num_envs,
            seed=seed,
            **params,
        )

    def default_params(self) -> dict[str, Any]:
        """Baseline discovery params — ``targets_respawn=False`` is required for M1/M3."""
        return {
            "n_agents": 5,
            "n_targets": 7,
            "agents_per_target": 2,
            "covering_range": 0.25,
            "targets_respawn": False,
            "shared_reward": True,
        }

    def has_comm(self) -> bool:
        """Discovery has no comm channel by default; M5 is N/A unless dim_c > 0."""
        return False

    def success_predicate(self, rollout: Any) -> Any:
        """Per-episode success Boolean — wired up at F2.3."""
        return None

    def coverage_progress(self, rollout: Any) -> Any | None:
        """Per-episode coverage scalar — wired up at F2.3."""
        return None

    def utilization_predicate(self, state: Any) -> Any:
        """Per-(env, agent) utilization Boolean — wired up at F2.3."""
        return None
