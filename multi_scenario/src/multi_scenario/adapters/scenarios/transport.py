"""VMAS transport scenario adapter — implements the Scenario port.

Success (M1) is the binary "all packages delivered to their goals during the
episode" — read from BenchMARL's universal ``episode_terminated`` rollout
key (transport's ``done()`` fires when all packages are on goal). Same
template as navigation.

M6 coverage progress is stubbed for now; sharpening it (e.g. fraction of
packages on goal at episode end) needs per-package position/goal extraction
into the rollout dict — out of scope for F4.3.
"""

from typing import Any

import torch
import vmas

from multi_scenario.domain.models import ScenarioSection


class VmasTransportAdapter:
    """Scenario adapter wrapping ``vmas.make_env(scenario='transport', ...)``."""

    # M6/M8 stub-return None for F4.3; the unused args are required by the
    # Scenario Protocol.
    # pylint: disable=unused-argument

    name = "transport"

    def make_env(self, cfg: ScenarioSection, num_envs: int, seed: int) -> Any:
        """Build the VMAS transport env; cfg.params override default_params."""
        params = {**self.default_params(), **cfg.params}
        return vmas.make_env(
            scenario="transport",
            num_envs=num_envs,
            seed=seed,
            **params,
        )

    def default_params(self) -> dict[str, Any]:
        """Baseline transport params — single heavy package needs cooperative push."""
        return {
            "n_agents": 4,
            "n_packages": 1,
            "package_mass": 50,
        }

    def has_comm(self) -> bool:
        """Transport has no comm channel; M5 is N/A."""
        return False

    def success_predicate(self, rollout: Any) -> Any:
        """M1: True per episode if env terminated naturally (= all packages on goal)."""
        terminated = rollout.get("episode_terminated")
        if terminated is None:
            return None
        return torch.as_tensor(terminated).bool()

    def coverage_progress(self, rollout: Any) -> Any | None:
        """M6: stub — sharper transport coverage lands in a later feature."""
        return None

    def utilization_predicate(self, state: Any) -> Any:
        """M8: per-(env, agent) utilization — stub, lands in a later feature."""
        return None
