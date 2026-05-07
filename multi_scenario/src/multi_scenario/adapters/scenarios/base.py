"""Shared scaffolding for VMAS-backed scenario adapters.

Concrete subclasses set ``name`` (which doubles as the VMAS scenario name)
and override ``default_params`` plus whichever DI primitives the scenario
naturally produces (``success_predicate``, ``coverage_progress``,
``has_comm``, ``utilization_predicate``). Everything else inherits the
``None`` / ``False`` defaults so adapters stay declarative.

The ``_terminated_based_success`` helper covers the common pattern where
M1 = "env terminated naturally" (navigation, transport, future scenarios
with binary done-on-success semantics).
"""

from typing import Any

import torch
import vmas

from multi_scenario.domain.models import ScenarioSection


class VmasScenarioBase:
    """Base class for VMAS-backed scenario adapters."""

    # Subclasses override `name` and the DI primitives they naturally produce;
    # the unused-args disable matches the Scenario Protocol surface.
    # pylint: disable=unused-argument

    name: str = "base"

    def make_env(self, cfg: ScenarioSection, num_envs: int, seed: int) -> Any:
        """Build the VMAS env using ``self.name`` and merged params."""
        params = {**self.default_params(), **cfg.params}
        return vmas.make_env(
            scenario=self.name,
            num_envs=num_envs,
            seed=seed,
            **params,
        )

    def default_params(self) -> dict[str, Any]:
        """Baseline scenario kwargs — subclass overrides as needed."""
        return {}

    def has_comm(self) -> bool:
        """No comm channel by default; M5 is N/A."""
        return False

    def success_predicate(self, rollout: Any) -> Any:
        """M1 default: None (no scenario-natural binary success metric)."""
        return None

    def coverage_progress(self, rollout: Any) -> Any | None:
        """M6 default: None — scenario-specific path lands in subclasses."""
        return None

    def utilization_predicate(self, state: Any) -> Any:
        """M8 default: None — stubbed, lands in a later feature."""
        return None

    @staticmethod
    def _terminated_based_success(rollout: Any) -> Any:
        """Helper: read the universal ``episode_terminated`` rollout key.

        Used by scenarios where M1 = "env terminated naturally during the
        episode" (navigation: all agents on goal; transport: all packages
        delivered). Returns None if the rollout lacks the key.
        """
        terminated = rollout.get("episode_terminated")
        if terminated is None:
            return None
        return torch.as_tensor(terminated).bool()
