"""VMAS discovery scenario adapter — implements the Scenario port.

Discovery has bespoke M1/M6 paths: ``success_predicate`` derives from the
``targets_covered`` cumsum max (project memory: NOT from the ``terminated``
signal — documented bug from rendezvous_comm); ``coverage_progress``
returns the max-fraction-ever-covered per episode.

Rollout-shape contract (extends F2.2's universal contract):

- ``rollout["targets_covered"]``: ``Tensor[n_episodes, T]`` — per-step covered count.
- ``rollout["n_targets"]``: ``int`` — used as the success threshold (M1)
  and as the normaliser for coverage progress (M6).

The BenchMARL adapter (F2.4) populates these from VMAS info dicts.
"""

from typing import Any

import torch

from multi_scenario.adapters.scenarios.base import VmasScenarioBase


class VmasDiscoveryAdapter(VmasScenarioBase):
    """Scenario adapter wrapping ``vmas.make_env(scenario='discovery', ...)``."""

    name = "discovery"

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

    def success_predicate(self, rollout: Any) -> Any:
        """M1: True per episode if all targets were covered at any step.

        Project memory: derived from ``targets_covered`` cumsum max — *not*
        from the ``terminated`` signal (documented bug from rendezvous_comm).
        Returns None when the rollout lacks the required keys.
        """
        targets_covered = rollout.get("targets_covered")
        n_targets = rollout.get("n_targets")
        if targets_covered is None or n_targets is None:
            return None
        max_per_episode = torch.as_tensor(targets_covered).max(dim=-1).values
        return max_per_episode >= n_targets

    def coverage_progress(self, rollout: Any) -> Any | None:
        """M6: max fraction of targets ever covered, per episode.

        ``= (targets_covered.max(dim=-1).values / n_targets)``. Returns None
        when the rollout lacks the required keys.
        """
        targets_covered = rollout.get("targets_covered")
        n_targets = rollout.get("n_targets")
        if targets_covered is None or n_targets is None:
            return None
        max_per_episode = torch.as_tensor(targets_covered).float().max(dim=-1).values
        return max_per_episode / float(n_targets)
