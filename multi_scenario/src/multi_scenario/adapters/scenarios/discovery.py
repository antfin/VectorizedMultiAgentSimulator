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
        """Baseline discovery params — ``targets_respawn=False`` is required for M1/M3.

        ``max_steps`` is the standard VMAS env episode cap; included here so
        the F7.7.B2 data-driven Submit form surfaces it as a knob alongside
        the scenario-specific tunables.
        """
        return {
            "n_agents": 5,
            "n_targets": 7,
            "agents_per_target": 2,
            "covering_range": 0.25,
            "targets_respawn": False,
            "shared_reward": True,
            "max_steps": 100,
        }

    def success_predicate(self, rollout: Any) -> Any:
        """M1: True per episode if the cumulative coverage-event total ever ≥ n_targets.

        ``rollout['targets_covered']`` (after the adapter's cumsum in
        :meth:`BenchmarlBaseAdapter._extract_targets_covered`) is the
        running coverage-event total per step. ``max(dim=-1)`` over
        time then ``>= n_targets`` matches rendezvous_comm's M1::

            # rendezvous_comm/src/metrics.py:109,135
            self.targets_covered_total += info[0]["targets_covered"]
            task_done = self.targets_covered_total >= self.n_targets

        Returns None when the rollout lacks the required keys.

        Note: cumulative coverage events can exceed ``n_targets``
        within an episode (empirically observed in Phase 5b — targets
        get re-covered despite the teleport-out logic). The cumsum
        ``>= n_targets`` criterion is rendezvous_comm's reported
        ``success_rate``; comparable to their 0.88 number for S3b-local.
        """
        targets_covered = rollout.get("targets_covered")
        n_targets = rollout.get("n_targets")
        if targets_covered is None or n_targets is None:
            return None
        max_per_episode = torch.as_tensor(targets_covered).max(dim=-1).values
        return max_per_episode >= n_targets

    def coverage_progress(self, rollout: Any) -> Any | None:
        """M6: clamped(cumulative coverage events, max=n_targets) / n_targets.

        Matches rendezvous_comm's coverage_progress::

            # rendezvous_comm/src/metrics.py:165-166
            targets_covered = self.targets_covered_total.clamp(max=self.n_targets)
            coverage_progress = (targets_covered / self.n_targets).mean().item()

        The ``clamp(max=n_targets)`` is the load-bearing piece — without
        it M6 can exceed 1.0 (Phase 5a's M6=1.97 was the symptom). With
        clamp, M6 in [0, 1].
        """
        targets_covered = rollout.get("targets_covered")
        n_targets = rollout.get("n_targets")
        if targets_covered is None or n_targets is None:
            return None
        max_per_episode = torch.as_tensor(targets_covered).float().max(dim=-1).values
        # Clamp BEFORE division so the metric stays in [0, 1].
        return max_per_episode.clamp(max=float(n_targets)) / float(n_targets)
