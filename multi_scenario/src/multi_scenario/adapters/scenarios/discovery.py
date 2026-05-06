"""VMAS discovery scenario adapter — implements the Scenario port.

``success_predicate`` (M1) and ``coverage_progress`` (M6) are wired up;
``utilization_predicate`` (M8) is still stubbed and lands later.
``has_comm`` defaults False (no comm channel); flip via
``cfg.params['dim_c']`` if/when comm-enabled discovery scenarios are added.

Rollout-shape contract (extends F2.2's universal contract):

- ``rollout["targets_covered"]``: ``Tensor[n_episodes, T]`` — per-step covered count.
- ``rollout["n_targets"]``: ``int`` — used both as the success threshold (M1)
  and as the normaliser for coverage progress (M6).

The BenchMARL adapter (F2.4) is responsible for populating these from VMAS
info dicts.
"""

from typing import Any

import torch
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

    def utilization_predicate(self, state: Any) -> Any:
        """M8: per-(env, agent) utilization — stub, lands in a later feature."""
        return None
