"""CommonMetricsBundle — the always-on M1-M9 metric dict.

Routing per §3.5.3 / F1.6:

- **Universal** (computed here): M2 (return), M3 (steps), M4 (collisions).
- **Scenario-DI** (delegated to ``Scenario`` predicates): M1, M6, M8 — return
  None whenever the scenario predicate returns None (e.g. F2.1 stubs).
- **Comm-gated**: M5 — None when ``scenario.has_comm()`` is False.
- **Stubbed for now**: M7 (sample efficiency, end-of-run from eval-curve data)
  and M9 (spatial spread, needs position field in the rollout). They land in
  later features.

Expected ``rollout`` shape (a dict-of-tensors); adapters that produce rollouts
(BenchMARL at F2.4) aggregate their native shape into this dict::

    rollout = {
        "episode_returns":    Tensor[n_episodes],   # sum of rewards per episode
        "episode_lengths":    Tensor[n_episodes],   # termination step per episode
        "episode_collisions": Tensor[n_episodes],   # agent-agent collisions per episode
    }
"""

from typing import Any

import torch

from multi_scenario.domain.ports import Scenario


class CommonMetricsBundle:
    """Concrete MetricsBundle implementing the always-on M1-M9 dict."""

    # Single public method by design; the per-metric helpers stay private.
    # pylint: disable=too-few-public-methods

    def compute(self, rollout: Any, scenario: Scenario) -> dict[str, float | None]:
        """Compute the M1-M9 dict; values are float or None (= not applicable)."""
        # M5 is gated by has_comm; even when True, token extraction from the
        # rollout isn't implemented until comm scenarios land.
        m5: float | None = None
        return {
            "M1_success_rate": _mean_or_none(scenario.success_predicate(rollout)),
            "M2_avg_return": _mean_or_none(rollout.get("episode_returns")),
            "M3_steps": _mean_or_none(rollout.get("episode_lengths")),
            "M4_collisions": _mean_or_none(rollout.get("episode_collisions")),
            "M5_tokens": m5 if scenario.has_comm() else None,
            "M6_coverage_progress": _mean_or_none(scenario.coverage_progress(rollout)),
            "M7_sample_efficiency": None,
            "M8_agent_utilization": _mean_or_none(
                scenario.utilization_predicate(rollout)
            ),
            "M9_spatial_spread": None,
        }


def _mean_or_none(values: Any) -> float | None:
    """Return ``float(values.float().mean())`` or None if values is None."""
    if values is None:
        return None
    return float(torch.as_tensor(values).float().mean())
