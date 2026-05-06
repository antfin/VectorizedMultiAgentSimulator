"""F2.2 tests: CommonMetricsBundle — always-on M1-M9 dict from rollout + scenario."""

# Fake scenarios stub the four DI primitives; their args are required by the
# Protocol but unused in these tests.
# pylint: disable=missing-function-docstring,unused-argument,too-few-public-methods

from typing import Any

import torch

from multi_scenario.adapters.metrics.common import CommonMetricsBundle
from multi_scenario.domain.ports import MetricsBundle


class _NoopScenario:
    """All DI primitives return None / no-comm — exercises the F2.1 stub path."""

    name = "noop"

    def make_env(self, cfg, num_envs, seed):
        return None

    def default_params(self):
        return {}

    def has_comm(self):
        return False

    def success_predicate(self, rollout):
        return None

    def coverage_progress(self, rollout):
        return None

    def utilization_predicate(self, state):
        return None


class _SuccessScenario(_NoopScenario):
    """success_predicate returns a real bool tensor — exercises M1 happy path."""

    name = "success"

    def __init__(self, success_mask: torch.Tensor) -> None:
        self._mask = success_mask

    def success_predicate(self, rollout: Any) -> Any:
        return self._mask


def test_implements_metrics_bundle_protocol():
    """CommonMetricsBundle satisfies the MetricsBundle port."""
    assert isinstance(CommonMetricsBundle(), MetricsBundle)


def test_universal_metrics_compute_from_episode_data():
    """M2/M3/M4 are means over the per-episode tensors."""
    rollout = {
        "episode_returns": torch.tensor([12.0, 9.0, 15.0]),
        "episode_lengths": torch.tensor([90, 100, 80]),
        "episode_collisions": torch.tensor([0, 1, 2]),
    }
    out = CommonMetricsBundle().compute(rollout, _NoopScenario())
    assert out["M2_avg_return"] == 12.0
    assert out["M3_steps"] == 90.0
    assert out["M4_collisions"] == 1.0


def test_m1_computed_when_predicate_returns_tensor():
    """When the scenario produces a success bool tensor, M1 is its mean."""
    scenario = _SuccessScenario(torch.tensor([True, False, True, True]))
    out = CommonMetricsBundle().compute({}, scenario)
    assert out["M1_success_rate"] == 0.75


def test_m1_returns_none_when_predicate_returns_none():
    """The F2.1 stub case — scenario predicate is None → M1 is None."""
    out = CommonMetricsBundle().compute({}, _NoopScenario())
    assert out["M1_success_rate"] is None


def test_m5_none_when_scenario_has_no_comm():
    """M5 is N/A whenever the scenario reports no comm channel."""
    out = CommonMetricsBundle().compute({}, _NoopScenario())
    assert out["M5_tokens"] is None
