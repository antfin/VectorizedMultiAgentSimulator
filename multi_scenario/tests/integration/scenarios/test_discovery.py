"""F2.1 + F2.3 integration tests: VmasDiscoveryAdapter — env build + DI predicates."""

import torch

from multi_scenario.adapters.scenarios.discovery import VmasDiscoveryAdapter
from multi_scenario.domain.models import ScenarioSection
from multi_scenario.domain.ports import Scenario


def test_implements_scenario_protocol():
    """The adapter satisfies the Scenario port (runtime-checkable)."""
    assert isinstance(VmasDiscoveryAdapter(), Scenario)


def test_default_params_has_required_invariants():
    """default_params bakes in the documented invariants."""
    defaults = VmasDiscoveryAdapter().default_params()
    assert defaults["targets_respawn"] is False
    assert defaults["shared_reward"] is True
    assert defaults["agents_per_target"] == 2
    assert defaults["covering_range"] == 0.25


def test_make_env_builds_with_correct_n_agents():
    """make_env constructs a VMAS env honouring the requested n_agents."""
    adapter = VmasDiscoveryAdapter()
    cfg = ScenarioSection(
        type="discovery",
        params={"n_agents": 2, "n_targets": 2, "agents_per_target": 2},
    )
    env = adapter.make_env(cfg, num_envs=4, seed=0)
    assert env.n_agents == 2
    assert len(env.agents) == 2
    assert env.num_envs == 4


def test_success_predicate_true_when_all_targets_covered_at_some_step():
    """Per-episode success when per-step covered count peaks at n_targets."""
    rollout = {
        "n_targets": 3,
        # episode 0 reaches 3 mid-trajectory; episode 1 stays at 2.
        "targets_covered": torch.tensor([[1, 2, 3, 2], [1, 2, 2, 1]]),
    }
    out = VmasDiscoveryAdapter().success_predicate(rollout)
    assert torch.equal(out, torch.tensor([True, False]))


def test_success_predicate_false_when_max_below_n_targets():
    """All episodes peak below n_targets → all False."""
    rollout = {
        "n_targets": 5,
        "targets_covered": torch.tensor([[1, 2, 3], [0, 1, 2]]),
    }
    out = VmasDiscoveryAdapter().success_predicate(rollout)
    assert torch.equal(out, torch.tensor([False, False]))


def test_success_predicate_returns_none_when_data_missing():
    """Missing targets_covered or n_targets → None (graceful degradation)."""
    assert VmasDiscoveryAdapter().success_predicate({}) is None
    assert VmasDiscoveryAdapter().success_predicate({"n_targets": 3}) is None
    assert (
        VmasDiscoveryAdapter().success_predicate({"targets_covered": torch.zeros(2, 3)})
        is None
    )


def test_coverage_progress_returns_max_fraction():
    """M6 = max-over-time covered count, normalised by n_targets."""
    rollout = {
        "n_targets": 4,
        "targets_covered": torch.tensor([[1, 3, 2], [0, 2, 4]]),
    }
    out = VmasDiscoveryAdapter().coverage_progress(rollout)
    expected = torch.tensor([3 / 4, 4 / 4])
    assert torch.allclose(out, expected)


def test_utilization_predicate_still_stubbed():
    """M8 stays None until a later feature implements it."""
    assert VmasDiscoveryAdapter().utilization_predicate(state={"any": "thing"}) is None
