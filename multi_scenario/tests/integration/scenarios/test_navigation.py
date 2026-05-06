"""F4.1 integration tests: VmasNavigationAdapter — env build + DI predicates."""

import torch

from multi_scenario.adapters.scenarios.navigation import VmasNavigationAdapter
from multi_scenario.domain.models import ScenarioSection
from multi_scenario.domain.ports import Scenario


def test_implements_scenario_protocol():
    """The adapter satisfies the Scenario port (runtime-checkable)."""
    assert isinstance(VmasNavigationAdapter(), Scenario)


def test_default_params_has_required_invariants():
    """default_params bakes in the documented invariants."""
    defaults = VmasNavigationAdapter().default_params()
    assert defaults["n_agents"] == 4
    assert defaults["agents_with_same_goal"] == 1
    assert defaults["observe_all_goals"] is False
    assert defaults["shared_rew"] is True


def test_make_env_builds_with_correct_n_agents():
    """make_env constructs a VMAS navigation env honouring the requested n_agents."""
    adapter = VmasNavigationAdapter()
    cfg = ScenarioSection(type="navigation", params={"n_agents": 3})
    env = adapter.make_env(cfg, num_envs=2, seed=0)
    assert env.n_agents == 3
    assert len(env.agents) == 3
    assert env.num_envs == 2


def test_success_predicate_reads_episode_terminated():
    """M1 mirrors the universal episode_terminated rollout flag."""
    rollout = {"episode_terminated": torch.tensor([True, False, True])}
    out = VmasNavigationAdapter().success_predicate(rollout)
    assert torch.equal(out, torch.tensor([True, False, True]))


def test_success_predicate_returns_none_when_data_missing():
    """Missing episode_terminated → None (graceful degradation)."""
    assert VmasNavigationAdapter().success_predicate({}) is None


def test_coverage_progress_still_stubbed():
    """M6 stays None for F4.1; sharper coverage lands later."""
    assert VmasNavigationAdapter().coverage_progress({}) is None


def test_utilization_predicate_still_stubbed():
    """M8 stays None until a later feature implements it."""
    assert VmasNavigationAdapter().utilization_predicate(state={}) is None


def test_has_comm_false():
    """No comm channel — M5 is N/A."""
    assert VmasNavigationAdapter().has_comm() is False
