"""F4.2 integration tests: VmasFlockingAdapter — env build + DI predicates (all None)."""

from multi_scenario.adapters.scenarios.flocking import VmasFlockingAdapter
from multi_scenario.domain.models import ScenarioSection
from multi_scenario.domain.ports import Scenario


def test_implements_scenario_protocol():
    """The adapter satisfies the Scenario port (runtime-checkable)."""
    assert isinstance(VmasFlockingAdapter(), Scenario)


def test_default_params_has_required_invariants():
    """default_params bakes in the documented invariants."""
    defaults = VmasFlockingAdapter().default_params()
    assert defaults["n_agents"] == 4
    assert defaults["collision_reward"] == -0.1
    assert defaults["dist_shaping_factor"] == 1


def test_make_env_builds_with_correct_n_agents():
    """make_env constructs a VMAS flocking env honouring the requested n_agents."""
    adapter = VmasFlockingAdapter()
    cfg = ScenarioSection(type="flocking", params={"n_agents": 3})
    env = adapter.make_env(cfg, num_envs=2, seed=0)
    assert env.n_agents == 3
    assert len(env.agents) == 3
    assert env.num_envs == 2


def test_success_predicate_returns_none():
    """M1: flocking has no natural binary success metric."""
    assert VmasFlockingAdapter().success_predicate({"any": "rollout"}) is None


def test_coverage_progress_returns_none():
    """M6: not meaningful for flocking."""
    assert VmasFlockingAdapter().coverage_progress({}) is None


def test_utilization_predicate_returns_none():
    """M8: still stubbed."""
    assert VmasFlockingAdapter().utilization_predicate(state={}) is None


def test_has_comm_false():
    """No comm channel — M5 is N/A."""
    assert VmasFlockingAdapter().has_comm() is False
