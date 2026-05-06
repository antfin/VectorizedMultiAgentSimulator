"""F2.1 integration tests: VmasDiscoveryAdapter — real env build via vmas."""

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
