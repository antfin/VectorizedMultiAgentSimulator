"""F2.6 + F7.7.B1 tests: name → adapter factory registries + listing API."""

import pytest

from multi_scenario.application.factories import (
    available_algorithms,
    available_runners,
    available_scenarios,
    available_storages,
    make_algorithm,
    make_scenario,
    make_storage,
    runner_spec,
    RunnerSpec,
)
from multi_scenario.domain.ports import Algorithm, Scenario, Storage


def test_make_scenario_returns_discovery_adapter():
    """make_scenario('discovery') yields a Scenario with the right name."""
    s = make_scenario("discovery")
    assert isinstance(s, Scenario)
    assert s.name == "discovery"


def test_make_algorithm_returns_mappo_adapter():
    """make_algorithm('mappo') yields an Algorithm with the right name."""
    a = make_algorithm("mappo")
    assert isinstance(a, Algorithm)
    assert a.name == "mappo"


def test_make_storage_returns_local_adapter():
    """make_storage('fs') yields a Storage with the right name."""
    s = make_storage("fs")
    assert isinstance(s, Storage)
    assert s.name == "fs"


def test_unknown_scenario_raises_with_helpful_message():
    """Unknown scenario name → ValueError listing the registered names."""
    with pytest.raises(ValueError, match="unknown scenario"):
        make_scenario("bogus")


def test_unknown_algorithm_raises_with_helpful_message():
    """Unknown algorithm name → ValueError listing the registered names."""
    with pytest.raises(ValueError, match="unknown algorithm"):
        make_algorithm("bogus")


def test_unknown_storage_raises_with_helpful_message():
    """Unknown storage name → ValueError listing the registered names."""
    with pytest.raises(ValueError, match="unknown storage"):
        make_storage("bogus")


# ── F7.7.B1: listing API for the data-driven frontend ─────────────


def test_available_scenarios_lists_known_names_alphabetically():
    """Frontend reads this to populate the scenario picker dropdown."""
    names = available_scenarios()
    assert names == sorted(names)
    assert {"discovery", "navigation", "transport", "flocking"} <= set(names)


def test_available_algorithms_lists_known_names_alphabetically():
    names = available_algorithms()
    assert names == sorted(names)
    assert {"mappo", "iddpg", "ippo", "isac", "maddpg", "masac"} <= set(names)


def test_available_storages_lists_fs():
    """Today only ``fs`` is registered — adding S3 here will surface in the UI."""
    assert "fs" in available_storages()


def test_available_runners_lists_local_and_ovh():
    assert set(available_runners()) == {"local", "ovh"}


def test_runner_spec_local_does_not_require_ovh_cfg():
    """Local submits never load configs/ovh.yaml — saves the user a confusing error."""
    spec = runner_spec("local")
    assert isinstance(spec, RunnerSpec)
    assert spec.requires_ovh_cfg is False


def test_runner_spec_ovh_requires_ovh_cfg():
    """OVH submits need configs/ovh.yaml; UI should preflight-load it before submit."""
    assert runner_spec("ovh").requires_ovh_cfg is True


def test_runner_spec_unknown_raises():
    with pytest.raises(ValueError, match="unknown runner"):
        runner_spec("bogus")


@pytest.mark.parametrize("scen_name", available_scenarios())
def test_every_scenario_default_params_returns_jsonable_primitives(scen_name):
    """Schema-driven form requires every default to be a primitive (no tensors)."""
    params = make_scenario(scen_name).default_params()
    assert isinstance(params, dict)
    for key, value in params.items():
        assert isinstance(
            value, (str, int, float, bool, list, dict)
        ), f"{scen_name}.{key} default is non-primitive: {type(value).__name__}"


@pytest.mark.parametrize("algo_name", available_algorithms())
def test_every_algorithm_has_default_params_method(algo_name):
    """Every Algorithm must expose default_params() — Protocol contract for B2."""
    algo = make_algorithm(algo_name)
    params = algo.default_params()
    assert isinstance(params, dict)
