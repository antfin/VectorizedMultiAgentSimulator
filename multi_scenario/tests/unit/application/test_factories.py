"""F2.6 tests: name → adapter factory registries."""

import pytest

from multi_scenario.application.factories import (
    make_algorithm,
    make_scenario,
    make_storage,
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
