"""Name → adapter factories.

Centralised registries keyed by the ``type`` field on each cfg section
(``cfg.scenario.type``, ``cfg.algorithm.type``, ``cfg.runtime.storage.type``).
Each ``make_*`` raises a clean ValueError on unknown names so config typos
fail loudly with the registered names listed.

When a new adapter lands (next algorithm, next scenario, S3 storage, …) it
gets one line added here.
"""

from typing import Callable

from multi_scenario.adapters.algorithms.mappo import MappoAdapter
from multi_scenario.adapters.scenarios.discovery import VmasDiscoveryAdapter
from multi_scenario.adapters.storage.local import LocalStorageAdapter
from multi_scenario.domain.ports import Algorithm, Scenario, Storage

_SCENARIOS: dict[str, Callable[[], Scenario]] = {
    "discovery": VmasDiscoveryAdapter,
}

_ALGORITHMS: dict[str, Callable[[], Algorithm]] = {
    "mappo": MappoAdapter,
}

_STORAGES: dict[str, Callable[[], Storage]] = {
    "fs": LocalStorageAdapter,
}


def make_scenario(name: str) -> Scenario:
    """Construct the registered Scenario adapter for ``name``."""
    if name not in _SCENARIOS:
        raise ValueError(f"unknown scenario: {name!r}; known: {sorted(_SCENARIOS)}")
    return _SCENARIOS[name]()


def make_algorithm(name: str) -> Algorithm:
    """Construct the registered Algorithm adapter for ``name``."""
    if name not in _ALGORITHMS:
        raise ValueError(f"unknown algorithm: {name!r}; known: {sorted(_ALGORITHMS)}")
    return _ALGORITHMS[name]()


def make_storage(name: str) -> Storage:
    """Construct the registered Storage adapter for ``name``."""
    if name not in _STORAGES:
        raise ValueError(f"unknown storage: {name!r}; known: {sorted(_STORAGES)}")
    return _STORAGES[name]()
