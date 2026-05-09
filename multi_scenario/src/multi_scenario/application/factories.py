"""Name → adapter factories.

Centralised registries keyed by the ``type`` field on each cfg section
(``cfg.scenario.type``, ``cfg.algorithm.type``, ``cfg.runtime.storage.type``,
``cfg.runtime.runner.type``). Each ``make_*`` raises a clean ValueError on
unknown names so config typos fail loudly with the registered names listed.

The :func:`available_*` listing helpers mirror each registry — the Submit
page's data-driven form reads them at render time so adding a new adapter
in the backend automatically surfaces in the UI without touching any
frontend code (F7.7.B2). Together with the ``default_params()`` method on
each adapter port, the frontend stays generic.

When a new adapter lands (next algorithm, next scenario, S3 storage, …) it
gets one line added here.
"""

from dataclasses import dataclass
from typing import Callable

from multi_scenario.adapters.algorithms.iddpg import IddpgAdapter
from multi_scenario.adapters.algorithms.ippo import IppoAdapter
from multi_scenario.adapters.algorithms.isac import IsacAdapter
from multi_scenario.adapters.algorithms.maddpg import MaddpgAdapter
from multi_scenario.adapters.algorithms.mappo import MappoAdapter
from multi_scenario.adapters.algorithms.masac import MasacAdapter
from multi_scenario.adapters.scenarios.discovery import VmasDiscoveryAdapter
from multi_scenario.adapters.scenarios.flocking import VmasFlockingAdapter
from multi_scenario.adapters.scenarios.navigation import VmasNavigationAdapter
from multi_scenario.adapters.scenarios.transport import VmasTransportAdapter
from multi_scenario.adapters.storage.local import LocalStorageAdapter
from multi_scenario.domain.ports import Algorithm, Scenario, Storage

_SCENARIOS: dict[str, Callable[[], Scenario]] = {
    "discovery": VmasDiscoveryAdapter,
    "flocking": VmasFlockingAdapter,
    "navigation": VmasNavigationAdapter,
    "transport": VmasTransportAdapter,
}

_ALGORITHMS: dict[str, Callable[[], Algorithm]] = {
    "iddpg": IddpgAdapter,
    "ippo": IppoAdapter,
    "isac": IsacAdapter,
    "maddpg": MaddpgAdapter,
    "mappo": MappoAdapter,
    "masac": MasacAdapter,
}

_STORAGES: dict[str, Callable[[], Storage]] = {
    "fs": LocalStorageAdapter,
}


@dataclass(frozen=True)
class RunnerSpec:
    """Descriptive metadata about a runner the user can submit to.

    The Submit page reads :attr:`requires_ovh_cfg` to decide whether to load
    ``configs/ovh.yaml`` on this submit target — keeps the page free of
    hardcoded ``if target == "ovh"`` branches.
    """

    name: str
    requires_ovh_cfg: bool = False  # only True for runners that talk to OVH


_RUNNERS: dict[str, RunnerSpec] = {
    "local": RunnerSpec(name="local", requires_ovh_cfg=False),
    "ovh": RunnerSpec(name="ovh", requires_ovh_cfg=True),
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


# ── Listing API for the data-driven frontend (F7.7.B1/B2) ───────────


def available_scenarios() -> list[str]:
    """Sorted names of every registered scenario adapter."""
    return sorted(_SCENARIOS)


def available_algorithms() -> list[str]:
    """Sorted names of every registered algorithm adapter."""
    return sorted(_ALGORITHMS)


def available_storages() -> list[str]:
    """Sorted names of every registered storage adapter."""
    return sorted(_STORAGES)


def available_runners() -> list[str]:
    """Sorted names of every registered submit-target runner."""
    return sorted(_RUNNERS)


def runner_spec(name: str) -> RunnerSpec:
    """Look up :class:`RunnerSpec` metadata by name; raises on unknown."""
    if name not in _RUNNERS:
        raise ValueError(f"unknown runner: {name!r}; known: {sorted(_RUNNERS)}")
    return _RUNNERS[name]
