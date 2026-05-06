"""Domain ports — Protocols that adapters must satisfy.

Every Protocol here is torch/vmas/benchmarl-agnostic. Tensor-shaped values
appear as ``Any`` so the domain doesn't pull in those libraries (enforced by
F1.12). Concrete adapters in ``adapters/`` know the real types.

Public surface re-exports each Protocol so callers say
``from multi_scenario.domain.ports import Scenario, Algorithm, MetricsBundle``.
"""

from .algorithm import Algorithm
from .metrics import MetricsBundle
from .scenario import Scenario

__all__ = ["Algorithm", "MetricsBundle", "Scenario"]
