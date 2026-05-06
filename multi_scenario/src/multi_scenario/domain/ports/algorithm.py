"""Algorithm port — Protocol that algorithm adapters must satisfy."""

from typing import Any, Protocol, runtime_checkable

from multi_scenario.domain.models import ExperimentConfig


@runtime_checkable
class Algorithm(Protocol):
    """Domain port for MARL algorithms.

    Implementations live in ``adapters/algorithms/`` (typically wrapping a
    BenchMARL ``Experiment``). ``env`` is whatever ``Scenario.make_env``
    produced; ``artifact`` is whatever ``train`` returned (the algorithm
    decides its own state representation). The Protocol stays
    torch/benchmarl-agnostic by typing both as ``Any``.
    """

    name: str

    def train(self, env: Any, cfg: ExperimentConfig) -> Any:
        """Train and return a serialisable artifact (policy state + metadata)."""

    def evaluate(self, artifact: Any, env: Any, cfg: ExperimentConfig) -> Any:
        """Run evaluation episodes; return rollout data for metric computation."""
