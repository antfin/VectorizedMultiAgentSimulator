"""Domain ports — Protocols that adapters must satisfy.

Every Protocol here is torch/vmas/benchmarl-agnostic. Tensor-shaped values
appear as ``Any`` so the domain doesn't pull in those libraries (enforced by
F1.12). Concrete adapters in ``adapters/`` know the real types.
"""

from typing import Any, Protocol, runtime_checkable

from .models import ExperimentConfig, ScenarioSection


@runtime_checkable
class Scenario(Protocol):
    """Domain port for VMAS-like scenarios.

    Implementations live in ``adapters/scenarios/`` and are free to import
    vmas / torch. The four metric primitives (``has_comm``,
    ``success_predicate``, ``coverage_progress``, ``utilization_predicate``)
    feed the always-on ``MetricsBundle`` (§3.5.3) — every scenario must
    provide them. ``coverage_progress`` may return ``None`` for scenarios
    where M6 isn't applicable (e.g. flocking, transport).
    """

    name: str

    def make_env(self, cfg: ScenarioSection, num_envs: int, seed: int) -> Any:
        """Construct and return the underlying environment."""

    def default_params(self) -> dict[str, Any]:
        """Default scenario.params used when YAML omits a key."""

    def has_comm(self) -> bool:
        """Whether this scenario exposes a communication channel (drives M5 applicability)."""

    def success_predicate(self, rollout: Any) -> Any:
        """Per-episode success Boolean tensor (drives M1)."""

    def coverage_progress(self, rollout: Any) -> Any | None:
        """Per-episode coverage scalar tensor, or None when not applicable (drives M6)."""

    def utilization_predicate(self, state: Any) -> Any:
        """Per-(env, agent) Boolean indicating 'doing useful work' (drives M8)."""


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
