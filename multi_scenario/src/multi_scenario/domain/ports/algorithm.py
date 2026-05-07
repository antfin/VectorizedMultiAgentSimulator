"""Algorithm port — Protocol that algorithm adapters must satisfy."""

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from multi_scenario.domain.models import ExperimentConfig


@runtime_checkable
class Algorithm(Protocol):
    """Domain port for MARL algorithms.

    Implementations live in ``adapters/algorithms/`` (typically wrapping a
    BenchMARL ``Experiment``). ``env`` is whatever ``Scenario.make_env``
    produced; ``artifact`` is whatever ``train`` returned (the algorithm
    decides its own state representation). ``run_dir`` is the per-run
    folder so that algorithms can write any native output beneath it
    (e.g. BenchMARL's ``save_folder`` → ``run_dir / "output" / "benchmarl"``).
    Implementations may ignore ``run_dir`` if they don't need it. The
    Protocol stays torch/benchmarl-agnostic by typing tensor-shaped values
    as ``Any``.
    """

    name: str

    def train(
        self,
        env: Any,
        cfg: ExperimentConfig,
        run_dir: Path | None = None,
        resume_from: Path | None = None,
    ) -> Any:
        """Train and return a serialisable artifact (policy state + metadata).

        ``resume_from`` (F5.7): when set, points at a checkpoint file the
        adapter should load before continuing training. Adapters that don't
        support resume can ignore this kwarg.
        """

    def evaluate(
        self,
        artifact: Any,
        env: Any,
        cfg: ExperimentConfig,
        run_dir: Path | None = None,
    ) -> Any:
        """Run evaluation episodes; return rollout data for metric computation."""
