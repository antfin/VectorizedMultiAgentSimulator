"""F1.6 tests: Scenario port (Protocol) — runtime-checkable."""

# The fake classes below exist only to satisfy the Protocol's structural shape.
# Their methods have no business logic to document, don't use the
# protocol-required arguments, and the *incomplete* fakes have intentionally
# few public methods (that's what makes them incomplete). Suppress for this file.
# pylint: disable=missing-function-docstring,unused-argument,too-few-public-methods

from typing import Any

from multi_scenario.domain.models import ExperimentConfig, ScenarioSection
from multi_scenario.domain.ports import Algorithm, Scenario


class _FakeScenario:
    """A complete fake scenario covering every member of the Scenario protocol."""

    name = "fake"

    def make_env(self, cfg: ScenarioSection, num_envs: int, seed: int) -> Any:
        return ("env", cfg, num_envs, seed)

    def default_params(self) -> dict[str, Any]:
        return {}

    def has_comm(self) -> bool:
        return False

    def success_predicate(self, rollout: Any) -> Any:
        return None

    def coverage_progress(self, rollout: Any) -> Any | None:
        return None

    def utilization_predicate(self, state: Any) -> Any:
        return None


class _IncompleteScenario:
    """Missing `utilization_predicate` — should fail isinstance."""

    name = "incomplete"

    def make_env(self, cfg: ScenarioSection, num_envs: int, seed: int) -> Any:
        return None

    def default_params(self) -> dict[str, Any]:
        return {}

    def has_comm(self) -> bool:
        return False

    def success_predicate(self, rollout: Any) -> Any:
        return None

    def coverage_progress(self, rollout: Any) -> Any | None:
        return None


class _FakeAlgorithm:
    """A complete fake algorithm covering every member of the Algorithm protocol."""

    name = "fake"

    def train(self, env: Any, cfg: ExperimentConfig) -> Any:
        return ("artifact", env, cfg)

    def evaluate(self, artifact: Any, env: Any, cfg: ExperimentConfig) -> Any:
        return ("rollout", artifact, env, cfg)


class _IncompleteAlgorithm:
    """Missing `evaluate` — should fail isinstance."""

    name = "incomplete"

    def train(self, env: Any, cfg: ExperimentConfig) -> Any:
        return None


def test_fake_scenario_implements_protocol():
    """A class exposing every protocol member passes isinstance(Scenario)."""
    assert isinstance(_FakeScenario(), Scenario)


def test_incomplete_scenario_fails_protocol_check():
    """Missing one protocol member → isinstance returns False."""
    assert not isinstance(_IncompleteScenario(), Scenario)


def test_fake_algorithm_implements_protocol():
    """A class exposing every protocol member passes isinstance(Algorithm)."""
    assert isinstance(_FakeAlgorithm(), Algorithm)


def test_incomplete_algorithm_fails_protocol_check():
    """Missing one protocol member → isinstance returns False."""
    assert not isinstance(_IncompleteAlgorithm(), Algorithm)
