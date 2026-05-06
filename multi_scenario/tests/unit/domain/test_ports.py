"""F1.6 tests: Scenario port (Protocol) — runtime-checkable."""

# The fake classes below exist only to satisfy the Protocol's structural shape.
# Their methods have no business logic to document and don't use the
# protocol-required arguments — pylint's noisy here, suppress for this file.
# pylint: disable=missing-function-docstring,unused-argument

from typing import Any

from multi_scenario.domain.models import ScenarioSection
from multi_scenario.domain.ports import Scenario


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


def test_fake_scenario_implements_protocol():
    """A class exposing every protocol member passes isinstance(Scenario)."""
    assert isinstance(_FakeScenario(), Scenario)


def test_incomplete_scenario_fails_protocol_check():
    """Missing one protocol member → isinstance returns False."""
    assert not isinstance(_IncompleteScenario(), Scenario)
