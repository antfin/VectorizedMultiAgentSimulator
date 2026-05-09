"""F1.6 tests: Scenario port (Protocol) — runtime-checkable."""

# The fake classes below exist only to satisfy the Protocol's structural shape.
# Their methods have no business logic to document, don't use the
# protocol-required arguments, and the *incomplete* fakes have intentionally
# few public methods (that's what makes them incomplete). Suppress for this file.
# pylint: disable=missing-function-docstring,unused-argument,too-few-public-methods

from pathlib import Path
from typing import Any

from multi_scenario.domain.models import (
    ExperimentConfig,
    ExperimentResult,
    Provenance,
    RunStateRecord,
    ScenarioSection,
)
from multi_scenario.domain.ports import (
    Algorithm,
    Logger,
    MetricsBundle,
    Runner,
    Scenario,
    Storage,
)


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

    def default_params(self) -> dict[str, Any]:
        return {}

    def train(
        self,
        env: Any,
        cfg: ExperimentConfig,
        run_dir: Path | None = None,
        resume_from: Path | None = None,
    ) -> Any:
        return ("artifact", env, cfg, run_dir, resume_from)

    def evaluate(
        self,
        artifact: Any,
        env: Any,
        cfg: ExperimentConfig,
        run_dir: Path | None = None,
    ) -> Any:
        return ("rollout", artifact, env, cfg, run_dir)


class _IncompleteAlgorithm:
    """Missing `evaluate` — should fail isinstance."""

    name = "incomplete"

    def train(
        self,
        env: Any,
        cfg: ExperimentConfig,
        run_dir: Path | None = None,
        resume_from: Path | None = None,
    ) -> Any:
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


class _FakeMetricsBundle:
    """A complete fake bundle covering every member of the MetricsBundle protocol."""

    def compute(self, rollout: Any, scenario: Scenario) -> dict[str, float | None]:
        return {"M1_success_rate": 0.4, "M5_tokens": None}


class _IncompleteMetricsBundle:
    """No `compute` method — should fail isinstance."""

    name = "incomplete"


def test_fake_metrics_bundle_implements_protocol():
    """A class exposing every protocol member passes isinstance(MetricsBundle)."""
    assert isinstance(_FakeMetricsBundle(), MetricsBundle)


def test_incomplete_metrics_bundle_fails_protocol_check():
    """Missing the protocol member → isinstance returns False."""
    assert not isinstance(_IncompleteMetricsBundle(), MetricsBundle)


class _FakeStorage:
    """A complete fake storage covering every member of the Storage protocol."""

    name = "fake"

    def save_config(self, run_dir: Path, config: ExperimentConfig) -> None:
        pass

    def save_provenance(self, run_dir: Path, provenance: Provenance) -> None:
        pass

    def save_result(self, run_dir: Path, result: ExperimentResult) -> None:
        pass

    def save_run_state(self, run_dir: Path, state: RunStateRecord) -> None:
        pass

    def load_config(self, run_dir: Path) -> ExperimentConfig:  # type: ignore[empty-body]
        return None  # type: ignore[return-value]

    def load_provenance(self, run_dir: Path) -> Provenance:  # type: ignore[empty-body]
        return None  # type: ignore[return-value]

    def load_result(self, run_dir: Path) -> ExperimentResult:  # type: ignore[empty-body]
        return None  # type: ignore[return-value]

    def load_run_state(self, run_dir: Path) -> RunStateRecord:  # type: ignore[empty-body]
        return None  # type: ignore[return-value]


class _IncompleteStorage:
    """Missing `load_run_state` — should fail isinstance."""

    name = "incomplete"

    def save_config(self, run_dir: Path, config: ExperimentConfig) -> None:
        pass

    def save_provenance(self, run_dir: Path, provenance: Provenance) -> None:
        pass

    def save_result(self, run_dir: Path, result: ExperimentResult) -> None:
        pass

    def save_run_state(self, run_dir: Path, state: RunStateRecord) -> None:
        pass

    def load_config(self, run_dir: Path) -> ExperimentConfig:  # type: ignore[empty-body]
        return None  # type: ignore[return-value]

    def load_provenance(self, run_dir: Path) -> Provenance:  # type: ignore[empty-body]
        return None  # type: ignore[return-value]

    def load_result(self, run_dir: Path) -> ExperimentResult:  # type: ignore[empty-body]
        return None  # type: ignore[return-value]


def test_fake_storage_implements_protocol():
    """A class exposing every protocol member passes isinstance(Storage)."""
    assert isinstance(_FakeStorage(), Storage)


def test_incomplete_storage_fails_protocol_check():
    """Missing one protocol member → isinstance returns False."""
    assert not isinstance(_IncompleteStorage(), Storage)


class _FakeLogger:
    """A complete fake logger covering every member of the Logger protocol."""

    def info(self, msg: str) -> None:
        pass

    def debug(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        pass

    def error(self, msg: str) -> None:
        pass


class _IncompleteLogger:
    """Missing `error` — should fail isinstance."""

    def info(self, msg: str) -> None:
        pass

    def debug(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        pass


def test_fake_logger_implements_protocol():
    """A class exposing every protocol member passes isinstance(Logger)."""
    assert isinstance(_FakeLogger(), Logger)


def test_incomplete_logger_fails_protocol_check():
    """Missing one protocol member → isinstance returns False."""
    assert not isinstance(_IncompleteLogger(), Logger)


class _FakeRunner:
    """A complete fake runner covering every member of the Runner protocol."""

    name = "fake"
    supports_resume = True

    def run(  # type: ignore[empty-body]
        self, cfg: ExperimentConfig, run_dir: Path, resume_from: Path | None = None
    ) -> ExperimentResult:
        return None  # type: ignore[return-value]


class _IncompleteRunner:
    """No `run` method — should fail isinstance."""

    name = "incomplete"
    supports_resume = False


def test_fake_runner_implements_protocol():
    """A class exposing every protocol member passes isinstance(Runner)."""
    assert isinstance(_FakeRunner(), Runner)


def test_incomplete_runner_fails_protocol_check():
    """Missing the protocol member → isinstance returns False."""
    assert not isinstance(_IncompleteRunner(), Runner)
