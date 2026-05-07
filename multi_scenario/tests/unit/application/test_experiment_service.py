"""F1.11 tests: ExperimentService — full pipeline use case with port fakes."""

# The fakes below stand in for the five ports; their methods are intentionally
# minimal stubs and don't use every argument, so suppress pylint's noise.
# pylint: disable=missing-function-docstring,unused-argument,too-few-public-methods

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from multi_scenario.application.experiment_service import ExperimentService
from multi_scenario.domain.models import (
    ExperimentConfig,
    ExperimentResult,
    LibraryVersions,
    Provenance,
    RunState,
    ScenarioSection,
)


class _FakeScenario:
    name = "discovery"

    def make_env(self, cfg: ScenarioSection, num_envs: int, seed: int) -> Any:
        return {"env": True, "num_envs": num_envs, "seed": seed}

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


class _FakeAlgorithm:
    name = "mappo"
    last_resume_from: Path | None = None  # captured for F5.7 resume tests

    def train(
        self,
        env: Any,
        cfg: ExperimentConfig,
        run_dir: Path | None = None,
        resume_from: Path | None = None,
    ) -> Any:
        type(self).last_resume_from = resume_from
        return {"trained": True}

    def evaluate(
        self,
        artifact: Any,
        env: Any,
        cfg: ExperimentConfig,
        run_dir: Path | None = None,
    ) -> Any:
        return {"rollout": True}


class _FakeMetricsBundle:
    def compute(self, rollout: Any, scenario: Any) -> dict[str, float | None]:
        return {"M1_success_rate": 0.42, "M2_avg_return": 12.34, "M5_tokens": None}


class _InMemoryStorage:
    name = "in_memory"

    def __init__(self) -> None:
        self.config: ExperimentConfig | None = None
        self.provenance: Provenance | None = None
        self.result: ExperimentResult | None = None
        self.state_history: list = []

    def save_config(self, run_dir: Path, config: ExperimentConfig) -> None:
        self.config = config

    def save_provenance(self, run_dir: Path, provenance: Provenance) -> None:
        self.provenance = provenance

    def save_result(self, run_dir: Path, result: ExperimentResult) -> None:
        self.result = result

    def save_run_state(self, run_dir: Path, state: Any) -> None:
        self.state_history.append(state)

    def load_config(self, run_dir: Path) -> ExperimentConfig:
        assert self.config is not None
        return self.config

    def load_provenance(self, run_dir: Path) -> Provenance:
        assert self.provenance is not None
        return self.provenance

    def load_result(self, run_dir: Path) -> ExperimentResult:
        assert self.result is not None
        return self.result

    def load_run_state(self, run_dir: Path) -> Any:
        return self.state_history[-1]


class _ListLogger:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def info(self, msg: str) -> None:
        self.messages.append(("info", msg))

    def debug(self, msg: str) -> None:
        self.messages.append(("debug", msg))

    def warning(self, msg: str) -> None:
        self.messages.append(("warning", msg))

    def error(self, msg: str) -> None:
        self.messages.append(("error", msg))


def _stub_provenance() -> Provenance:
    return Provenance(
        config_hash="sha256:test",
        code_hash="sha256:test",
        hashed_source_files=[],
        git_sha="0",
        git_dirty=False,
        created_at=datetime(2026, 5, 6, 14, 23, tzinfo=timezone.utc),
        library_versions=LibraryVersions(
            python="3.11",
            torch="2.4",
            vmas="1.4",
            benchmarl="1.3",
            multi_scenario="0.0.1",
        ),
    )


def test_run_completes_full_pipeline_with_fakes(fake_config_builder, tmp_path: Path):
    """ExperimentService runs the full lifecycle with port fakes only."""
    storage = _InMemoryStorage()
    logger = _ListLogger()
    service = ExperimentService(
        scenario=_FakeScenario(),
        algorithm=_FakeAlgorithm(),
        metrics=_FakeMetricsBundle(),
        storage=storage,
        logger=logger,
    )

    cfg = ExperimentConfig.model_validate(fake_config_builder())
    result = service.run(cfg, run_dir=tmp_path, provenance=_stub_provenance())

    # Returned result has the expected identity + metrics.
    assert isinstance(result, ExperimentResult)
    assert result.run_id == "test_s0"
    assert result.scenario == "discovery"
    assert result.algorithm == "mappo"
    metric_names = {m.name for m in result.metrics}
    assert {"M1_success_rate", "M2_avg_return", "M5_tokens"} <= metric_names

    # Storage saw the config / provenance / result.
    assert storage.config is cfg
    assert storage.provenance is not None
    assert storage.result is result

    # Lifecycle: INITIALIZING → RUNNING → DONE (at minimum).
    states = [s.state for s in storage.state_history]
    assert states[0] == RunState.INITIALIZING
    assert states[-1] == RunState.DONE

    # Logger received some info messages along the way.
    info_msgs = [m for level, m in logger.messages if level == "info"]
    assert any("training" in m for m in info_msgs)
    assert any("evaluating" in m for m in info_msgs)


def test_eval_episodes_writer_called_with_rollout(fake_config_builder, tmp_path: Path):
    """When ``eval_episodes_writer`` is provided, it receives the rollout from evaluate()."""
    captured: list[tuple[Path, object]] = []

    def writer(run_dir: Path, rollout: object) -> None:
        captured.append((run_dir, rollout))

    service = ExperimentService(
        scenario=_FakeScenario(),
        algorithm=_FakeAlgorithm(),
        metrics=_FakeMetricsBundle(),
        storage=_InMemoryStorage(),
        logger=_ListLogger(),
        eval_episodes_writer=writer,
    )
    cfg = ExperimentConfig.model_validate(fake_config_builder())
    service.run(cfg, run_dir=tmp_path, provenance=_stub_provenance())

    # _FakeAlgorithm.evaluate returns the rollout dict {"rollout": True}.
    assert len(captured) == 1
    received_run_dir, received_rollout = captured[0]
    assert received_run_dir == tmp_path
    assert received_rollout == {"rollout": True}
