"""F1.2 tests: MetricRecord and ExperimentResult models."""

import pytest
from pydantic import ValidationError

from multi_scenario.domain.models import ExperimentResult, MetricRecord


def _minimal_result() -> ExperimentResult:
    """Build a minimal valid ExperimentResult shared across tests."""
    return ExperimentResult(
        run_id="disc_baseline_smoke_mappo_s0",
        exp_id="disc_baseline_smoke_mappo",
        scenario="discovery",
        algorithm="mappo",
        seed=0,
        run_timestamp="20260506_1423",
        metrics=[
            MetricRecord(name="M1_success_rate", value=0.42),
            MetricRecord(name="M5_tokens", value=None),
        ],
        config_snapshot={"n_agents": 2, "lr": 0.0003},
        n_envs=1,
        n_eval_episodes=10,
    )


def test_metric_record_with_value():
    """A MetricRecord with a numeric value constructs cleanly."""
    m = MetricRecord(name="M1_success_rate", value=0.42)
    assert m.name == "M1_success_rate"
    assert m.value == 0.42


def test_metric_record_with_null():
    """A MetricRecord can carry None to mean 'not applicable'."""
    m = MetricRecord(name="M5_tokens", value=None)
    assert m.value is None


def test_metric_record_rejects_extra():
    """Unknown fields on a MetricRecord raise ValidationError."""
    with pytest.raises(ValidationError):
        MetricRecord(name="M1", value=0.4, foo="bar")  # type: ignore[call-arg]


def test_experiment_result_minimal():
    """ExperimentResult constructs with the required fields and optional defaults."""
    r = _minimal_result()
    assert r.run_id == "disc_baseline_smoke_mappo_s0"
    assert r.scenario == "discovery"
    assert len(r.metrics) == 2
    assert r.convergence_frame is None


def test_experiment_result_metrics_serialise_as_dict():
    """model_dump produces metrics as {name: value} dict, not list of records."""
    r = _minimal_result()
    dumped = r.model_dump()
    assert dumped["metrics"] == {"M1_success_rate": 0.42, "M5_tokens": None}


def test_experiment_result_roundtrip():
    """model_dump followed by model_validate preserves the model exactly."""
    r = _minimal_result()
    r2 = ExperimentResult.model_validate(r.model_dump())
    assert r == r2


def test_experiment_result_to_flat_dict():
    """to_flat_dict produces a single-level dict suitable for one runs.csv row."""
    r = _minimal_result()
    flat = r.to_flat_dict()
    assert flat["run_id"] == "disc_baseline_smoke_mappo_s0"
    assert flat["scenario"] == "discovery"
    assert flat["algorithm"] == "mappo"
    assert flat["seed"] == 0
    assert flat["M1_success_rate"] == 0.42
    assert flat["M5_tokens"] is None
    assert flat["n_agents"] == 2
    assert flat["lr"] == 0.0003
    assert flat["n_envs"] == 1
    assert flat["n_eval_episodes"] == 10
