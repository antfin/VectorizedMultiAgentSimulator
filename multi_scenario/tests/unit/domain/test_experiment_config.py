"""F1.1 tests: ExperimentConfig parsing, strict validation, YAML loading."""

from pathlib import Path

import pytest

from multi_scenario.domain.models import ExperimentConfig
from pydantic import ValidationError


def _minimal_dict() -> dict:
    """Smallest dict that satisfies ExperimentConfig's required sections."""
    return {
        "experiment": {"id": "test", "seed": 0},
        "scenario": {"type": "discovery", "params": {}},
        "algorithm": {"type": "mappo", "params": {}},
        "training": {"max_iters": 1},
        "evaluation": {"interval_iters": 1, "episodes": 1},
    }


def test_roundtrip_dict():
    """model_validate followed by model_dump preserves the data."""
    d = _minimal_dict()
    cfg = ExperimentConfig.model_validate(d)
    assert cfg.experiment.id == "test"
    assert cfg.scenario.type == "discovery"
    assert cfg.algorithm.type == "mappo"
    assert cfg.training.max_iters == 1
    assert cfg.evaluation.episodes == 1
    assert cfg.runtime is None
    cfg2 = ExperimentConfig.model_validate(cfg.model_dump())
    assert cfg == cfg2


def test_rejects_missing_section():
    """Dropping a required top-level section raises ValidationError."""
    d = _minimal_dict()
    del d["scenario"]
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(d)


def test_rejects_unknown_top_level():
    """An unknown top-level key is rejected (extra='forbid')."""
    d = _minimal_dict()
    d["foo"] = "bar"
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(d)


def test_rejects_unknown_nested():
    """An unknown nested key under a section is rejected."""
    d = _minimal_dict()
    d["experiment"]["bogus"] = "x"
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(d)


def test_runtime_optional():
    """Runtime is optional; absence parses to None."""
    cfg = ExperimentConfig.model_validate(_minimal_dict())
    assert cfg.runtime is None


def test_runtime_when_present(tmp_path: Path):
    """Runtime parses with runner + storage when supplied."""
    d = _minimal_dict()
    d["runtime"] = {
        "runner": {"type": "local", "params": {}},
        "storage": {"type": "fs", "path": str(tmp_path), "params": {}},
    }
    cfg = ExperimentConfig.model_validate(d)
    assert cfg.runtime is not None
    assert cfg.runtime.runner.type == "local"
    assert cfg.runtime.storage.path == str(tmp_path)


def test_from_yaml_parses_example(repo_root: Path):
    """The shipped docs/getting_started/example_config.yaml round-trips through ExperimentConfig."""
    cfg = ExperimentConfig.from_yaml(
        repo_root / "docs" / "getting_started" / "example_config.yaml"
    )
    assert cfg.experiment.id == "disc_baseline_smoke"
    assert cfg.scenario.type == "discovery"
    assert cfg.algorithm.type == "mappo"
    assert cfg.scenario.params["n_agents"] == 2
    assert cfg.runtime is not None
    assert cfg.runtime.storage.type == "fs"
