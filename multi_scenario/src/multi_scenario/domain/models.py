"""Domain models for multi_scenario.

Strictly validated Pydantic models that describe an experiment configuration.
This module imports only stdlib + pydantic + yaml — no torch, no vmas, no
benchmarl, no streamlit, no boto3 (enforced by F1.12).
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

_STRICT = ConfigDict(extra="forbid")


class ExperimentSection(BaseModel):
    """Run identity (id, seed) and human-readable metadata (name, description)."""

    model_config = _STRICT

    id: str
    seed: int = 0
    name: str | None = None
    description: str | None = None


class ScenarioSection(BaseModel):
    """Scenario adapter selection (`type`) and its make_env parameters."""

    model_config = _STRICT

    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class AlgorithmSection(BaseModel):
    """Algorithm adapter selection (`type`) and its hyperparameters."""

    model_config = _STRICT

    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class TrainingSection(BaseModel):
    """Training-loop orchestration: algorithm-agnostic knobs (iters, envs, device)."""

    model_config = _STRICT

    max_iters: int
    num_envs: int = 1
    device: str = "cpu"


class EvaluationSection(BaseModel):
    """Evaluation cadence (interval_iters) and scope (episodes per eval)."""

    model_config = _STRICT

    interval_iters: int
    episodes: int


class RunnerSection(BaseModel):
    """Runner adapter selection — where the experiment executes (local, ovh, ...)."""

    model_config = _STRICT

    type: str = "local"
    params: dict[str, Any] = Field(default_factory=dict)


class StorageSection(BaseModel):
    """Storage adapter selection — where run outputs are written and read (fs, s3, ...)."""

    model_config = _STRICT

    type: str = "fs"
    path: str
    params: dict[str, Any] = Field(default_factory=dict)


class RuntimeSection(BaseModel):
    """Optional infrastructure: runner + storage. Defaults are computed by ExperimentService."""

    model_config = _STRICT

    runner: RunnerSection = Field(default_factory=RunnerSection)
    storage: StorageSection


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration. Loaded from YAML; strictly validated."""

    model_config = _STRICT

    experiment: ExperimentSection
    scenario: ScenarioSection
    algorithm: AlgorithmSection
    training: TrainingSection
    evaluation: EvaluationSection
    runtime: RuntimeSection | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load and validate an ExperimentConfig from a YAML file."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
