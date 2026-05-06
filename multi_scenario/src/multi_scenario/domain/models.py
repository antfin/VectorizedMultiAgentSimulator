"""Domain models for multi_scenario.

Strictly validated Pydantic models that describe an experiment configuration.
This module imports only stdlib + pydantic + yaml — no torch, no vmas, no
benchmarl, no streamlit, no boto3 (enforced by F1.12).
"""

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer

_STRICT = ConfigDict(extra="forbid")
_FROZEN = ConfigDict(extra="forbid", frozen=True)

_RUN_ID_RE = re.compile(r"^(.+)_s(\d+)$")
_FOLDER_RE = re.compile(r"^(.+)__(\d{8}_\d{4})$")
_EXP_ID_CHARS_RE = re.compile(r"^[A-Za-z0-9_-]+$")


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


class MetricRecord(BaseModel):
    """A single named metric value; None means not applicable for this run."""

    model_config = _STRICT

    name: str
    value: float | None


class ExperimentResult(BaseModel):
    """Final outcome of one experiment run; serialised to output/metrics.json."""

    model_config = _STRICT

    run_id: str
    exp_id: str
    scenario: str
    algorithm: str
    seed: int
    run_timestamp: str
    metrics: list[MetricRecord]
    config_snapshot: dict[str, Any]
    n_envs: int
    n_eval_episodes: int
    convergence_frame: int | None = None

    @field_validator("metrics", mode="before")
    @classmethod
    def _coerce_metrics(cls, v: Any) -> Any:
        # Accept a {name: value} dict on input and rewrap as list[MetricRecord]
        # so that round-tripping through the dict-shaped serialiser works.
        if isinstance(v, dict):
            return [{"name": name, "value": value} for name, value in v.items()]
        return v

    @model_serializer(mode="wrap")
    def _serialise_metrics_as_dict(self, handler: Any) -> dict[str, Any]:
        # Output `metrics` as a flat {name: value} dict on the wire while keeping
        # the in-memory representation as list[MetricRecord]. Affects both
        # model_dump() and model_dump_json().
        data = handler(self)
        metrics = data.get("metrics")
        if isinstance(metrics, list):
            data["metrics"] = {item["name"]: item["value"] for item in metrics}
        return data

    def to_flat_dict(self) -> dict[str, Any]:
        """Single-level dict suitable for one row of runs.csv."""
        flat: dict[str, Any] = {
            "run_id": self.run_id,
            "exp_id": self.exp_id,
            "scenario": self.scenario,
            "algorithm": self.algorithm,
            "seed": self.seed,
            "run_timestamp": self.run_timestamp,
            "n_envs": self.n_envs,
            "n_eval_episodes": self.n_eval_episodes,
            "convergence_frame": self.convergence_frame,
        }
        for record in self.metrics:
            flat[record.name] = record.value
        flat.update(self.config_snapshot)
        return flat


class RunId(BaseModel):
    """Parametric run identity rendered as ``<exp_id>_s<seed>``."""

    model_config = _FROZEN

    exp_id: str
    seed: int

    @field_validator("exp_id")
    @classmethod
    def _validate_exp_id(cls, v: str) -> str:
        if not v:
            raise ValueError("exp_id cannot be empty")
        if "__" in v:
            # Reserved as the separator between run_id and timestamp in folder names.
            raise ValueError("exp_id cannot contain '__'")
        if not _EXP_ID_CHARS_RE.match(v):
            raise ValueError(f"exp_id must be alphanumeric with _ or - only: {v!r}")
        return v

    @field_validator("seed")
    @classmethod
    def _validate_seed(cls, v: int) -> int:
        if v < 0:
            raise ValueError("seed must be non-negative")
        return v

    def __str__(self) -> str:
        return f"{self.exp_id}_s{self.seed}"

    def folder_name(self, timestamp: str) -> str:
        """Run folder name = ``<run_id>__<timestamp>``."""
        return f"{self}__{timestamp}"

    @classmethod
    def from_string(cls, s: str) -> "RunId":
        """Parse ``<exp_id>_s<seed>`` back to a RunId."""
        match = _RUN_ID_RE.match(s)
        if not match:
            raise ValueError(f"invalid run_id format: {s!r}")
        return cls(exp_id=match.group(1), seed=int(match.group(2)))

    @classmethod
    def from_folder_name(cls, folder: str) -> tuple["RunId", str]:
        """Parse ``<run_id>__<timestamp>`` back to (RunId, timestamp)."""
        match = _FOLDER_RE.match(folder)
        if not match:
            raise ValueError(f"invalid folder format: {folder!r}")
        return cls.from_string(match.group(1)), match.group(2)
