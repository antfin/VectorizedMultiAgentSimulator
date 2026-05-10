"""ExperimentConfig and its section models — strict YAML/JSON parsing.

This module imports only stdlib + pydantic + yaml — no torch, no vmas, no
benchmarl, no streamlit, no boto3 (enforced by F1.12).

F7.7.D1: every numeric carries an explicit ``gt`` / ``ge`` / ``le`` so
nonsense values (negative seeds, zero ``max_iters``, ``minibatch_size``
larger than ``frames_per_batch``, ``device`` outside ``{cpu, cuda}``,
``experiment.id`` containing path-unfriendly characters) are rejected at
parse time with a clean Pydantic error. The registry-aware cross-field
checks (``scenario.type ∈ available_scenarios()`` etc.) live on the
top-level :class:`ExperimentConfig` so they only fire after Pydantic has
validated each section in isolation.
"""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from ._common import STRICT

#: ``experiment.id`` becomes a folder name (``<exp_id>_s<seed>__<timestamp>``)
#: so it must avoid path separators / shell metacharacters. Allowed:
#: alphanumerics + ``_`` + ``-``. Empty string forbidden via ``min_length``.
_EXPERIMENT_ID_PATTERN = r"^[A-Za-z0-9_\-]+$"


class ExperimentSection(BaseModel):
    """Run identity (id, seed) and human-readable metadata (name, description)."""

    model_config = STRICT

    id: str = Field(min_length=1, pattern=_EXPERIMENT_ID_PATTERN)
    seed: int = Field(default=0, ge=0)
    name: str | None = None
    description: str | None = None


class ScenarioSection(BaseModel):
    """Scenario adapter selection (`type`) and its make_env parameters."""

    model_config = STRICT

    type: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class AlgorithmSection(BaseModel):
    """Algorithm adapter selection (`type`) and its hyperparameters."""

    model_config = STRICT

    type: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class TrainingSection(BaseModel):
    """Training-loop orchestration: algorithm-agnostic knobs.

    Includes the universal knobs BenchMARL puts on its ExperimentConfig
    (``lr``, ``gamma``, ``frames_per_batch``, ``minibatch_size``,
    ``n_minibatch_iters``, ``share_policy_params``). Algorithm-specific
    knobs (``lmbda`` for PPO, ``entropy_coef``, ``clip_epsilon``, …) stay
    in ``cfg.algorithm.params``.

    F7.7.D1 cross-field invariant: ``minibatch_size <= frames_per_batch``.
    BenchMARL silently floors this in some code paths; we'd rather fail
    loudly at config parse than have downstream training behave oddly.
    """

    model_config = STRICT

    max_iters: int = Field(gt=0)
    num_envs: int = Field(default=1, gt=0)
    device: Literal["cpu", "cuda"] = "cpu"
    lr: float = Field(default=3e-4, gt=0)
    gamma: float = Field(default=0.99, ge=0, le=1)
    frames_per_batch: int = Field(default=6000, gt=0)
    minibatch_size: int = Field(default=400, gt=0)
    n_minibatch_iters: int = Field(default=45, gt=0)
    share_policy_params: bool = True
    # F5.7: BenchMARL writes a checkpoint every N iters (sparse). Set to 0 to
    # disable interval checkpoints (final-only via checkpoint_at_end). Smoke
    # runs auto-disable in the adapter; this is the cadence for non-smoke runs.
    checkpoint_interval_iters: int = Field(default=10, ge=0)
    # F8.2.G: how many checkpoint snapshots BenchMARL retains on disk.
    # BenchMARL's default is 3, which silently overwrites earlier snapshots —
    # the ER1 dry-run lost the iter-125 peak (only kept iters 150 / 160 / 167).
    # Set high enough that an entire ~167-iter run keeps everything; F8.5.D's
    # best-checkpoint-policy callback needs the full history to find the peak.
    # Smoke runs ignore this (no checkpoints written).
    keep_checkpoints_num: int = Field(default=1000, gt=0)

    @model_validator(mode="after")
    def _minibatch_fits_in_batch(self) -> "TrainingSection":
        if self.minibatch_size > self.frames_per_batch:
            raise ValueError(
                f"minibatch_size ({self.minibatch_size}) must be ≤ "
                f"frames_per_batch ({self.frames_per_batch})"
            )
        return self


class EvaluationSection(BaseModel):
    """Evaluation cadence (interval_iters) and scope (episodes per eval)."""

    model_config = STRICT

    interval_iters: int = Field(gt=0)
    episodes: int = Field(gt=0)


class RunnerSection(BaseModel):
    """Runner adapter selection — where the experiment executes.

    ``type`` is the *dispatch* layer (the Python class that wraps the
    submission), not what runs *inside* the host:

    - ``local`` → ``LocalRunner`` — runs the BenchMARL training loop in the
      current Python process. Use when training on the user's machine.
    - ``ovh``   → ``OvhRunner``   — shells out ``ovhai job run`` which boots
      an OVH container that then invokes ``LocalRunner`` *inside*. Use when
      training on OVH GPU nodes.

    The CLI ``--runner`` flag overrides this YAML field at submit time
    (rare; mostly useful for forcing a YAML to run locally for debugging).
    """

    model_config = STRICT

    type: Literal["local", "ovh"] = "local"
    params: dict[str, Any] = Field(default_factory=dict)


class StorageSection(BaseModel):
    """Storage adapter selection — where run outputs are written and read (fs, s3, ...)."""

    model_config = STRICT

    type: str = Field(default="fs", min_length=1)
    path: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class RuntimeSection(BaseModel):
    """Optional infrastructure: runner + storage. Defaults are computed by ExperimentService."""

    model_config = STRICT

    runner: RunnerSection = Field(default_factory=RunnerSection)
    storage: StorageSection


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration. Loaded from YAML; strictly validated."""

    model_config = STRICT

    experiment: ExperimentSection
    scenario: ScenarioSection
    algorithm: AlgorithmSection
    training: TrainingSection
    evaluation: EvaluationSection
    runtime: RuntimeSection | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load and validate an ExperimentConfig from a YAML file.

        Schema-only validation. Registry-aware checks (``scenario.type ∈
        available_scenarios()`` etc.) live in
        :func:`multi_scenario.application.config_validation.validate_known_types`
        — they belong in the application layer because they cross the domain
        boundary into ``application.factories``. Entry points that need both
        call ``from_yaml`` then the validator, or use the
        ``application.config_loader.load_experiment_config`` wrapper.
        """
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
