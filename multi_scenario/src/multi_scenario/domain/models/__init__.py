"""Domain models for multi_scenario.

Strict pydantic models split by concept across submodules. The package surface
re-exports the public names so callers keep using
``from multi_scenario.domain.models import X``.
"""

from .config import (
    AlgorithmSection,
    EvaluationSection,
    ExperimentConfig,
    ExperimentSection,
    RunnerSection,
    RuntimeSection,
    ScenarioSection,
    StorageSection,
    TrainingSection,
)
from .result import ExperimentResult, MetricRecord
from .run_id import RunId
from .run_state import RunState, RunStateRecord, RunStateTransition

__all__ = [
    "AlgorithmSection",
    "EvaluationSection",
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentSection",
    "MetricRecord",
    "RunId",
    "RunState",
    "RunStateRecord",
    "RunStateTransition",
    "RunnerSection",
    "RuntimeSection",
    "ScenarioSection",
    "StorageSection",
    "TrainingSection",
]
