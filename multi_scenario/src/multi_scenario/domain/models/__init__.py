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
from .provenance import LibraryVersions, Provenance
from .report import ReportLinks, ReportVideos, RunReport
from .result import ExperimentResult, MetricRecord
from .rng_state import RngState
from .run_id import RunId
from .run_state import RunState, RunStateRecord, RunStateTransition
from .runs_manifest import ManifestRunEntry, ManifestScope, RankingEntry, RunsManifest

__all__ = [
    "AlgorithmSection",
    "EvaluationSection",
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentSection",
    "LibraryVersions",
    "ManifestRunEntry",
    "ManifestScope",
    "MetricRecord",
    "Provenance",
    "RankingEntry",
    "ReportLinks",
    "ReportVideos",
    "RngState",
    "RunId",
    "RunReport",
    "RunState",
    "RunStateRecord",
    "RunStateTransition",
    "RunnerSection",
    "RuntimeSection",
    "RunsManifest",
    "ScenarioSection",
    "StorageSection",
    "TrainingSection",
]
