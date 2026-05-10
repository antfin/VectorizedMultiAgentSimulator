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
    LeroSection,
    LlmSection,
    RunnerSection,
    RuntimeSection,
    ScenarioSection,
    StorageSection,
    TrainingSection,
)
from .eval_run import EvalRunRecord
from .ovh_job_config import OvhGpuModel, OvhJobConfig
from .provenance import LibraryVersions, Provenance
from .report import BenchmarlLinks, ReportLinks, ReportVideos, RunReport
from .result import ExperimentResult, MetricRecord
from .rng_state import RngState
from .run_id import RunId
from .run_state import RunState, RunStateRecord, RunStateTransition
from .runs_manifest import ManifestRunEntry, ManifestScope, RankingEntry, RunsManifest
from .s3_storage_config import S3StorageConfig

__all__ = [
    "AlgorithmSection",
    "BenchmarlLinks",
    "EvalRunRecord",
    "EvaluationSection",
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentSection",
    "LeroSection",
    "LibraryVersions",
    "LlmSection",
    "ManifestRunEntry",
    "ManifestScope",
    "MetricRecord",
    "OvhGpuModel",
    "OvhJobConfig",
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
    "S3StorageConfig",
    "ScenarioSection",
    "StorageSection",
    "TrainingSection",
]
