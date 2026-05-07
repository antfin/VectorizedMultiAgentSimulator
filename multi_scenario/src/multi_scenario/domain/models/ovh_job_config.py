"""OVH AI Training job-spec config — loaded from ``configs/ovh.yaml``.

Holds the deployment-side knobs that don't belong in per-experiment YAMLs:
GPU type / count, region, container image, S3 buckets + mount points, and
the entry-point script. The experiment YAML stays portable across runners;
this file pins the OVH-specific defaults.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from ._common import STRICT


class OvhGpuModel(BaseModel):
    """One known GPU model (validated against ``gpu_type`` choice)."""

    model_config = STRICT

    eur_per_hour: float = 0.0
    description: str = ""


class OvhJobConfig(BaseModel):
    """OVH-side job parameters — independent of the experiment YAML."""

    model_config = STRICT

    region: str
    image: str
    gpu_type: str
    n_gpu: int = 1
    bucket_code: str
    bucket_results: str
    mount_code: str = "/workspace/code"
    mount_results: str = "/workspace/results"
    default_runner: str = "python -m multi_scenario.cli run"
    default_extra_cli: str = "--device cuda"
    poll_interval_sec: float = 30.0
    timeout_sec: float = 7200.0  # 2h default; long enough for most matrix cells
    gpu_models: dict[str, OvhGpuModel] = {}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "OvhJobConfig":
        """Load an OVH job config from a YAML file (typically ``configs/ovh.yaml``)."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    def validate_gpu_choice(self) -> None:
        """Raise if ``gpu_type`` isn't in the declared ``gpu_models`` registry."""
        if self.gpu_models and self.gpu_type not in self.gpu_models:
            raise ValueError(
                f"unknown gpu_type: {self.gpu_type!r}; known: {sorted(self.gpu_models)}"
            )

    def estimate_cost_eur(self, hours: float) -> float | None:
        """Best-effort cost estimate from ``gpu_models[gpu_type].eur_per_hour``."""
        gpu = self.gpu_models.get(self.gpu_type)
        if gpu is None or gpu.eur_per_hour <= 0:
            return None
        return gpu.eur_per_hour * self.n_gpu * hours


def _yaml_safe_dump(obj: Any) -> str:
    """Stable yaml dump used for example-config generation in tests."""
    return yaml.safe_dump(obj, sort_keys=False, default_flow_style=False)
