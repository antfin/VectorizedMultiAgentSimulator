"""S3 storage adapter config — bucket / prefix / region / endpoint.

Holds S3-specific deployment knobs; an experiment's per-run YAML stays portable
across runners. Loaded by ``S3StorageAdapter`` to construct boto3 clients.

``endpoint_url`` is the OVH S3-compatible endpoint when targeting OVH Object
Storage (``https://s3.<region>.io.cloud.ovh.net``); leave None for AWS S3.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel

from ._common import STRICT


class S3StorageConfig(BaseModel):
    """S3 deployment-side config — bucket + prefix + region + optional endpoint."""

    model_config = STRICT

    bucket: str
    prefix: str
    region: str = "gra"
    endpoint_url: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "S3StorageConfig":
        """Load an S3 storage config from a YAML file."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)
