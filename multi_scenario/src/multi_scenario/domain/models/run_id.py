"""Parametric run identifier rendered as ``<exp_id>_s<seed>``."""

import re

from pydantic import BaseModel, field_validator

from ._common import FROZEN

_RUN_ID_RE = re.compile(r"^(.+)_s(\d+)$")
_FOLDER_RE = re.compile(r"^(.+)__(\d{8}_\d{4})$")
_EXP_ID_CHARS_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class RunId(BaseModel):
    """Parametric run identity rendered as ``<exp_id>_s<seed>``."""

    model_config = FROZEN

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
