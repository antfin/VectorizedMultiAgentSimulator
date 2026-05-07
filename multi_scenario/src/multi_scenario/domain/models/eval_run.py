"""Eval-only run record — emitted as ``output/eval_runs/<tag>.json`` (F5.8).

Sibling artefact to ``output/metrics.json``: re-evaluations of a trained policy
without retraining. Multiple eval runs can coexist on the same run_dir, each
with its own ``<tag>.json`` file (e.g. ``eval_20260507_1500.json`` or a
user-supplied tag like ``post_hoc_extra_episodes.json``).

The schema mirrors ``ExperimentResult`` for the metric block + run identity,
plus eval-only fields: ``eval_id`` (the file's tag), ``eval_timestamp``,
``policy_checkpoint`` (relative path under run_dir).
"""

from typing import Any

from pydantic import BaseModel, field_validator, model_serializer

from ._common import STRICT
from .result import MetricRecord


class EvalRunRecord(BaseModel):
    """One re-evaluation of a trained policy; serialised to ``output/eval_runs/<tag>.json``."""

    model_config = STRICT

    eval_id: str
    run_id: str
    scenario: str
    algorithm: str
    seed: int
    eval_timestamp: str
    n_eval_episodes: int
    metrics: list[MetricRecord]
    policy_checkpoint: str

    @field_validator("metrics", mode="before")
    @classmethod
    def _coerce_metrics(cls, v: Any) -> Any:
        # Accept a {name: value} dict on input — same pattern as ExperimentResult.
        if isinstance(v, dict):
            return [{"name": name, "value": value} for name, value in v.items()]
        return v

    @model_serializer(mode="wrap")
    def _serialise_metrics_as_dict(self, handler: Any) -> dict[str, Any]:
        # Output `metrics` as a flat {name: value} dict on the wire while keeping
        # the in-memory representation as list[MetricRecord]. Mirrors
        # ExperimentResult so consumers can read both files the same way.
        data = handler(self)
        metrics = data.get("metrics")
        if isinstance(metrics, list):
            data["metrics"] = {item["name"]: item["value"] for item in metrics}
        return data
