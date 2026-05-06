"""Run-level result models — emitted as output/metrics.json."""

from typing import Any

from pydantic import BaseModel, field_validator, model_serializer

from ._common import STRICT


class MetricRecord(BaseModel):
    """A single named metric value; None means not applicable for this run."""

    model_config = STRICT

    name: str
    value: float | None


class ExperimentResult(BaseModel):
    """Final outcome of one experiment run; serialised to output/metrics.json."""

    model_config = STRICT

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
