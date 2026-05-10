"""F9.0 — Aggregate run summary written at the end of a LERO run.

Two complementary records:

- :class:`FallbackEntry` — one row per attempt in the full-training
  fallback chain (rank 0 = best inner-loop candidate; rank 1+ = next
  attempts when rank 0 crashed during full training).
- :class:`LeroRunSummary` — top-level summary written to
  ``<run_dir>/output/lero/final_summary.json`` aggregating cost,
  best candidate, fallback chain, and iteration trajectory.
"""

from typing import Literal

from pydantic import BaseModel, Field

from multi_scenario.domain.lero.candidate import CandidateMetrics, Verdict
from multi_scenario.domain.models._common import STRICT


class FallbackEntry(BaseModel):
    """One row of the full-training fallback chain.

    Mirrors the rendezvous_comm ``fallback_chain.json`` shape so the
    F8.4 reproducibility comparison can diff our chain against theirs.
    """

    model_config = STRICT

    rank: int = Field(ge=0)
    iteration: int = Field(ge=0)
    candidate_idx: int = Field(ge=0)
    eval_metrics: CandidateMetrics
    outcome: Literal["success", "crashed", "skipped"]
    #: Exception text when ``outcome == "crashed"``; ``None`` otherwise.
    error: str | None = None


class LeroRunSummary(BaseModel):
    """Top-level summary of a completed LERO run."""

    model_config = STRICT

    #: ``cfg.experiment.id`` of the parent experiment.
    exp_id: str
    seed: int = Field(ge=0)
    n_iterations_completed: int = Field(ge=0)
    n_candidates_total: int = Field(ge=0)
    #: Aggregated USD spend across every LLM call in this run.
    total_cost_usd: float = Field(ge=0.0)
    #: Best inner-loop candidate's metrics + verdict (the rank-0 entry
    #: in :attr:`fallback_chain` before full training).
    best_candidate_metrics: CandidateMetrics
    best_candidate_verdict: Verdict
    fallback_chain: list[FallbackEntry] = Field(default_factory=list)
    #: True when full training succeeded on at least one rank in the
    #: chain. False = every attempt crashed; the run produced inner-loop
    #: results but no full-training final.
    full_training_succeeded: bool
