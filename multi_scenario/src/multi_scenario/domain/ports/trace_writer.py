"""F9.3 — :class:`TraceWriter` Protocol — durably persist LERO traces.

The orchestrator writes through this Protocol; the filesystem adapter
materialises the per-(iter, cand, attempt) layout under
``<run_dir>/output/lero/iter_<n>/cand_<m>/attempt_<a>/``. Decoupling
means a future S3-backed trace writer can land without orchestrator
changes — the contract is "give me these dataclasses, I'll get them
durably stored".

Persistence properties the implementer must guarantee:

- **Atomic per-write** — a crashed process never leaves a half-written
  trace file behind. Implementers use write-to-tempfile-then-rename
  (POSIX) or equivalent.
- **Idempotent on re-write** — overwriting the same trace path with
  the same payload is a no-op (allows safe resume).
- **Single source of truth for path layout** — callers don't construct
  ``iter_<n>/cand_<m>/`` paths themselves; the writer owns that.

Methods are split into per-record-type writes (``write_prompt``,
``write_response``, ``write_reasoning``, ``write_candidate``) plus
two aggregate writes (``write_evolution_history``, ``write_summary``)
so the orchestrator can call each at the natural point in the loop
without bundling state.
"""

from pathlib import Path
from typing import Protocol

from multi_scenario.domain.lero import (
    CandidateResult,
    LeroRunSummary,
    PromptTrace,
    ReasoningTrace,
    ResponseTrace,
)


class TraceWriter(Protocol):
    """Durable trace persistence for the LERO orchestrator."""

    def write_prompt(
        self,
        *,
        run_dir: Path,
        iteration: int,
        candidate_idx: int,
        attempt: int,
        trace: PromptTrace,
    ) -> None:
        """Persist a prompt trace at ``iter_<n>/cand_<m>/attempt_<a>/prompt.json``."""
        ...

    def write_response(
        self,
        *,
        run_dir: Path,
        iteration: int,
        candidate_idx: int,
        attempt: int,
        trace: ResponseTrace,
    ) -> None:
        """Persist a response trace alongside the matching prompt."""
        ...

    def write_reasoning(
        self,
        *,
        run_dir: Path,
        iteration: int,
        candidate_idx: int,
        attempt: int,
        trace: ReasoningTrace,
    ) -> None:
        """Persist provider-separated reasoning when the model emits it."""
        ...

    def write_candidate_result(
        self,
        *,
        run_dir: Path,
        result: CandidateResult,
    ) -> None:
        """Persist the iter+cand-keyed final :class:`CandidateResult`."""
        ...

    def write_evolution_history(
        self,
        *,
        run_dir: Path,
        results: list[CandidateResult],
    ) -> None:
        """Persist the cross-iteration sorted history.

        Mirrors rendezvous_comm's ``evolution_history.json``: the
        cumulative list of every CandidateResult so far, in the order
        the prompt composer would see them as feedback.
        """
        ...

    def write_summary(
        self,
        *,
        run_dir: Path,
        summary: LeroRunSummary,
    ) -> None:
        """Persist the final aggregate at ``output/lero/final_summary.json``."""
        ...
