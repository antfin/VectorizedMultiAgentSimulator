"""F9.3 — :class:`FilesystemTraceWriter` — disk-backed :class:`TraceWriter`.

Layout (matches rendezvous_comm so post-hoc analysis tools port cleanly)::

    <run_dir>/output/lero/
    ├── iter_0/
    │   ├── cand_0/
    │   │   ├── attempt_0/
    │   │   │   ├── prompt.json     (PromptTrace)
    │   │   │   ├── response.json   (ResponseTrace)
    │   │   │   └── reasoning.json  (ReasoningTrace; only when emitted)
    │   │   └── result.json         (CandidateResult)
    │   ├── cand_1/
    │   └── …
    ├── iter_1/
    ├── …
    ├── evolution_history.json      (cumulative CandidateResult list)
    └── final_summary.json          (LeroRunSummary)

Atomic writes: every file lands via write-temp-then-rename so a
crashed process can't leave half-written JSON behind. Idempotent: a
re-write of the same path with the same payload is a no-op (the
orchestrator's resume path relies on this).
"""

import os
from pathlib import Path

from multi_scenario.domain.lero import (
    CandidateResult,
    LeroRunSummary,
    PromptTrace,
    ReasoningTrace,
    ResponseTrace,
)


class FilesystemTraceWriter:
    """Per-run, per-iter, per-candidate trace persistence on local disk.

    The four per-trace methods all take five keyword-only arguments
    (``run_dir``, ``iteration``, ``candidate_idx``, ``attempt``,
    ``trace``) because that's the Protocol's contract — the kwargs-only
    signature makes call sites self-documenting at the orchestrator.
    """

    # Single :meth:`_atomic_write_json` is the one place the
    # write-temp-then-rename pattern lives — all the public methods
    # serialise their payload to JSON and delegate.
    # The kwargs-only Protocol signatures legitimately need 5+ args; the
    # max-args lint rule isn't a useful warning for these.
    # pylint: disable=too-few-public-methods,too-many-arguments,too-many-positional-arguments

    def write_prompt(
        self,
        *,
        run_dir: Path,
        iteration: int,
        candidate_idx: int,
        attempt: int,
        trace: PromptTrace,
    ) -> None:
        """Persist a :class:`PromptTrace` at the canonical per-attempt path."""
        path = (
            self._attempt_dir(run_dir, iteration, candidate_idx, attempt)
            / "prompt.json"
        )
        self._atomic_write_json(path, trace.model_dump_json(indent=2))

    def write_response(
        self,
        *,
        run_dir: Path,
        iteration: int,
        candidate_idx: int,
        attempt: int,
        trace: ResponseTrace,
    ) -> None:
        """Persist a :class:`ResponseTrace` alongside the matching prompt."""
        path = (
            self._attempt_dir(run_dir, iteration, candidate_idx, attempt)
            / "response.json"
        )
        self._atomic_write_json(path, trace.model_dump_json(indent=2))

    def write_reasoning(
        self,
        *,
        run_dir: Path,
        iteration: int,
        candidate_idx: int,
        attempt: int,
        trace: ReasoningTrace,
    ) -> None:
        """Persist provider-separated reasoning (only when the model emits it)."""
        path = (
            self._attempt_dir(run_dir, iteration, candidate_idx, attempt)
            / "reasoning.json"
        )
        self._atomic_write_json(path, trace.model_dump_json(indent=2))

    def write_candidate_result(
        self,
        *,
        run_dir: Path,
        result: CandidateResult,
    ) -> None:
        """Persist a :class:`CandidateResult` at ``iter_<n>/cand_<m>/result.json``."""
        path = (
            self._cand_dir(
                run_dir, result.candidate.iteration, result.candidate.candidate_idx
            )
            / "result.json"
        )
        self._atomic_write_json(path, result.model_dump_json(indent=2))

    def write_evolution_history(
        self,
        *,
        run_dir: Path,
        results: list[CandidateResult],
    ) -> None:
        """Persist the cumulative cross-iteration list as a JSON array."""
        path = self._lero_root(run_dir) / "evolution_history.json"
        # Tiny wrapper so the file is a JSON array of CandidateResult.
        body = (
            "[\n" + ",\n".join(r.model_dump_json(indent=2) for r in results) + "\n]\n"
        )
        self._atomic_write_json(path, body)

    def write_summary(
        self,
        *,
        run_dir: Path,
        summary: LeroRunSummary,
    ) -> None:
        """Persist the top-level :class:`LeroRunSummary` aggregate."""
        path = self._lero_root(run_dir) / "final_summary.json"
        self._atomic_write_json(path, summary.model_dump_json(indent=2))

    # ── path helpers ──────────────────────────────────────────────────

    @staticmethod
    def _lero_root(run_dir: Path) -> Path:
        return run_dir / "output" / "lero"

    @classmethod
    def _iter_dir(cls, run_dir: Path, iteration: int) -> Path:
        return cls._lero_root(run_dir) / f"iter_{iteration}"

    @classmethod
    def _cand_dir(cls, run_dir: Path, iteration: int, candidate_idx: int) -> Path:
        return cls._iter_dir(run_dir, iteration) / f"cand_{candidate_idx}"

    @classmethod
    def _attempt_dir(
        cls, run_dir: Path, iteration: int, candidate_idx: int, attempt: int
    ) -> Path:
        return cls._cand_dir(run_dir, iteration, candidate_idx) / f"attempt_{attempt}"

    # ── atomic write ──────────────────────────────────────────────────

    @staticmethod
    def _atomic_write_json(path: Path, body: str) -> None:
        """Write-tempfile-then-rename. POSIX atomic for same-volume renames.

        Idempotent: if the file already exists with identical bytes,
        skip the write entirely (saves the rename's syscall pair on
        resume paths that re-process unchanged candidates).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        # Skip write when payload is byte-identical (idempotent resume).
        if path.is_file():
            try:
                if path.read_text(encoding="utf-8") == body:
                    return
            except OSError:
                # Race with another writer; fall through to overwrite.
                pass
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(body, encoding="utf-8")
        os.replace(tmp, path)
