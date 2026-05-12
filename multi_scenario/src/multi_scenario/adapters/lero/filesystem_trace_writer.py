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

from multi_scenario.adapters.lero.evolution_doc import render_evolution_doc
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
        """Persist a :class:`PromptTrace` at the canonical per-attempt path.

        Also extracts the rendered system / user_initial / user_feedback
        messages into separate markdown files under
        ``output/lero/prompts/iter_<n>/``. These are the human-readable
        counterparts the evolution_doc.md links to (Phase 5).

        Idempotent: per-iter markdown files only get written on the
        first call for that iter (typically iteration 0 / candidate 0
        / attempt 0). Subsequent prompts in the same iter share the
        composed system + initial_user (the composer never varies them
        within an iter), so re-writes are no-ops.
        """
        path = (
            self._attempt_dir(run_dir, iteration, candidate_idx, attempt)
            / "prompt.json"
        )
        self._atomic_write_json(path, trace.model_dump_json(indent=2))
        # Markdown side-files for the evolution doc's relative links.
        self._write_prompt_markdown_files(run_dir, iteration, trace)

    def write_response(
        self,
        *,
        run_dir: Path,
        iteration: int,
        candidate_idx: int,
        attempt: int,
        trace: ResponseTrace,
    ) -> None:
        """Persist a :class:`ResponseTrace` alongside the matching prompt.

        Also writes the raw response text as ``response.md`` under
        ``output/lero/prompts/iter_<n>/cand_<m>/`` so the evolution doc
        can link directly to the LLM output without users having to
        ``jq -r '.text'`` the JSON. (Phase 5.)
        """
        path = (
            self._attempt_dir(run_dir, iteration, candidate_idx, attempt)
            / "response.json"
        )
        self._atomic_write_json(path, trace.model_dump_json(indent=2))
        # Markdown side-file. Only on attempt_0 — retries (rare) overwrite.
        md_path = (
            self._lero_root(run_dir)
            / "prompts"
            / f"iter_{iteration}"
            / f"cand_{candidate_idx}"
            / "response.md"
        )
        self._atomic_write_text(md_path, trace.text)

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
        """Persist a :class:`CandidateResult` at ``iter_<n>/cand_<m>/result.json``.

        Also extracts the candidate's ``obs_source`` / ``reward_source``
        to ``.py`` files under ``output/lero/prompts/iter_<n>/cand_<m>/``
        when present (LLM may emit either / both / neither). These are
        the files the evolution_doc.md links to as the "code" column of
        the per-iter table. (Phase 5.)
        """
        path = (
            self._cand_dir(
                run_dir, result.candidate.iteration, result.candidate.candidate_idx
            )
            / "result.json"
        )
        self._atomic_write_json(path, result.model_dump_json(indent=2))
        # Code side-files. Only emit a file when the candidate actually
        # produced that source — keeps the prompts/ tree free of empty
        # placeholders.
        prompts_cand_dir = (
            self._lero_root(run_dir)
            / "prompts"
            / f"iter_{result.candidate.iteration}"
            / f"cand_{result.candidate.candidate_idx}"
        )
        if result.candidate.code.obs_source:
            self._atomic_write_text(
                prompts_cand_dir / "obs_source.py",
                result.candidate.code.obs_source,
            )
        if result.candidate.code.reward_source:
            self._atomic_write_text(
                prompts_cand_dir / "reward_source.py",
                result.candidate.code.reward_source,
            )

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
        """Persist the top-level :class:`LeroRunSummary` aggregate.

        ALSO writes ``evolution_doc.md`` next to ``final_summary.json``
        — the human-readable narrative with relative links into the
        ``prompts/`` folder built up by the per-prompt / per-response /
        per-result writes above. (Phase 5.)
        """
        path = self._lero_root(run_dir) / "final_summary.json"
        self._atomic_write_json(path, summary.model_dump_json(indent=2))
        # Evolution doc reads from final_summary + the on-disk history.
        # If history.json doesn't exist yet (very rare; orchestrator
        # writes it before summary on the normal path), skip silently.
        history_path = self._lero_root(run_dir) / "evolution_history.json"
        history: list[CandidateResult] = []
        if history_path.is_file():
            # pylint: disable=import-outside-toplevel
            import json as _json

            for d in _json.loads(history_path.read_text(encoding="utf-8")):
                history.append(CandidateResult.model_validate(d))
        doc_md = render_evolution_doc(summary=summary, history=history)
        self._atomic_write_text(self._lero_root(run_dir) / "evolution_doc.md", doc_md)

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

    # ── markdown side-files for the evolution doc ─────────────────────

    @classmethod
    def _write_prompt_markdown_files(
        cls, run_dir: Path, iteration: int, trace: PromptTrace
    ) -> None:
        """Extract system / user_initial / user_feedback messages to .md files.

        :class:`InitialAndFeedbackComposer`'s contract: ``messages``
        starts with a ``system`` message then one or two ``user``
        messages. The first user is always ``user_initial``; on iter > 0
        a second user message carries the feedback block.

        Idempotent via :meth:`_atomic_write_text` — re-writing the same
        bytes is a no-op, so calls from each candidate in the same iter
        don't multiply disk I/O.
        """
        prompts_iter_dir = cls._lero_root(run_dir) / "prompts" / f"iter_{iteration}"
        system_msgs = [m for m in trace.messages if m.get("role") == "system"]
        user_msgs = [m for m in trace.messages if m.get("role") == "user"]
        if system_msgs:
            cls._atomic_write_text(
                prompts_iter_dir / "system.md", system_msgs[0]["content"]
            )
        if user_msgs:
            cls._atomic_write_text(
                prompts_iter_dir / "user_initial.md", user_msgs[0]["content"]
            )
        if len(user_msgs) >= 2:
            cls._atomic_write_text(
                prompts_iter_dir / "user_feedback.md", user_msgs[1]["content"]
            )

    # ── atomic write ──────────────────────────────────────────────────

    @staticmethod
    def _atomic_write_json(path: Path, body: str) -> None:
        """Write-tempfile-then-rename. POSIX atomic for same-volume renames.

        Idempotent: if the file already exists with identical bytes,
        skip the write entirely (saves the rename's syscall pair on
        resume paths that re-process unchanged candidates).
        """
        FilesystemTraceWriter._atomic_write_text(path, body)

    @staticmethod
    def _atomic_write_text(path: Path, body: str) -> None:
        """Write a text blob via temp-then-rename, idempotent on identical bytes.

        Same semantics as :meth:`_atomic_write_json` but content-type-
        agnostic — used for markdown side-files and extracted code
        files. Both paths share this implementation so the atomicity
        guarantees stay consistent.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            try:
                if path.read_text(encoding="utf-8") == body:
                    return
            except OSError:
                pass
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(body, encoding="utf-8")
        os.replace(tmp, path)
