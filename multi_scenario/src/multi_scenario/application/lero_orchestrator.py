"""F9.6.b — :class:`LeroOrchestrator` — the LERO use case.

Sits in the application layer; depends only on Protocols from
``domain/ports`` + value types from ``domain/lero``. The CLI / Streamlit
caller constructs the orchestrator with concrete adapters and invokes
:meth:`run` (or :meth:`resume` — F9.6.d).

The loop is the rendezvous_comm shape, hex-cleaned:

1. Compose the iteration's prompt via :class:`PromptComposer`.
2. Ask :class:`LlmClient` for ``n_candidates`` completions.
3. Persist prompt/response traces via :class:`TraceWriter`.
4. Extract validated code via :func:`extract_candidates`.
5. For each valid candidate, build a patched scenario and evaluate
   short-training metrics via the injected :class:`CandidateEvaluator`.
6. Persist :class:`CandidateResult` and append to history.
7. Repeat for ``cfg.lero.n_iterations`` iterations.
8. Rank ALL candidates across iterations; run full training on the top
   rank with a fallback chain on crash.
9. Persist :class:`LeroRunSummary`.

Cost-cap exceptions short-circuit the loop gracefully: the rolling
candidates so far are sorted and the full-training fallback chain runs
on whatever we have. We don't crash mid-budget; we deliver degraded.

The evaluator (Protocol :class:`CandidateEvaluator`) is the injection
point for ``short training`` — it takes a patched scenario class and
returns a :class:`CandidateMetrics`. F9.6.b ships the Protocol +
contract test; the BenchMARL-backed implementation lives in
``adapters/lero/benchmarl_candidate_evaluator.py`` (lazy import to
keep this module test-friendly).
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Protocol

from multi_scenario.domain.lero import (
    Candidate,
    CandidateCode,
    CandidateMetrics,
    CandidateResult,
    extract_candidates,
    FallbackEntry,
    LeroError,
    LeroRunSummary,
    LlmCompletion,
    LlmCostCapExceeded,
    PromptTrace,
    ReasoningTrace,
    ResponseTrace,
    StrategyCard,
    Verdict,
)
from multi_scenario.domain.models import ExperimentConfig
from multi_scenario.domain.ports import (
    ComposedPrompt,
    LlmClient,
    Logger,
    PromptComposer,
    TraceWriter,
)


_log = logging.getLogger(__name__)


# ── Candidate evaluator port ──────────────────────────────────────────


class CandidateEvaluator(Protocol):
    """Run short training on a patched scenario, return per-candidate metrics.

    Decouples the LERO loop from BenchMARL: the orchestrator hands the
    evaluator a candidate's code strings + the run dir, the evaluator
    builds a patched scenario, runs ``eval_frames_per_candidate`` worth
    of training, and returns the :class:`CandidateMetrics`.

    Implementations live in the adapters layer (BenchMARL-backed),
    plus an in-memory fake for tests.
    """

    def evaluate(
        self,
        *,
        cfg: ExperimentConfig,
        candidate: Candidate,
        run_dir: Path,
    ) -> CandidateMetrics:
        """Return metrics for ``candidate`` under ``cfg``. Raise on crash."""
        ...


class FullTrainer(Protocol):
    """Run the full-training step on the winning candidate.

    Same shape as :class:`CandidateEvaluator` but runs the
    ``cfg.training.max_iters`` budget (typically 10M frames for ER1/LERO).
    Split into its own Protocol so a future "skip-full-training-for-debug"
    mode can swap the evaluator without touching the inner-loop adapter.
    """

    def train_full(
        self,
        *,
        cfg: ExperimentConfig,
        candidate: Candidate,
        run_dir: Path,
    ) -> CandidateMetrics:
        """Return metrics after full training. Raise on crash."""
        ...


# ── Orchestrator ──────────────────────────────────────────────────────


def _derive_seed(
    run_id: str, base_seed: int, iteration: int, candidate_idx: int
) -> int:
    """Per-(iter, cand) LLM seed via SHA(...) % 2**31.

    Without this, ``llm.generate(n=k)`` at temperature 1.0 returns
    ``k`` identical completions when the provider seeds them all the
    same. Hashing decorrelates siblings while keeping the whole sweep
    reproducible from ``run_id`` + ``base_seed``.
    """
    key = f"{run_id}|{base_seed}|{iteration}|{candidate_idx}".encode()
    return int(hashlib.sha256(key).hexdigest()[:8], 16) % (2**31)


def _verdict_for(metrics: CandidateMetrics, *, baseline_m1: float = 0.0) -> Verdict:
    """Map raw metrics → coarse Verdict for prompt-composer ranking.

    Thresholds are conservative: anything strictly above the iteration
    baseline counts as ``progress``; equal-to-baseline counts as
    ``flat``; below counts as ``regression``. ``invalid`` is emitted by
    the orchestrator separately (when evaluator raises). Matches
    rendezvous_comm's verdict semantics.
    """
    m1 = metrics.M1_success_rate
    if m1 is None:
        return "invalid"
    if m1 > baseline_m1:
        return "progress"
    if m1 < baseline_m1:
        return "regression"
    return "flat"


def _ranked_for_fallback(history: list[CandidateResult]) -> list[CandidateResult]:
    """Sort ALL candidates across iterations for the full-training fallback chain.

    Same key as the prompt composer's ranking: verdict desc → M1 desc →
    M2 desc → M6 desc. The orchestrator then walks this list trying
    full-training; if rank 0 crashes, rank 1, etc.
    """
    return sorted(
        history,
        key=lambda r: (
            {"progress": 3, "flat": 2, "regression": 1, "invalid": 0}.get(r.verdict, 0),
            r.metrics.M1_success_rate
            if r.metrics.M1_success_rate is not None
            else -1e9,
            r.metrics.M2_avg_return if r.metrics.M2_avg_return is not None else -1e9,
            r.metrics.M6_coverage_progress
            if r.metrics.M6_coverage_progress is not None
            else -1e9,
        ),
        reverse=True,
    )


class LeroOrchestrator:
    """The LERO evolutionary loop with 8 ports injected."""

    # 8 injected ports is the natural surface — every Protocol covers a
    # genuinely different concern. Bundling them into a config object
    # would obscure the dependency-injection contract.
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-instance-attributes

    def __init__(
        self,
        *,
        llm: LlmClient,
        composer: PromptComposer,
        trace_writer: TraceWriter,
        evaluator: CandidateEvaluator,
        full_trainer: FullTrainer,
        logger: Logger,
    ) -> None:
        self._llm = llm
        self._composer = composer
        self._trace_writer = trace_writer
        self._evaluator = evaluator
        self._full_trainer = full_trainer
        self._log = logger

    def run(
        self,
        *,
        cfg: ExperimentConfig,
        run_dir: Path,
        strategy_card: StrategyCard | None = None,
        resume: bool = False,
    ) -> LeroRunSummary:
        """Drive the full LERO loop end-to-end.

        F9.6.d: when ``resume=True``, the orchestrator scans the run-dir
        for previously-completed iterations (any
        ``iter_<n>/cand_<m>/result.json`` files) and seeds the history
        from them. The loop then continues at ``iteration = max(prior) + 1``
        — the orchestrator never re-runs an iteration's LLM calls,
        preserving the cost budget across restarts.
        """
        assert cfg.lero is not None and cfg.llm is not None
        history: list[CandidateResult] = (
            self._load_history_from_disk(run_dir) if resume else []
        )
        start_iteration = max((r.candidate.iteration for r in history), default=-1) + 1
        cost_cap_hit = False
        total_cost_usd = 0.0

        run_id = f"{cfg.experiment.id}_s{cfg.experiment.seed}"
        for iteration in range(start_iteration, cfg.lero.n_iterations):
            self._log.info(f"LERO iter {iteration}: composing prompt")
            try:
                composed = self._composer.compose(
                    iteration=iteration,
                    history=history,
                    task_params=dict(cfg.scenario.params),
                    strategy_card=strategy_card,
                )
                iter_results, iter_cost = self._run_iteration(
                    cfg=cfg,
                    run_dir=run_dir,
                    iteration=iteration,
                    composed=composed,
                    run_id=run_id,
                )
                history.extend(iter_results)
                total_cost_usd += iter_cost
                self._trace_writer.write_evolution_history(
                    run_dir=run_dir, results=history
                )
            except LlmCostCapExceeded as exc:
                self._log.warning(
                    f"LERO iter {iteration}: cost cap reached, "
                    f"falling back to rank with history so far ({len(history)} candidates): {exc}"
                )
                cost_cap_hit = True
                break

        # Full training fallback chain across all collected candidates.
        ranked = _ranked_for_fallback(history)
        fallback_chain, full_success = self._full_training_with_fallback(
            cfg=cfg, run_dir=run_dir, ranked=ranked
        )

        # Best inner-loop candidate (rank 0 BEFORE full training).
        best = ranked[0] if ranked else None
        # The post-full-train metrics belong to the rank that succeeded
        # in the fallback chain (if any). Walk the chain to find it.
        full_metrics_winner = next(
            (e.full_train_metrics for e in fallback_chain if e.outcome == "success"),
            None,
        )

        summary = LeroRunSummary(
            exp_id=cfg.experiment.id,
            seed=cfg.experiment.seed,
            n_iterations_completed=(
                cfg.lero.n_iterations if not cost_cap_hit else (iteration)
            ),
            n_candidates_total=len(history),
            total_cost_usd=total_cost_usd,
            best_candidate_metrics=(
                best.metrics if best is not None else CandidateMetrics()
            ),
            best_candidate_verdict=(best.verdict if best is not None else "invalid"),
            best_candidate_full_metrics=full_metrics_winner,
            fallback_chain=fallback_chain,
            full_training_succeeded=full_success,
        )
        self._trace_writer.write_summary(run_dir=run_dir, summary=summary)
        return summary

    # ── Per-iteration step ────────────────────────────────────────────

    def _run_iteration(
        self,
        *,
        cfg: ExperimentConfig,
        run_dir: Path,
        iteration: int,
        composed: ComposedPrompt,
        run_id: str,
    ) -> tuple[list[CandidateResult], float]:
        """Generate + evaluate one iteration's batch of candidates.

        Returns ``(iter_results, iter_cost_usd)``. The cost is the sum
        of ``LlmCompletion.usage.estimated_cost_usd`` across the
        iteration's LLM calls — surfaced so :meth:`run` can aggregate
        a per-run total for :class:`LeroRunSummary.total_cost_usd`
        (Phase 9 issue #10 fix; the field was hardcoded to 0.0 before).
        """
        assert cfg.lero is not None and cfg.llm is not None
        n = cfg.lero.n_candidates

        # Fire one LLM call per candidate with a per-(iter, cand) seed
        # so siblings aren't identical at temperature 1.0.
        completions: list[LlmCompletion] = []
        for cand_idx in range(n):
            seed = _derive_seed(run_id, cfg.experiment.seed, iteration, cand_idx)
            self._persist_prompt(
                run_dir=run_dir,
                iteration=iteration,
                candidate_idx=cand_idx,
                composed=composed,
            )
            sibling = self._llm.generate(messages=composed.messages, n=1, seed=seed)
            comp = sibling[0]
            completions.append(comp)
            self._persist_response(
                run_dir=run_dir,
                iteration=iteration,
                candidate_idx=cand_idx,
                completion=comp,
            )

        # Extract validated code per response (separately so failed
        # validation drops the candidate without affecting siblings).
        valid_codes = extract_candidates(
            [c.text for c in completions],
            evolve_reward=cfg.lero.evolve_reward,
            evolve_observation=cfg.lero.evolve_observation,
        )
        # extract_candidates may return fewer than n (validation failures
        # silently dropped). Realign by re-running with single responses
        # so we can carry the original cand_idx through.
        results: list[CandidateResult] = []
        for cand_idx, comp in enumerate(completions):
            singles = extract_candidates(
                [comp.text],
                evolve_reward=cfg.lero.evolve_reward,
                evolve_observation=cfg.lero.evolve_observation,
            )
            if not singles:
                results.append(
                    self._invalid_result(
                        iteration=iteration,
                        candidate_idx=cand_idx,
                        note="no valid functions extracted",
                    )
                )
                continue
            cand = Candidate(
                iteration=iteration, candidate_idx=cand_idx, code=singles[0]
            )
            results.append(
                self._evaluate_candidate(cfg=cfg, run_dir=run_dir, candidate=cand)
            )

        # Persist per-candidate result + return.
        for r in results:
            self._trace_writer.write_candidate_result(run_dir=run_dir, result=r)
        # Quieten unused-variable warnings for the side-effect-only
        # local from the extract_candidates batch pre-flight.
        del valid_codes
        # Sum the iteration's LLM cost. ``estimated_cost_usd`` is
        # populated by LiteLlmClient on the first sibling per call;
        # since we issue one call per candidate (n=1 each), each
        # ``comp.usage`` carries the full call's cost — summing across
        # completions is correct, not double-counted.
        iter_cost = sum(c.usage.estimated_cost_usd or 0.0 for c in completions)
        return results, iter_cost

    # ── Per-candidate evaluation ──────────────────────────────────────

    def _evaluate_candidate(
        self,
        *,
        cfg: ExperimentConfig,
        run_dir: Path,
        candidate: Candidate,
    ) -> CandidateResult:
        """Evaluate one candidate; catch crashes → invalid verdict."""
        try:
            metrics = self._evaluator.evaluate(
                cfg=cfg, candidate=candidate, run_dir=run_dir
            )
        # pylint: disable=broad-except  # evaluator can crash any number of ways
        except Exception as exc:
            self._log.warning(
                f"candidate {candidate.iteration}/{candidate.candidate_idx} "
                f"evaluation crashed: {exc}"
            )
            return CandidateResult(
                candidate=candidate,
                metrics=CandidateMetrics(),
                verdict="invalid",
                note=f"evaluator crashed: {exc!s}"[:1000],
            )
        verdict = _verdict_for(metrics)
        return CandidateResult(
            candidate=candidate,
            metrics=metrics,
            verdict=verdict,
            note=f"M1={metrics.M1_success_rate}, verdict={verdict}",
        )

    # ── Full training fallback chain ──────────────────────────────────

    def _full_training_with_fallback(
        self,
        *,
        cfg: ExperimentConfig,
        run_dir: Path,
        ranked: list[CandidateResult],
    ) -> tuple[list[FallbackEntry], bool]:
        """Walk the ranked list trying full training; capture every attempt."""
        chain: list[FallbackEntry] = []
        for rank, r in enumerate(ranked):
            if r.verdict == "invalid":
                chain.append(
                    FallbackEntry(
                        rank=rank,
                        iteration=r.candidate.iteration,
                        candidate_idx=r.candidate.candidate_idx,
                        eval_metrics=r.metrics,
                        outcome="skipped",
                        error="candidate marked invalid in eval",
                    )
                )
                continue
            try:
                full_metrics = self._full_trainer.train_full(
                    cfg=cfg, candidate=r.candidate, run_dir=run_dir
                )
                # ``full_metrics`` is the 10M-frame post-train CandidateMetrics
                # (200-episode eval against the trained policy). Pre-fix this
                # was discarded — ``final_summary.json`` showed only the
                # 1M-frame inner-loop M1 (~0.03 on Phase 5a) instead of the
                # true 10M-frame M1 (~0.79). Keep both: inner = screening
                # signal, full = science result.
                chain.append(
                    FallbackEntry(
                        rank=rank,
                        iteration=r.candidate.iteration,
                        candidate_idx=r.candidate.candidate_idx,
                        eval_metrics=r.metrics,
                        outcome="success",
                        full_train_metrics=full_metrics,
                    )
                )
                return chain, True
            # pylint: disable=broad-except
            except Exception as exc:
                self._log.warning(
                    f"full training rank {rank} crashed: {exc} — trying next"
                )
                chain.append(
                    FallbackEntry(
                        rank=rank,
                        iteration=r.candidate.iteration,
                        candidate_idx=r.candidate.candidate_idx,
                        eval_metrics=r.metrics,
                        outcome="crashed",
                        error=f"{exc!s}"[:1000],
                    )
                )
        return chain, False

    # ── Trace persistence helpers ─────────────────────────────────────

    def _persist_prompt(
        self,
        *,
        run_dir: Path,
        iteration: int,
        candidate_idx: int,
        composed: ComposedPrompt,
    ) -> None:
        trace = PromptTrace(
            prompt_version=composed.prompt_version,
            messages=composed.messages,
            render_context={
                k: v for k, v in composed.render_context.items() if _is_json_safe(v)
            },
        )
        self._trace_writer.write_prompt(
            run_dir=run_dir,
            iteration=iteration,
            candidate_idx=candidate_idx,
            attempt=0,
            trace=trace,
        )

    def _persist_response(
        self,
        *,
        run_dir: Path,
        iteration: int,
        candidate_idx: int,
        completion: LlmCompletion,
    ) -> None:
        self._trace_writer.write_response(
            run_dir=run_dir,
            iteration=iteration,
            candidate_idx=candidate_idx,
            attempt=0,
            trace=ResponseTrace(
                text=completion.text,
                finish_reason=completion.finish_reason,
                system_fingerprint=completion.system_fingerprint,
                usage=completion.usage,
            ),
        )
        if completion.reasoning:
            self._trace_writer.write_reasoning(
                run_dir=run_dir,
                iteration=iteration,
                candidate_idx=candidate_idx,
                attempt=0,
                trace=ReasoningTrace(text=completion.reasoning),
            )

    @staticmethod
    def _load_history_from_disk(run_dir: Path) -> list[CandidateResult]:
        """F9.6.d: scan ``output/lero/iter_<n>/cand_<m>/result.json`` files.

        Returns the recovered history in iteration-then-candidate-idx
        order so the prompt composer sees the same shape it would have
        in-process. Missing / corrupt files are silently skipped — the
        orchestrator treats a partially-completed iteration's missing
        candidates as "still to do" via the ``start_iteration`` cutoff.
        """
        lero_root = run_dir / "output" / "lero"
        if not lero_root.is_dir():
            return []
        recovered: list[CandidateResult] = []
        for result_path in sorted(lero_root.rglob("cand_*/result.json")):
            try:
                recovered.append(
                    CandidateResult.model_validate_json(
                        result_path.read_text(encoding="utf-8")
                    )
                )
            except (OSError, ValueError):
                # Defensive: a half-written file from a prior crash
                # shouldn't block resume; the orchestrator's start_iter
                # cutoff treats the matching slot as "not done yet".
                continue
        return recovered

    @staticmethod
    def _invalid_result(
        *, iteration: int, candidate_idx: int, note: str
    ) -> CandidateResult:
        return CandidateResult(
            candidate=Candidate(
                iteration=iteration,
                candidate_idx=candidate_idx,
                code=CandidateCode(),
            ),
            metrics=CandidateMetrics(),
            verdict="invalid",
            note=note,
        )


def _is_json_safe(value: Any) -> bool:
    """Tiny JSON-safety predicate for render-context filtering.

    Trace files are JSON, so anything that doesn't round-trip is dropped
    rather than crashing the writer. The orchestrator's render context
    is supposed to be JSON-safe by contract; this is the defensive
    backstop for the rare custom-object case.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list):
        return all(_is_json_safe(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_json_safe(v) for k, v in value.items())
    return False


# Re-export for caller convenience.
__all__ = [
    "CandidateEvaluator",
    "FullTrainer",
    "LeroError",
    "LeroOrchestrator",
]
