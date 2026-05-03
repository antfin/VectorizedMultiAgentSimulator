"""v5 inner loop — S3b-local-style iterative refinement, but with
v5 weighted fitness, best+worst feedback, and a tried-and-failed
registry that persists across iterations.

This module reuses ``LeroLoop`` for prompt building and per-candidate
evaluation. The iteration logic, fitness, feedback, and registry are
v5-specific.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..codegen import CandidateCode
from ..inner_llm import CandidateGenerationFailed, InnerLLM
from ..loop import LeroLoop, _derive_seed
from ..meta.v4_analyzer import classify_shape
from .feedback import build_v5_inner_feedback
from .inner_fitness import m6_slope as m6_slope_fn, weighted_fitness
from .registry import Registry, RegistryEntry

_log = logging.getLogger("rendezvous.lero.v5.inner")


# ── Output container ─────────────────────────────────────────────


@dataclass
class CandidateOutcome:
    candidate: CandidateCode
    metrics: Dict[str, float]
    fitness: float
    shape: str
    iter_idx: int


@dataclass
class InnerResult:
    best: Optional[CandidateOutcome]
    worst: Optional[CandidateOutcome]
    all_outcomes: List[CandidateOutcome] = field(default_factory=list)
    registry: Registry = field(default_factory=Registry)
    did_stagnate: bool = False
    n_iters_run: int = 0


# ── Helpers ─────────────────────────────────────────────────────


def _short_handle(cand: CandidateCode, iter_idx: int, rank: int) -> str:
    src = (cand.obs_source or "") + " " + (cand.reward_source or "")
    by_lines = [ln.strip() for ln in src.splitlines()
                if ln.strip().startswith("def ") or "=" in ln]
    if by_lines:
        snippet = by_lines[0][:60].replace("def ", "").replace("(", "_").rstrip()
        snippet = "".join(c if c.isalnum() or c == "_" else "_" for c in snippet)
        return f"i{iter_idx}_r{rank}_{snippet[:24]}"
    return f"i{iter_idx}_r{rank}_anon"


def _summarize_outcome(metrics: Dict[str, float], shape: str) -> str:
    m1 = float(metrics.get("M1_success_rate", 0.0))
    m6 = float(metrics.get("M6_coverage_progress", 0.0))
    if shape == "monotonic_rise":
        return f"stable rise to M1={m1:.3f}; M6 ended at {m6:.3f}"
    if shape == "peak_collapse":
        return f"peaked then collapsed; final M1={m1:.3f}"
    if shape == "flat_zero":
        return f"never escaped flat-zero; M6={m6:.3f}"
    return f"shape={shape}; M1={m1:.3f}; M6={m6:.3f}"


def _build_metrics_history(
    metrics: Dict[str, float],
) -> List[Dict[str, float]]:
    """Single-point history fallback when per-iter trajectory is unavailable.
    classify_shape returns flat_zero for single-point histories with
    M1<0.02, monotonic_rise otherwise — both reasonable defaults at
    1M-eval where peak ≈ final by construction.
    """
    return [{
        "M1": float(metrics.get("M1_success_rate", 0.0)),
        "M6": float(metrics.get("M6_coverage_progress", 0.0)),
        "frame": 0,
    }]


# ── Main loop ───────────────────────────────────────────────────


def run_inner_loop(
    base_loop: LeroLoop,
    metaprompt_version: str,
    n_iterations: int,
    n_candidates_per_iter: int,
    seed: int,
    output_dir: Path,
    task_overrides: Optional[Dict[str, Any]] = None,
    algorithm: str = "mappo",
    prior_registry: Optional[Registry] = None,
    pre_eval_validator: Optional[
        "Callable[[CandidateCode], Optional[List[str]]]"
    ] = None,
) -> InnerResult:
    """Run a v5 inner loop.

    `base_loop` is a LeroLoop configured with the right eval_frames /
    LLM client / etc. We override its `prompt_loader.template_dir`
    by re-instantiating PromptLoader with `metaprompt_version` (which
    must resolve under the loader's _PROMPTS_DIR — outer loop is
    responsible for setting that up).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Repoint the loop's prompt loader at this metaprompt version
    from ..prompts.loader import PromptLoader
    base_loop.prompt_loader = PromptLoader(version=metaprompt_version)

    messages = base_loop._build_initial_messages(task_overrides)

    registry = prior_registry or Registry()
    all_outcomes: List[CandidateOutcome] = []

    inner_llm = InnerLLM(
        base_loop.llm,
        evolve_reward=base_loop.lero.evolve_reward,
        evolve_observation=base_loop.lero.evolve_observation,
        use_structured=False,
    )

    for iter_idx in range(n_iterations):
        iter_dir = output_dir / f"iter_{iter_idx}"
        iter_dir.mkdir(exist_ok=True)
        _log.info("--- v5 inner iter %d/%d ---", iter_idx + 1, n_iterations)

        # 1. Generate N candidates
        candidates: List[CandidateCode] = []
        for c_idx in range(n_candidates_per_iter):
            seed_base = _derive_seed(
                run_id=output_dir.name,
                seed=seed,
                iteration=iter_idx,
                candidate_idx=c_idx,
                level="v5_inner",
            )
            try:
                cand = inner_llm.generate(messages, seed_base=seed_base)
                candidates.append(cand)
            except CandidateGenerationFailed as e:
                _log.warning("v5 inner gen failed cand=%d/%d: %s",
                             c_idx + 1, n_candidates_per_iter, e)

        if not candidates:
            _log.warning("v5 inner iter %d: no valid candidates", iter_idx)
            continue

        # 2a. Save full candidate source code per file. Restored from
        # v3 loop.py (which saved candidate_{j}_obs.py / _reward.py)
        # because v5 inner_loop dropped this and we lost the ability
        # to audit candidates after the run.
        for j, cand in enumerate(candidates):
            if cand.obs_source:
                (iter_dir / f"candidate_{j}_obs.py").write_text(cand.obs_source)
            if cand.reward_source:
                (iter_dir / f"candidate_{j}_reward.py").write_text(
                    cand.reward_source
                )

        # 2b. Evaluate each candidate. v9.1 §2.1: optional pre-eval
        # validator AST-checks mandatory_features (e.g. role_one_hot)
        # BEFORE training. Fast-fails bad candidates with synthetic
        # failure metrics, saving ~9 min training per rejected one.
        evaluated: List[Tuple[CandidateCode, Dict[str, float]]] = []
        for j, cand in enumerate(candidates):
            if pre_eval_validator is not None:
                issues = pre_eval_validator(cand) or []
                if issues:
                    _log.warning(
                        "v5 inner cand=%d REJECTED pre-eval "
                        "(skipping training): %s",
                        j, issues,
                    )
                    (iter_dir / f"candidate_{j}_rejected.json").write_text(
                        json.dumps({
                            "issues": issues,
                            "reason": "pre_eval_validator",
                        }, indent=2)
                    )
                    # Synthetic failure metrics (M1=0, M6=0). The scoring
                    # step assigns fitness=-99 via _build_metrics_history.
                    m_fail = {
                        "M1_success_rate": 0.0,
                        "M6_coverage_progress": 0.0,
                        "M2_avg_return": -99.0,
                        "M3_avg_steps": 0.0,
                        "M4_avg_collisions": 0.0,
                        "_pre_eval_rejected": True,
                        "_pre_eval_issues": "; ".join(issues),
                    }
                    evaluated.append((cand, m_fail))
                    continue
            try:
                m = base_loop._evaluate_candidate(
                    cand, task_overrides, algorithm, seed,
                    iter_dir=iter_dir, candidate_idx=j,
                )
                evaluated.append((cand, m))
            except Exception as e:  # noqa: BLE001
                _log.error("v5 inner eval failed cand=%d: %s", j, e)

        if not evaluated:
            _log.warning("v5 inner iter %d: all evals failed", iter_idx)
            continue

        # 3. Score: weighted fitness over (metrics, shape, m6_slope)
        scored: List[Tuple[CandidateCode, Dict, float, str]] = []
        for cand, m in evaluated:
            history = _build_metrics_history(m)
            shape = classify_shape(history)
            slope = m6_slope_fn(history)
            fit = weighted_fitness(m, shape_tag=shape, m6_slope=slope)
            scored.append((cand, m, fit, shape))

        scored.sort(key=lambda x: x[2], reverse=True)

        # 4. Record into registry + global outcomes
        for rank, (cand, m, fit, shape) in enumerate(scored, 1):
            entry = RegistryEntry(
                iter_idx=iter_idx,
                handle=_short_handle(cand, iter_idx, rank),
                summary=_summarize_outcome(m, shape),
                fitness=fit,
                M1=float(m.get("M1_success_rate", 0.0)),
                shape=shape,
                code_excerpt=(cand.obs_source or cand.reward_source or "")[:1500],
            )
            registry.add(entry)
            all_outcomes.append(CandidateOutcome(
                candidate=cand, metrics=m, fitness=fit,
                shape=shape, iter_idx=iter_idx,
            ))

        best_fit_this_iter = scored[0][2]
        registry.record_iter_best_fitness(best_fit_this_iter)

        _log.info(
            "v5 inner iter %d: best fitness=%+.3f shape=%s M1=%.3f",
            iter_idx, best_fit_this_iter,
            scored[0][3], scored[0][1].get("M1_success_rate", 0.0),
        )

        # 5. Stagnation check + feedback for next iter
        pivot = registry.stagnated(window=2, eps=0.05)
        if pivot:
            _log.warning("v5 inner iter %d: STAGNATION → pivot prompt",
                         iter_idx)

        # 6. Build feedback + sliding-window message update
        feedback = build_v5_inner_feedback(
            candidates_with_metrics=scored,
            registry=registry,
            iter_idx=iter_idx,
            n_next_candidates=n_candidates_per_iter,
            pivot=pivot,
        )
        (iter_dir / "feedback.txt").write_text(feedback)

        messages = [
            messages[0],  # system
            messages[1],  # initial user prompt
            {"role": "assistant", "content": scored[0][0].source},
            {"role": "user", "content": feedback},
        ]

    if not all_outcomes:
        return InnerResult(best=None, worst=None, registry=registry,
                           did_stagnate=False, n_iters_run=0)

    all_outcomes.sort(key=lambda o: o.fitness, reverse=True)
    return InnerResult(
        best=all_outcomes[0],
        worst=all_outcomes[-1],
        all_outcomes=all_outcomes,
        registry=registry,
        did_stagnate=registry.stagnated(window=2, eps=0.05),
        n_iters_run=len(registry.fitness_trajectory),
    )
