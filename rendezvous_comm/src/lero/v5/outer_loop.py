"""v5 outer loop — N_OUTER metaprompt refinement steps.

For each outer iter:
    1. Run an S3b-local-style inner loop (4 × 3 × eval_frames by default)
    2. Update OuterRegistry with best/worst inner outcome
    3. Check outer stagnation
    4. If not last iter: meta-LLM refines metaprompt slot files

After N_OUTER, deep-train the GLOBAL best inner candidate at full_frames.
"""

from __future__ import annotations

import copy
import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...config import ExperimentSpec
from ..llm_client import LLMClient
from ..loop import LeroLoop
from .inner_loop import InnerResult, run_inner_loop
from .meta_refiner import refine_metaprompt
from .registry import Registry, RegistryEntry

_log = logging.getLogger("rendezvous.lero.v5.outer")


@dataclass
class OuterIterRecord:
    iter_idx: int
    prompt_dir: Path
    inner: InnerResult
    diagnosis: str = ""


@dataclass
class V5OuterConfig:
    n_outer: int
    n_inner_iter: int
    n_inner_candidates_per_iter: int
    eval_frames: int
    full_frames: int
    pivot_eps: float = 0.05
    base_prompt_version: str = "v2_fewshot_modular_v2"
    meta_model: str = "gpt-5.4-mini"
    meta_temperature: float = 1.0
    task_summary: str = ""


def _redirect_prompt_loader(prompts_root: Path) -> None:
    """Same trick as v4: repoint PromptLoader._PROMPTS_DIR at a writable
    location so we can materialize per-outer-iter metaprompts there."""
    from ..prompts import loader as _l
    prompts_root.mkdir(parents=True, exist_ok=True)
    _l._PROMPTS_DIR = prompts_root


def _outer_registry_entry(iter_idx: int, inner: InnerResult,
                          metaprompt_dir: Path) -> RegistryEntry:
    """Map an inner result into a single outer-registry entry."""
    if inner.best is None:
        return RegistryEntry(
            iter_idx=iter_idx,
            handle=f"outer_iter_{iter_idx}_dead",
            summary="inner produced zero valid candidates",
            fitness=-1.0, M1=0.0, shape="flat_zero",
            code_excerpt="",
        )
    diag_path = metaprompt_dir / "_refiner_diagnosis.md"
    diagnosis = diag_path.read_text().strip() if diag_path.exists() else ""
    handle_hint = ""
    if diagnosis:
        first_line = diagnosis.splitlines()[0][:60]
        handle_hint = "_" + "".join(
            c if c.isalnum() else "_" for c in first_line
        )[:40]
    return RegistryEntry(
        iter_idx=iter_idx,
        handle=f"outer_iter_{iter_idx}{handle_hint}",
        summary=(
            f"best inner: M1={inner.best.metrics.get('M1_success_rate', 0):.3f} "
            f"shape={inner.best.shape}; "
            f"fitness trajectory: "
            f"{[round(f, 3) for f in inner.registry.fitness_trajectory]}; "
            f"diagnosis: {diagnosis[:300]}"
        ),
        fitness=inner.best.fitness,
        M1=float(inner.best.metrics.get("M1_success_rate", 0.0)),
        shape=inner.best.shape,
        code_excerpt=(
            inner.best.candidate.obs_source
            or inner.best.candidate.reward_source
            or ""
        )[:1500],
    )


def run_v5_outer_loop(
    spec: ExperimentSpec,
    base_loop: LeroLoop,
    cfg: V5OuterConfig,
    meta_llm: LLMClient,
    output_dir: Path,
    seed: int = 0,
    task_overrides: Optional[Dict[str, Any]] = None,
    algorithm: str = "mappo",
) -> Dict[str, Any]:
    """Top-level v5 entry. Returns dict with final metrics + provenance."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Materialize prompts under the run output dir; redirect PromptLoader
    # to look there. This MUST happen before any PromptLoader is built.
    prompts_root = output_dir / "prompts"
    _redirect_prompt_loader(prompts_root)

    # Copy base prompt version into the writable root so PromptLoader
    # can find it via version=<dirname>.
    from ..prompts import loader as _l_mod  # local mutated module
    _orig_prompts_dir = Path(_l_mod.__file__).parent
    base_src = _orig_prompts_dir / cfg.base_prompt_version
    if not base_src.exists():
        raise FileNotFoundError(
            f"base prompt version not found: {base_src}"
        )
    seed_prompt_name = f"v5_outer_0_seed{seed}"
    seed_prompt_dir = prompts_root / seed_prompt_name
    if seed_prompt_dir.exists():
        shutil.rmtree(seed_prompt_dir)
    shutil.copytree(base_src, seed_prompt_dir)

    outer_registry = Registry()
    iter_records: List[OuterIterRecord] = []
    current_prompt_name = seed_prompt_name
    current_prompt_dir = seed_prompt_dir

    t_start = time.monotonic()

    for outer_idx in range(cfg.n_outer):
        _log.info("=== v5 OUTER ITER %d/%d ===", outer_idx + 1, cfg.n_outer)
        inner_out_dir = output_dir / f"outer_{outer_idx:02d}_inner"
        inner_result = run_inner_loop(
            base_loop=base_loop,
            metaprompt_version=current_prompt_name,
            n_iterations=cfg.n_inner_iter,
            n_candidates_per_iter=cfg.n_inner_candidates_per_iter,
            seed=seed,
            output_dir=inner_out_dir,
            task_overrides=task_overrides,
            algorithm=algorithm,
        )
        iter_records.append(OuterIterRecord(
            iter_idx=outer_idx,
            prompt_dir=current_prompt_dir,
            inner=inner_result,
        ))

        entry = _outer_registry_entry(outer_idx, inner_result,
                                       current_prompt_dir)
        outer_registry.add(entry)
        outer_registry.record_iter_best_fitness(entry.fitness)

        _log.info(
            "v5 outer iter %d: best inner M1=%.3f fitness=%+.3f shape=%s",
            outer_idx, entry.M1, entry.fitness, entry.shape,
        )

        # Refine metaprompt for next iter (skip on the last)
        if outer_idx == cfg.n_outer - 1:
            break
        next_prompt_name = f"v5_outer_{outer_idx + 1}_seed{seed}"
        next_prompt_dir = prompts_root / next_prompt_name
        outer_pivot = outer_registry.stagnated(window=2,
                                                eps=cfg.pivot_eps)
        refine_metaprompt(
            prev_prompt_dir=current_prompt_dir,
            next_prompt_dir=next_prompt_dir,
            outer_registry=outer_registry,
            last_inner=inner_result,
            meta_llm=meta_llm,
            task_summary=cfg.task_summary or spec.description or "",
            pivot=outer_pivot,
        )
        current_prompt_name = next_prompt_name
        current_prompt_dir = next_prompt_dir

    # Pick global best across all outer iters
    candidates_all = []
    for rec in iter_records:
        for o in rec.inner.all_outcomes:
            candidates_all.append((rec.iter_idx, o))
    if not candidates_all:
        _log.error("v5: no valid candidates produced across all outer iters")
        return {"_error": "no valid candidates"}
    candidates_all.sort(key=lambda x: x[1].fitness, reverse=True)
    global_best_outer_idx, global_best = candidates_all[0]

    _log.info(
        "v5: global best from outer_iter=%d fitness=%+.3f M1=%.3f shape=%s",
        global_best_outer_idx, global_best.fitness,
        global_best.metrics.get("M1_success_rate", 0), global_best.shape,
    )

    # Deep-train the winner at full_frames
    deep_dir = output_dir / "deep_train"
    deep_dir.mkdir(exist_ok=True)
    base_loop.output_dir = deep_dir
    base_loop.lero = copy.copy(base_loop.lero)
    base_loop.lero.full_frames = cfg.full_frames
    final_metrics = base_loop._full_training(
        global_best.candidate, task_overrides, algorithm, seed,
    )

    elapsed = time.monotonic() - t_start
    final_metrics["_v5_elapsed_s"] = elapsed
    final_metrics["_v5_global_best_outer_idx"] = global_best_outer_idx
    final_metrics["_v5_global_best_fitness"] = global_best.fitness
    final_metrics["_v5_global_best_M1_inner"] = float(
        global_best.metrics.get("M1_success_rate", 0.0)
    )
    final_metrics["_v5_outer_fitness_trajectory"] = list(
        outer_registry.fitness_trajectory
    )

    (output_dir / "v5_summary.json").write_text(
        json.dumps(final_metrics, indent=2, default=str)
    )

    _log.info(
        "=== v5 DONE === elapsed=%.0fs final_M1=%.3f outer_traj=%s",
        elapsed,
        final_metrics.get("M1_success_rate", 0),
        [round(f, 3) for f in outer_registry.fitness_trajectory],
    )
    return final_metrics
