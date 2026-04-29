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
import os
import pickle
import shutil
import time
from dataclasses import dataclass, field
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
class V5Checkpoint:
    """Persisted state of an in-progress v5 run.

    Saved after each phase boundary so a killed run can resume cleanly:
      - after inner of outer_idx finishes  → outer_idx_completed=k
      - after meta-refine for outer_idx=k → refine_completed_for=k
                                            (next outer's prompt dir ready)
      - after deep-train                   → deep_train_done=True
    """
    schema_version: int = 1
    seed: int = 0
    outer_idx_completed: int = -1     # last outer whose inner finished
    refine_completed_for: int = -1    # last outer whose meta-refine ran
    outer_registry: Registry = field(default_factory=Registry)
    iter_records: List["OuterIterRecord"] = field(default_factory=list)
    current_prompt_name: str = ""     # metaprompt to use NEXT
    current_prompt_dir_str: str = ""  # absolute path string
    deep_train_done: bool = False
    final_metrics: Optional[Dict[str, Any]] = None
    elapsed_s_so_far: float = 0.0


_CHECKPOINT_BASENAME = "_v5_checkpoint.pkl"


def _checkpoint_path(output_dir: Path) -> Path:
    return Path(output_dir) / _CHECKPOINT_BASENAME


def _save_checkpoint(output_dir: Path, ckpt: V5Checkpoint) -> None:
    """Atomic save: write to .tmp, then rename."""
    path = _checkpoint_path(output_dir)
    tmp = path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(ckpt, f)
    os.replace(tmp, path)
    _log.info(
        "v5 checkpoint saved: outer_done=%d refine_done=%d deep=%s",
        ckpt.outer_idx_completed,
        ckpt.refine_completed_for,
        ckpt.deep_train_done,
    )


def _load_checkpoint(output_dir: Path) -> Optional[V5Checkpoint]:
    path = _checkpoint_path(output_dir)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    _log.info(
        "v5 checkpoint loaded: outer_done=%d refine_done=%d deep=%s",
        ckpt.outer_idx_completed,
        ckpt.refine_completed_for,
        ckpt.deep_train_done,
    )
    return ckpt


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
    resume: bool = False,
) -> Dict[str, Any]:
    """Top-level v5 entry. Returns dict with final metrics + provenance.

    With ``resume=True``, picks up from ``_v5_checkpoint.pkl`` in
    output_dir if present. Phase boundaries that get checkpointed:
      - after each inner loop completes
      - after each meta-refine completes
      - after deep-train completes
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Materialize prompts under the run output dir; redirect PromptLoader
    # to look there. This MUST happen before any PromptLoader is built.
    prompts_root = output_dir / "prompts"
    _redirect_prompt_loader(prompts_root)

    from ..prompts import loader as _l_mod
    _orig_prompts_dir = Path(_l_mod.__file__).parent
    base_src = _orig_prompts_dir / cfg.base_prompt_version
    if not base_src.exists():
        raise FileNotFoundError(
            f"base prompt version not found: {base_src}"
        )

    # Resume-or-init checkpoint state
    ckpt: Optional[V5Checkpoint] = None
    if resume:
        ckpt = _load_checkpoint(output_dir)
    if ckpt is None:
        seed_prompt_name = f"v5_outer_0_seed{seed}"
        seed_prompt_dir = prompts_root / seed_prompt_name
        if seed_prompt_dir.exists():
            shutil.rmtree(seed_prompt_dir)
        shutil.copytree(base_src, seed_prompt_dir)
        ckpt = V5Checkpoint(
            seed=seed,
            current_prompt_name=seed_prompt_name,
            current_prompt_dir_str=str(seed_prompt_dir.resolve()),
        )
        _save_checkpoint(output_dir, ckpt)
    else:
        _log.info(
            "v5 RESUME — skipping outers ≤ %d", ckpt.outer_idx_completed
        )

    outer_registry = ckpt.outer_registry
    iter_records = ckpt.iter_records
    current_prompt_name = ckpt.current_prompt_name
    current_prompt_dir = Path(ckpt.current_prompt_dir_str)

    t_start = time.monotonic()

    for outer_idx in range(cfg.n_outer):
        if outer_idx <= ckpt.outer_idx_completed:
            _log.info("=== v5 OUTER ITER %d/%d (skipped — already done) ===",
                       outer_idx + 1, cfg.n_outer)
            continue

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

        # CHECKPOINT — after inner finishes (the long phase). Resume
        # from here means: meta-refine + next outer iter remain to do.
        ckpt.outer_idx_completed = outer_idx
        ckpt.elapsed_s_so_far += time.monotonic() - t_start
        t_start = time.monotonic()
        _save_checkpoint(output_dir, ckpt)

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
        # CHECKPOINT — after meta-refine. Resume from here means: next
        # outer iter's inner loop is what restarts.
        ckpt.refine_completed_for = outer_idx
        ckpt.current_prompt_name = current_prompt_name
        ckpt.current_prompt_dir_str = str(current_prompt_dir.resolve())
        _save_checkpoint(output_dir, ckpt)

    # If a previous run already finished deep-train, return cached.
    if ckpt.deep_train_done and ckpt.final_metrics is not None:
        _log.info("v5 RESUME — deep-train already complete, returning cached "
                  "final metrics.")
        return ckpt.final_metrics

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

    # CHECKPOINT — deep-train done, run is complete.
    ckpt.deep_train_done = True
    ckpt.final_metrics = final_metrics
    ckpt.elapsed_s_so_far += time.monotonic() - t_start
    _save_checkpoint(output_dir, ckpt)

    _log.info(
        "=== v5 DONE === elapsed=%.0fs final_M1=%.3f outer_traj=%s",
        elapsed,
        final_metrics.get("M1_success_rate", 0),
        [round(f, 3) for f in outer_registry.fitness_trajectory],
    )
    return final_metrics
