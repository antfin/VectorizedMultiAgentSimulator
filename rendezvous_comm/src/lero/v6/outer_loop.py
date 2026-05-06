"""v6 outer loop — single-strategy, simplicity-first, inner-only validation.

Each outer iter:
  1. Apply prior decision's evolve_observation / evolve_reward flags
     to base_loop.lero
  2. Run inner LERO loop (4 × 3 × 1M, S3b-local style) with current
     metaprompt
  3. Update outer registry
  4. Call meta-strategist → V6MetaDecision (with code-side cross-check
     and policy enforcement)
  5. If found_good → STOP (early stop)
  6. Otherwise: write next metaprompt slot files, update flags for
     next iter, persist checkpoint

NO deep-train inside the loop. The deep_train_winner() callable is
present but commented out — uncomment + call separately once a
found_good candidate exists (see docs/v6_plan.md §4.5).
"""

from __future__ import annotations

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
from ..v5.inner_loop import InnerResult, run_inner_loop
from ..v5.registry import Registry, RegistryEntry
from .decision import V6MetaDecision
from .meta_strategist import decide_and_enforce

_log = logging.getLogger("rendezvous.lero.v6.outer")


@dataclass
class V6Checkpoint:
    schema_version: int = 1
    seed: int = 0
    outer_idx_completed: int = -1  # last outer whose inner finished
    outer_registry: Registry = field(default_factory=Registry)
    iter_records: List["V6OuterIterRecord"] = field(default_factory=list)
    last_decision: Optional[V6MetaDecision] = None  # drives next outer's flags
    current_prompt_name: str = ""
    current_prompt_dir_str: str = ""
    early_stopped: bool = False
    final_summary: Optional[Dict[str, Any]] = None
    elapsed_s_so_far: float = 0.0


@dataclass
class V6OuterIterRecord:
    iter_idx: int
    prompt_dir: Path
    inner: InnerResult
    decision: V6MetaDecision  # the decision MADE AFTER this iter (drives next)
    decision_raw_text: str  # full meta-LLM response for postmortem


_CHECKPOINT_BASENAME = "_v6_checkpoint.pkl"


def _checkpoint_path(output_dir: Path) -> Path:
    return Path(output_dir) / _CHECKPOINT_BASENAME


def _save_checkpoint(output_dir: Path, ckpt: V6Checkpoint) -> None:
    path = _checkpoint_path(output_dir)
    tmp = path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(ckpt, f)
    os.replace(tmp, path)
    _log.info(
        "v6 checkpoint saved: outer_done=%d early_stopped=%s",
        ckpt.outer_idx_completed,
        ckpt.early_stopped,
    )


def _load_checkpoint(output_dir: Path) -> Optional[V6Checkpoint]:
    path = _checkpoint_path(output_dir)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    _log.info(
        "v6 checkpoint loaded: outer_done=%d early_stopped=%s",
        ckpt.outer_idx_completed,
        ckpt.early_stopped,
    )
    return ckpt


def _redirect_prompt_loader(prompts_root: Path) -> None:
    from ..prompts import loader as _l

    prompts_root.mkdir(parents=True, exist_ok=True)
    _l._PROMPTS_DIR = prompts_root


def _read_slot_files(prompt_dir: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for slot in ("guidance_observation", "guidance_reward", "guidance_shared"):
        p = prompt_dir / f"{slot}.txt"
        out[slot] = p.read_text() if p.exists() else ""
    return out


def _apply_slot_edits(
    prev_dir: Path,
    next_dir: Path,
    slot_edits: Dict[str, str],
) -> None:
    if next_dir.exists():
        shutil.rmtree(next_dir)
    shutil.copytree(prev_dir, next_dir)
    for slot, text in slot_edits.items():
        (next_dir / f"{slot}.txt").write_text((text or "").rstrip() + "\n")


def _build_outer_registry_entry(
    iter_idx: int,
    inner: InnerResult,
    decision: V6MetaDecision,
) -> RegistryEntry:
    if inner.best is None:
        return RegistryEntry(
            iter_idx=iter_idx,
            handle=f"v6_outer_{iter_idx}_dead",
            summary="inner produced zero valid candidates",
            fitness=-1.0,
            M1=0.0,
            shape="flat_zero",
            code_excerpt="",
        )
    flag_str = (
        f"obs={decision.next_evolve_observation},"
        f"rew={decision.next_evolve_reward},"
        f"complexity={decision.complexity_level}"
    )
    return RegistryEntry(
        iter_idx=iter_idx,
        handle=f"v6_outer_{iter_idx}_{decision.next_mode}",
        summary=(
            f"classification={decision.classification}; "
            f"next_mode={decision.next_mode}; flags=({flag_str}); "
            f"best inner: M1={inner.best.metrics.get('M1_success_rate', 0):.3f} "
            f"shape={inner.best.shape}; rationale={decision.rationale[:200]}"
        ),
        fitness=inner.best.fitness,
        M1=float(inner.best.metrics.get("M1_success_rate", 0.0)),
        shape=inner.best.shape,
        code_excerpt=(
            inner.best.candidate.obs_source or inner.best.candidate.reward_source or ""
        )[:1500],
    )


@dataclass
class V6OuterConfig:
    max_outer: int = 5
    n_inner_iter: int = 4
    n_inner_candidates_per_iter: int = 3
    eval_frames: int = 1_000_000
    base_prompt_version: str = "v2_fewshot_modular_v2_local"
    meta_model: str = "gpt-5.4-mini"
    meta_temperature: float = 1.0
    task_summary: str = ""
    # Cold-start bootstrap (added 2026-04-30 after V4 prompt-lab finding):
    # call meta-LLM ONCE before outer iter 0 with no prior inner result,
    # so V4-style operational guidance is active from outer 0's first
    # inner candidate. Without this, outer 0's inner runs with empty
    # guidance regardless of meta system prompt content.
    cold_start_bootstrap: bool = True


def run_v6_outer_loop(
    spec: ExperimentSpec,
    base_loop: LeroLoop,
    cfg: V6OuterConfig,
    meta_llm: LLMClient,
    output_dir: Path,
    seed: int = 0,
    task_overrides: Optional[Dict[str, Any]] = None,
    algorithm: str = "mappo",
    resume: bool = False,
) -> Dict[str, Any]:
    """v6 main entry. Returns summary dict.

    No deep-train. Output is the inner-stage best candidate(s); a
    separate manual step (see docs/v6_plan.md §4.5) deep-trains them
    if classification == found_good.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts_root = output_dir / "prompts"
    _redirect_prompt_loader(prompts_root)

    from ..prompts import loader as _l_mod

    _orig_prompts_dir = Path(_l_mod.__file__).parent
    base_src = _orig_prompts_dir / cfg.base_prompt_version
    if not base_src.exists():
        raise FileNotFoundError(f"base prompt version not found: {base_src}")

    ckpt: Optional[V6Checkpoint] = None
    if resume:
        ckpt = _load_checkpoint(output_dir)
    if ckpt is None:
        seed_prompt_name = f"v6_outer_0_seed{seed}"
        seed_prompt_dir = prompts_root / seed_prompt_name
        if seed_prompt_dir.exists():
            shutil.rmtree(seed_prompt_dir)
        shutil.copytree(base_src, seed_prompt_dir)

        # Cold-start bootstrap: call meta-LLM with no prior inner
        # result so V4-style operational guidance is active from
        # outer 0's first inner candidate. See docs/v6_plan.md §4.x.
        if cfg.cold_start_bootstrap:
            _log.info(
                "v6 COLD-START bootstrap: meta-LLM call with no "
                "prior inner (outer 0 will start with seeded "
                "guidance instead of empty slots)"
            )
            bootstrap_decision, bootstrap_raw = decide_and_enforce(
                meta_llm=meta_llm,
                task_summary=cfg.task_summary or spec.description or "",
                current_slots=_read_slot_files(seed_prompt_dir),
                last_inner=None,
                outer_registry=Registry(),
                outer_idx=0,
                prior_complexity=1,
                prior_classification=None,
            )
            for slot, text in bootstrap_decision.slot_edits.items():
                (seed_prompt_dir / f"{slot}.txt").write_text(
                    (text or "").rstrip() + "\n"
                )
            (seed_prompt_dir / "_bootstrap_response.txt").write_text(bootstrap_raw)
            (seed_prompt_dir / "_bootstrap_decision.json").write_text(
                json.dumps(
                    {
                        "classification": bootstrap_decision.classification,
                        "next_mode": bootstrap_decision.next_mode,
                        "rationale": bootstrap_decision.rationale,
                        "complexity_level": bootstrap_decision.complexity_level,
                        "slot_edits_keys": list(bootstrap_decision.slot_edits.keys()),
                        "enforcement_notes": bootstrap_decision.enforcement_notes,
                    },
                    indent=2,
                    default=str,
                )
            )
            _log.info(
                "v6 cold-start: emitted %d slot edits, complexity=%d, " "rationale: %s",
                len(bootstrap_decision.slot_edits),
                bootstrap_decision.complexity_level,
                bootstrap_decision.rationale[:120],
            )
        ckpt = V6Checkpoint(
            seed=seed,
            current_prompt_name=seed_prompt_name,
            current_prompt_dir_str=str(seed_prompt_dir.resolve()),
        )
        _save_checkpoint(output_dir, ckpt)
    else:
        _log.info("v6 RESUME — skipping outers ≤ %d", ckpt.outer_idx_completed)

    outer_registry = ckpt.outer_registry
    iter_records = ckpt.iter_records
    current_prompt_name = ckpt.current_prompt_name
    current_prompt_dir = Path(ckpt.current_prompt_dir_str)
    last_decision = ckpt.last_decision
    early_stopped = ckpt.early_stopped

    t_start = time.monotonic()

    for outer_idx in range(cfg.max_outer):
        if outer_idx <= ckpt.outer_idx_completed:
            continue
        if early_stopped:
            break

        _log.info("=== v6 OUTER ITER %d/%d ===", outer_idx + 1, cfg.max_outer)

        # Apply flags from prior decision (or hard-locked obs-only at iter 0)
        if outer_idx == 0:
            base_loop.lero.evolve_observation = True
            base_loop.lero.evolve_reward = False
            _log.info("v6 outer 0: locked to obs-only (simplicity-first)")
        else:
            assert last_decision is not None, "no prior decision after iter 0"
            base_loop.lero.evolve_observation = last_decision.next_evolve_observation
            base_loop.lero.evolve_reward = last_decision.next_evolve_reward
            _log.info(
                "v6 outer %d flags: evolve_obs=%s evolve_rew=%s",
                outer_idx,
                base_loop.lero.evolve_observation,
                base_loop.lero.evolve_reward,
            )

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

        # Meta-strategist call (with code-side cross-check + enforcement).
        prior_complexity = last_decision.complexity_level if last_decision else 1
        prior_classification = last_decision.classification if last_decision else None
        decision, raw_text = decide_and_enforce(
            meta_llm=meta_llm,
            task_summary=cfg.task_summary or spec.description or "",
            current_slots=_read_slot_files(current_prompt_dir),
            last_inner=inner_result,
            outer_registry=outer_registry,
            outer_idx=outer_idx,
            prior_complexity=prior_complexity,
            prior_classification=prior_classification,
        )

        # Persist meta-LLM response + enforcement notes for postmortem.
        (inner_out_dir / "_decision.json").write_text(
            json.dumps(
                {
                    "classification": decision.classification,
                    "next_mode": decision.next_mode,
                    "rationale": decision.rationale,
                    "next_evolve_observation": decision.next_evolve_observation,
                    "next_evolve_reward": decision.next_evolve_reward,
                    "complexity_level": decision.complexity_level,
                    "slot_edits_keys": list(decision.slot_edits.keys()),
                    "enforcement_notes": decision.enforcement_notes,
                },
                indent=2,
                default=str,
            )
        )
        (inner_out_dir / "_meta_response.txt").write_text(raw_text)

        # Record outer iter
        iter_records.append(
            V6OuterIterRecord(
                iter_idx=outer_idx,
                prompt_dir=current_prompt_dir,
                inner=inner_result,
                decision=decision,
                decision_raw_text=raw_text,
            )
        )
        entry = _build_outer_registry_entry(outer_idx, inner_result, decision)
        outer_registry.add(entry)
        outer_registry.record_iter_best_fitness(entry.fitness)

        ckpt.outer_idx_completed = outer_idx
        ckpt.last_decision = decision
        last_decision = decision
        ckpt.elapsed_s_so_far += time.monotonic() - t_start
        t_start = time.monotonic()

        _log.info(
            "v6 outer %d done: classification=%s M1=%.3f shape=%s " "next_mode=%s",
            outer_idx,
            decision.classification,
            entry.M1,
            entry.shape,
            decision.next_mode,
        )

        # Early stop on found_good or LLM decided to stop.
        if decision.next_mode == "stop":
            early_stopped = True
            ckpt.early_stopped = True
            _save_checkpoint(output_dir, ckpt)
            break

        # Otherwise: compose next metaprompt directory and continue.
        next_prompt_name = f"v6_outer_{outer_idx + 1}_seed{seed}"
        next_prompt_dir = prompts_root / next_prompt_name
        _apply_slot_edits(
            prev_dir=current_prompt_dir,
            next_dir=next_prompt_dir,
            slot_edits=decision.slot_edits,
        )
        # Save the rationale + decision metadata alongside the new prompt
        (next_prompt_dir / "_decision_rationale.md").write_text(
            f"# Outer iter {outer_idx} → {outer_idx + 1} decision\n\n"
            f"**classification:** {decision.classification}\n"
            f"**next_mode:** {decision.next_mode}\n"
            f"**evolve_observation:** {decision.next_evolve_observation}\n"
            f"**evolve_reward:** {decision.next_evolve_reward}\n"
            f"**complexity_level:** {decision.complexity_level}\n"
            f"**enforcement notes:** "
            f"{decision.enforcement_notes or 'none'}\n\n"
            f"## Rationale\n\n{decision.rationale}\n"
        )

        current_prompt_name = next_prompt_name
        current_prompt_dir = next_prompt_dir
        ckpt.current_prompt_name = current_prompt_name
        ckpt.current_prompt_dir_str = str(current_prompt_dir.resolve())
        _save_checkpoint(output_dir, ckpt)

    # === Pick global best across all completed outer iters ===
    candidates_all = []
    for rec in iter_records:
        for o in rec.inner.all_outcomes:
            candidates_all.append((rec.iter_idx, o))
    if not candidates_all:
        return {"_error": "no valid candidates", "early_stopped": early_stopped}
    candidates_all.sort(key=lambda x: x[1].fitness, reverse=True)
    global_best_outer_idx, global_best = candidates_all[0]

    summary = {
        "early_stopped": early_stopped,
        "outer_iters_completed": ckpt.outer_idx_completed + 1,
        "global_best_outer_idx": global_best_outer_idx,
        "global_best_M1": float(global_best.metrics.get("M1_success_rate", 0.0)),
        "global_best_shape": global_best.shape,
        "global_best_fitness": global_best.fitness,
        "global_best_M6": float(global_best.metrics.get("M6_coverage_progress", 0.0)),
        "outer_classification_trajectory": [
            r.decision.classification for r in iter_records
        ],
        "outer_complexity_trajectory": [
            r.decision.complexity_level for r in iter_records
        ],
        "outer_flag_trajectory": [
            (r.decision.next_evolve_observation, r.decision.next_evolve_reward)
            for r in iter_records
        ],
        "outer_fitness_trajectory": list(outer_registry.fitness_trajectory),
        "elapsed_s_total": ckpt.elapsed_s_so_far + (time.monotonic() - t_start),
    }
    ckpt.final_summary = summary
    _save_checkpoint(output_dir, ckpt)

    (output_dir / "v6_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    _log.info(
        "=== v6 DONE === outer_done=%d global_best_M1=%.3f shape=%s "
        "early_stopped=%s",
        ckpt.outer_idx_completed + 1,
        summary["global_best_M1"],
        summary["global_best_shape"],
        early_stopped,
    )

    # ===================================================================
    # DEEP-TRAIN — DISABLED IN v6.
    # The function below is implemented but commented out per
    # docs/v6_plan.md §4.5. Once a found_good candidate is identified,
    # uncomment + run from a separate runner / notebook to validate
    # against S3b-local.
    # ===================================================================
    # if early_stopped and decision.classification == "found_good":
    #     deep_summary = deep_train_winner(
    #         base_loop=base_loop,
    #         winner_candidate=global_best.candidate,
    #         output_dir=output_dir / "deep_train",
    #         full_frames=10_000_000,
    #         task_overrides=task_overrides,
    #         algorithm=algorithm,
    #         seed=seed,
    #     )
    #     summary["deep_train"] = deep_summary

    return summary


# def deep_train_winner(
#     base_loop: LeroLoop,
#     winner_candidate,
#     output_dir: Path,
#     full_frames: int = 10_000_000,
#     task_overrides=None,
#     algorithm: str = "mappo",
#     seed: int = 0,
# ) -> Dict[str, Any]:
#     """10M deep-train of the v6 inner-stage winner. Disabled by default.
#     Call this manually after v6 produces a found_good candidate."""
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
#     base_loop.output_dir = output_dir
#     base_loop.lero = copy.copy(base_loop.lero)
#     base_loop.lero.full_frames = full_frames
#     return base_loop._full_training(
#         winner_candidate, task_overrides, algorithm, seed,
#     )
