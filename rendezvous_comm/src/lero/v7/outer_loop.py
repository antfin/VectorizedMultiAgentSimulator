"""v7 outer loop — bundle + grounded reflection + comprehensive logging.

Architecture:

    cold-start:
      - enumerate_bundle(meta_llm) → V7StrategyBundle
      - save _bundle_init.json + _bundle_init_response.txt
      - apply chosen strategy's translation hint to outer 0's slots

    for outer_idx in 0..MAX_OUTER:
      - save _active_strategy.json + _bundle_state_before.json
      - run inner_loop (which now saves candidate_{j}_obs.py per iter)
      - diagnose_inner_result → V7Diagnosis (auto from AST + metrics)
      - save _diagnosis.json + _diagnosis_full.txt
      - reflect_and_decide(meta_llm, bundle, inner, diagnosis) → V7ReflectionDecision
      - save _reflection_prompt.txt + _reflection_response.txt + _decision.json
      - apply decision:
          stop → break
          refine_current_strategy → update slots, keep strategy
          refine_inner_prompt_for_current → update slots, keep strategy (more concrete)
          switch_to_next_strategy → bundle.next_best_idx(), apply new strategy's hint
      - save _bundle_state_after.json
      - record outcome in bundle history
      - checkpoint

    end: write _bundle_history.json + v7_summary.json
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
from ..v5.inner_loop import InnerResult, run_inner_loop
from ..v5.registry import Registry, RegistryEntry
from .diagnosis import V7Diagnosis, diagnose_inner_result
from .meta_strategist import (
    V7ReflectionDecision,
    enumerate_bundle,
    reflect_and_decide,
)
from .strategy import V7Strategy, V7StrategyBundle

_log = logging.getLogger("rendezvous.lero.v7.outer")


@dataclass
class V7OuterConfig:
    max_outer: int = 5
    n_inner_iter: int = 4
    n_inner_candidates_per_iter: int = 3
    eval_frames: int = 1_000_000
    base_prompt_version: str = "v2_fewshot_modular_v2_local"
    meta_model: str = "gpt-5.4-mini"
    meta_temperature: float = 0.8
    task_summary: str = ""


@dataclass
class V7Checkpoint:
    schema_version: int = 1
    seed: int = 0
    outer_idx_completed: int = -1
    bundle: V7StrategyBundle = field(default_factory=V7StrategyBundle)
    elapsed_s_so_far: float = 0.0
    early_stopped: bool = False
    final_summary: Optional[Dict[str, Any]] = None


_CHECKPOINT_BASENAME = "_v7_checkpoint.pkl"


def _save_checkpoint(output_dir: Path, ckpt: V7Checkpoint) -> None:
    p = output_dir / _CHECKPOINT_BASENAME
    tmp = p.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(ckpt, f)
    os.replace(tmp, p)
    _log.info(
        "v7 checkpoint: outer_done=%d early_stopped=%s",
        ckpt.outer_idx_completed, ckpt.early_stopped,
    )


def _load_checkpoint(output_dir: Path) -> Optional[V7Checkpoint]:
    p = output_dir / _CHECKPOINT_BASENAME
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


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


def _write_slots(prompt_dir: Path, slots: Dict[str, str]) -> None:
    for slot, text in slots.items():
        (prompt_dir / f"{slot}.txt").write_text((text or "").rstrip() + "\n")


def _strategy_to_slots(strategy: V7Strategy) -> Dict[str, str]:
    """Translate a V7Strategy's translation hint into slot text. The
    full hint goes into guidance_observation; guidance_shared stays
    empty by default (the meta-LLM may add it during reflection)."""
    return {
        "guidance_observation": strategy.lero_translation_hint,
        "guidance_shared": "",
        "guidance_reward": "",
    }


def _strategy_dict(s: V7Strategy) -> Dict[str, Any]:
    return {
        "name": s.name,
        "full_solution": s.full_solution,
        "lero_codability": s.lero_codability,
        "rl_trainability": s.rl_trainability,
        "combined_score": s.combined_score,
        "attempts": s.attempts,
        "last_outcome": s.last_outcome,
        "last_M1": s.last_inner_M1,
        "last_M6": s.last_inner_M6,
        "last_pattern_present": s.last_pattern_present,
        "excluded": s.excluded,
        "lero_translation_hint": s.lero_translation_hint,
        "success_signature": {
            "ast_pattern_description": s.success_signature.ast_pattern_description,
            "expected_M1_at_1M": s.success_signature.expected_M1_at_1M,
            "expected_M6_at_1M_min": s.success_signature.expected_M6_at_1M_min,
        },
    }


def _bundle_dict(b: V7StrategyBundle) -> Dict[str, Any]:
    return {
        "chosen_idx": b.chosen_idx,
        "current_name": b.current().name if b.current() else None,
        "strategies": [_strategy_dict(s) for s in b.strategies],
        "history": b.history,
    }


# ── Main entry ─────────────────────────────────────────────────


def run_v7_outer_loop(
    spec: ExperimentSpec,
    base_loop: LeroLoop,
    cfg: V7OuterConfig,
    meta_llm: LLMClient,
    output_dir: Path,
    seed: int = 0,
    task_overrides: Optional[Dict[str, Any]] = None,
    algorithm: str = "mappo",
    resume: bool = False,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts_root = output_dir / "prompts"
    _redirect_prompt_loader(prompts_root)

    from ..prompts import loader as _l_mod
    _orig_prompts_dir = Path(_l_mod.__file__).parent
    base_src = _orig_prompts_dir / cfg.base_prompt_version
    if not base_src.exists():
        raise FileNotFoundError(f"base prompt not found: {base_src}")

    ckpt: Optional[V7Checkpoint] = None
    if resume:
        ckpt = _load_checkpoint(output_dir)

    # === Cold-start: bundle enumeration ===
    if ckpt is None:
        _log.info("v7 COLD-START: enumerating strategy bundle")
        bundle, raw_bundle = enumerate_bundle(
            meta_llm,
            cfg.task_summary or spec.description or "",
        )
        (output_dir / "_bundle_init.json").write_text(
            json.dumps(_bundle_dict(bundle), indent=2, default=str)
        )
        (output_dir / "_bundle_init_response.txt").write_text(raw_bundle)
        _log.info(
            "v7 bundle: %d strategies, chosen='%s' (score=%.1f)",
            len(bundle.strategies),
            bundle.current().name,
            bundle.current().combined_score,
        )
        # Apply chosen strategy's slots to outer 0
        seed_prompt_name = f"v7_outer_0_seed{seed}"
        seed_prompt_dir = prompts_root / seed_prompt_name
        if seed_prompt_dir.exists():
            shutil.rmtree(seed_prompt_dir)
        shutil.copytree(base_src, seed_prompt_dir)
        _write_slots(seed_prompt_dir, _strategy_to_slots(bundle.current()))
        ckpt = V7Checkpoint(seed=seed, bundle=bundle)
        _save_checkpoint(output_dir, ckpt)
    else:
        _log.info("v7 RESUME — outer_done=%d", ckpt.outer_idx_completed)

    bundle = ckpt.bundle

    t_start = time.monotonic()

    for outer_idx in range(cfg.max_outer):
        if outer_idx <= ckpt.outer_idx_completed:
            continue
        if ckpt.early_stopped:
            break

        _log.info("=== v7 OUTER ITER %d/%d ===",
                   outer_idx + 1, cfg.max_outer)

        # Save bundle state before this iter
        outer_dir = output_dir / f"outer_{outer_idx:02d}"
        outer_dir.mkdir(exist_ok=True)
        active = bundle.current()
        (outer_dir / "_active_strategy.json").write_text(
            json.dumps(_strategy_dict(active), indent=2, default=str)
        )
        (outer_dir / "_bundle_state_before.json").write_text(
            json.dumps(_bundle_dict(bundle), indent=2, default=str)
        )

        # Outer 0 always obs-only; reward-evolve disabled across all v7
        # outers in this initial design (matching v6 evolve_reward=false).
        base_loop.lero.evolve_observation = True
        base_loop.lero.evolve_reward = False

        # Determine which prompt dir to use this round
        prompt_name = f"v7_outer_{outer_idx}_seed{seed}"
        prompt_dir = prompts_root / prompt_name

        # Run inner LERO with current slots
        inner_out_dir = outer_dir / "inner"
        inner_result = run_inner_loop(
            base_loop=base_loop,
            metaprompt_version=prompt_name,
            n_iterations=cfg.n_inner_iter,
            n_candidates_per_iter=cfg.n_inner_candidates_per_iter,
            seed=seed,
            output_dir=inner_out_dir,
            task_overrides=task_overrides,
            algorithm=algorithm,
        )

        # Diagnose
        diag = diagnose_inner_result(inner_result, active)
        (outer_dir / "_diagnosis.json").write_text(json.dumps({
            "label": diag.label,
            "pattern_present": diag.pattern_present,
            "metrics_signature_match": diag.metrics_signature_match,
            "rationale": diag.rationale,
            "inner_M1": diag.inner_M1,
            "inner_M6": diag.inner_M6,
        }, indent=2, default=str))
        (outer_dir / "_diagnosis_full.txt").write_text(
            f"=== V7 DIAGNOSIS ===\n"
            f"label: {diag.label}\n"
            f"pattern_present: {diag.pattern_present}\n"
            f"metrics_signature_match: {diag.metrics_signature_match}\n"
            f"inner_M1: {diag.inner_M1:.3f}\n"
            f"inner_M6: {diag.inner_M6:.3f}\n"
            f"\nrationale:\n{diag.rationale}\n"
        )
        _log.info(
            "v7 outer %d diag: label=%s pattern=%s M1=%.3f M6=%.3f",
            outer_idx, diag.label, diag.pattern_present,
            diag.inner_M1, diag.inner_M6,
        )

        # If achieved or no candidates, short-circuit
        if diag.label == "achieved":
            ckpt.early_stopped = True
            bundle.record_outcome(
                outer_idx, diag.label, diag.inner_M1,
                diag.inner_M6, diag.pattern_present,
            )
            ckpt.outer_idx_completed = outer_idx
            _save_checkpoint(output_dir, ckpt)
            break

        # Reflect with the meta-LLM (skipped if too_early)
        if diag.label == "too_early":
            _log.warning("v7 outer %d: too_early — no inner result", outer_idx)
            bundle.record_outcome(
                outer_idx, "too_early", 0.0, 0.0, False,
            )
            ckpt.outer_idx_completed = outer_idx
            _save_checkpoint(output_dir, ckpt)
            continue

        decision, raw_reflect = reflect_and_decide(
            meta_llm, bundle, inner_result, diag,
        )
        (outer_dir / "_reflection_response.txt").write_text(raw_reflect)
        (outer_dir / "_decision.json").write_text(json.dumps({
            "next_action": decision.next_action,
            "rationale": decision.rationale,
            "slot_edits_keys": list(decision.slot_edits.keys()),
            "bundle_demote": decision.bundle_demote,
            "bundle_add_names": [s.name for s in decision.bundle_add],
        }, indent=2, default=str))
        _log.info(
            "v7 outer %d decision: action=%s rationale=%s",
            outer_idx, decision.next_action, decision.rationale[:160],
        )

        # Record outcome on the active strategy
        bundle.record_outcome(
            outer_idx, diag.label, diag.inner_M1,
            diag.inner_M6, diag.pattern_present,
        )

        # Apply bundle updates
        if decision.bundle_demote:
            for name in decision.bundle_demote:
                for s in bundle.strategies:
                    if s.name == name:
                        s.last_outcome = "rl_too_hard"  # exclude
        if decision.bundle_add:
            bundle.strategies.extend(decision.bundle_add)

        # Apply next_action
        if decision.next_action == "stop":
            ckpt.early_stopped = True
            ckpt.outer_idx_completed = outer_idx
            _save_checkpoint(output_dir, ckpt)
            break

        next_prompt_name = f"v7_outer_{outer_idx + 1}_seed{seed}"
        next_prompt_dir = prompts_root / next_prompt_name
        if next_prompt_dir.exists():
            shutil.rmtree(next_prompt_dir)
        shutil.copytree(base_src, next_prompt_dir)

        if decision.next_action == "switch_to_next_strategy":
            nb = bundle.next_best_idx(exclude_current=True)
            if nb is None:
                _log.warning(
                    "v7 outer %d: switch requested but no eligible "
                    "strategy left; stopping",
                    outer_idx,
                )
                ckpt.early_stopped = True
                ckpt.outer_idx_completed = outer_idx
                _save_checkpoint(output_dir, ckpt)
                break
            bundle.chosen_idx = nb
            _write_slots(next_prompt_dir, _strategy_to_slots(bundle.current()))
            _log.info(
                "v7 outer %d: SWITCHED to strategy '%s'",
                outer_idx, bundle.current().name,
            )
        elif decision.next_action in (
            "refine_current_strategy",
            "refine_inner_prompt_for_current",
        ):
            current_slots = _read_slot_files(prompt_dir)
            new_slots = dict(current_slots)
            for k, v in decision.slot_edits.items():
                new_slots[k] = v
            _write_slots(next_prompt_dir, new_slots)
        else:
            _log.warning(
                "v7 outer %d: unknown next_action %r — defaulting to refine",
                outer_idx, decision.next_action,
            )
            current_slots = _read_slot_files(prompt_dir)
            _write_slots(next_prompt_dir, current_slots)

        (outer_dir / "_bundle_state_after.json").write_text(
            json.dumps(_bundle_dict(bundle), indent=2, default=str)
        )

        ckpt.outer_idx_completed = outer_idx
        ckpt.bundle = bundle
        ckpt.elapsed_s_so_far += time.monotonic() - t_start
        t_start = time.monotonic()
        _save_checkpoint(output_dir, ckpt)

    # === Final summary ===
    summary = {
        "early_stopped": ckpt.early_stopped,
        "outer_iters_completed": ckpt.outer_idx_completed + 1,
        "bundle_chosen_at_end": bundle.current().name if bundle.current() else None,
        "bundle_history": bundle.history,
        "bundle_final": _bundle_dict(bundle),
        "elapsed_s_total": ckpt.elapsed_s_so_far + (
            time.monotonic() - t_start
        ),
    }
    (output_dir / "v7_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    (output_dir / "_bundle_history.json").write_text(
        json.dumps(bundle.history, indent=2, default=str)
    )
    ckpt.final_summary = summary
    _save_checkpoint(output_dir, ckpt)

    _log.info(
        "=== v7 DONE === outer_done=%d early_stopped=%s",
        ckpt.outer_idx_completed + 1, ckpt.early_stopped,
    )
    return summary
