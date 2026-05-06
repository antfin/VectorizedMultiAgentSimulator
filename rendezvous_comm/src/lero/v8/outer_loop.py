"""v8 outer_loop — density-first, bounded features, per docs/v8_plan.md.

Inherits v7's structure but uses v8's diagnosis (with too_many_features +
over_gated) and v8's meta_strategist (with density-first prompt + 3-4
feature working fewshot). Adds the `trim_features` and
`replace_gated_with_dense` actions to the outer-iter decision tree.

Comprehensive logging carries over from v7: every meta-LLM
prompt/response saved, every inner candidate code saved, every
diagnosis logged.
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
from typing import Any, Dict, Optional

from ...config import ExperimentSpec
from ..llm_client import LLMClient
from ..loop import LeroLoop
from ..v5.inner_loop import run_inner_loop
from ..v7.outer_loop import (
    _bundle_dict,
    _read_slot_files,
    _redirect_prompt_loader,
    _strategy_dict,
    _strategy_to_slots,
    _write_slots,
)
from ..v7.strategy import V7StrategyBundle
from .diagnosis import diagnose_inner_result_v8
from .meta_strategist import (
    enumerate_bundle_v8,
    reflect_and_decide_v8,
)

_log = logging.getLogger("rendezvous.lero.v8.outer")


@dataclass
class V8OuterConfig:
    max_outer: int = 5
    n_inner_iter: int = 4
    n_inner_candidates_per_iter: int = 3
    eval_frames: int = 1_000_000
    base_prompt_version: str = "v2_fewshot_modular_v2_local"
    meta_model: str = "gpt-5.4-mini"
    meta_temperature: float = 0.8
    task_summary: str = ""
    # v8 feature-budget knobs (config-tunable per docs/v8_plan.md §2.1)
    feature_count_target_min: int = 8
    feature_count_target_max: int = 12
    feature_count_cap: int = 15
    gated_feature_cap: int = 2


@dataclass
class V8Checkpoint:
    schema_version: int = 2
    seed: int = 0
    outer_idx_completed: int = -1
    bundle: V7StrategyBundle = field(default_factory=V7StrategyBundle)
    elapsed_s_so_far: float = 0.0
    early_stopped: bool = False
    final_summary: Optional[Dict[str, Any]] = None


_CHECKPOINT_BASENAME = "_v8_checkpoint.pkl"


def _save_ckpt(output_dir: Path, ckpt: V8Checkpoint) -> None:
    p = output_dir / _CHECKPOINT_BASENAME
    tmp = p.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(ckpt, f)
    os.replace(tmp, p)
    _log.info(
        "v8 ckpt: outer_done=%d early=%s", ckpt.outer_idx_completed, ckpt.early_stopped
    )


def _load_ckpt(output_dir: Path) -> Optional[V8Checkpoint]:
    p = output_dir / _CHECKPOINT_BASENAME
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def run_v8_outer_loop(
    spec: ExperimentSpec,
    base_loop: LeroLoop,
    cfg: V8OuterConfig,
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

    ckpt: Optional[V8Checkpoint] = None
    if resume:
        ckpt = _load_ckpt(output_dir)

    # === Cold-start ===
    if ckpt is None:
        _log.info(
            "v8 COLD-START: bundle enum (target=%d-%d, cap=%d, gated_cap=%d)",
            cfg.feature_count_target_min,
            cfg.feature_count_target_max,
            cfg.feature_count_cap,
            cfg.gated_feature_cap,
        )
        bundle, raw_bundle = enumerate_bundle_v8(
            meta_llm,
            cfg.task_summary or spec.description or "",
            cfg.feature_count_target_min,
            cfg.feature_count_target_max,
            cfg.feature_count_cap,
            cfg.gated_feature_cap,
        )
        (output_dir / "_bundle_init.json").write_text(
            json.dumps(_bundle_dict(bundle), indent=2, default=str)
        )
        (output_dir / "_bundle_init_response.txt").write_text(raw_bundle)
        seed_prompt_name = f"v8_outer_0_seed{seed}"
        seed_prompt_dir = prompts_root / seed_prompt_name
        if seed_prompt_dir.exists():
            shutil.rmtree(seed_prompt_dir)
        shutil.copytree(base_src, seed_prompt_dir)
        _write_slots(seed_prompt_dir, _strategy_to_slots(bundle.current()))
        ckpt = V8Checkpoint(seed=seed, bundle=bundle)
        _save_ckpt(output_dir, ckpt)
    else:
        _log.info("v8 RESUME — outer_done=%d", ckpt.outer_idx_completed)

    bundle = ckpt.bundle
    t_start = time.monotonic()

    for outer_idx in range(cfg.max_outer):
        if outer_idx <= ckpt.outer_idx_completed:
            continue
        if ckpt.early_stopped:
            break

        _log.info("=== v8 OUTER ITER %d/%d ===", outer_idx + 1, cfg.max_outer)
        outer_dir = output_dir / f"outer_{outer_idx:02d}"
        outer_dir.mkdir(exist_ok=True)
        active = bundle.current()
        (outer_dir / "_active_strategy.json").write_text(
            json.dumps(_strategy_dict(active), indent=2, default=str)
        )
        (outer_dir / "_bundle_state_before.json").write_text(
            json.dumps(_bundle_dict(bundle), indent=2, default=str)
        )

        base_loop.lero.evolve_observation = True
        base_loop.lero.evolve_reward = False

        prompt_name = f"v8_outer_{outer_idx}_seed{seed}"
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

        diag = diagnose_inner_result_v8(
            inner_result,
            active,
            cfg.feature_count_cap,
            cfg.gated_feature_cap,
        )
        (outer_dir / "_diagnosis.json").write_text(
            json.dumps(
                {
                    "label": diag.label,
                    "pattern_present": diag.pattern_present,
                    "metrics_signature_match": diag.metrics_signature_match,
                    "rationale": diag.rationale,
                    "inner_M1": diag.inner_M1,
                    "inner_M6": diag.inner_M6,
                    "n_features": diag.n_features,
                    "n_gated": diag.n_gated,
                    "n_dense": diag.n_dense,
                },
                indent=2,
                default=str,
            )
        )
        (outer_dir / "_diagnosis_full.txt").write_text(
            f"=== V8 DIAGNOSIS ===\n"
            f"label: {diag.label}\n"
            f"pattern_present: {diag.pattern_present}\n"
            f"metrics_match: {diag.metrics_signature_match}\n"
            f"M1={diag.inner_M1:.3f} M6={diag.inner_M6:.3f}\n"
            f"n_features={diag.n_features} (cap={cfg.feature_count_cap})\n"
            f"n_gated={diag.n_gated} (cap={cfg.gated_feature_cap})\n"
            f"n_dense={diag.n_dense}\n\n"
            f"rationale:\n{diag.rationale}\n"
        )
        _log.info(
            "v8 outer %d diag: label=%s n_feat=%d gated=%d dense=%d " "M1=%.3f M6=%.3f",
            outer_idx,
            diag.label,
            diag.n_features,
            diag.n_gated,
            diag.n_dense,
            diag.inner_M1,
            diag.inner_M6,
        )

        if diag.label == "achieved":
            ckpt.early_stopped = True
            bundle.record_outcome(
                outer_idx,
                diag.label,
                diag.inner_M1,
                diag.inner_M6,
                diag.pattern_present,
            )
            ckpt.outer_idx_completed = outer_idx
            _save_ckpt(output_dir, ckpt)
            break

        if diag.label == "too_early":
            bundle.record_outcome(outer_idx, "too_early", 0.0, 0.0, False)
            ckpt.outer_idx_completed = outer_idx
            _save_ckpt(output_dir, ckpt)
            continue

        decision, raw_reflect = reflect_and_decide_v8(
            meta_llm,
            bundle,
            inner_result,
            diag,
            cfg.feature_count_target_min,
            cfg.feature_count_target_max,
            cfg.feature_count_cap,
            cfg.gated_feature_cap,
        )
        (outer_dir / "_reflection_response.txt").write_text(raw_reflect)
        (outer_dir / "_decision.json").write_text(
            json.dumps(
                {
                    "next_action": decision.next_action,
                    "rationale": decision.rationale,
                    "slot_edits_keys": list(decision.slot_edits.keys()),
                    "bundle_demote": decision.bundle_demote,
                    "bundle_add_names": [s.name for s in decision.bundle_add],
                },
                indent=2,
                default=str,
            )
        )

        bundle.record_outcome(
            outer_idx,
            diag.label,
            diag.inner_M1,
            diag.inner_M6,
            diag.pattern_present,
        )
        if decision.bundle_demote:
            for name in decision.bundle_demote:
                for s in bundle.strategies:
                    if s.name == name:
                        s.last_outcome = "rl_too_hard"
        if decision.bundle_add:
            bundle.strategies.extend(decision.bundle_add)

        if decision.next_action == "stop":
            ckpt.early_stopped = True
            ckpt.outer_idx_completed = outer_idx
            _save_ckpt(output_dir, ckpt)
            break

        # Compose next prompt dir
        next_prompt_name = f"v8_outer_{outer_idx + 1}_seed{seed}"
        next_prompt_dir = prompts_root / next_prompt_name
        if next_prompt_dir.exists():
            shutil.rmtree(next_prompt_dir)
        shutil.copytree(base_src, next_prompt_dir)

        if decision.next_action == "switch_to_next_strategy":
            nb = bundle.next_best_idx(exclude_current=True)
            if nb is None:
                _log.warning("v8 outer %d: no eligible strategy left; stop", outer_idx)
                ckpt.early_stopped = True
                ckpt.outer_idx_completed = outer_idx
                _save_ckpt(output_dir, ckpt)
                break
            bundle.chosen_idx = nb
            _write_slots(next_prompt_dir, _strategy_to_slots(bundle.current()))
            _log.info("v8 outer %d: SWITCHED to '%s'", outer_idx, bundle.current().name)
        else:
            # refine_current_strategy / refine_inner_prompt_for_current /
            # trim_features / replace_gated_with_dense — all apply slot_edits
            prompt_dir = prompts_root / prompt_name
            current_slots = _read_slot_files(prompt_dir)
            new_slots = dict(current_slots)
            for k, v in decision.slot_edits.items():
                new_slots[k] = v
            _write_slots(next_prompt_dir, new_slots)
            _log.info("v8 outer %d: %s applied", outer_idx, decision.next_action)

        (outer_dir / "_bundle_state_after.json").write_text(
            json.dumps(_bundle_dict(bundle), indent=2, default=str)
        )

        ckpt.outer_idx_completed = outer_idx
        ckpt.bundle = bundle
        ckpt.elapsed_s_so_far += time.monotonic() - t_start
        t_start = time.monotonic()
        _save_ckpt(output_dir, ckpt)

    summary = {
        "early_stopped": ckpt.early_stopped,
        "outer_iters_completed": ckpt.outer_idx_completed + 1,
        "bundle_chosen_at_end": bundle.current().name if bundle.current() else None,
        "bundle_history": bundle.history,
        "bundle_final": _bundle_dict(bundle),
        "feature_count_cap": cfg.feature_count_cap,
        "gated_feature_cap": cfg.gated_feature_cap,
        "elapsed_s_total": ckpt.elapsed_s_so_far + (time.monotonic() - t_start),
    }
    (output_dir / "v8_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    (output_dir / "_bundle_history.json").write_text(
        json.dumps(bundle.history, indent=2, default=str)
    )
    ckpt.final_summary = summary
    _save_ckpt(output_dir, ckpt)
    _log.info(
        "=== v8 DONE === outer_done=%d early=%s",
        ckpt.outer_idx_completed + 1,
        ckpt.early_stopped,
    )
    return summary
