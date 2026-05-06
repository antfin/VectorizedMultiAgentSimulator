"""v9 outer loop — serial bundle coverage + memory + combined reflect.

Per docs/v9_plan.md:

  - max_outer = bundle_size (typically 3-5). Tries every strategy
    serially until `achieved` or all are excluded.
  - Inner stagnation pivot is removed (§6.2). Inner runs to completion.
  - Diagnose + reflect collapsed into one LLM call (§6.3).
  - MemoryStore appended each iter; last N=3 read before each reflect.
  - AST analyzer facts computed locally and embedded in reflect prompt.
"""

from __future__ import annotations

import json
import logging
import pickle
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...config import ExperimentSpec
from ..llm_client import LLMClient
from ..loop import LeroLoop
from ..prompts.loader import PromptLoader
from ..v5.inner_loop import InnerResult, run_inner_loop
from ..v6_prompt_lab.analyzer import (
    analyze_inner_code,
    count_dense_features,
    count_gated_features,
)
from .memory import MemoryRow, MemoryStore
from .meta_strategist import (
    enumerate_bundle_v9,
    reflect_decide_v9,
)
from .strategy import V9Bundle, V9Strategy

_log = logging.getLogger("rendezvous.lero.v9.outer")


# ── Config + checkpoint ─────────────────────────────────────────


@dataclass
class V9OuterConfig:
    n_inner_iter: int = 3
    n_inner_candidates_per_iter: int = 3
    eval_frames: int = 1_000_000
    base_prompt_version: str = "v3_modular_taskdomain"
    meta_model: str = "gpt-5.4-mini"
    meta_temperature: float = 0.8
    task_summary: str = ""
    memory_lookback_n: int = 3
    # If True, max_outer overrides the bundle_size cap (used by smoke tests).
    max_outer_override: Optional[int] = None


@dataclass
class V9Checkpoint:
    schema_version: int = 1
    seed: int = 0
    outer_idx_completed: int = -1
    bundle: Optional[V9Bundle] = None
    elapsed_s_so_far: float = 0.0
    early_stopped: bool = False
    final_summary: Optional[Dict[str, Any]] = None


_CHECKPOINT_BASENAME = "_v9_checkpoint.pkl"


def _save_ckpt(output_dir: Path, ckpt: V9Checkpoint) -> None:
    with (output_dir / _CHECKPOINT_BASENAME).open("wb") as f:
        pickle.dump(ckpt, f)


def _load_ckpt(output_dir: Path) -> Optional[V9Checkpoint]:
    p = output_dir / _CHECKPOINT_BASENAME
    if not p.exists():
        return None
    with p.open("rb") as f:
        return pickle.load(f)


# ── Slot helpers (v3 prompt slots) ─────────────────────────────


_V9_SLOT_NAMES = ("inferable_hints", "examples", "feedback")


def _redirect_prompt_loader(prompts_root: Path) -> None:
    from ..prompts import loader as _l

    prompts_root.mkdir(parents=True, exist_ok=True)
    _l._PROMPTS_DIR = prompts_root


def _read_slot_files(prompt_dir: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for slot in _V9_SLOT_NAMES:
        p = prompt_dir / f"{slot}.txt"
        out[slot] = p.read_text() if p.exists() else ""
    return out


def _write_slots(prompt_dir: Path, slots: Dict[str, str]) -> None:
    for slot, text in slots.items():
        (prompt_dir / f"{slot}.txt").write_text((text or "").rstrip() + "\n")


def _strategy_to_slots(strategy: V9Strategy) -> Dict[str, str]:
    return {
        "inferable_hints": strategy.artifacts.inferable_hints_text,
        "examples": strategy.artifacts.examples_text,
        # feedback.txt in v3 has placeholder $strategy_feedback_reminder
        # that's substituted at render time. We persist the raw template
        # in a sibling slot so other code can read it; the actual
        # feedback.txt stays static.
        "feedback": strategy.artifacts.feedback_template,
    }


def _strategy_dict(s: V9Strategy) -> Dict[str, Any]:
    return {
        "name": s.name,
        "full_solution": s.full_solution,
        "lero_codability": s.lero_codability,
        "rl_trainability": s.rl_trainability,
        "combined_score": s.combined_score,
        "attempts": s.attempts,
        "last_outcome": s.last_outcome,
        "last_M1": s.last_M1,
        "last_M6": s.last_M6,
        "last_pattern_present": s.last_pattern_present,
        "excluded": s.excluded,
        "chain_of_thought": {
            "why_it_works": s.chain_of_thought.why_it_works,
            "what_is_needed": s.chain_of_thought.what_is_needed,
            "failure_modes": s.chain_of_thought.failure_modes,
        },
        "artifacts": {
            "inferable_hints_text": s.artifacts.inferable_hints_text,
            "examples_text": s.artifacts.examples_text,
            "feedback_template": s.artifacts.feedback_template,
        },
        "success_signature": {
            "ast_pattern_description": s.success_signature.ast_pattern_description,
            "expected_M1_at_1M": s.success_signature.expected_M1_at_1M,
            "expected_M6_at_1M_min": s.success_signature.expected_M6_at_1M_min,
        },
    }


def _bundle_dict(b: V9Bundle) -> Dict[str, Any]:
    return {
        "chosen_idx": b.chosen_idx,
        "current_name": b.current().name if b.strategies else None,
        "task_understanding": b.task_understanding,
        "strategies": [_strategy_dict(s) for s in b.strategies],
        "history": b.history,
    }


# ── Analyzer facts ─────────────────────────────────────────────


def detect_pathological_refine(
    memory_rows: List[Dict[str, Any]],
    current_strategy_name: str,
    n: int = 2,
) -> bool:
    """Return True if the last `n` memory rows are all on
    `current_strategy_name` AND `M6` is monotonically non-increasing
    across them.

    This is the v9 pathological-refine signature: the meta-LLM is stuck
    refining a strategy that isn't improving (regressing or stable-bad).

    Pure function — no side effects — so it's unit-testable in isolation.

    Args:
        memory_rows: list of dicts (e.g. from MemoryStore.read_recent),
            each containing `strategy_name` and `actual["M6"]`.
        current_strategy_name: name of the bundle's currently chosen
            strategy. Detector only fires if all last-n rows match.
        n: how many recent outers must show the regression. Default 2.

    Returns:
        True iff len(memory_rows) >= n AND all last-n rows are on the
        same strategy as `current_strategy_name` AND M6 is non-increasing
        across them. False otherwise (including empty/short memory or
        mixed strategies).
    """
    if len(memory_rows) < n or n < 1:
        return False
    last_n = memory_rows[-n:]
    if not all(r.get("strategy_name") == current_strategy_name for r in last_n):
        return False
    m6_seq = [float((r.get("actual") or {}).get("M6", 0.0)) for r in last_n]
    return all(m6_seq[i + 1] <= m6_seq[i] for i in range(len(m6_seq) - 1))


def make_pre_eval_validator(
    task_domain: Dict[str, Any],
):
    """v9.1 §2.1: build a candidate pre-eval validator from
    `task_domain.mandatory_features`.

    Returns a callable suitable for `run_inner_loop(pre_eval_validator=...)`.
    The callable receives a CandidateCode and returns a list of issue
    strings (empty list / None = pass).

    Currently checks for two mandatory features:
      - role_one_hot: F.one_hot(...) or torch.zeros + [:, agent_idx]=1
      - cross_source_signal: AST detects ops touching both lidar streams

    Both are mandatory per task_domains/rendezvous_k2.yaml. Missing them
    means the candidate cannot encode role differentiation or coordination,
    so we save the ~9 min of training time.
    """
    mandatory = (task_domain or {}).get("mandatory_features") or []
    mandatory_names = {m["name"] for m in mandatory}
    budget = (task_domain or {}).get("feature_budget") or {}
    hard_cap = int(budget.get("hard_cap", 0)) or None

    def validate(cand) -> List[str]:
        issues: List[str] = []
        code = cand.obs_source or ""
        if not code:
            return ["empty obs_source"]

        if "role_one_hot" in mandatory_names:
            role_present = bool(
                "F.one_hot(" in code
                or "one_hot(" in code
                or "[:, agent_idx] = 1" in code
                or "[:, agent_idx]=1" in code
                or "[:,agent_idx] = 1" in code
                or "[:,agent_idx]=1" in code
            )
            if not role_present:
                issues.append(
                    "missing mandatory_feature 'role_one_hot' — "
                    "shared-policy MAPPO needs F.one_hot(agent_idx, "
                    "n_agents) or torch.zeros(B, n_agents); "
                    "one_hot[:, agent_idx] = 1.0"
                )

        if "cross_source_signal" in mandatory_names:
            ana = analyze_inner_code(code)
            if not ana.touches_both_lidars:
                issues.append(
                    "missing mandatory_feature 'cross_source_signal' — "
                    "no operation combining target-derived and "
                    "agent-derived sensor scalars"
                )

        # v9.1 §2.2: feature_budget hard cap. Soft: only fire if the
        # AST estimator reliably counted features (>0) AND > hard_cap.
        # Estimator returning 0 = "couldn't measure", not "0 features".
        if hard_cap and hard_cap > 0:
            ana = analyze_inner_code(code) if "ana" not in dir() else ana
            n_feat = ana.n_returned_features
            if n_feat > 0 and n_feat > hard_cap:
                issues.append(
                    f"feature_budget exceeded: AST counted "
                    f"{n_feat} features (hard_cap={hard_cap})"
                )

        return issues

    return validate


def detect_falsification_failure(
    memory_rows: List[Dict[str, Any]],
    current_strategy_name: str,
    expected_M1: float,
    n_attempts: int = 2,
    threshold_factor: float = 0.5,
) -> bool:
    """v9.1 §2.7 falsification gate.

    Return True iff the last `n_attempts` memory rows on
    `current_strategy_name` ALL have actual M1 below
    `threshold_factor * expected_M1`. Implies the LLM's predicted
    `expected_M1_at_1M` is empirically falsified at this point.

    Used to override `refine_current` → `switch_to_next` when the
    meta-LLM's per-outer label hasn't escalated despite consistent
    underperformance.

    Pure function — unit-testable in isolation.

    Args:
        memory_rows: list of memory dicts (from MemoryStore).
        current_strategy_name: bundle's currently chosen strategy.
        expected_M1: the strategy's success_signature.expected_M1_at_1M.
        n_attempts: how many recent attempts on the same strategy must
            all undershoot before firing. Default 2.
        threshold_factor: actual M1 must be below
            `threshold_factor * expected_M1` for each attempt.
            Default 0.5 (i.e., must be at least half of expected).

    Returns:
        True iff there are ≥`n_attempts` memory rows on this strategy
        AND every one shows M1 < threshold_factor * expected_M1.
    """
    if expected_M1 <= 0 or n_attempts < 1:
        return False
    same_strategy_rows = [
        r for r in memory_rows if r.get("strategy_name") == current_strategy_name
    ]
    if len(same_strategy_rows) < n_attempts:
        return False
    target = threshold_factor * expected_M1
    recent = same_strategy_rows[-n_attempts:]
    actual_m1s = [float((r.get("actual") or {}).get("M1", 0.0)) for r in recent]
    return all(m < target for m in actual_m1s)


def _compute_facts(inner: Optional[InnerResult]) -> Dict[str, Any]:
    """Run the AST analyzer on the inner-result best candidate and
    return a flat facts dict the meta-LLM can read."""
    if inner is None or inner.best is None:
        return {
            "inner_present": False,
            "M1": 0.0,
            "M6": 0.0,
            "n_features": 0,
            "n_gated": 0,
            "n_dense": 0,
            "touches_both_lidars": False,
            "role_one_hot_present": False,
        }
    metrics = inner.best.metrics
    code = inner.best.candidate.obs_source or ""
    ana = analyze_inner_code(code)
    n_gated = count_gated_features(code)
    n_dense = count_dense_features(ana, code)
    role_one_hot = bool(
        "F.one_hot(" in code
        or "one_hot(" in code
        or "[:, agent_idx] = 1" in code
        or "[:, agent_idx]=1" in code
    )
    return {
        "inner_present": True,
        "M1": float(metrics.get("M1_success_rate", 0.0)),
        "M6": float(metrics.get("M6_coverage_progress", 0.0)),
        "n_features": ana.n_returned_features,
        "n_gated": n_gated,
        "n_dense": n_dense,
        "touches_both_lidars": ana.touches_both_lidars,
        "role_one_hot_present": role_one_hot,
        "feature_stack_score": ana.feature_stack_score,
        "n_cross_source_ops": ana.n_cross_source,
    }


# ── Main entry ─────────────────────────────────────────────────


def run_v9_outer_loop(
    spec: ExperimentSpec,
    base_loop: LeroLoop,
    cfg: V9OuterConfig,
    meta_llm: LLMClient,
    output_dir: Path,
    seed: int = 0,
    task_overrides: Optional[Dict[str, Any]] = None,
    algorithm: str = "mappo",
    resume: bool = False,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from ..prompts import loader as _l_mod

    _orig_prompts_dir = Path(_l_mod.__file__).parent
    base_src = _orig_prompts_dir / cfg.base_prompt_version
    if not base_src.exists():
        raise FileNotFoundError(f"base prompt not found: {base_src}")

    # Construct the meta loader BEFORE redirecting. The loader's
    # template_dir is captured on __init__ so it survives the redirect.
    # task_domain() uses self.template_dir.parent which is the original
    # prompts root (where task_domains/ lives).
    loader = PromptLoader(version=cfg.base_prompt_version)

    # Now redirect the module-level _PROMPTS_DIR so per-outer prompt
    # copies live under the run output dir (used by the inner loop).
    prompts_root = output_dir / "prompts"
    _redirect_prompt_loader(prompts_root)

    # Memory store
    mem = MemoryStore(output_dir / "_meta_memory.jsonl")

    ckpt = _load_ckpt(output_dir) if resume else None

    # === Cold-start: bundle enumeration ===
    if ckpt is None or ckpt.bundle is None:
        _log.info(
            "v9 COLD-START: bundle enumeration (task_domain=%s)",
            (loader.task_domain() or {}).get("name", "?"),
        )
        bundle, raw_bundle = enumerate_bundle_v9(
            meta_llm,
            loader,
            cfg.task_summary or spec.description or "",
        )
        (output_dir / "_bundle_init.json").write_text(
            json.dumps(_bundle_dict(bundle), indent=2, default=str)
        )
        (output_dir / "_bundle_init_response.txt").write_text(raw_bundle)

        # Apply chosen strategy's slots to outer 0
        seed_prompt_name = f"v9_outer_0_seed{seed}"
        seed_prompt_dir = prompts_root / seed_prompt_name
        if seed_prompt_dir.exists():
            shutil.rmtree(seed_prompt_dir)
        shutil.copytree(base_src, seed_prompt_dir)
        _write_slots(seed_prompt_dir, _strategy_to_slots(bundle.current()))
        ckpt = V9Checkpoint(seed=seed, bundle=bundle)
        _save_ckpt(output_dir, ckpt)
    else:
        _log.info("v9 RESUME — outer_done=%d", ckpt.outer_idx_completed)

    bundle = ckpt.bundle
    assert bundle is not None
    bundle_size = len(bundle.strategies)
    max_outer = (
        cfg.max_outer_override if cfg.max_outer_override is not None else bundle_size
    )

    t_start = time.monotonic()

    for outer_idx in range(max_outer):
        if outer_idx <= ckpt.outer_idx_completed:
            continue
        if ckpt.early_stopped:
            break

        # ── Fail-safe: detect pathological refine_current loop ─────────
        # See `detect_pathological_refine`. Overrides the meta-LLM
        # decision and switches to the next pending strategy BEFORE
        # running another inner loop on a regressing strategy.
        if outer_idx > 0:
            cur_name = bundle.current().name
            recent = mem.read_recent(2)
            if detect_pathological_refine(recent, cur_name, n=2):
                m6_seq = [float((r.get("actual") or {}).get("M6", 0.0)) for r in recent]
                _log.warning(
                    "v9 outer %d: FORCED SWITCH — strategy '%s' "
                    "showed M6=%s across last 2 outers (monotonic "
                    "non-increasing). Marking excluded.",
                    outer_idx,
                    cur_name,
                    m6_seq,
                )
                bundle.current().excluded = True
                nb = bundle.next_pending_idx()
                if nb is None:
                    _log.warning(
                        "v9 outer %d: no pending strategy left after "
                        "forced switch; stopping",
                        outer_idx,
                    )
                    ckpt.early_stopped = True
                    ckpt.outer_idx_completed = outer_idx - 1
                    ckpt.bundle = bundle
                    _save_ckpt(output_dir, ckpt)
                    break
                bundle.chosen_idx = nb
                forced_prompt_name = f"v9_outer_{outer_idx}_seed{seed}"
                forced_prompt_dir = prompts_root / forced_prompt_name
                if forced_prompt_dir.exists():
                    shutil.rmtree(forced_prompt_dir)
                shutil.copytree(base_src, forced_prompt_dir)
                _write_slots(
                    forced_prompt_dir,
                    _strategy_to_slots(bundle.current()),
                )
                _log.info(
                    "v9 outer %d: switched to '%s' via fail-safe",
                    outer_idx,
                    bundle.current().name,
                )

        _log.info(
            "=== v9 OUTER ITER %d/%d (strategy=%s) ===",
            outer_idx + 1,
            max_outer,
            bundle.current().name,
        )

        outer_dir = output_dir / f"outer_{outer_idx:02d}"
        outer_dir.mkdir(exist_ok=True)
        active = bundle.current()
        active.attempts += 1
        (outer_dir / "_active_strategy.json").write_text(
            json.dumps(_strategy_dict(active), indent=2, default=str)
        )
        (outer_dir / "_bundle_state_before.json").write_text(
            json.dumps(_bundle_dict(bundle), indent=2, default=str)
        )

        base_loop.lero.evolve_observation = True
        base_loop.lero.evolve_reward = False

        prompt_name = f"v9_outer_{outer_idx}_seed{seed}"
        prompt_dir = prompts_root / prompt_name

        # Run inner LERO with current slots. v9.1 §2.1: pre-eval
        # validator AST-rejects mandatory_features-missing candidates
        # before training (saves ~9 min/cand).
        inner_out_dir = outer_dir / "inner"
        td_for_validator = loader.task_domain() or {}
        pre_eval = make_pre_eval_validator(td_for_validator)
        inner_result = run_inner_loop(
            base_loop=base_loop,
            metaprompt_version=prompt_name,
            n_iterations=cfg.n_inner_iter,
            n_candidates_per_iter=cfg.n_inner_candidates_per_iter,
            seed=seed,
            output_dir=inner_out_dir,
            task_overrides=task_overrides,
            algorithm=algorithm,
            pre_eval_validator=pre_eval,
        )

        # Compute analyzer facts locally (replaces v8 diagnose).
        facts = _compute_facts(inner_result)
        (outer_dir / "_facts.json").write_text(json.dumps(facts, indent=2, default=str))
        _log.info(
            "v9 outer %d facts: M1=%.3f M6=%.3f n_feat=%d gated=%d "
            "dense=%d role_one_hot=%s",
            outer_idx,
            facts["M1"],
            facts["M6"],
            facts["n_features"],
            facts["n_gated"],
            facts["n_dense"],
            facts["role_one_hot_present"],
        )

        # Read memory (last N) for the reflect call
        memory_rows = mem.read_recent(cfg.memory_lookback_n)

        # Combined diagnose+reflect+decide single LLM call
        decision, raw_reflect = reflect_decide_v9(
            meta_llm,
            loader,
            bundle,
            facts,
            memory_rows,
        )
        (outer_dir / "_reflection_response.txt").write_text(raw_reflect)

        # v9.1 §2.7 falsification-gate action override.
        # If the meta-LLM picked refine_current AND the strategy has
        # already been attempted ≥2 times AND every attempt undershot
        # 0.5×expected_M1, force switch_to_next regardless of the LLM's
        # per-outer label. The LLM's CoT in production showed it
        # correctly diagnosed "stuck" but the action map kept it
        # locked into refine — this gate flips it.
        original_action = decision.next_action
        gate_fired = False
        if decision.next_action == "refine_current":
            recent_with_current_facts = list(memory_rows) + [
                {
                    "strategy_name": active.name,
                    "actual": {"M1": facts["M1"]},
                }
            ]
            if detect_falsification_failure(
                memory_rows=recent_with_current_facts,
                current_strategy_name=active.name,
                expected_M1=active.success_signature.expected_M1_at_1M,
                n_attempts=2,
                threshold_factor=0.5,
            ):
                _log.warning(
                    "v9 outer %d: §2.7 OVERRIDE — strategy '%s' has "
                    "expected_M1=%.3f but actual M1 stayed below "
                    "0.5×expected for ≥2 attempts. Flipping "
                    "refine_current → switch_to_next.",
                    outer_idx,
                    active.name,
                    active.success_signature.expected_M1_at_1M,
                )
                decision.next_action = "switch_to_next"
                decision.rationale = (
                    "[v9.1 §2.7 override] "
                    + decision.rationale
                    + " — falsification gate fired (≥2 attempts below "
                    f"0.5×expected_M1={active.success_signature.expected_M1_at_1M:.3f})"
                )
                gate_fired = True

        (outer_dir / "_decision.json").write_text(
            json.dumps(
                {
                    "next_action": decision.next_action,
                    "next_action_original_from_llm": original_action,
                    "falsification_gate_fired": gate_fired,
                    "label": decision.label,
                    "rationale": decision.rationale,
                    "memory_recall": decision.memory_recall,
                    "diff_vs_predicted": decision.diff_vs_predicted,
                    "reflection_chain_of_thought": decision.reflection_cot,
                    "slot_edits_keys": list(decision.slot_edits.keys()),
                    "bundle_demote": decision.bundle_demote,
                    "bundle_add_names": [s.name for s in decision.bundle_add],
                },
                indent=2,
                default=str,
            )
        )
        _log.info(
            "v9 outer %d decision: label=%s action=%s%s",
            outer_idx,
            decision.label,
            decision.next_action,
            " (overridden from refine_current)" if gate_fired else "",
        )

        # Record outcome on the active strategy
        active.last_outcome = decision.label
        active.last_M1 = facts["M1"]
        active.last_M6 = facts["M6"]
        active.last_pattern_present = facts["touches_both_lidars"]
        bundle.history.append(
            {
                "outer_idx": outer_idx,
                "strategy_name": active.name,
                "label": decision.label,
                "M1": facts["M1"],
                "M6": facts["M6"],
                "pattern_present": facts["touches_both_lidars"],
                "role_one_hot_present": facts["role_one_hot_present"],
            }
        )

        # Append to memory
        mem.append(
            MemoryRow(
                outer_idx=outer_idx,
                ts=time.strftime("%Y-%m-%dT%H:%M:%S"),
                strategy_name=active.name,
                predicted={
                    "M1": active.success_signature.expected_M1_at_1M,
                    "M6": active.success_signature.expected_M6_at_1M_min,
                    "what_is_needed": active.chain_of_thought.what_is_needed,
                },
                actual={
                    "M1": facts["M1"],
                    "M6": facts["M6"],
                    "diagnosis_label": decision.label,
                    "n_features": facts["n_features"],
                    "role_one_hot_present": facts["role_one_hot_present"],
                },
                delta={
                    "M1": (facts["M1"] - active.success_signature.expected_M1_at_1M),
                    "M6": (
                        facts["M6"] - active.success_signature.expected_M6_at_1M_min
                    ),
                },
                chain_of_thought={
                    "why_it_works": active.chain_of_thought.why_it_works,
                    "what_is_needed": active.chain_of_thought.what_is_needed,
                    "failure_modes": active.chain_of_thought.failure_modes,
                },
                post_hoc_reflection=decision.reflection_cot,
            )
        )

        # Apply bundle updates (demote / add)
        for name in decision.bundle_demote:
            for s in bundle.strategies:
                if s.name == name:
                    s.excluded = True
        if decision.bundle_add:
            bundle.strategies.extend(decision.bundle_add)

        # Apply next_action
        if decision.next_action == "stop":
            ckpt.early_stopped = True
            ckpt.outer_idx_completed = outer_idx
            ckpt.bundle = bundle
            _save_ckpt(output_dir, ckpt)
            break

        # Prepare next outer's prompt dir
        next_prompt_name = f"v9_outer_{outer_idx + 1}_seed{seed}"
        next_prompt_dir = prompts_root / next_prompt_name
        if next_prompt_dir.exists():
            shutil.rmtree(next_prompt_dir)
        shutil.copytree(base_src, next_prompt_dir)

        if decision.next_action == "switch_to_next":
            # Mark current excluded, pick next pending strategy
            active.excluded = True
            nb = bundle.next_pending_idx()
            if nb is None:
                _log.warning(
                    "v9 outer %d: switch requested but no pending "
                    "strategy left; stopping",
                    outer_idx,
                )
                ckpt.early_stopped = True
                ckpt.outer_idx_completed = outer_idx
                ckpt.bundle = bundle
                _save_ckpt(output_dir, ckpt)
                break
            bundle.chosen_idx = nb
            new_active = bundle.current()

            # v9.1 §2.10: bundle enum only authors artifacts for the
            # original chosen strategy. Strategies activated via
            # switch_to_next have empty V9Artifacts(). Lazily author
            # them now via one extra LLM call.
            arts = new_active.artifacts
            if not (arts.inferable_hints_text and arts.examples_text):
                _log.info(
                    "v9 outer %d: §2.10 lazy-authoring artifacts for '%s'",
                    outer_idx,
                    new_active.name,
                )
                from .meta_strategist import author_artifacts_for_strategy

                try:
                    new_arts = author_artifacts_for_strategy(
                        meta_llm,
                        loader,
                        new_active,
                    )
                    new_active.artifacts = new_arts
                except Exception as e:  # noqa: BLE001
                    _log.error(
                        "v9 outer %d: §2.10 artifact authoring failed "
                        "for '%s': %s — falling back to prior outer's "
                        "slot text",
                        outer_idx,
                        new_active.name,
                        e,
                    )
                    # Fallback: copy the prior outer's slot files into
                    # the new strategy's artifacts so something works.
                    prev_slots = _read_slot_files(prompt_dir)
                    new_active.artifacts = V9Artifacts(
                        inferable_hints_text=prev_slots.get("inferable_hints", ""),
                        examples_text=prev_slots.get("examples", ""),
                        feedback_template=prev_slots.get("feedback", ""),
                    )

            _write_slots(
                next_prompt_dir,
                _strategy_to_slots(new_active),
            )
            _log.info(
                "v9 outer %d: SWITCHED to '%s'",
                outer_idx,
                new_active.name,
            )
        elif decision.next_action == "refine_current":
            # v9.1 §2.3: validate slot_edits structurally before
            # applying. Reject prose-only edits that drop python
            # examples or strip the inferable_concepts list.
            from .slot_validator import validate_slot_edits

            current_slots = _read_slot_files(prompt_dir)
            td_for_validate = loader.task_domain() or {}
            validation_results = validate_slot_edits(
                slot_edits=decision.slot_edits,
                task_domain=td_for_validate,
                prev_slots=current_slots,
            )
            # Build the actual edits to apply: caller's edit if it
            # passed, else keep previous slot text.
            new_slots = dict(current_slots)
            applied_edits: Dict[str, str] = {}
            rejection_log: Dict[str, List[str]] = {}
            for k, v in decision.slot_edits.items():
                if k not in _V9_SLOT_NAMES or not v:
                    continue
                vr = validation_results.get(k)
                if vr is not None and not vr.passed:
                    rejection_log[k] = vr.issues
                    _log.warning(
                        "v9 outer %d: REJECTING slot_edit '%s' " "(keeping prev) — %s",
                        outer_idx,
                        k,
                        "; ".join(vr.issues),
                    )
                    continue
                new_slots[k] = v
                applied_edits[k] = v
            (outer_dir / "_slot_edit_validation.json").write_text(
                json.dumps(
                    {
                        "rejected": rejection_log,
                        "applied": list(applied_edits.keys()),
                        "validator_metrics": {
                            k: vr.metrics for k, vr in validation_results.items()
                        },
                    },
                    indent=2,
                    default=str,
                )
            )
            _write_slots(next_prompt_dir, new_slots)
            # Persist only APPLIED edits back onto the strategy artifacts
            # (rejected edits don't pollute the bundle history).
            if "inferable_hints" in applied_edits:
                active.artifacts.inferable_hints_text = applied_edits["inferable_hints"]
            if "examples" in applied_edits:
                active.artifacts.examples_text = applied_edits["examples"]
            if "feedback_template" in applied_edits:
                active.artifacts.feedback_template = applied_edits["feedback_template"]
        else:
            _log.warning(
                "v9 outer %d: unknown next_action %r — defaulting to refine",
                outer_idx,
                decision.next_action,
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
        _save_ckpt(output_dir, ckpt)

    # === Final summary ===
    summary = {
        "early_stopped": ckpt.early_stopped,
        "outer_iters_completed": ckpt.outer_idx_completed + 1,
        "bundle_size": bundle_size,
        "bundle_chosen_at_end": (bundle.current().name if bundle.strategies else None),
        "bundle_history": bundle.history,
        "bundle_final": _bundle_dict(bundle),
        "memory_rows": len(mem),
        "elapsed_s_total": ckpt.elapsed_s_so_far + (time.monotonic() - t_start),
    }
    (output_dir / "v9_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    ckpt.final_summary = summary
    _save_ckpt(output_dir, ckpt)
    _log.info(
        "=== v9 DONE === outers=%d/%d early=%s strategy=%s",
        ckpt.outer_idx_completed + 1,
        max_outer,
        ckpt.early_stopped,
        bundle.current().name if bundle.strategies else None,
    )
    return summary
