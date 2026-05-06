"""Phase 1 harness — drives v6 meta_strategist + inner_llm with
synthetic inputs and returns prompt-quality metrics in seconds.

No RL training, no scenario, no PyTorch beyond the analyzer's AST
checks. Each call:

  meta-LLM (~5-15s)  →  inner LLM × N (~5-10s each)  →  analyze + judge
                                                          (≤3s + ~3s/cand)

A typical N=9 sweep takes ~60-90s total LLM cost, ~$0.10-$0.30.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..codegen import CandidateCode
from ..inner_llm import CandidateGenerationFailed, InnerLLM
from ..llm_client import LLMClient
from ..prompts.loader import PromptLoader
from ..v5.inner_loop import CandidateOutcome, InnerResult
from ..v5.registry import Registry, RegistryEntry
from ..v6.decision import V6MetaDecision, classify_inner_result, enforce_decision
from ..v6.meta_strategist import _build_meta_prompt, _parse_decision
from .analyzer import (
    CodeAnalysis,
    analyze_inner_code,
)
from .judge import JudgeResult, avg_judge_score, judge_batch

_log = logging.getLogger("rendezvous.lero.v6_prompt_lab.harness")


# ── Synthetic input factories ────────────────────────────────────


def make_synthetic_inner_result(
    best_obs_code: str,
    worst_obs_code: str = "",
    best_M1: float = 0.000,
    best_M6: float = 0.130,
    best_fitness: float = 0.060,
    best_shape: str = "flat_zero",
    fitness_trajectory: Optional[List[float]] = None,
) -> InnerResult:
    """Build an InnerResult that looks like what v6's meta-LLM would
    receive after a typical no-signal inner run.

    Defaults match the empirical v6 outer-0 footprint (M1=0, M6 ~ 0.13,
    fitness ~ 0.06, shape flat_zero).
    """
    if fitness_trajectory is None:
        fitness_trajectory = [0.060, 0.055, 0.060]

    best_cand = CandidateCode(
        obs_source=best_obs_code,
        reward_source=None,
        raw_response="<synthetic>",
    )
    worst_cand = CandidateCode(
        obs_source=worst_obs_code or best_obs_code,
        reward_source=None,
        raw_response="<synthetic>",
    )

    best_outcome = CandidateOutcome(
        candidate=best_cand,
        metrics={
            "M1_success_rate": best_M1,
            "M6_coverage_progress": best_M6,
            "M2_avg_return": -3.0,
            "M4_avg_collisions": 5.0,
        },
        fitness=best_fitness,
        shape=best_shape,
        iter_idx=0,
    )
    worst_outcome = CandidateOutcome(
        candidate=worst_cand,
        metrics={
            "M1_success_rate": 0.0,
            "M6_coverage_progress": 0.05,
            "M2_avg_return": -3.5,
            "M4_avg_collisions": 8.0,
        },
        fitness=-0.50,
        shape="flat_zero",
        iter_idx=0,
    )

    registry = Registry()
    registry.fitness_trajectory = list(fitness_trajectory)
    registry.add(
        RegistryEntry(
            iter_idx=0,
            handle="i0_r1_synthetic_best",
            summary="synthetic best",
            fitness=best_fitness,
            M1=best_M1,
            shape=best_shape,
            code_excerpt=best_obs_code[:1500],
        )
    )
    registry.add(
        RegistryEntry(
            iter_idx=0,
            handle="i0_r3_synthetic_worst",
            summary="synthetic worst",
            fitness=-0.50,
            M1=0.0,
            shape="flat_zero",
            code_excerpt=worst_obs_code[:1500],
        )
    )

    return InnerResult(
        best=best_outcome,
        worst=worst_outcome,
        all_outcomes=[best_outcome, worst_outcome],
        registry=registry,
        did_stagnate=False,
        n_iters_run=len(fitness_trajectory),
    )


def load_v6_old_best_obs(
    run: str = "run4_mini_replicate_20260430_0417",
    outer: str = "00",
    iter_dir: str = "iter_2",
) -> str:
    """Pull a real v6 run's best obs code as the synthetic-input
    distribution (a typical 'no signal' candidate)."""
    f = Path(
        f"results/lero_v6/lero_v6_rendezvous_k2_2x3/{run}/"
        f"outer_{outer}_inner/{iter_dir}/feedback.txt"
    )
    text = f.read_text()
    import re

    m = re.search(
        r"Best candidate.*?observation code:\s*```python(.+?)```",
        text,
        re.DOTALL,
    )
    if not m:
        return ""
    return m.group(1).strip()


# ── Custom-system meta-LLM call ──────────────────────────────────


def _call_meta_with_custom_system(
    system_prompt: str,
    user_prompt: str,
    meta_llm: LLMClient,
) -> Tuple[V6MetaDecision, str]:
    """Call meta-LLM with a custom system prompt and parse the
    V6MetaDecision. Mirrors v6.meta_strategist.call_meta_strategist
    but allows the caller to override _META_SYSTEM."""
    last_err: Optional[Exception] = None
    raw_text = ""
    decision: Optional[V6MetaDecision] = None
    for attempt in range(1, 4):
        raw_text = meta_llm.generate(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            n=1,
        )[0]
        try:
            decision = _parse_decision(raw_text)
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
    if decision is None:
        raise ValueError(f"meta parse failed: {last_err}")
    return decision, raw_text


# ── Slot-edit application ────────────────────────────────────────


def _read_base_slots(prompt_version: str) -> Dict[str, str]:
    base_dir = Path(__file__).parent.parent / "prompts" / prompt_version
    out: Dict[str, str] = {}
    for slot in ("guidance_observation", "guidance_reward", "guidance_shared"):
        p = base_dir / f"{slot}.txt"
        out[slot] = p.read_text() if p.exists() else ""
    return out


def _apply_slot_edits(
    base_slots: Dict[str, str],
    edits: Dict[str, str],
) -> Dict[str, str]:
    new = dict(base_slots)
    for k, v in edits.items():
        new[k] = (v or "").rstrip() + "\n"
    return new


# ── Inner LLM with custom slots ──────────────────────────────────


def _build_inner_messages_with_slots(
    prompt_version: str,
    slot_overrides: Dict[str, str],
    task_overrides: Dict,
    inner_llm: InnerLLM,
) -> List[Dict[str, str]]:
    """Render the inner prompt by writing slot_overrides into a temp
    copy of the base prompt directory, then loading via PromptLoader.
    Cleans up the temp copy after rendering.
    """
    import shutil
    import tempfile

    base_dir = Path(__file__).parent.parent / "prompts" / prompt_version
    with tempfile.TemporaryDirectory() as tmp:
        new_dir = Path(tmp) / prompt_version
        shutil.copytree(base_dir, new_dir)
        for slot, text in slot_overrides.items():
            (new_dir / f"{slot}.txt").write_text((text or "").rstrip() + "\n")
        # Repoint loader at temp parent
        from ..prompts import loader as _l

        orig = _l._PROMPTS_DIR
        try:
            _l._PROMPTS_DIR = Path(tmp)
            loader = PromptLoader(version=prompt_version)
            system_text = loader.render(
                "system.txt",
                experiment_context="",
                covering_range=task_overrides.get("covering_range", 0.25),
            )
            user_text = loader.render(
                "initial_user.txt",
                output_spec_variant="obs_only",
                **task_overrides,
                experiment_context="",
                agent_lidar_description="",
                comm_description="",
                reward_description="",
                coordination_guidance="",
                comm_state_description="",
                obs_lidar_agents="",
                obs_comm_state="",
                comm_obs_guidance="",
                scenario_reward_code="",
                scenario_observation_code="",
            )
        finally:
            _l._PROMPTS_DIR = orig
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]


# ── End-to-end harness call ──────────────────────────────────────


@dataclass
class PromptTrialResult:
    variant_name: str
    meta_decision: V6MetaDecision
    meta_raw_text: str
    inner_candidates: List[CandidateCode] = field(default_factory=list)
    analyses: List[CodeAnalysis] = field(default_factory=list)
    judge_results: List[JudgeResult] = field(default_factory=list)
    n_generation_failures: int = 0
    elapsed_s: float = 0.0

    @property
    def cross_source_rate(self) -> float:
        if not self.analyses:
            return 0.0
        return sum(1 for a in self.analyses if a.touches_both_lidars) / len(
            self.analyses
        )

    @property
    def avg_cross_source_ops(self) -> float:
        if not self.analyses:
            return 0.0
        return sum(a.n_cross_source for a in self.analyses) / len(self.analyses)

    @property
    def avg_judge(self) -> float:
        return avg_judge_score(self.judge_results)

    def summary_dict(self) -> Dict:
        return {
            "variant_name": self.variant_name,
            "n_inner_candidates": len(self.inner_candidates),
            "n_generation_failures": self.n_generation_failures,
            "cross_source_rate": self.cross_source_rate,
            "avg_cross_source_ops": self.avg_cross_source_ops,
            "avg_judge_total": self.avg_judge,
            "judge_breakdown_avg": self._avg_breakdown(),
            "meta_classification": self.meta_decision.classification,
            "meta_next_mode": self.meta_decision.next_mode,
            "meta_complexity": self.meta_decision.complexity_level,
            "meta_evolve_obs": self.meta_decision.next_evolve_observation,
            "meta_evolve_rew": self.meta_decision.next_evolve_reward,
            "meta_slot_edits_keys": list(self.meta_decision.slot_edits.keys()),
            "meta_enforcement_notes": self.meta_decision.enforcement_notes,
            "elapsed_s": self.elapsed_s,
        }

    def _avg_breakdown(self) -> Dict[str, float]:
        if not self.judge_results:
            return {}
        keys = self.judge_results[0].score_breakdown.keys()
        return {
            k: sum(r.score_breakdown.get(k, 0) for r in self.judge_results)
            / len(self.judge_results)
            for k in keys
        }


def run_harness_trial(
    variant_name: str,
    meta_system_prompt: str,
    base_prompt_version: str,
    synthetic_inner: InnerResult,
    task_summary: str,
    task_overrides: Dict,
    n_inner_candidates: int,
    meta_llm: LLMClient,
    inner_llm_client: LLMClient,
    judge_llm: LLMClient,
    outer_idx: int = 1,  # >0 so reward unlock is allowed
    prior_complexity: int = 1,
    prior_classification: str = "no_signal_simple",
) -> PromptTrialResult:
    """One end-to-end harness trial:

      1. Call meta-LLM with `meta_system_prompt` + standard v6 user prompt
         constructed from `synthetic_inner`.
      2. Apply enforcement. Get V6MetaDecision.
      3. Apply slot_edits to base prompt → inner prompt.
      4. Generate `n_inner_candidates` inner candidates.
      5. AST-analyze + LLM-judge each.
      6. Return PromptTrialResult.

    Default outer_idx=1 so reward-unlock is policy-allowed if the meta
    chooses to add reward; phase-2 variants test obs-only behavior so
    the meta usually stays obs-only anyway.
    """
    t0 = time.monotonic()

    # 1. Build user prompt for meta-LLM
    base_slots = _read_base_slots(base_prompt_version)
    user_prompt = _build_meta_prompt(
        task_summary=task_summary,
        current_slots=base_slots,
        last_inner=synthetic_inner,
        outer_registry=Registry(),
        outer_idx=outer_idx,
        prior_complexity=prior_complexity,
        prior_classification=prior_classification,
    )

    # 2. Call meta-LLM with custom system, parse, enforce
    raw_decision, raw_text = _call_meta_with_custom_system(
        system_prompt=meta_system_prompt,
        user_prompt=user_prompt,
        meta_llm=meta_llm,
    )
    code_class = classify_inner_result(
        best_M1=float(synthetic_inner.best.metrics.get("M1_success_rate", 0.0)),
        best_shape=synthetic_inner.best.shape,
        fitness_trajectory=list(synthetic_inner.registry.fitness_trajectory),
        best_M6=float(synthetic_inner.best.metrics.get("M6_coverage_progress", 0.0)),
        prior_complexity=prior_complexity,
    )
    decision = enforce_decision(
        raw=raw_decision,
        outer_idx=outer_idx,
        prior_complexity=prior_complexity,
        prior_classification=prior_classification,
        code_classification=code_class,
    )

    # 3. Apply slot edits → render inner prompt
    new_slots = _apply_slot_edits(base_slots, decision.slot_edits)
    inner_llm = InnerLLM(
        inner_llm_client,
        evolve_reward=decision.next_evolve_reward,
        evolve_observation=decision.next_evolve_observation,
        use_structured=False,
    )
    messages = _build_inner_messages_with_slots(
        prompt_version=base_prompt_version,
        slot_overrides=new_slots,
        task_overrides=task_overrides,
        inner_llm=inner_llm,
    )

    # 4. Generate inner candidates
    candidates: List[CandidateCode] = []
    n_failures = 0
    for c_idx in range(n_inner_candidates):
        try:
            c = inner_llm.generate(messages, seed_base=10000 + c_idx)
            candidates.append(c)
        except CandidateGenerationFailed as e:
            n_failures += 1
            _log.warning("inner gen failed cand=%d: %s", c_idx, e)

    # 5. Analyze + judge
    analyses = [analyze_inner_code(c.obs_source or "") for c in candidates]
    judge_results = judge_batch(
        [c.obs_source or "" for c in candidates],
        judge_llm,
    )

    elapsed = time.monotonic() - t0
    return PromptTrialResult(
        variant_name=variant_name,
        meta_decision=decision,
        meta_raw_text=raw_text,
        inner_candidates=candidates,
        analyses=analyses,
        judge_results=judge_results,
        n_generation_failures=n_failures,
        elapsed_s=elapsed,
    )
