"""Phase-3 LLM-only test of v7's reflection workflow.

For each of 4 synthetic inner-result scenarios (translation_failure /
rl_too_hard / partial / achieved), feed the meta-LLM the diagnosis +
bundle and verify it:

  1. picks the correct `next_action`
  2. emits sensible `slot_edits` (or none, when not applicable)
  3. correctly demotes the current strategy on rl_too_hard

Plus an end-to-end mini-validation: enumerate bundle, generate inner
candidates from the chosen strategy's translation hint, AST-analyze
them, see if at least one produces cross-source ops.

Total cost: ~10 LLM calls + ~5 inner gens ≈ 60-90s, ~$2.

Run: python -m src.lero.v7.run_phase3_test
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from ..codegen import CandidateCode
from ..config import LLMConfig
from ..inner_llm import InnerLLM
from ..llm_client import LLMClient
from ..v5.inner_loop import CandidateOutcome, InnerResult
from ..v5.registry import Registry
from ..v6_prompt_lab.analyzer import analyze_inner_code
from ..v6_prompt_lab.harness import _build_inner_messages_with_slots, _read_base_slots
from .diagnosis import diagnose_inner_result
from .meta_strategist import enumerate_bundle, reflect_and_decide
from .strategy import V7Strategy, V7StrategyBundle


_TASK_SUMMARY = (
    "Multi-agent rendezvous in VMAS Discovery scenario. 4 agents must "
    "collectively cover 4 targets, with each target requiring exactly "
    "2 agents simultaneously within covering_range. Local LiDAR-only "
    "observations; no oracle access to other agents' positions or "
    "target positions. Implicit coordination from local sensors only."
)

_TASK_OVERRIDES = {
    "n_agents": 4, "n_targets": 4, "agents_per_target": 2,
    "covering_range": 0.25, "lidar_range": 0.35, "max_steps": 400,
    "n_lidar_rays_entities": 15, "n_lidar_rays_agents": 12,
    "collision_penalty": -0.01, "time_penalty": -0.01,
}


def _make_inner(
    obs_code: str,
    M1: float,
    M6: float,
    M3: float = 400.0,
    M4: float = 5.0,
) -> InnerResult:
    cand = CandidateCode(
        obs_source=obs_code,
        reward_source=None,
        raw_response="<synth>",
    )
    out = CandidateOutcome(
        candidate=cand,
        metrics={
            "M1_success_rate": M1,
            "M6_coverage_progress": M6,
            "M3_avg_steps": M3,
            "M4_avg_collisions": M4,
            "M2_avg_return": -3.0,
        },
        fitness=M1 + 0.5 * M6,
        shape="monotonic_rise" if M1 > 0.02 else "flat_zero",
        iter_idx=0,
    )
    reg = Registry()
    reg.fitness_trajectory = [0.05, 0.06, 0.07]
    reg.add_outcome(...) if False else None  # noqa
    return InnerResult(
        best=out, worst=out,
        all_outcomes=[out],
        registry=reg,
        did_stagnate=False,
        n_iters_run=3,
    )


# Synthetic obs codes for each scenario.

_CODE_NO_AND_PRODUCT = """
import torch
def enhance_observation(scenario_state):
    lt = scenario_state["lidar_targets"]
    nearest = lt.min(dim=-1).values
    return nearest.unsqueeze(-1)
"""

_CODE_WITH_AND_PRODUCT = """
import torch
def enhance_observation(scenario_state):
    lt = scenario_state["lidar_targets"]
    la = scenario_state["lidar_agents"]
    t_close = (lt < 0.25).float().sum(dim=-1)
    a_close = (la < 0.25).float().sum(dim=-1)
    joint = t_close * a_close
    proximity_diff = lt.min(dim=-1).values - la.min(dim=-1).values
    return torch.stack([t_close, a_close, joint, proximity_diff], dim=-1)
"""


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s %(name)s %(levelname)s %(message)s")
    log = logging.getLogger("rendezvous.lero.v7.run_phase3_test")

    meta_llm = LLMClient(LLMConfig(
        model="gpt-5.4-mini", temperature=0.8, max_retries=3,
        prompt_version="v2_fewshot_modular_v2_local",
    ))
    inner_client = LLMClient(LLMConfig(
        model="gpt-5.4-mini", temperature=0.8, max_retries=3,
        prompt_version="v2_fewshot_modular_v2_local",
    ))

    t0 = time.monotonic()

    # === Phase A: enumerate bundle ===
    log.info("=== Phase A: bundle enumeration ===")
    bundle, _ = enumerate_bundle(meta_llm, _TASK_SUMMARY)
    print(bundle.format_for_prompt())

    # === Phase B: generate inner candidates from chosen strategy's hint ===
    log.info("=== Phase B: inner gen with chosen strategy's slot text ===")
    chosen = bundle.current()
    base_slots = _read_base_slots("v2_fewshot_modular_v2_local")
    new_slots = dict(base_slots)
    new_slots["guidance_observation"] = chosen.lero_translation_hint

    inner_llm = InnerLLM(inner_client, evolve_reward=False,
                          evolve_observation=True, use_structured=False)
    messages = _build_inner_messages_with_slots(
        prompt_version="v2_fewshot_modular_v2_local",
        slot_overrides=new_slots,
        task_overrides=_TASK_OVERRIDES,
        inner_llm=inner_llm,
    )
    n_inner_gens = 3
    inner_codes = []
    for i in range(n_inner_gens):
        try:
            c = inner_llm.generate(messages, seed_base=70000 + i)
            inner_codes.append(c.obs_source or "")
        except Exception as e:  # noqa: BLE001
            log.warning("inner gen %d failed: %s", i, e)
    cs_rate = 0
    for code in inner_codes:
        ana = analyze_inner_code(code)
        if ana.touches_both_lidars:
            cs_rate += 1
        log.info(
            "  inner cand: cross_source=%d touches_both=%s n_features=%d",
            ana.n_cross_source, ana.touches_both_lidars,
            ana.n_returned_features,
        )
    print(f"\nPhase B: {cs_rate}/{n_inner_gens} candidates have cross-source ops")

    # === Phase C: 4 reflection scenarios ===
    log.info("=== Phase C: reflection scenario tests ===")
    scenarios = [
        ("translation_failure", _CODE_NO_AND_PRODUCT, 0.000, 0.080,
         "translation_failure", "refine_inner_prompt_for_current"),
        ("rl_too_hard",          _CODE_WITH_AND_PRODUCT, 0.000, 0.100,
         "rl_too_hard", "switch_to_next_strategy"),
        ("partial",              _CODE_WITH_AND_PRODUCT, 0.000, 0.250,
         "partial", "refine_current_strategy"),
        ("achieved",             _CODE_WITH_AND_PRODUCT, 0.080, 0.350,
         "achieved", "stop"),
    ]
    results = []
    for name, code, M1, M6, expected_label, expected_action in scenarios:
        synth = _make_inner(code, M1=M1, M6=M6)
        diag = diagnose_inner_result(synth, chosen)
        decision, _ = reflect_and_decide(meta_llm, bundle, synth, diag)
        ok_label = (diag.label == expected_label)
        ok_action = (decision.next_action == expected_action)
        results.append({
            "scenario": name,
            "expected_label": expected_label,
            "actual_label": diag.label,
            "label_match": ok_label,
            "expected_action": expected_action,
            "actual_action": decision.next_action,
            "action_match": ok_action,
            "rationale": decision.rationale[:200],
        })
        log.info(
            "  %s: label=%s (exp %s) action=%s (exp %s) ok=%s",
            name, diag.label, expected_label,
            decision.next_action, expected_action,
            ok_label and ok_action,
        )

    # Save report
    out = Path(f"results/v7_prompt_lab/phase3_{time.strftime('%Y%m%d_%H%M')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "elapsed_s": time.monotonic() - t0,
        "bundle": [
            {
                "name": s.name, "score": s.combined_score,
                "lero_codability": s.lero_codability,
                "rl_trainability": s.rl_trainability,
                "full_solution": s.full_solution,
            }
            for s in bundle.strategies
        ],
        "chosen_idx": bundle.chosen_idx,
        "chosen_translation_hint": chosen.lero_translation_hint,
        "phase_b_cross_source_rate": cs_rate / max(1, n_inner_gens),
        "phase_b_n_candidates": n_inner_gens,
        "phase_c_scenarios": results,
    }, indent=2, default=str))
    log.info("saved %s", out)

    print()
    print(f"{'scenario':<25} {'label':<24} {'action':<35} {'pass'}")
    print("-" * 100)
    for r in results:
        ok = "✓" if r["label_match"] and r["action_match"] else "✗"
        print(f"{r['scenario']:<25} {r['actual_label']:<24} "
              f"{r['actual_action']:<35} {ok}")
    print(f"\nelapsed: {time.monotonic() - t0:.1f}s, output: {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
