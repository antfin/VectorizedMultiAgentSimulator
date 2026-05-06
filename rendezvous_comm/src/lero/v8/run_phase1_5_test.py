"""Phase-1.5 LLM-only test of v8's density-first prompts and reflection.

Verifies the v8 design before committing to a 3h Mac RL run:

  Phase A: enumerate v8 bundle with default caps (8-12, hard 15, gated 2)
           - 3-5 strategies returned
           - chosen strategy's lero_translation_hint contains a ```python```
             fenced 3-4 feature working fewshot
           - fewshot does NOT use S3b-local handles (anti-cheat)

  Phase B: generate inner candidates from chosen strategy's hint, verify:
           - n_features ≤ 15 (cap)
           - n_gated ≤ 2 (gated cap)
           - n_dense ≥ 3 (density target)
           - touches_both_lidars (cross-source pattern still present)

  Phase C: feed reflect_and_decide_v8 four synthetic scenarios — verify
           it picks the correct v8 next_action for each:
             too_many_features → trim_features
             over_gated        → replace_gated_with_dense
             rl_too_hard       → switch_to_next_strategy
             achieved          → stop

Total cost: ~10 LLM calls + ~5 inner gens ≈ 60-90s, ~$2-3.

Run: python -m src.lero.v8.run_phase1_5_test
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from ..codegen import CandidateCode
from ..config import LLMConfig
from ..inner_llm import InnerLLM
from ..llm_client import LLMClient
from ..v5.inner_loop import CandidateOutcome, InnerResult
from ..v5.registry import Registry
from ..v6_prompt_lab.analyzer import (
    analyze_inner_code,
    count_dense_features,
    count_gated_features,
)
from ..v6_prompt_lab.harness import _build_inner_messages_with_slots, _read_base_slots
from .diagnosis import diagnose_inner_result_v8
from .meta_strategist import enumerate_bundle_v8, reflect_and_decide_v8


_TASK_SUMMARY = (
    "Multi-agent rendezvous in VMAS Discovery. 4 agents must collectively "
    "cover 4 targets, with each target requiring exactly 2 agents "
    "simultaneously within covering_range=0.25. Local LiDAR-only "
    "observations; no oracle access. Implicit coordination from local "
    "sensors only."
)

_TASK_OVERRIDES = {
    "n_agents": 4,
    "n_targets": 4,
    "agents_per_target": 2,
    "covering_range": 0.25,
    "lidar_range": 0.35,
    "max_steps": 400,
    "n_lidar_rays_entities": 15,
    "n_lidar_rays_agents": 12,
    "collision_penalty": -0.01,
    "time_penalty": -0.01,
}

# Default v8 caps (matching configs/lero_v8/rendezvous_k2_2x3.yaml).
_FEATURE_TARGET_MIN = 8
_FEATURE_TARGET_MAX = 12
_FEATURE_CAP = 15
_GATED_CAP = 2

# S3b-local winning handles — must NOT appear verbatim in v8 fewshot
# (anti-cheat boundary).
_FORBIDDEN_HANDLES = [
    "hold_signal",
    "approach_signal",
    "settle_signal",
    "rendezvous_pressure",
    "t_close_mean",
    "t_dispersion",
]


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
    return InnerResult(
        best=out,
        worst=out,
        all_outcomes=[out],
        registry=reg,
        did_stagnate=False,
        n_iters_run=3,
    )


# ── Synthetic obs codes to exercise each v8 diagnosis label ──────

# Few features, dense, no cross-source — should yield translation_failure
# (we don't test this in this script; we focus on the new v8 labels).

# 20 features (over cap=15) — should yield "too_many_features".
_CODE_TOO_MANY = """
import torch
def enhance_observation(scenario_state):
    lt = scenario_state["lidar_targets"]
    la = scenario_state["lidar_agents"]
    cover_r = float(scenario_state["covering_range"])
    f01 = lt.min(dim=-1).values.unsqueeze(-1)
    f02 = la.min(dim=-1).values.unsqueeze(-1)
    f03 = lt.mean(dim=-1).unsqueeze(-1)
    f04 = la.mean(dim=-1).unsqueeze(-1)
    f05 = lt.std(dim=-1).unsqueeze(-1)
    f06 = la.std(dim=-1).unsqueeze(-1)
    f07 = (lt < cover_r).float().sum(dim=-1).unsqueeze(-1)
    f08 = (la < cover_r).float().sum(dim=-1).unsqueeze(-1)
    near_t = (lt < cover_r).float().sum(dim=-1)
    near_a = (la < cover_r).float().sum(dim=-1)
    cs = (near_t * near_a).unsqueeze(-1)
    f10 = (lt.max(dim=-1).values).unsqueeze(-1)
    f11 = (la.max(dim=-1).values).unsqueeze(-1)
    f12 = (lt.median(dim=-1).values).unsqueeze(-1)
    f13 = (la.median(dim=-1).values).unsqueeze(-1)
    f14 = scenario_state["agent_pos"]
    f15 = scenario_state["agent_vel"]
    f16 = (lt.min(dim=-1).values - la.min(dim=-1).values).unsqueeze(-1)
    f17 = ((lt < cover_r).float().mean(dim=-1)).unsqueeze(-1)
    f18 = ((la < cover_r).float().mean(dim=-1)).unsqueeze(-1)
    f19 = (lt[:, 0] - la[:, 0]).unsqueeze(-1)
    f20 = (lt[:, -1] - la[:, -1]).unsqueeze(-1)
    return torch.cat([f01, f02, f03, f04, f05, f06, f07, f08,
                      cs, f10, f11, f12, f13, f14, f15,
                      f16, f17, f18, f19, f20], dim=-1)
"""

# 7 features, mostly gated, n_dense<3 → "over_gated"
_CODE_OVER_GATED = """
import torch
def enhance_observation(scenario_state):
    lt = scenario_state["lidar_targets"]
    la = scenario_state["lidar_agents"]
    cover_r = float(scenario_state["covering_range"])
    near_t = (lt < cover_r).float().sum(dim=-1)
    near_a = (la < cover_r).float().sum(dim=-1)
    t_close = (lt < cover_r).float()
    a_close = (la < cover_r).float()
    g1 = (near_t * near_a).unsqueeze(-1)
    g2 = (near_t * 1.0).unsqueeze(-1)
    g3 = (1.0 * near_a).unsqueeze(-1)
    g4 = (t_close * a_close).sum(dim=-1).unsqueeze(-1)
    g5 = (a_close * 2.0).sum(dim=-1).unsqueeze(-1)
    f6 = lt.min(dim=-1).values.unsqueeze(-1)
    f7 = la.min(dim=-1).values.unsqueeze(-1)
    return torch.cat([g1, g2, g3, g4, g5, f6, f7], dim=-1)
"""

# Cross-source pattern present, simple, M1=0 → "rl_too_hard"
_CODE_RL_HARD = """
import torch
def enhance_observation(scenario_state):
    lt = scenario_state["lidar_targets"]
    la = scenario_state["lidar_agents"]
    cover_r = float(scenario_state["covering_range"])
    nearest_t = lt.min(dim=-1).values.unsqueeze(-1)
    nearest_a = la.min(dim=-1).values.unsqueeze(-1)
    diff = (lt.min(dim=-1).values - la.min(dim=-1).values).unsqueeze(-1)
    mean_t = lt.mean(dim=-1).unsqueeze(-1)
    std_t = lt.std(dim=-1).unsqueeze(-1)
    return torch.cat([nearest_t, nearest_a, diff, mean_t, std_t], dim=-1)
"""

# Same code but M1=0.10 → "achieved"
_CODE_ACHIEVED = _CODE_RL_HARD


def _check_fewshot_quality(hint_text: str) -> dict:
    """Inspect chosen strategy's translation hint for v8 invariants."""
    fenced = re.findall(r"```python(.*?)```", hint_text, re.DOTALL)
    has_fenced = bool(fenced)
    fewshot = fenced[0] if fenced else ""
    fewshot_ana = analyze_inner_code(fewshot) if fewshot.strip() else None
    fewshot_n_features = fewshot_ana.n_returned_features if fewshot_ana else 0
    fewshot_n_gated = count_gated_features(fewshot)
    fewshot_n_dense = count_dense_features(fewshot_ana, fewshot) if fewshot_ana else 0
    forbidden_hits = [h for h in _FORBIDDEN_HANDLES if h in hint_text]
    return {
        "has_fenced_python_block": has_fenced,
        "fewshot_n_features": fewshot_n_features,
        "fewshot_n_gated": fewshot_n_gated,
        "fewshot_n_dense": fewshot_n_dense,
        "fewshot_in_3_4_range": 3 <= fewshot_n_features <= 4,
        "anti_cheat_clean": not forbidden_hits,
        "forbidden_hits": forbidden_hits,
        "fewshot_excerpt": fewshot[:400],
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("rendezvous.lero.v8.phase1_5")

    meta_llm = LLMClient(
        LLMConfig(
            model="gpt-5.4-mini",
            temperature=0.8,
            max_retries=3,
            prompt_version="v2_fewshot_modular_v2_local",
        )
    )
    inner_client = LLMClient(
        LLMConfig(
            model="gpt-5.4-mini",
            temperature=0.8,
            max_retries=3,
            prompt_version="v2_fewshot_modular_v2_local",
        )
    )

    t0 = time.monotonic()

    # === Phase A: bundle enumeration with v8 caps ===
    log.info(
        "=== Phase A: v8 bundle enumeration "
        "(target=%d-%d, cap=%d, gated_cap=%d) ===",
        _FEATURE_TARGET_MIN,
        _FEATURE_TARGET_MAX,
        _FEATURE_CAP,
        _GATED_CAP,
    )
    bundle, bundle_raw = enumerate_bundle_v8(
        meta_llm,
        _TASK_SUMMARY,
        feature_target_min=_FEATURE_TARGET_MIN,
        feature_target_max=_FEATURE_TARGET_MAX,
        feature_cap=_FEATURE_CAP,
        gated_cap=_GATED_CAP,
    )
    chosen = bundle.current()
    print(bundle.format_for_prompt())

    fewshot_check = _check_fewshot_quality(chosen.lero_translation_hint)
    print("\n[Phase A] fewshot quality:")
    for k, v in fewshot_check.items():
        if k != "fewshot_excerpt":
            print(f"  {k}: {v}")
    print(f"  fewshot_excerpt:\n{fewshot_check['fewshot_excerpt']}")

    # === Phase B: inner gen with chosen strategy's slot text ===
    log.info("=== Phase B: inner gen with v8 chosen strategy ===")
    base_slots = _read_base_slots("v2_fewshot_modular_v2_local")
    new_slots = dict(base_slots)
    new_slots["guidance_observation"] = chosen.lero_translation_hint

    inner_llm = InnerLLM(
        inner_client,
        evolve_reward=False,
        evolve_observation=True,
        use_structured=False,
    )
    messages = _build_inner_messages_with_slots(
        prompt_version="v2_fewshot_modular_v2_local",
        slot_overrides=new_slots,
        task_overrides=_TASK_OVERRIDES,
        inner_llm=inner_llm,
    )
    n_inner_gens = 3
    inner_reports = []
    for i in range(n_inner_gens):
        try:
            c = inner_llm.generate(messages, seed_base=80000 + i)
            code = c.obs_source or ""
        except Exception as e:  # noqa: BLE001
            log.warning("inner gen %d failed: %s", i, e)
            inner_reports.append({"err": str(e)})
            continue
        ana = analyze_inner_code(code)
        n_g = count_gated_features(code)
        n_d = count_dense_features(ana, code)
        rep = {
            "n_features": ana.n_returned_features,
            "n_gated": n_g,
            "n_dense": n_d,
            "touches_both_lidars": ana.touches_both_lidars,
            "n_cross_source": ana.n_cross_source,
            "feat_under_cap": ana.n_returned_features <= _FEATURE_CAP,
            "gated_under_cap": n_g <= _GATED_CAP,
            "dense_meets_min": n_d >= 3,
        }
        inner_reports.append(rep)
        log.info("  inner cand %d: %s", i, rep)

    valid = [r for r in inner_reports if "err" not in r]
    summary_b = {
        "n_valid": len(valid),
        "all_under_feat_cap": all(r["feat_under_cap"] for r in valid),
        "all_under_gated_cap": all(r["gated_under_cap"] for r in valid),
        "all_dense_meets_min": all(r["dense_meets_min"] for r in valid),
        "all_cross_source": all(r["touches_both_lidars"] for r in valid),
        "avg_n_features": (sum(r["n_features"] for r in valid) / max(1, len(valid))),
        "avg_n_gated": sum(r["n_gated"] for r in valid) / max(1, len(valid)),
        "avg_n_dense": sum(r["n_dense"] for r in valid) / max(1, len(valid)),
    }
    print("\n[Phase B] aggregate:")
    for k, v in summary_b.items():
        print(f"  {k}: {v}")

    # === Phase C: 4 v8 reflection scenarios ===
    log.info("=== Phase C: v8 reflection scenario tests ===")
    scenarios = [
        (
            "too_many_features",
            _CODE_TOO_MANY,
            0.000,
            0.080,
            "too_many_features",
            "trim_features",
        ),
        (
            "over_gated",
            _CODE_OVER_GATED,
            0.000,
            0.100,
            "over_gated",
            "replace_gated_with_dense",
        ),
        (
            "rl_too_hard",
            _CODE_RL_HARD,
            0.000,
            0.100,
            "rl_too_hard",
            "switch_to_next_strategy",
        ),
        ("achieved", _CODE_ACHIEVED, 0.080, 0.350, "achieved", "stop"),
    ]
    results = []
    for name, code, M1, M6, expected_label, expected_action in scenarios:
        synth = _make_inner(code, M1=M1, M6=M6)
        diag = diagnose_inner_result_v8(
            synth,
            chosen,
            feature_count_cap=_FEATURE_CAP,
            gated_feature_cap=_GATED_CAP,
        )
        try:
            decision, _ = reflect_and_decide_v8(
                meta_llm,
                bundle,
                synth,
                diag,
                feature_target_min=_FEATURE_TARGET_MIN,
                feature_target_max=_FEATURE_TARGET_MAX,
                feature_cap=_FEATURE_CAP,
                gated_cap=_GATED_CAP,
            )
            actual_action = decision.next_action
            rationale = decision.rationale[:200]
        except Exception as e:  # noqa: BLE001
            log.warning("reflect failed for %s: %s", name, e)
            actual_action = f"<error: {e}>"
            rationale = ""

        ok_label = diag.label == expected_label
        ok_action = actual_action == expected_action
        results.append(
            {
                "scenario": name,
                "expected_label": expected_label,
                "actual_label": diag.label,
                "label_match": ok_label,
                "n_features": diag.n_features,
                "n_gated": diag.n_gated,
                "n_dense": diag.n_dense,
                "expected_action": expected_action,
                "actual_action": actual_action,
                "action_match": ok_action,
                "rationale": rationale,
            }
        )
        log.info(
            "  %s: label=%s (exp %s) action=%s (exp %s) ok=%s "
            "[n_feat=%d n_gated=%d n_dense=%d]",
            name,
            diag.label,
            expected_label,
            actual_action,
            expected_action,
            ok_label and ok_action,
            diag.n_features,
            diag.n_gated,
            diag.n_dense,
        )

    # === Save report ===
    out = Path(f"results/v8_prompt_lab/phase1_5_{time.strftime('%Y%m%d_%H%M')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "elapsed_s": time.monotonic() - t0,
                "caps": {
                    "feature_target_min": _FEATURE_TARGET_MIN,
                    "feature_target_max": _FEATURE_TARGET_MAX,
                    "feature_cap": _FEATURE_CAP,
                    "gated_cap": _GATED_CAP,
                },
                "phase_a_bundle": [
                    {
                        "name": s.name,
                        "score": s.combined_score,
                        "lero_codability": s.lero_codability,
                        "rl_trainability": s.rl_trainability,
                        "full_solution": s.full_solution,
                    }
                    for s in bundle.strategies
                ],
                "phase_a_chosen_idx": bundle.chosen_idx,
                "phase_a_chosen_translation_hint": chosen.lero_translation_hint,
                "phase_a_fewshot_check": fewshot_check,
                "phase_b_inner_reports": inner_reports,
                "phase_b_summary": summary_b,
                "phase_c_scenarios": results,
            },
            indent=2,
            default=str,
        )
    )
    log.info("saved %s", out)

    # Print summary table
    print()
    print(f"{'scenario':<25} {'label':<24} {'action':<35} {'pass'}")
    print("-" * 100)
    all_pass = True
    for r in results:
        ok = r["label_match"] and r["action_match"]
        all_pass = all_pass and ok
        mark = "PASS" if ok else "FAIL"
        print(
            f"{r['scenario']:<25} {r['actual_label']:<24} "
            f"{r['actual_action']:<35} {mark}"
        )

    phase_a_pass = (
        fewshot_check["has_fenced_python_block"] and fewshot_check["anti_cheat_clean"]
    )
    phase_b_pass = (
        summary_b["all_under_feat_cap"]
        and summary_b["all_under_gated_cap"]
        and summary_b["all_cross_source"]
        and summary_b["n_valid"] >= 1
    )
    print()
    print(
        f"Phase A (fewshot quality + anti-cheat): "
        f"{'PASS' if phase_a_pass else 'FAIL'}"
    )
    print(
        f"Phase B (inner caps respected, cross-source present): "
        f"{'PASS' if phase_b_pass else 'FAIL'}"
    )
    print(f"Phase C (reflection actions correct): " f"{'PASS' if all_pass else 'FAIL'}")
    print(f"\nelapsed: {time.monotonic() - t0:.1f}s, output: {out}")
    return 0 if (phase_a_pass and phase_b_pass and all_pass) else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
