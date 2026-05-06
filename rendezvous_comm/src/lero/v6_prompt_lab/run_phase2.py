"""Phase 2 runner — sweep V0..V4 meta-system-prompt variants.

Each variant tests one structural addition to the v6 meta-system
prompt. The hypothesis (from prompt_evolution_analysis.md / v6 trilogy):
the missing trigger is operational vocabulary (cross-source operations
phrased as decisions). V1..V4 progressively add layers of that
vocabulary while staying anti-cheat (no winning-feature names).

Run: python -m src.lero.v6_prompt_lab.run_phase2

The runner saves a JSON report and prints a side-by-side comparison.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from ..config import LLMConfig
from ..llm_client import LLMClient
from ..v6.meta_strategist import _META_SYSTEM as V0_META_SYSTEM
from .harness import (
    PromptTrialResult,
    load_v6_old_best_obs,
    make_synthetic_inner_result,
    run_harness_trial,
)


# ── Variants ────────────────────────────────────────────────────


_V1_ADDITION = """

## Operational guidance (added in V1)

When you write `slot_edits` text describing operations, prefer ones that COMBINE information across multiple input channels. The inner LLM defaults to writing within-channel statistics (target-only summaries OR agent-only summaries); your guidance should explicitly nudge it toward operations that touch BOTH the target and the agent sensor channels in a single expression. Patterns to invite (without naming features): products, boolean conjunctions, ratios, differences between channel-derived quantities.
"""

_V2_ADDITION = (
    _V1_ADDITION
    + """

## Decision-shaped framing (added in V2)

Phrase operations as DECISIONS the policy needs to make, not as descriptive statistics. Examples of decision-shaped phrasing (template only, do not copy verbatim): "stay vs move", "alone vs paired", "scout vs converge", "this target or another one". Each operation you mention in slot_edits should map to a decision the agent could plausibly make from the resulting feature.
"""
)

_V3_ADDITION = (
    _V2_ADDITION
    + """

## Parallel-bullet pattern (added in V3)

When you describe operations in `guidance_observation`, USE PARALLEL BULLETS for the target sensor channel and the agent sensor channel. Whenever you describe an operation on one channel, immediately describe the analogous operation on the other channel. This priming makes the inner LLM see them as a pair and naturally combine them. Example structure (do not copy content):
  - <op> applied to target sensor → <one-line description>
  - <same op> applied to agent sensor → <one-line description>
  - <one cross-channel combination of the two> → <decision name>
"""
)

_V4_ADDITION = (
    _V3_ADDITION
    + """

## Operations palette slot (added in V4)

You may now also write a fourth slot named `guidance_observation` (you already do). In addition, place a 3-5 line "operations palette" sub-section at the top of `guidance_observation` listing 3-5 PATTERN TEMPLATES — each pattern is one sentence, names a generic operation (product, mask, ratio, gating), and points at how it relates to a decision. Do not name specific features.
"""
)


VARIANTS = {
    "V0_baseline": V0_META_SYSTEM,
    "V1_cross_source": V0_META_SYSTEM + _V1_ADDITION,
    "V2_plus_decision": V0_META_SYSTEM + _V2_ADDITION,
    "V3_plus_parallel_bullets": V0_META_SYSTEM + _V3_ADDITION,
    "V4_plus_ops_palette": V0_META_SYSTEM + _V4_ADDITION,
}


# ── Anti-cheat re-grep (final guard) ────────────────────────────


_FORBIDDEN = (
    "hold_signal",
    "hold_target_signal",
    "approach_signal",
    "approach_target_signal",
    "crowd_signal",
    "sparsity_signal",
    "gap_to_partner",
    "pair_formation_zone",
    "nearest_unassigned",
    "nearest unassigned helper",
    "second agent needed",
    "I am the second",
)


def assert_anticheat_clean(text: str, label: str) -> None:
    low = text.lower()
    for tok in _FORBIDDEN:
        if tok.lower() in low:
            raise AssertionError(f"ANTI-CHEAT VIOLATION in {label}: '{tok}' present")


# ── Runner ──────────────────────────────────────────────────────


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-candidates", type=int, default=3)
    p.add_argument("--model", type=str, default="gpt-5.4-mini")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    log = logging.getLogger("rendezvous.lero.v6_prompt_lab.run_phase2")

    # Anti-cheat grep on every variant before any LLM call
    for name, sys_prompt in VARIANTS.items():
        assert_anticheat_clean(sys_prompt, name)
    log.info("anti-cheat grep clean on all %d variants", len(VARIANTS))

    inner_cfg = LLMConfig(
        model=args.model,
        temperature=args.temperature,
        max_retries=3,
        prompt_version="v2_fewshot_modular_v2_local",
    )
    meta_cfg = LLMConfig(
        model=args.model,
        temperature=args.temperature,
        max_retries=3,
        prompt_version="v2_fewshot_modular_v2_local",
    )
    judge_cfg = LLMConfig(
        model=args.model,
        temperature=0.3,
        max_retries=3,
        prompt_version="v2_fewshot_modular_v2_local",
    )
    inner_llm = LLMClient(inner_cfg)
    meta_llm = LLMClient(meta_cfg)
    judge_llm = LLMClient(judge_cfg)

    obs_code = load_v6_old_best_obs()
    synthetic = make_synthetic_inner_result(best_obs_code=obs_code)

    task_summary = (
        "Multi-agent rendezvous in VMAS Discovery scenario. 4 agents must "
        "collectively cover 4 targets, with each target requiring exactly "
        "2 agents simultaneously within covering_range. Local LiDAR-only "
        "observations; no oracle access to other agents' positions or "
        "target positions. Implicit coordination from local sensors only."
    )
    task_overrides = {
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

    t_start = time.monotonic()
    results: dict[str, PromptTrialResult] = {}
    for name, sys_prompt in VARIANTS.items():
        log.info("=== running variant %s ===", name)
        trial = run_harness_trial(
            variant_name=name,
            meta_system_prompt=sys_prompt,
            base_prompt_version="v2_fewshot_modular_v2_local",
            synthetic_inner=synthetic,
            task_summary=task_summary,
            task_overrides=task_overrides,
            n_inner_candidates=args.n_candidates,
            meta_llm=meta_llm,
            inner_llm_client=inner_llm,
            judge_llm=judge_llm,
        )
        # Anti-cheat: also grep meta-LLM output + slot edits
        for slot, text in trial.meta_decision.slot_edits.items():
            assert_anticheat_clean(text, f"{name}.slot_edits.{slot}")
        results[name] = trial
        log.info(
            "%s: cross_source_rate=%.2f avg_cs_ops=%.2f avg_judge=%.2f (%.1fs)",
            name,
            trial.cross_source_rate,
            trial.avg_cross_source_ops,
            trial.avg_judge,
            trial.elapsed_s,
        )

    elapsed = time.monotonic() - t_start

    # Save report
    out = (
        args.out or f"results/v6_prompt_lab/phase2_{time.strftime('%Y%m%d_%H%M')}.json"
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(
        json.dumps(
            {
                "elapsed_s_total": elapsed,
                "n_candidates_per_variant": args.n_candidates,
                "model": args.model,
                "variants": {
                    name: {
                        **trial.summary_dict(),
                        "meta_rationale": trial.meta_decision.rationale,
                        "meta_guidance_observation": trial.meta_decision.slot_edits.get(
                            "guidance_observation", ""
                        ),
                        "meta_guidance_shared": trial.meta_decision.slot_edits.get(
                            "guidance_shared", ""
                        ),
                    }
                    for name, trial in results.items()
                },
            },
            indent=2,
            default=str,
        )
    )
    log.info("saved report → %s", out)

    # Print comparison table
    print()
    print(
        f"{'variant':<28} {'cs_rate':>8} {'avg_cs':>8} {'avg_judge':>10} "
        f"{'class':>20} {'mode':>22} {'elapsed':>8}"
    )
    print("-" * 110)
    for name, t in results.items():
        s = t.summary_dict()
        print(
            f"{name:<28} {s['cross_source_rate']:>8.2f} "
            f"{s['avg_cross_source_ops']:>8.2f} {s['avg_judge_total']:>10.2f} "
            f"{s['meta_classification']:>20} {s['meta_next_mode']:>22} "
            f"{s['elapsed_s']:>7.1f}s"
        )
    print(f"\ntotal: {elapsed:.1f}s, output: {out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
