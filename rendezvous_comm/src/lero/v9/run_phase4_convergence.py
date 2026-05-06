"""v9 Phase-4 prompt-lab convergence sweep.

Per docs/v9_plan.md §7. LLM-only test: does v9 meta-prompt produce inner
candidates that match S3b-local's structural quality?

For each trial we:
  1. Call enumerate_bundle_v9 → get task_understanding + 3-5 strategies
     each with chain_of_thought + 1 chosen with artifacts text
  2. Render the v3_modular_taskdomain prompt with the chosen strategy's
     inferable_hints + examples slot text injected
  3. Generate N=3 inner candidates with that rendered prompt
  4. AST/regex analyze each candidate

Pass criteria (per §7):
  - role_one_hot rate ≥ 80% across inner candidates
  - inferable_hints text mentions ≥ 7 of task_domain.inferable_concepts
  - examples slot has ≥ 2 fenced ```python``` blocks
  - at least 1 example contains role one-hot

Cost: 1 meta-call + 3 inner-calls per trial ≈ 30s, $0.50.
For a 5-trial sweep: ~3 min, ~$3.

Run: python -m src.lero.v9.run_phase4_convergence
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List

from ..config import LLMConfig
from ..inner_llm import InnerLLM
from ..llm_client import LLMClient
from ..prompts.loader import PromptLoader
from ..v6_prompt_lab.analyzer import (
    analyze_inner_code,
    count_dense_features,
    count_gated_features,
)
from .meta_strategist import enumerate_bundle_v9


_TASK_OVERRIDES = {
    "n_agents": 4,
    "n_targets": 4,
    "agents_per_target": 2,
    "covering_range": 0.25,
    "lidar_range": 0.35,
    "max_steps": 400,
    "n_lidar_rays_entities": 15,
    "n_lidar_rays_agents": 12,
}

_TASK_SUMMARY = (
    "Multi-agent rendezvous in VMAS Discovery. 4 agents must cover "
    "4 targets; each target needs exactly 2 agents simultaneously."
)


def _render_inner_prompt(
    base_prompt_version: str,
    slot_overrides: Dict[str, str],
    task_overrides: Dict,
):
    """Copy base prompt to temp, write slot overrides, render system+user
    via the loader (with task_domain substitution working). Returns the
    rendered messages list."""
    from ..prompts import loader as _l

    base_dir = Path(_l.__file__).parent / base_prompt_version
    with tempfile.TemporaryDirectory() as tmp:
        new_dir = Path(tmp) / base_prompt_version
        shutil.copytree(base_dir, new_dir)
        # Also copy task_domains so loader can find rendezvous_k2.yaml
        td_src = Path(_l.__file__).parent / "task_domains"
        if td_src.exists():
            td_dst = Path(tmp) / "task_domains"
            shutil.copytree(td_src, td_dst)
        for slot, text in slot_overrides.items():
            (new_dir / f"{slot}.txt").write_text((text or "").rstrip() + "\n")
        orig = _l._PROMPTS_DIR
        try:
            _l._PROMPTS_DIR = Path(tmp)
            loader = PromptLoader(version=base_prompt_version)
            system_text = loader.render(
                "system.txt",
                **task_overrides,
            )
            user_text = loader.render(
                "initial_user.txt",
                output_spec_variant="obs_only",
                **task_overrides,
            )
        finally:
            _l._PROMPTS_DIR = orig
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]


def _measure_inferable_hints(text: str, concepts: List[Dict]) -> Dict:
    """How many of task_domain.inferable_concepts are mentioned in the
    inferable_hints text?"""
    hits = 0
    matches = []
    for c in concepts:
        # Match either the concept phrase OR the idiom keyword
        phrase = c["concept"].lower()
        idiom = c["idiom"].lower()
        idiom_keywords = re.findall(r"[a-z_]{4,}", idiom)
        phrase_match = phrase in text.lower()
        idiom_match = any(k in text.lower() for k in idiom_keywords)
        if phrase_match or idiom_match:
            hits += 1
            matches.append(c["concept"])
    return {
        "concepts_total": len(concepts),
        "concepts_hit": hits,
        "concepts_matched": matches,
    }


def _measure_examples(text: str) -> Dict:
    """Count fenced ```python blocks; check if at least one has role
    one-hot."""
    blocks = re.findall(r"```python(.*?)```", text, re.DOTALL)
    n_blocks = len(blocks)
    role_in_examples = 0
    for b in blocks:
        if re.search(r"(F\.one_hot|one_hot\s*\(|\[:\s*,\s*agent_idx\]\s*=\s*1)", b):
            role_in_examples += 1
    return {
        "n_python_blocks": n_blocks,
        "blocks_with_role_one_hot": role_in_examples,
    }


def _measure_inner_candidate(code: str) -> Dict:
    ana = analyze_inner_code(code)
    n_g = count_gated_features(code)
    n_d = count_dense_features(ana, code)
    role_match = bool(
        re.search(r"(F\.one_hot|one_hot\s*\(|\[:\s*,\s*agent_idx\]\s*=\s*1)", code)
    )
    return {
        "n_features": ana.n_returned_features,
        "n_gated": n_g,
        "n_dense": n_d,
        "n_cross_source": ana.n_cross_source,
        "touches_both_lidars": ana.touches_both_lidars,
        "role_one_hot_present": role_match,
    }


def run_one_trial(
    meta_llm: LLMClient,
    inner_client: LLMClient,
    loader: PromptLoader,
    trial_idx: int,
    n_inner: int = 3,
    base_prompt_version: str = "v3_modular_taskdomain",
) -> Dict:
    log = logging.getLogger(f"v9.phase4.trial{trial_idx}")

    log.info("Phase A: bundle enumeration")
    bundle, raw = enumerate_bundle_v9(
        meta_llm,
        loader,
        _TASK_SUMMARY,
    )
    chosen = bundle.current()
    arts = chosen.artifacts

    td = loader.task_domain() or {}
    concepts = td.get("inferable_concepts") or []
    hints_check = _measure_inferable_hints(arts.inferable_hints_text, concepts)
    examples_check = _measure_examples(arts.examples_text)

    log.info(
        "  chosen='%s' score=%.1f | hints %d/%d concepts | "
        "examples blocks=%d role_in=%d",
        chosen.name,
        chosen.combined_score,
        hints_check["concepts_hit"],
        hints_check["concepts_total"],
        examples_check["n_python_blocks"],
        examples_check["blocks_with_role_one_hot"],
    )

    log.info("Phase B: %d inner candidates", n_inner)
    inner_llm = InnerLLM(
        inner_client,
        evolve_reward=False,
        evolve_observation=True,
        use_structured=False,
    )
    messages = _render_inner_prompt(
        base_prompt_version=base_prompt_version,
        slot_overrides={
            "inferable_hints": arts.inferable_hints_text,
            "examples": arts.examples_text,
        },
        task_overrides=_TASK_OVERRIDES,
    )

    inner_reports = []
    for i in range(n_inner):
        try:
            c = inner_llm.generate(messages, seed_base=90000 + i)
            code = c.obs_source or ""
        except Exception as e:  # noqa: BLE001
            log.warning("inner gen %d failed: %s", i, e)
            inner_reports.append({"err": str(e)})
            continue
        meas = _measure_inner_candidate(code)
        inner_reports.append(meas)
        log.info("  cand %d: %s", i, meas)

    valid = [r for r in inner_reports if "err" not in r]
    role_rate = sum(1 for r in valid if r["role_one_hot_present"]) / max(1, len(valid))
    cs_rate = sum(1 for r in valid if r["touches_both_lidars"]) / max(1, len(valid))

    return {
        "trial_idx": trial_idx,
        "chosen_strategy": chosen.name,
        "chosen_combined_score": chosen.combined_score,
        "task_understanding": bundle.task_understanding,
        "hints_check": hints_check,
        "examples_check": examples_check,
        "inner_reports": inner_reports,
        "inner_role_rate": role_rate,
        "inner_cross_source_rate": cs_rate,
        "n_inner_valid": len(valid),
        "raw_bundle_response_len": len(raw),
    }


def main(n_trials: int = 5) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("v9.phase4")

    meta_llm = LLMClient(
        LLMConfig(
            model="gpt-5.4-mini",
            temperature=0.8,
            max_retries=3,
            prompt_version="v3_modular_taskdomain",
        )
    )
    inner_client = LLMClient(
        LLMConfig(
            model="gpt-5.4-mini",
            temperature=0.8,
            max_retries=3,
            prompt_version="v3_modular_taskdomain",
        )
    )

    loader = PromptLoader(version="v3_modular_taskdomain")

    t0 = time.monotonic()
    results = []
    for i in range(n_trials):
        log.info("=" * 60)
        log.info("=== TRIAL %d/%d ===", i + 1, n_trials)
        try:
            r = run_one_trial(meta_llm, inner_client, loader, i)
            results.append(r)
        except Exception as e:  # noqa: BLE001
            log.error("trial %d failed: %s", i, e)
            results.append({"trial_idx": i, "err": str(e)})

    # Aggregate
    valid_trials = [r for r in results if "err" not in r]
    n = len(valid_trials)
    if n > 0:
        avg_role = sum(r["inner_role_rate"] for r in valid_trials) / n
        avg_cs = sum(r["inner_cross_source_rate"] for r in valid_trials) / n
        avg_hints = sum(r["hints_check"]["concepts_hit"] for r in valid_trials) / n
        avg_blocks = (
            sum(r["examples_check"]["n_python_blocks"] for r in valid_trials) / n
        )
    else:
        avg_role = avg_cs = avg_hints = avg_blocks = 0.0

    summary = {
        "n_trials": n_trials,
        "n_valid": n,
        "elapsed_s": time.monotonic() - t0,
        "avg_inner_role_one_hot_rate": avg_role,
        "avg_inner_cross_source_rate": avg_cs,
        "avg_hints_concepts_hit": avg_hints,
        "avg_examples_n_blocks": avg_blocks,
        "pass_role_rate_80pct": avg_role >= 0.80,
        "pass_hints_concepts_7": avg_hints >= 7,
        "pass_examples_2blocks": avg_blocks >= 2,
        "trials": results,
    }

    out = Path(f"results/v9_prompt_lab/phase4_{time.strftime('%Y%m%d_%H%M')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    log.info("saved %s", out)

    print()
    print("=" * 70)
    print(
        f"v9 Phase 4 convergence — {n}/{n_trials} valid trials, "
        f"{summary['elapsed_s']:.0f}s"
    )
    print(
        f"  avg inner role_one_hot rate: {avg_role*100:.0f}%   "
        f"(target ≥80%) {'PASS' if avg_role>=0.80 else 'FAIL'}"
    )
    print(f"  avg inner cross_source rate: {avg_cs*100:.0f}%")
    print(
        f"  avg hints concepts hit: {avg_hints:.1f}/7   "
        f"{'PASS' if avg_hints>=7 else 'FAIL'}"
    )
    print(
        f"  avg examples blocks: {avg_blocks:.1f}   "
        f"(target ≥2) {'PASS' if avg_blocks>=2 else 'FAIL'}"
    )

    all_pass = (
        summary["pass_role_rate_80pct"]
        and summary["pass_hints_concepts_7"]
        and summary["pass_examples_2blocks"]
    )
    return 0 if all_pass else 1


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    sys.exit(main(n_trials=n))
