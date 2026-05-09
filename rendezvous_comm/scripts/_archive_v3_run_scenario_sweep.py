#!/usr/bin/env python3
"""Run the meta-LLM scenario sweep across multiple LLMs.

Emits a markdown report to stdout + a JSON dump to a file. Useful
for iterating on Strategist/Editor/Critic prompts WITHOUT running
real RL training.

Usage:
    cd rendezvous_comm
    python scripts/run_scenario_sweep.py \
        --scenarios A1_flat_zero_baseline,A2_reward_hack_shape \
        --output /tmp/sweep.json

    # Run every scenario against every LLM:
    python scripts/run_scenario_sweep.py

Cost estimate: ~€0.01 per scenario per LLM (gpt-5.4-mini), scaling by
model price. 15 scenarios × 4 LLMs ≈ €1 for a full sweep.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# Auto-load .env so OVH_AI_ENDPOINTS_* and API keys are available
try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

from src.lero.config import LLMConfig  # noqa: E402
from src.lero.llm_client import LLMClient  # noqa: E402
from tests.scenarios.fixtures import ALL_SCENARIOS  # noqa: E402
from tests.scenarios.runner import check_expectations, run_scenario  # noqa: E402


# Label convention:
#   "<base>"    → auto-detect reasoning_variant from model name
#   "<base>-r"  → FORCE reasoning_variant=True
#   "<base>-nr" → FORCE reasoning_variant=False (control)
LLM_CONFIGS = {
    "gpt-5.4-mini": dict(model="gpt-5.4-mini"),
    "gpt-4o": dict(model="gpt-4o"),
    "o4-mini": dict(model="o4-mini"),  # auto → reasoning prompts
    "o4-mini-nr": dict(
        model="o4-mini", force_non_reasoning=True
    ),  # control: same model, verbose prompts
    "ovh-llama-3.3-70b": dict(
        model=f"openai/{os.environ.get('OVH_AI_ENDPOINTS_MODEL', 'Meta-Llama-3_3-70B-Instruct')}",
        api_base=os.environ.get("OVH_AI_ENDPOINTS_URL"),
        api_key=os.environ.get("OVH_AI_ENDPOINTS_ACCESS_TOKEN"),
    ),
    "ovh-gpt-oss-120b": dict(
        model="openai/gpt-oss-120b",
        api_base=os.environ.get("OVH_AI_ENDPOINTS_URL"),
        api_key=os.environ.get("OVH_AI_ENDPOINTS_ACCESS_TOKEN"),
    ),
    "ovh-gpt-oss-120b-nr": dict(
        model="openai/gpt-oss-120b",
        api_base=os.environ.get("OVH_AI_ENDPOINTS_URL"),
        api_key=os.environ.get("OVH_AI_ENDPOINTS_ACCESS_TOKEN"),
        force_non_reasoning=True,  # control
    ),
    "ovh-Qwen3-Coder-30B": dict(
        model="openai/Qwen3-Coder-30B-A3B-Instruct",
        api_base=os.environ.get("OVH_AI_ENDPOINTS_URL"),
        api_key=os.environ.get("OVH_AI_ENDPOINTS_ACCESS_TOKEN"),
    ),
}


def _make_llm(label: str) -> LLMClient:
    cfg_kw = LLM_CONFIGS[label]
    cfg = LLMConfig(
        model=cfg_kw["model"],
        temperature=1.0,
        max_retries=2,
        api_base=cfg_kw.get("api_base"),
        api_key=cfg_kw.get("api_key"),
    )
    return LLMClient(cfg)


def _reasoning_variant_for(label: str) -> Optional[bool]:
    """Decide reasoning_variant from label flags.

    None = auto-detect (let runner.py decide from model name).
    True / False = override.
    """
    cfg_kw = LLM_CONFIGS[label]
    if cfg_kw.get("force_non_reasoning"):
        return False
    if cfg_kw.get("force_reasoning"):
        return True
    return None  # auto


def summarize_checks(checks: Dict) -> str:
    parts = []
    for name, detail in checks.items():
        mark = "PASS" if detail.get("ok") else "FAIL"
        parts.append(f"{name}:{mark}")
    return " | ".join(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios",
        type=str,
        default="",
        help="Comma-separated scenario names (default: all).",
    )
    parser.add_argument(
        "--llms",
        type=str,
        default=",".join(LLM_CONFIGS.keys()),
        help="Comma-separated LLM labels.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp/sweep_results.json",
        help="Path for full JSON dump.",
    )
    args = parser.parse_args(argv)

    scenarios = (
        [s.strip() for s in args.scenarios.split(",") if s.strip()]
        if args.scenarios
        else list(ALL_SCENARIOS.keys())
    )
    llm_labels = [s.strip() for s in args.llms.split(",") if s.strip()]

    for s in scenarios:
        if s not in ALL_SCENARIOS:
            print(f"ERROR: scenario {s!r} not in registry.", file=sys.stderr)
            return 2
    for l in llm_labels:
        if l not in LLM_CONFIGS:
            print(
                f"ERROR: llm label {l!r} not in {list(LLM_CONFIGS.keys())}.",
                file=sys.stderr,
            )
            return 2

    print("# Meta-LLM Scenario Sweep")
    print(f"Scenarios: {len(scenarios)}  |  LLMs: {llm_labels}")
    print()

    results = []
    llms: Dict[str, LLMClient] = {}
    for label in llm_labels:
        try:
            llms[label] = _make_llm(label)
        except Exception as e:
            print(f"!! Failed to init {label}: {e}")
            continue

    for name in scenarios:
        scenario = ALL_SCENARIOS[name]
        print(f"## {name}")
        print(f"_{scenario['description']}_")
        print()
        print(
            "| LLM | pass/total | target_slot | include_signals | critic | revisions | latency |"
        )
        print("|---|---|---|---|---|---|---|")
        for label, llm in llms.items():
            try:
                res = run_scenario(
                    name,
                    scenario,
                    label,
                    llm,
                    reasoning_variant=_reasoning_variant_for(label),
                )
            except Exception as e:
                print(f"| {label} | ERROR | `{type(e).__name__}: {e}` | | | | |")
                continue
            checks = check_expectations(res, scenario.get("expected", {}))
            passed = sum(1 for c in checks.values() if c.get("ok"))
            total = len(checks)
            card = res.strategy_card or {}
            critique = res.critique or {}
            print(
                f"| {label} | {passed}/{total} "
                f"| {card.get('target_slot','-')} "
                f"| {','.join(card.get('include_signals') or [])} "
                f"| {critique.get('overall_quality','-')} "
                f"| {res.critique_revisions} "
                f"| {res.latency_seconds:.1f}s |"
            )
            results.append(
                {
                    "scenario": name,
                    "llm": label,
                    "checks": checks,
                    "result": {
                        "strategy_card": res.strategy_card,
                        "editor_new_slot": ((res.editor_new_slot or "")[:500]),
                        "critique": res.critique,
                        "critique_revisions": res.critique_revisions,
                        "latency_seconds": res.latency_seconds,
                        "tokens_estimate": res.total_tokens_estimate,
                        "strategy_parse_error": res.strategy_parse_error,
                        "editor_parse_error": res.editor_parse_error,
                        "critique_parse_error": res.critique_parse_error,
                    },
                }
            )
        print()

    # Summary section
    print("## Summary by LLM")
    print("| LLM | scenarios | total checks | passed | pass_rate | total_latency |")
    print("|---|---|---|---|---|---|")
    for label in llm_labels:
        label_rows = [r for r in results if r["llm"] == label]
        total = sum(len(r["checks"]) for r in label_rows)
        passed = sum(1 for r in label_rows for c in r["checks"].values() if c.get("ok"))
        latency = sum(r["result"]["latency_seconds"] for r in label_rows)
        if total == 0:
            rate = "—"
        else:
            rate = f"{passed / total:.1%}"
        print(
            f"| {label} | {len(label_rows)} | {total} | {passed} "
            f"| {rate} | {latency:.1f}s |"
        )

    Path(args.output).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nFull JSON dump: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
