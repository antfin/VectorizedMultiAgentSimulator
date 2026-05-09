"""Real-LLM scenario runner for v4 tests.

Each scenario calls the REAL meta-LLM with a synthetic
(BootstrapCard, round_history) and asserts properties of the
returned StrategyBundle. Use sparingly — each call is ~$0.001 with
gpt-5.4-mini.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from src.lero.llm_client import LLMClient
from src.lero.meta.v4_schemas import BootstrapCard, RoundResult, StrategyBundle
from src.lero.meta.v4_strategist import emit_strategies


@dataclass
class V4ScenarioResult:
    name: str
    bundle: Optional[StrategyBundle]
    parse_error: Optional[str]
    expectations_passed: Dict[str, bool]


def run_strategist_scenario(
    name: str,
    bootstrap: BootstrapCard,
    history: List[RoundResult],
    expectations: Dict[str, Callable[[StrategyBundle], bool]],
    meta_llm: LLMClient,
    n_strategies: int = 3,
    fairness: str = "Use only local sensors. Forbidden keys raise FairnessViolation.",
) -> V4ScenarioResult:
    """Run one strategist call against a synthetic context, evaluate
    expectations.

    `expectations` maps a name → predicate(StrategyBundle) → bool.
    """
    try:
        bundle = emit_strategies(
            bootstrap=bootstrap,
            round_history=history,
            meta_llm=meta_llm,
            round_idx=len(history),
            n_strategies=n_strategies,
            fairness_excerpt=fairness,
        )
    except Exception as e:
        return V4ScenarioResult(
            name=name,
            bundle=None,
            parse_error=f"{type(e).__name__}: {e}",
            expectations_passed={},
        )

    passed = {}
    for exp_name, predicate in expectations.items():
        try:
            passed[exp_name] = bool(predicate(bundle))
        except Exception:
            passed[exp_name] = False
    return V4ScenarioResult(
        name=name,
        bundle=bundle,
        parse_error=None,
        expectations_passed=passed,
    )
