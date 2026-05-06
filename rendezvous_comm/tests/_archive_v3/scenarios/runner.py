"""Scenario harness runner.

Executes one scenario end-to-end through the REAL Strategist + Editor
+ Critic pipeline (no RL). Returns the observed decisions for assertion.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from src.lero.config import LLMConfig
from src.lero.llm_client import LLMClient
from src.lero.meta.critique import (
    critique_and_revise,
)
from src.lero.meta.mutation import (
    build_editor_prompt,
    parse_mutation_response,
)
from src.lero.meta.strategy import (
    build_strategist_prompt,
    parse_strategy_card,
)


@dataclass
class ScenarioResult:
    scenario_name: str
    llm_label: str
    strategy_card: Optional[Dict[str, Any]]
    strategy_parse_error: Optional[str]
    editor_new_slot: Optional[str]
    editor_rationale: Optional[str]
    editor_parse_error: Optional[str]
    critique: Optional[Dict[str, Any]]
    critique_revisions: int
    critique_parse_error: Optional[str]
    latency_seconds: float
    total_tokens_estimate: int


def _make_llm(
    model: str, api_base: Optional[str] = None, api_key: Optional[str] = None
) -> LLMClient:
    cfg = LLMConfig(
        model=model,
        temperature=1.0,
        max_retries=2,
        api_base=api_base,
        api_key=api_key,
    )
    return LLMClient(cfg)


def _call(llm: LLMClient) -> Callable[[List[Dict[str, str]]], str]:
    def _go(messages):
        return llm.generate(messages, n=1)[0]

    return _go


def run_scenario(
    name: str,
    scenario: Dict[str, Any],
    llm_label: str,
    meta_llm: LLMClient,
    fairness_text: str = (
        "Use only local sensors (lidar_targets, lidar_agents, "
        "own position/velocity, received messages). Do not access "
        "oracle state (absolute agent positions of others, absolute "
        "landmark positions). Rewards must stay within |r| <= 50."
    ),
    reasoning_variant: Optional[bool] = None,
) -> ScenarioResult:
    """Run the full Strategist → Editor → Critic chain on a fake scenario.

    ``reasoning_variant``: when True (or auto-detected from the model
    name), routes to the slim Strategist + Editor prompts intended
    for o-series / gpt-oss / Claude-thinking models.
    """
    from src.lero.meta._reasoning import is_reasoning_model

    if reasoning_variant is None:
        reasoning_variant = is_reasoning_model(meta_llm.config.model)

    t0 = time.monotonic()
    tokens_est = 0

    # 1. Strategist
    strat_prompt = build_strategist_prompt(
        history=scenario["history"],
        mutation_log_entries=scenario.get("priors", []),
        top_candidates=scenario["candidates"],
        seed_bias=scenario.get("bias", "exploratory"),
        fail_mode=scenario["fail_mode"],
        fairness_slot_excerpt=fairness_text,
        reasoning_variant=reasoning_variant,
    )
    tokens_est += len(strat_prompt) // 4
    strat_card = None
    strat_err = None
    try:
        strat_raw = meta_llm.generate(
            [
                {
                    "role": "system",
                    "content": "You are a careful research engineer deciding "
                    "how to improve a multi-agent RL prompt "
                    "template. Output only the requested YAML.",
                },
                {"role": "user", "content": strat_prompt},
            ],
            n=1,
        )[0]
        tokens_est += len(strat_raw) // 4
        card = parse_strategy_card(strat_raw)
        strat_card = card.to_dict()
    except Exception as e:
        strat_err = f"{type(e).__name__}: {e}"
        return ScenarioResult(
            scenario_name=name,
            llm_label=llm_label,
            strategy_card=None,
            strategy_parse_error=strat_err,
            editor_new_slot=None,
            editor_rationale=None,
            editor_parse_error=None,
            critique=None,
            critique_revisions=0,
            critique_parse_error=None,
            latency_seconds=time.monotonic() - t0,
            total_tokens_estimate=tokens_est,
        )

    # 2. Editor (or injected override if the scenario pins Editor output)
    editor_new_slot = None
    editor_rationale = None
    editor_err = None
    if "editor_override" in scenario:
        editor_new_slot = scenario["editor_override"]
        editor_rationale = "injected for harness test"
    else:
        editor_prompt = build_editor_prompt(
            parent_version="v2_fewshot_modular_v2",
            strategy_card=card,
            top_candidates=scenario["candidates"],
            loader=None,  # no real loader — avoids filesystem touch
            prior_slot_versions=scenario.get("priors", []),
            behavioral_block="",
            reasoning_variant=reasoning_variant,
        )
        tokens_est += len(editor_prompt) // 4
        try:
            editor_raw = meta_llm.generate(
                [
                    {
                        "role": "system",
                        "content": "You are a careful prompt-engineering "
                        "assistant. Follow the output format exactly.",
                    },
                    {"role": "user", "content": editor_prompt},
                ],
                n=1,
            )[0]
            tokens_est += len(editor_raw) // 4
            editor_new_slot, editor_rationale, _ = parse_mutation_response(
                editor_raw,
                card.target_slot,
            )
        except Exception as e:
            editor_err = f"{type(e).__name__}: {e}"

    # 3. Critic (if we have an Editor output to critique)
    critique_dict = None
    revisions = 0
    critic_err = None
    if editor_new_slot:
        try:
            outcome = critique_and_revise(
                strategy_card=card,
                editor_new_slot=editor_new_slot,
                editor_rationale=editor_rationale or "",
                editor_expected="small",
                fairness_text=fairness_text,
                prior_slot_versions=scenario.get("priors", []),
                critic_llm_call=_call(meta_llm),
                editor_revise_call=None,  # harness doesn't re-invoke editor
                max_revisions=1,
            )
            revisions = outcome.revisions
            critique_dict = outcome.critique.model_dump()
            tokens_est += 4 * 2000  # rough estimate for Critic prompt+response
        except ValueError as e:
            # The Critic raised ValueError because it returned
            # overall_quality="reject". That IS a critique outcome
            # (the strongest possible flag), not a parsing failure.
            # Synthesize a "reject" critique so harness scoring sees
            # the verdict instead of treating it as a parse error.
            critique_dict = {
                "overall_quality": "reject",
                "addresses_focus": False,
                "addresses_focus_reason": str(e)[:200],
                "cites_specific_features": [],
                "has_fairness_restatement": False,
                "has_fairness_restatement_reason": "",
                "diverges_from_priors": False,
                "suggested_edits": [],
                "suggested_signal_change": "keep",
            }
        except Exception as e:
            critic_err = f"{type(e).__name__}: {e}"

    return ScenarioResult(
        scenario_name=name,
        llm_label=llm_label,
        strategy_card=strat_card,
        strategy_parse_error=strat_err,
        editor_new_slot=editor_new_slot,
        editor_rationale=editor_rationale,
        editor_parse_error=editor_err,
        critique=critique_dict,
        critique_revisions=revisions,
        critique_parse_error=critic_err,
        latency_seconds=time.monotonic() - t0,
        total_tokens_estimate=tokens_est,
    )


# ── Assertion helpers ──────────────────────────────────────────


def check_expectations(
    result: ScenarioResult,
    expected: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate a ScenarioResult against the scenario's expected dict.

    Returns a dict of pass/fail flags per expectation.
    """
    checks: Dict[str, Any] = {}

    card = result.strategy_card or {}
    slot = card.get("target_slot")
    include = set(card.get("include_signals") or [])

    if "target_slot" in expected:
        want = set(expected["target_slot"])
        checks["target_slot"] = {
            "ok": slot in want,
            "got": slot,
            "want_any": list(want),
        }

    if "include_signals_contains_any" in expected:
        want = set(expected["include_signals_contains_any"])
        checks["include_signals_contains_any"] = {
            "ok": bool(include & want),
            "got": list(include),
            "want_any": list(want),
        }

    if "include_signals_equals" in expected:
        want = set(expected["include_signals_equals"])
        checks["include_signals_equals"] = {
            "ok": include == want,
            "got": list(include),
            "want": list(want),
        }

    if "confidence_in" in expected:
        conf = card.get("confidence")
        checks["confidence_in"] = {
            "ok": conf in expected["confidence_in"],
            "got": conf,
        }

    slot_text = (result.editor_new_slot or "").lower()

    if "editor_mentions_any" in expected:
        terms = [t.lower() for t in expected["editor_mentions_any"]]
        hits = [t for t in terms if t in slot_text]
        checks["editor_mentions_any"] = {
            "ok": bool(hits),
            "hits": hits,
            "want_any": terms,
        }

    if "editor_avoids_any" in expected:
        terms = [t.lower() for t in expected["editor_avoids_any"]]
        violations = [t for t in terms if t in slot_text]
        checks["editor_avoids_any"] = {
            "ok": not violations,
            "violations": violations,
            "forbidden": terms,
        }

    if "editor_must_not_contain_all" in expected:
        terms = [t.lower() for t in expected["editor_must_not_contain_all"]]
        all_present = all(t in slot_text for t in terms)
        checks["editor_must_not_contain_all"] = {
            "ok": not all_present,
            "all_present": all_present,
            "forbidden": terms,
        }

    critique = result.critique or {}

    if "critic_flags_fairness_restatement" in expected:
        want = expected["critic_flags_fairness_restatement"]
        checks["critic_flags_fairness_restatement"] = {
            "ok": critique.get("has_fairness_restatement") == want,
            "got": critique.get("has_fairness_restatement"),
            "want": want,
        }

    if "critic_quality_in" in expected:
        q = critique.get("overall_quality")
        checks["critic_quality_in"] = {
            "ok": q in expected["critic_quality_in"],
            "got": q,
        }

    if "critic_diverges_from_priors" in expected:
        want = expected["critic_diverges_from_priors"]
        checks["critic_diverges_from_priors"] = {
            "ok": critique.get("diverges_from_priors") == want,
            "got": critique.get("diverges_from_priors"),
            "want": want,
        }

    if "critic_cites_features_any" in expected:
        cited = [f.lower() for f in critique.get("cites_specific_features", [])]
        want = [t.lower() for t in expected["critic_cites_features_any"]]
        hits = [t for t in want if any(t in c for c in cited)]
        checks["critic_cites_features_any"] = {
            "ok": bool(hits),
            "hits": hits,
            "want_any": want,
            "cited": cited,
        }

    return checks
