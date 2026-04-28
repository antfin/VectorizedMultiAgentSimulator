"""Strategist that emits N strategies per round (LERO-MP v4).

Replaces v3's single-Editor-edit-one-slot pattern. Per round:
  - Read BootstrapCard + every prior round's results (including the
    actual best generated reward + obs code excerpts and per-candidate
    trajectory analysis tags).
  - Emit a StrategyBundle of N=3 strategies. Each strategy may target
    obs / reward / both, may revert reward to baseline if a prior
    round showed peak-collapse on a reward edit, and supplies a slot-
    edits dict the composer applies onto the bootstrap prompt.

DSPy migration: the ``emit_strategies`` callable is shape-compatible
with a ``dspy.Predict(StrategyBundleSignature)`` instance — swap one
class for the other in v5.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import List, Optional, Sequence

from ..llm_client import LLMClient
from .v4_schemas import (
    BootstrapCard,
    RoundResult,
    StrategyBundle,
    StrategyV4,
)

_log = logging.getLogger("rendezvous.lero.mp.v4.strategist")


# ── Prompt builder ──────────────────────────────────────────────


_SYSTEM = (
    "You are an RL strategist. You decide what 3 candidate prompts the "
    "next round of inner-LLM code generation should explore. You favor "
    "STABLE end-of-training behavior (avoid peak-then-collapse). You "
    "consider all prior-round evidence — best generated codes, "
    "trajectory tags (peak_collapse, monotonic_rise, etc.), stability "
    "scores. You may revert evolved reward to baseline if prior rounds "
    "showed reward-mediated collapse."
)


def _format_round_history(history: Sequence[RoundResult]) -> str:
    if not history:
        return "(this is round 0 — no prior rounds yet)"
    parts: List[str] = []
    for r in history:
        parts.append(f"## Round {r.round_idx}")
        parts.append(
            f"  Strategist's diversity rationale: "
            f"{r.bundle.diversity_rationale}"
        )
        parts.append(f"  Cross-strategy summary: {r.cross_round_summary}")
        for s in r.bundle.strategies:
            cand = next(
                (c for c in r.candidates if c.strategy_id == s.strategy_id),
                None,
            )
            parts.append(
                f"  --- {s.strategy_id} domain={s.target_domain} "
                f"revert_reward={s.revert_to_baseline_reward}"
            )
            parts.append(f"      idea: {s.high_level_idea}")
            parts.append(f"      expected: {s.expected_effect}")
            if cand:
                parts.append(
                    f"      OUTCOME: shape={cand.shape_tag}  "
                    f"final_M1={cand.final_M1:.3f}  "
                    f"peak_M1={cand.peak_M1:.3f}  "
                    f"score={cand.stability_score:+.3f}"
                )
                parts.append(f"      → {cand.qualitative_summary}")
        if r.best_strategy_id:
            mid = r.best_candidate_2M
            parts.append(
                f"  ★ Round winner {r.best_strategy_id} trained at "
                f"mid_frames: shape={mid.shape_tag} "
                f"final_M1={mid.final_M1:.3f} peak_M1={mid.peak_M1:.3f} "
                f"score={mid.stability_score:+.3f}"
            )
        parts.append("")
    return "\n".join(parts)


def _build_strategist_prompt(
    bootstrap: BootstrapCard,
    history: Sequence[RoundResult],
    n_strategies: int,
    fairness_excerpt: str,
) -> str:
    history_block = _format_round_history(history)
    return f"""[BOOTSTRAP CARD — the LLM's task understanding from Phase 0]
task_summary: {bootstrap.task_summary}
key_difficulty: {bootstrap.key_difficulty}
high_level_strategies_considered:
{chr(10).join("  - " + s for s in bootstrap.high_level_strategies_considered)}
proposed_initial_obs_features:
{chr(10).join("  - " + s for s in bootstrap.proposed_initial_obs_features) or "  (none)"}
proposed_initial_reward_components:
{chr(10).join("  - " + s for s in bootstrap.proposed_initial_reward_components) or "  (none)"}
fairness_audit: {bootstrap.fairness_audit}
assumptions:
{chr(10).join("  - " + a for a in bootstrap.assumptions) or "  (none)"}

[PRIOR ROUNDS — inspect every outcome]
{history_block}

[FAIRNESS CONTRACT — DO NOT re-encode into your strategies]
{fairness_excerpt or "(use only locally-observable state keys)"}

[YOUR TASK — TWO STEPS]

STEP 1 — DIAGNOSIS. Before proposing strategies, write a 3-5
sentence diagnosis of what prior rounds revealed:
  - What pattern is consistently failing across candidates?
  - What concept or feature seems MISSING from the codes generated
    so far? (Cite specific code excerpts if visible.)
  - What is the gap between best round outcome and the M1 ceiling
    we're aiming for?
If this is round 0 (no prior history), diagnose what the bootstrap
proposed and identify which abstractions are likely undertested.

STEP 2 — STRATEGY EMISSION. Emit EXACTLY {n_strategies} strategies
that EACH ADDRESS one specific gap from your Step-1 diagnosis. Not
"3 diverse strategies" — "3 strategies that target 3 specific
diagnosed gaps." Justify which diagnosis-gap each one targets.

Rules:
  - target_domain ∈ {{"reward", "observation", "both"}}.
  - slot_edits is a dict keyed by slot name
    (guidance_observation, guidance_reward, guidance_shared).
    Each value is the FULL replacement text for that slot
    (multi-line markdown is fine).
  - If any prior round had peak_collapse on a strategy with
    target_domain in {{"reward", "both"}}, AT LEAST ONE strategy this
    round MUST set revert_to_baseline_reward=True with revert_reason
    citing that prior round.
  - If a prior round produced a stable monotonic_rise candidate, at
    least one strategy should refine that candidate's direction rather
    than abandon it.
  - Stability over peaks. Optimize for END-of-training M1 at 10M, not
    intermediate peaks at 2-5M.

[OUTPUT FORMAT]
Reply with three sections:

### DIAGNOSIS
<3-5 sentences identifying the failure pattern + missing concept(s)>

### REASONING
<free-text 100–300 words explaining how the {n_strategies} strategies
EACH map to a specific diagnosis-gap>

### STRATEGY_BUNDLE
```json
{{
  "round_idx": <int>,
  "diversity_rationale": "<1-2 sentences>",
  "strategies": [
    {{
      "strategy_id": "S1",
      "high_level_idea": "...",
      "target_domain": "observation" | "reward" | "both",
      "revert_to_baseline_reward": true | false,
      "revert_reason": null | "...",
      "slot_edits": {{
        "guidance_observation": "..." (or omit),
        "guidance_reward": "..." (or omit),
        "guidance_shared": "..." (or omit)
      }},
      "expected_effect": "...",
      "rationale": "..."
    }},
    ... ({n_strategies} total)
  ]
}}
```
"""


# ── Output parsing ──────────────────────────────────────────────


def _parse_bundle(raw: str, expected_round: int) -> StrategyBundle:
    """Extract the JSON STRATEGY_BUNDLE section."""
    if "### STRATEGY_BUNDLE" not in raw:
        raise ValueError(
            "Strategist response missing '### STRATEGY_BUNDLE' header"
        )
    body = raw.split("### STRATEGY_BUNDLE", 1)[1]
    json_start = body.find("{")
    json_end = body.rfind("}")
    if json_start < 0 or json_end < json_start:
        raise ValueError(
            "STRATEGY_BUNDLE section did not contain valid JSON"
        )
    blob = body[json_start:json_end + 1]
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"STRATEGY_BUNDLE JSON parse failed: {e}. Body: {blob[:200]!r}"
        )
    # Force round_idx to caller's value if model produced wrong int
    data["round_idx"] = expected_round
    return StrategyBundle.model_validate(data)


# ── Main entry point ────────────────────────────────────────────


def emit_strategies(
    bootstrap: BootstrapCard,
    round_history: Sequence[RoundResult],
    meta_llm: LLMClient,
    round_idx: int,
    n_strategies: int = 3,
    fairness_excerpt: str = "",
) -> StrategyBundle:
    """Call the meta-LLM and parse the StrategyBundle.

    Idempotent on (bootstrap, round_history, meta_llm). Caching not
    implemented here — at temperature=1.0 each call is intentionally
    independent. Use the LLMCache layer for replay.
    """
    if n_strategies < 1:
        raise ValueError(f"n_strategies must be >= 1, got {n_strategies}")

    prompt = _build_strategist_prompt(
        bootstrap=bootstrap,
        history=round_history,
        n_strategies=n_strategies,
        fairness_excerpt=fairness_excerpt,
    )

    t0 = time.monotonic()
    last_err: Optional[Exception] = None
    bundle: Optional[StrategyBundle] = None
    for attempt in range(1, 4):
        raw = meta_llm.generate(
            [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            n=1,
        )[0]
        try:
            bundle = _parse_bundle(raw, expected_round=round_idx)
            break
        except ValueError as e:
            last_err = e
            _log.warning(
                "Strategist parse failed on attempt %d/3 (round=%d): %s",
                attempt, round_idx, e,
            )
    if bundle is None:
        raise ValueError(
            f"Strategist failed to produce parseable bundle after 3 "
            f"attempts (round={round_idx}). Last error: {last_err}"
        )
    elapsed = time.monotonic() - t0

    if len(bundle.strategies) != n_strategies:
        raise ValueError(
            f"Strategist returned {len(bundle.strategies)} strategies; "
            f"expected {n_strategies}"
        )

    # Sanity: enforce strategy_id naming convention so composer paths
    # are predictable.
    for i, s in enumerate(bundle.strategies, 1):
        expected_id = f"S{i}"
        if s.strategy_id != expected_id:
            _log.warning(
                "Strategist used id '%s' at position %d; renaming to %s.",
                s.strategy_id, i, expected_id,
            )
            s.strategy_id = expected_id

    # Sanity: revert_to_baseline_reward implies revert_reason
    for s in bundle.strategies:
        if s.revert_to_baseline_reward and not s.revert_reason:
            _log.warning(
                "Strategy %s set revert_to_baseline_reward=True but "
                "no revert_reason; clearing the flag for safety.",
                s.strategy_id,
            )
            s.revert_to_baseline_reward = False

    _log.info(
        "v4 strategist round=%d emitted %d strategies in %.1fs: %s",
        round_idx, len(bundle.strategies), elapsed,
        ", ".join(
            f"{s.strategy_id}({s.target_domain}"
            + (",revert" if s.revert_to_baseline_reward else "")
            + ")"
            for s in bundle.strategies
        ),
    )
    return bundle
