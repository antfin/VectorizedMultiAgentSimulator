"""End-to-end integration for the v3 inner+meta pipeline (LERO-MP v3 §Step 10).

Smoke test that stitches together retry loop, include_signals gate,
and Critic — all with stub LLMs. Does NOT run real RL training;
BenchMARL integration is covered by the live OVH dry-run.
"""

from __future__ import annotations

from typing import Dict, List
from pathlib import Path

import pytest

from src.lero.codegen import CandidateCode
from src.lero.inner_llm import InnerLLM
from src.lero.meta.behavioral_summary import format_behavioral_block
from src.lero.meta.critique import critique_and_revise
from src.lero.meta.strategy import StrategyCard


_VALID_OBS = """```python
def enhance_observation(scenario_state: dict):
    import torch
    return torch.zeros(1)
```"""

_VALID_REWARD = """```python
def compute_reward(scenario_state: dict):
    import torch
    return torch.zeros(1)
```"""


class _StubLLM:
    def __init__(self, responses: List[str]):
        self.responses = list(responses)
        self.n_calls = 0

    def generate(self, messages, n=1, seed_base=None):
        out = []
        for _ in range(n):
            self.n_calls += 1
            out.append(self.responses.pop(0) if self.responses else "")
        return out

    def generate_structured(self, *a, **kw):
        raise NotImplementedError


def test_inner_to_critic_end_to_end():
    """Full v3 pipeline: inner generates with retry → include_signals
    filter renders behavioral block → Critic reviews → revise → accept.
    """
    # 1. Inner LLM: first response fails, second succeeds.
    stub_inner = _StubLLM(["not a valid block", _VALID_OBS])
    inner = InnerLLM(
        stub_inner, evolve_reward=False, evolve_observation=True,
        max_attempts=3,
    )
    cand = inner.generate(
        [{"role": "system", "content": "s"},
         {"role": "user", "content": "u"}],
        seed_base=42,
    )
    assert cand.attempts == 2
    assert cand.obs_source is not None

    # 2. Behavioral block filters per Strategist's include_signals.
    metrics = {
        "M1_success_rate": 0.1, "M4_avg_collisions": 80,
        "M9_spatial_spread": 0.15,
    }
    block_scalar_only = format_behavioral_block(
        metrics, include_signals=["scalar"],
    )
    assert "Tier 1" in block_scalar_only
    assert "Tier 2" not in block_scalar_only

    block_with_curve = format_behavioral_block(
        metrics, include_signals=["scalar", "curve_shape"],
        trajectory=[0.0, 0.3, 0.3, 0.1],
    )
    assert "Tier 3" in block_with_curve

    # 3. Critic loop: first critique says revise, second says keep.
    import json

    def _cjson(q, edits=None):
        return json.dumps({
            "addresses_focus": True,
            "addresses_focus_reason": "named proximity_count",
            "cites_specific_features": ["proximity_count"],
            "has_fairness_restatement": False,
            "has_fairness_restatement_reason": "no markers",
            "diverges_from_priors": True,
            "suggested_edits": edits or [],
            "suggested_signal_change": "keep",
            "overall_quality": q,
        })

    responses = [_cjson("revise", ["also name gap feature"]), _cjson("keep")]

    def critic_call(messages):
        return responses.pop(0)

    def editor_revise(critique, current):
        return {
            "new_slot": current + "\nAlso: gap feature.",
            "rationale": "revised",
            "expected": "small",
        }

    card = StrategyCard(
        target_domain="observation",
        target_slot="guidance_observation",
        focus=["add proximity_count"],
        avoid=[],
        confidence="medium",
        rationale="M9=0.15 — clustered",
        include_signals=["scalar", "fingerprint"],
    )
    outcome = critique_and_revise(
        strategy_card=card,
        editor_new_slot="use proximity_count",
        editor_rationale="v1",
        editor_expected="small",
        fairness_text="local sensors only",
        prior_slot_versions=[],
        critic_llm_call=critic_call,
        editor_revise_call=editor_revise,
    )
    assert outcome.revisions == 1
    assert "gap feature" in outcome.accepted_slot
    assert outcome.critique.overall_quality == "keep"


def test_strategy_card_defaults_still_drive_scalar_only_filter():
    """If the Strategist didn't set include_signals, downstream should
    only see scalars — no accidental fingerprint/curve leakage."""
    card = StrategyCard(
        target_domain="observation",
        target_slot="guidance_observation",
    )
    out = format_behavioral_block(
        {"M1_success_rate": 0.1},
        include_signals=card.include_signals,
    )
    assert "Tier 1" in out
    assert "Tier 2" not in out
    assert "Tier 3" not in out
