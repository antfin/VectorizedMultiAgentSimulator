"""Tests for the TextGrad Editor Critic loop (LERO-MP v3 §4.2)."""

from __future__ import annotations

import json
from typing import List

import pytest

from src.lero.meta.critique import (
    CritiqueOutcome,
    critique_and_revise,
    parse_critique,
)
from src.lero.meta.strategy import StrategyCard


def _card():
    return StrategyCard(
        target_domain="observation",
        target_slot="guidance_observation",
        focus=["add proximity_count feature"],
        avoid=["oracle positions"],
        confidence="medium",
        rationale="baseline lacks proximity signal",
    )


def _critic_json(
    quality="keep", addresses=True, fairness=False, diverges=True,
    suggested_edits=None, signal_change="keep",
):
    return json.dumps({
        "addresses_focus": addresses,
        "addresses_focus_reason": "cites proximity_count by name",
        "cites_specific_features": ["proximity_count"],
        "has_fairness_restatement": fairness,
        "has_fairness_restatement_reason": "no fairness-markers detected",
        "diverges_from_priors": diverges,
        "suggested_edits": suggested_edits or [],
        "suggested_signal_change": signal_change,
        "overall_quality": quality,
    })


def test_parse_critique_valid():
    c = parse_critique(_critic_json())
    assert c.overall_quality == "keep"
    assert c.addresses_focus is True
    assert "proximity_count" in c.cites_specific_features


def test_parse_critique_tolerates_leading_prose():
    raw = "Here is my review:\n\n" + _critic_json()
    c = parse_critique(raw)
    assert c.overall_quality == "keep"


def test_parse_critique_defaults_missing_optionals():
    # Minimal valid JSON missing suggested_signal_change
    raw = json.dumps({
        "addresses_focus": True,
        "addresses_focus_reason": "x",
        "cites_specific_features": [],
        "has_fairness_restatement": False,
        "has_fairness_restatement_reason": "x",
        "diverges_from_priors": True,
        "suggested_edits": [],
        "overall_quality": "keep",
    })
    c = parse_critique(raw)
    assert c.suggested_signal_change == "keep"


def test_critique_and_revise_keep_path():
    """Critic says keep on first round → accept, no revision."""
    calls = []

    def critic(messages):
        calls.append(messages)
        return _critic_json(quality="keep")

    outcome = critique_and_revise(
        strategy_card=_card(),
        editor_new_slot="use proximity_count (count of lidar hits within covering_range)",
        editor_rationale="implements focus",
        editor_expected="small",
        fairness_text="local sensors only; oracle state forbidden; |r| <= 50",
        prior_slot_versions=[],
        critic_llm_call=critic,
    )
    assert outcome.accepted_slot.startswith("use proximity_count")
    assert outcome.revisions == 0
    assert len(calls) == 1


def test_critique_and_revise_revise_then_keep():
    responses = [
        _critic_json(quality="revise", suggested_edits=["name the gap feature"]),
        _critic_json(quality="keep"),
    ]

    def critic(messages):
        return responses.pop(0)

    revise_calls = []

    def editor_revise(critique, current_slot):
        revise_calls.append(critique)
        return {
            "new_slot": current_slot + "\nAlso add the gap feature.",
            "rationale": "revised per critique",
            "expected": "small",
        }

    outcome = critique_and_revise(
        strategy_card=_card(),
        editor_new_slot="use proximity_count",
        editor_rationale="first",
        editor_expected="small",
        fairness_text="local sensors only",
        prior_slot_versions=[],
        critic_llm_call=critic,
        editor_revise_call=editor_revise,
    )
    assert outcome.revisions == 1
    assert "gap feature" in outcome.accepted_slot
    assert len(revise_calls) == 1


def test_critique_and_revise_reject_raises():
    def critic(messages):
        return _critic_json(
            quality="reject", addresses=False, fairness=True,
        )

    with pytest.raises(ValueError):
        critique_and_revise(
            strategy_card=_card(),
            editor_new_slot="generic restatement",
            editor_rationale="",
            editor_expected="small",
            fairness_text="local sensors only",
            prior_slot_versions=[],
            critic_llm_call=critic,
        )


def test_critique_and_revise_respects_max_revisions():
    """If Critic keeps saying revise, we cap at max_revisions."""
    def critic(messages):
        return _critic_json(
            quality="revise", suggested_edits=["do more"],
        )

    def editor_revise(critique, current_slot):
        return {
            "new_slot": current_slot + " (more)",
            "rationale": "",
            "expected": "small",
        }

    outcome = critique_and_revise(
        strategy_card=_card(),
        editor_new_slot="start",
        editor_rationale="",
        editor_expected="small",
        fairness_text="",
        prior_slot_versions=[],
        critic_llm_call=critic,
        editor_revise_call=editor_revise,
        max_revisions=2,
    )
    assert outcome.revisions == 2
    assert outcome.accepted_slot.count(" (more)") == 2
