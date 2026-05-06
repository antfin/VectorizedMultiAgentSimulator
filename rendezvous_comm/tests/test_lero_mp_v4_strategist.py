"""Tests for v4 strategist — stub LLM, no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.lero.config import LLMConfig
from src.lero.meta.v4_composer import compose_prompt_for_strategy
from src.lero.meta.v4_strategist import _parse_bundle, emit_strategies
from src.lero.meta.v4_schemas import (
    BootstrapCard,
    StrategyV4,
)


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def bootstrap_card():
    return BootstrapCard(
        task_summary="rendezvous k=2",
        success_metric_understanding="M1 = simultaneous coverage rate",
        key_difficulty="2 agents must arrive at same target same step",
        failure_modes_anticipated=["ships_passing", "anti_crowding"],
        high_level_strategies_considered=["hold_signal", "proximity_count"],
        proposed_initial_obs_features=[
            "hold_signal",
            "proximity_count",
            "gap",
        ],
        proposed_initial_reward_components=[],
        fairness_audit="Uses lidar + own state only",
        assumptions=["agents are point particles"],
    )


def _stub_response(strategies, round_idx=0, diversity="diverse strategies"):
    bundle_dict = {
        "round_idx": round_idx,
        "diversity_rationale": diversity,
        "strategies": strategies,
    }
    return (
        "### REASONING\nthinking here\n\n"
        "### STRATEGY_BUNDLE\n```json\n" + json.dumps(bundle_dict, indent=2) + "\n```\n"
    )


def _strategy(sid="S1", domain="observation", revert=False, **kw):
    out = {
        "strategy_id": sid,
        "high_level_idea": kw.get("idea", "test idea"),
        "target_domain": domain,
        "revert_to_baseline_reward": revert,
        "revert_reason": kw.get("revert_reason"),
        "slot_edits": kw.get("slot_edits", {}),
        "expected_effect": kw.get("expected", "expected effect"),
        "rationale": kw.get("rationale", "rationale text"),
    }
    return out


class _StubLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.config = LLMConfig(model="gpt-5.4-mini", temperature=1.0)
        self.calls = 0

    def generate(self, messages, n=1, seed_base=None):
        out = []
        for _ in range(n):
            self.calls += 1
            if not self._responses:
                raise RuntimeError("stub exhausted")
            out.append(self._responses.pop(0))
        return out


# ── _parse_bundle ──────────────────────────────────────────────


def test_parse_bundle_basic():
    raw = _stub_response(
        [
            _strategy("S1", domain="observation"),
            _strategy("S2", domain="reward"),
            _strategy("S3", domain="both"),
        ]
    )
    b = _parse_bundle(raw, expected_round=0)
    assert b.round_idx == 0
    assert len(b.strategies) == 3
    assert b.strategies[0].target_domain == "observation"
    assert b.strategies[2].target_domain == "both"


def test_parse_bundle_overrides_round_idx():
    raw = _stub_response([_strategy("S1")], round_idx=999)
    b = _parse_bundle(raw, expected_round=2)
    assert b.round_idx == 2


def test_parse_bundle_rejects_missing_section():
    raw = "Just some prose, no headers"
    with pytest.raises(ValueError, match="STRATEGY_BUNDLE"):
        _parse_bundle(raw, expected_round=0)


# ── emit_strategies — happy path ───────────────────────────────


def test_emit_strategies_round_0_no_history(bootstrap_card):
    raw = _stub_response(
        [
            _strategy(
                "S1",
                domain="observation",
                slot_edits={"guidance_observation": "use proximity_count"},
            ),
            _strategy(
                "S2",
                domain="reward",
                slot_edits={"guidance_reward": "potential shaping"},
            ),
            _strategy("S3", domain="both"),
        ]
    )
    llm = _StubLLM([raw])
    bundle = emit_strategies(
        bootstrap=bootstrap_card,
        round_history=[],
        meta_llm=llm,
        round_idx=0,
        n_strategies=3,
    )
    assert len(bundle.strategies) == 3
    assert bundle.round_idx == 0
    assert llm.calls == 1


def test_emit_strategies_renames_off_convention_ids(bootstrap_card):
    raw = _stub_response(
        [
            _strategy("Alpha"),
            _strategy("Beta"),
            _strategy("Gamma"),
        ]
    )
    llm = _StubLLM([raw])
    bundle = emit_strategies(
        bootstrap=bootstrap_card,
        round_history=[],
        meta_llm=llm,
        round_idx=0,
        n_strategies=3,
    )
    ids = [s.strategy_id for s in bundle.strategies]
    assert ids == ["S1", "S2", "S3"]


def test_emit_strategies_clears_revert_without_reason(bootstrap_card):
    raw = _stub_response(
        [
            _strategy("S1", revert=True, revert_reason=None),
            _strategy("S2"),
            _strategy("S3"),
        ]
    )
    llm = _StubLLM([raw])
    bundle = emit_strategies(
        bootstrap=bootstrap_card,
        round_history=[],
        meta_llm=llm,
        round_idx=0,
        n_strategies=3,
    )
    # Safety: revert flag cleared when reason missing
    assert bundle.strategies[0].revert_to_baseline_reward is False


def test_emit_strategies_wrong_count_raises(bootstrap_card):
    raw = _stub_response([_strategy("S1")])  # only 1 strategy
    llm = _StubLLM([raw])
    with pytest.raises(ValueError, match="returned 1"):
        emit_strategies(
            bootstrap=bootstrap_card,
            round_history=[],
            meta_llm=llm,
            round_idx=0,
            n_strategies=3,
        )


# ── Composer ────────────────────────────────────────────────────


@pytest.fixture
def base_prompt_dir():
    return Path(
        "/Users/afin/Documents/Studio/PHD/Code/VectorizedMultiAgentSimulator/"
        "rendezvous_comm/src/lero/prompts/v2_fewshot_modular_v2"
    )


def test_composer_overlays_slot_edits(tmp_path, base_prompt_dir):
    s = StrategyV4(
        strategy_id="S1",
        high_level_idea="test",
        target_domain="observation",
        slot_edits={"guidance_observation": "USE proximity_count + gap"},
        expected_effect="improvement",
        rationale="testing",
    )
    target = compose_prompt_for_strategy(
        base_prompt_dir=base_prompt_dir,
        strategy=s,
        output_root=tmp_path,
        candidate_id="S1",
    )
    assert target.exists()
    obs = (target / "guidance_observation.txt").read_text()
    assert "USE proximity_count" in obs
    # Other slots unchanged from base
    assert (target / "fairness.txt").exists()


def test_composer_revert_clears_reward(tmp_path, base_prompt_dir):
    s = StrategyV4(
        strategy_id="S2",
        high_level_idea="revert reward",
        target_domain="observation",
        revert_to_baseline_reward=True,
        revert_reason="round 0 had peak_collapse on reward edit",
        slot_edits={"guidance_reward": "this should be cleared anyway"},
        expected_effect="stability",
        rationale="testing",
    )
    target = compose_prompt_for_strategy(
        base_prompt_dir=base_prompt_dir,
        strategy=s,
        output_root=tmp_path,
        candidate_id="S2",
    )
    rew = (target / "guidance_reward.txt").read_text()
    assert rew.strip() == ""


def test_composer_rejects_unknown_slot(tmp_path, base_prompt_dir):
    s = StrategyV4(
        strategy_id="S1",
        high_level_idea="bad",
        target_domain="observation",
        slot_edits={"some_other_slot": "text"},
        expected_effect="x",
        rationale="x",
    )
    with pytest.raises(ValueError, match="unknown slot"):
        compose_prompt_for_strategy(
            base_prompt_dir=base_prompt_dir,
            strategy=s,
            output_root=tmp_path,
            candidate_id="S1",
        )


def test_composer_overwrites_existing_target(tmp_path, base_prompt_dir):
    """Re-running for same candidate_id should clobber the old dir."""
    s = StrategyV4(
        strategy_id="S1",
        high_level_idea="v1",
        target_domain="observation",
        slot_edits={"guidance_observation": "first version"},
        expected_effect="x",
        rationale="x",
    )
    compose_prompt_for_strategy(
        base_prompt_dir=base_prompt_dir,
        strategy=s,
        output_root=tmp_path,
        candidate_id="S1",
    )
    s.slot_edits = {"guidance_observation": "second version"}
    target = compose_prompt_for_strategy(
        base_prompt_dir=base_prompt_dir,
        strategy=s,
        output_root=tmp_path,
        candidate_id="S1",
    )
    assert "second version" in (target / "guidance_observation.txt").read_text()
