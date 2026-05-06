"""Tests for v4 bootstrap module — stub LLM, no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.lero.config import LLMConfig
from src.lero.meta.v4_bootstrap import (
    _cache_key,
    _parse_response,
    bootstrap_from_description,
)


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def description_md(tmp_path):
    p = tmp_path / "task.md"
    p.write_text(
        "# Task: Multi-robot rendezvous\n\n"
        "4 agents must cover 4 targets simultaneously (k=2).\n\n"
        "## Available state\n"
        "- agent_pos, agent_vel, lidar_targets, lidar_agents\n",
    )
    return p


@pytest.fixture
def base_prompt_dir():
    return Path(
        "/Users/afin/Documents/Studio/PHD/Code/VectorizedMultiAgentSimulator/"
        "rendezvous_comm/src/lero/prompts/v2_fewshot_modular_v2"
    )


def _stub_response(card_data: dict, thoughts: str = "Some reasoning") -> str:
    return (
        "### THOUGHTS\n" + thoughts + "\n\n"
        "### BOOTSTRAP_CARD\n"
        "```json\n" + json.dumps(card_data, indent=2) + "\n"
        "```\n"
    )


_VALID_CARD = {
    "task_summary": "Multi-robot rendezvous, k=2, partial observability",
    "success_metric_understanding": "M1 = fraction of episodes solved",
    "key_difficulty": "Coordination requires 2 agents at same target simultaneously",
    "failure_modes_anticipated": [
        "ships passing in the night",
        "anti-crowding",
        "reward gaming at long training",
    ],
    "high_level_strategies_considered": [
        "pre-compute hold_signal for arrived agents",
        "encode proximity counts and intensity",
    ],
    "proposed_initial_obs_features": [
        "hold_signal = target_near AND partner_near",
        "approach_signal",
        "proximity_count",
    ],
    "proposed_initial_reward_components": [],
    "fairness_audit": "Uses only lidar + own state, no oracle keys",
    "assumptions": ["agents are point particles", "lidar rays are uniform"],
}


class _StubLLM:
    """Mimics LLMClient.generate."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.config = LLMConfig(
            model="gpt-5.4-mini",
            temperature=1.0,
        )
        self.calls = 0

    def generate(self, messages, n=1, seed_base=None):
        out = []
        for _ in range(n):
            self.calls += 1
            if not self._responses:
                raise RuntimeError("stub exhausted")
            out.append(self._responses.pop(0))
        return out


# ── _cache_key ─────────────────────────────────────────────────


def test_cache_key_stable():
    k1 = _cache_key("desc", "gpt-5.4-mini", 1.0)
    k2 = _cache_key("desc", "gpt-5.4-mini", 1.0)
    assert k1 == k2


def test_cache_key_changes_on_description():
    assert _cache_key("a", "gpt-5.4-mini", 1.0) != _cache_key("b", "gpt-5.4-mini", 1.0)


def test_cache_key_changes_on_model():
    assert _cache_key("a", "gpt-5.4-mini", 1.0) != _cache_key("a", "o4-mini", 1.0)


# ── _parse_response ────────────────────────────────────────────


def test_parse_response_extracts_thoughts_and_card():
    raw = _stub_response(_VALID_CARD, "I think this is hard because…")
    thoughts, card = _parse_response(raw)
    assert "I think this is hard" in thoughts
    assert card.task_summary == _VALID_CARD["task_summary"]
    assert "ships passing in the night" in card.failure_modes_anticipated


def test_parse_response_rejects_missing_thoughts():
    raw = "### BOOTSTRAP_CARD\n```json\n{}\n```"
    with pytest.raises(ValueError, match="### THOUGHTS"):
        _parse_response(raw)


def test_parse_response_rejects_missing_card():
    raw = "### THOUGHTS\nthinking here only\n"
    with pytest.raises(ValueError, match="BOOTSTRAP_CARD"):
        _parse_response(raw)


def test_parse_response_rejects_invalid_json():
    raw = "### THOUGHTS\nok\n### BOOTSTRAP_CARD\n```json\nnot json\n```"
    with pytest.raises(ValueError):
        _parse_response(raw)


# ── End-to-end with stub LLM ───────────────────────────────────


def test_bootstrap_creates_artifacts(tmp_path, description_md, base_prompt_dir):
    llm = _StubLLM([_stub_response(_VALID_CARD, "thoughts here")])
    result = bootstrap_from_description(
        description_path=description_md,
        meta_llm=llm,
        output_dir=tmp_path,
        base_prompt_dir=base_prompt_dir,
    )
    assert result.cache_hit is False
    assert llm.calls == 1
    # Card persisted
    card_file = tmp_path / "bootstrap" / "bootstrap_card.json"
    assert card_file.exists()
    # Thoughts persisted
    thoughts_file = tmp_path / "bootstrap" / "bootstrap_thoughts.md"
    assert thoughts_file.exists()
    assert "thoughts here" in thoughts_file.read_text()
    # Prompt directory materialized with the LLM's features
    prompt_dir = result.bootstrap_dir
    assert prompt_dir.exists()
    obs_text = (prompt_dir / "guidance_observation.txt").read_text()
    assert "hold_signal" in obs_text
    assert "proximity_count" in obs_text


def test_bootstrap_cache_hit_skips_llm(tmp_path, description_md, base_prompt_dir):
    cache_dir = tmp_path / "cache"
    llm = _StubLLM([_stub_response(_VALID_CARD)])

    # First call → cache miss → 1 LLM call
    bootstrap_from_description(
        description_path=description_md,
        meta_llm=llm,
        output_dir=tmp_path,
        base_prompt_dir=base_prompt_dir,
        cache_dir=cache_dir,
    )
    assert llm.calls == 1

    # Second call (same description) → cache hit → no new call
    llm2 = _StubLLM([])  # no responses; would fail if invoked
    result = bootstrap_from_description(
        description_path=description_md,
        meta_llm=llm2,
        output_dir=tmp_path / "run2",
        base_prompt_dir=base_prompt_dir,
        cache_dir=cache_dir,
    )
    assert result.cache_hit is True
    assert llm2.calls == 0


def test_bootstrap_cache_miss_on_changed_description(
    tmp_path,
    description_md,
    base_prompt_dir,
):
    cache_dir = tmp_path / "cache"
    llm = _StubLLM([_stub_response(_VALID_CARD), _stub_response(_VALID_CARD)])

    bootstrap_from_description(
        description_path=description_md,
        meta_llm=llm,
        output_dir=tmp_path,
        base_prompt_dir=base_prompt_dir,
        cache_dir=cache_dir,
    )

    # Modify the description
    description_md.write_text(description_md.read_text() + "\n\n## Extra section")

    bootstrap_from_description(
        description_path=description_md,
        meta_llm=llm,
        output_dir=tmp_path / "run2",
        base_prompt_dir=base_prompt_dir,
        cache_dir=cache_dir,
    )
    # Both calls hit the LLM (different cache keys)
    assert llm.calls == 2


def test_bootstrap_prompt_materialization_skips_empty_components(
    tmp_path,
    description_md,
    base_prompt_dir,
):
    """When the LLM proposes no reward components, guidance_reward.txt
    should be empty (LERO uses hand-crafted reward by default)."""
    card_data = dict(_VALID_CARD)
    card_data["proposed_initial_reward_components"] = []
    llm = _StubLLM([_stub_response(card_data)])
    result = bootstrap_from_description(
        description_path=description_md,
        meta_llm=llm,
        output_dir=tmp_path,
        base_prompt_dir=base_prompt_dir,
    )
    rew = (result.bootstrap_dir / "guidance_reward.txt").read_text()
    assert rew.strip() == ""


def test_bootstrap_prompt_includes_failure_modes_in_shared(
    tmp_path,
    description_md,
    base_prompt_dir,
):
    llm = _StubLLM([_stub_response(_VALID_CARD)])
    result = bootstrap_from_description(
        description_path=description_md,
        meta_llm=llm,
        output_dir=tmp_path,
        base_prompt_dir=base_prompt_dir,
    )
    sh = (result.bootstrap_dir / "guidance_shared.txt").read_text()
    assert "STABLE end-of-training" in sh
    assert "ships passing in the night" in sh
