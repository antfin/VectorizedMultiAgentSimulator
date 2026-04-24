"""Tests for reproducibility: cache, per-candidate seed, schemas (v3 §5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.lero.llm_cache import LLMCache
from src.lero.llm_client import _cache_key_for
from src.lero.schemas import (
    EditorCritique,
    EditorOutput,
    InnerLLMOutput,
    StrategyCard,
)


# ── Cache keys ───────────────────────────────────────────────────


def test_cache_key_stable_for_identical_inputs():
    kw = {
        "model": "gpt-5.4-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 1.0,
        "seed": 42,
        "response_format": None,
    }
    assert _cache_key_for(kw) == _cache_key_for(dict(kw))


def test_cache_key_distinguishes_seeds():
    base = {
        "model": "m", "temperature": 1.0,
        "messages": [{"role": "user", "content": "x"}],
        "response_format": None,
    }
    k1 = _cache_key_for({**base, "seed": 1})
    k2 = _cache_key_for({**base, "seed": 2})
    assert k1 != k2


def test_cache_key_distinguishes_messages():
    base = {
        "model": "m", "temperature": 1.0, "seed": 1,
        "response_format": None,
    }
    k1 = _cache_key_for({**base, "messages": [{"role": "user", "content": "a"}]})
    k2 = _cache_key_for({**base, "messages": [{"role": "user", "content": "b"}]})
    assert k1 != k2


# ── Cache modes ─────────────────────────────────────────────────


def test_cache_off_reads_and_writes_nothing(tmp_path: Path):
    c = LLMCache(mode="off", root=tmp_path)
    c.write("key123", "value")
    # off mode → no files created
    assert not any(tmp_path.iterdir())
    assert c.read("key123") is None


def test_cache_read_write_roundtrip(tmp_path: Path):
    c = LLMCache(mode="read_write", root=tmp_path)
    assert c.read("k1") is None
    c.write("k1", "hello")
    assert c.read("k1") == "hello"


def test_cache_read_only_blocks_writes(tmp_path: Path):
    writer = LLMCache(mode="read_write", root=tmp_path)
    writer.write("k1", "v1")

    reader = LLMCache(mode="read_only", root=tmp_path)
    assert reader.read("k1") == "v1"
    reader.write("k1", "v2_ignored")
    assert reader.read("k1") == "v1"  # write was a no-op


def test_cache_write_only_blocks_reads(tmp_path: Path):
    c = LLMCache(mode="write_only", root=tmp_path)
    c.write("k1", "v1")
    assert c.read("k1") is None  # no reads allowed
    # The file should still exist
    assert (tmp_path / "k1.txt").exists()


def test_cache_env_override(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LERO_LLM_CACHE_MODE", "read_only")
    c = LLMCache(mode="off", root=tmp_path)
    # env should override constructor
    assert c.mode == "read_only"


# ── Pydantic schema roundtrips ────────────────────────────────────


def test_inner_llm_output_schema():
    raw = {
        "obs_code": "def enhance_observation(s): return None",
        "reward_code": None,
        "rationale": "test",
    }
    out = InnerLLMOutput.model_validate(raw)
    assert out.obs_code.startswith("def enhance_observation")
    assert out.reward_code is None
    # Round trip
    dumped = out.model_dump()
    again = InnerLLMOutput.model_validate(dumped)
    assert again == out


def test_strategy_card_defaults_include_signals():
    raw = {
        "target_domain": "observation",
        "target_slot": "guidance_observation",
        "rationale": "x",
    }
    card = StrategyCard.model_validate(raw)
    assert card.include_signals == ["scalar"]
    assert card.focus == []


def test_strategy_card_rejects_invalid_slot():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        StrategyCard.model_validate({
            "target_domain": "reward",
            "target_slot": "not_a_valid_slot",
            "rationale": "x",
        })


def test_editor_output_schema():
    out = EditorOutput.model_validate({
        "new_slot_content": "use proximity_count",
        "rationale": "implements focus",
        "expected_improvement": "small",
    })
    assert out.new_slot_content == "use proximity_count"


def test_editor_critique_schema():
    c = EditorCritique.model_validate({
        "addresses_focus": True,
        "addresses_focus_reason": "names proximity_count",
        "cites_specific_features": ["proximity_count"],
        "has_fairness_restatement": False,
        "has_fairness_restatement_reason": "no markers",
        "diverges_from_priors": True,
        "suggested_edits": [],
        "suggested_signal_change": "keep",
        "overall_quality": "keep",
    })
    assert c.overall_quality == "keep"
    assert c.suggested_signal_change == "keep"
