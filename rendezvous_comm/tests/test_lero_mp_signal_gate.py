"""Tests for behavioral signals + include_signals gate (LERO-MP v3 §4.1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.lero.meta.behavioral_summary import (
    classify_learning_curve,
    fingerprint_from_csv,
    format_behavioral_block,
    format_tier1,
    outlier_flags,
)
from src.lero.meta.strategy import StrategyCard, parse_strategy_card


# ── Tier 1 scalars ───────────────────────────────────────────────


def test_format_tier1_covers_all_metrics():
    m = {
        "M1_success_rate": 0.5, "M2_avg_return": -1.2, "M3_avg_steps": 150,
        "M4_avg_collisions": 12, "M6_coverage_progress": 0.75,
        "M8_agent_utilization_cv": 0.33, "M9_spatial_spread": 0.42,
    }
    line = format_tier1(m)
    for tag in ("M1=", "M2=", "M3=", "M4=", "M6=", "M8=", "M9="):
        assert tag in line


def test_outlier_flags_fire_on_thresholds():
    high_coll = {"M4_avg_collisions": 60}
    assert "M4_high_collisions" in outlier_flags(high_coll)
    clustered = {"M9_spatial_spread": 0.1}
    assert "M9_clustered" in outlier_flags(clustered)
    scattered = {"M9_spatial_spread": 0.9}
    assert "M9_scattered" in outlier_flags(scattered)


def test_outlier_flags_empty_when_in_range():
    normal = {
        "M4_avg_collisions": 5, "M9_spatial_spread": 0.5,
        "M8_agent_utilization_cv": 0.1,
    }
    assert outlier_flags(normal) == []


# ── Tier 3 curve shape ───────────────────────────────────────────


@pytest.mark.parametrize("traj,tag", [
    ([0.0, 0.0, 0.01], "flat_zero"),
    ([0.0, 0.02, 0.04], "flat_nonzero"),
    ([0.0, 0.1, 0.3, 0.5], "monotonic_rise"),
    ([0.0, 0.4, 0.4, 0.1], "reward_hack_shape"),
    ([0.0, 0.4, 0.4, 0.2], "plateau_then_collapse"),
    ([0.0, 0.3, 0.1, 0.35, 0.1, 0.2], "plateau_then_collapse"),
])
def test_classify_learning_curve(traj, tag):
    assert classify_learning_curve(traj) == tag


# ── Include-signals filter ───────────────────────────────────────


def test_format_behavioral_block_scalar_only():
    m = {"M1_success_rate": 0.1, "M4_avg_collisions": 100}
    out = format_behavioral_block(m, include_signals=["scalar"])
    assert "Tier 1" in out
    assert "Tier 2" not in out
    assert "Tier 3" not in out


def test_format_behavioral_block_skips_curve_without_trajectory():
    m = {"M1_success_rate": 0.1}
    out = format_behavioral_block(
        m, include_signals=["scalar", "curve_shape"], trajectory=None,
    )
    # Without a trajectory or CSV, curve_shape tier is skipped.
    assert "Tier 3" not in out


def test_format_behavioral_block_with_curve_shape():
    m = {"M1_success_rate": 0.1}
    traj = [0.0, 0.3, 0.3, 0.05]
    out = format_behavioral_block(
        m, include_signals=["curve_shape"], trajectory=traj,
    )
    assert "Tier 3" in out
    assert "reward_hack_shape" in out


def test_format_behavioral_block_empty_when_nothing_requested():
    m = {"M1_success_rate": 0.5}
    out = format_behavioral_block(m, include_signals=[])
    assert "no behavioral signals included" in out


def test_format_behavioral_block_appends_outliers(tmp_path):
    m = {"M1_success_rate": 0.1, "M4_avg_collisions": 100}
    out = format_behavioral_block(m, include_signals=["scalar"])
    assert "Outliers:" in out
    assert "M4_high_collisions" in out


# ── Strategy card include_signals parsing ─────────────────────────


def test_parse_strategy_card_default_include_signals():
    yaml_text = """
```yaml
target_domain: observation
target_slot: guidance_observation
focus:
  - "add proximity count"
avoid: []
confidence: medium
rationale: "baseline evidence suggests observation missing"
```
"""
    card = parse_strategy_card(yaml_text)
    assert card.include_signals == ["scalar"]


def test_parse_strategy_card_accepts_custom_include_signals():
    yaml_text = """
```yaml
target_domain: reward
target_slot: guidance_reward
focus:
  - "add potential shaping"
avoid: []
confidence: large
rationale: "peak-collapse visible in all seeds"
include_signals:
  - scalar
  - fingerprint
signal_rationale: "need the peak-collapse trace to justify"
```
"""
    card = parse_strategy_card(yaml_text)
    assert set(card.include_signals) == {"scalar", "fingerprint"}
    assert "peak-collapse" in card.signal_rationale


def test_parse_strategy_card_rejects_invalid_tiers():
    yaml_text = """
```yaml
target_domain: reward
target_slot: guidance_reward
focus: []
avoid: []
confidence: small
rationale: "x"
include_signals:
  - "not_a_real_tier"
```
"""
    card = parse_strategy_card(yaml_text)
    # Invalid tiers silently dropped; default kicks in
    assert card.include_signals == ["scalar"]


# ── Fingerprint from CSV ─────────────────────────────────────────


def test_fingerprint_from_csv(tmp_path: Path):
    csv_path = tmp_path / "scalars.csv"
    csv_path.write_text(
        "frames,M1_success_rate\n"
        "0,0.0\n"
        "100000,0.15\n"
        "200000,0.40\n"
        "300000,0.10\n"
    )
    fp = fingerprint_from_csv(csv_path)
    assert fp is not None
    assert "peak_M1=0.400" in fp
    assert "shape=reward_hack_shape" in fp


def test_fingerprint_returns_none_on_missing_cols(tmp_path: Path):
    csv_path = tmp_path / "missing.csv"
    csv_path.write_text("foo,bar\n1,2\n")
    assert fingerprint_from_csv(csv_path) is None
