"""F9.1 — :class:`CostCapDecorator` rolling-window enforcement."""

# pylint: disable=missing-function-docstring,redefined-outer-name

import logging
from datetime import datetime, timedelta, timezone

import pytest

from multi_scenario.adapters.llm.cost_cap import CostCapDecorator
from multi_scenario.adapters.llm.fake_adapter import FakeLlmClient
from multi_scenario.adapters.llm.filesystem_cost_ledger import InMemoryCostLedger
from multi_scenario.domain.lero import LlmCompletion, LlmCostCapExceeded, LlmUsage
from multi_scenario.domain.models import LlmSection


@pytest.fixture
def fixed_now():
    return datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc)


def _completion(cost_usd: float = 1.0) -> LlmCompletion:
    return LlmCompletion(
        text="```python\ndef compute_reward(s): return s['agent_pos']\n```",
        usage=LlmUsage(
            prompt_tokens=100,
            completion_tokens=50,
            estimated_cost_usd=cost_usd,
        ),
    )


def _build(
    *, fake: FakeLlmClient, ledger: InMemoryCostLedger, cfg: LlmSection
) -> CostCapDecorator:
    return CostCapDecorator(fake, ledger=ledger, cfg_llm=cfg)


# ── Pass-through: no cap tripped ──────────────────────────────────────


def test_under_budget_passes_through_and_records(fixed_now):
    ledger = InMemoryCostLedger(_now=lambda: fixed_now)
    cfg = LlmSection(model="gpt-4o-mini", cost_cap_per_day_eur=10.0)
    fake = FakeLlmClient().register_always(_completion(cost_usd=1.0))
    capped = _build(fake=fake, ledger=ledger, cfg=cfg)

    completions = capped.generate(messages=[{"role": "user", "content": "x"}])
    assert len(completions) == 1
    # 1 USD × 0.92 = 0.92 EUR → recorded.
    assert ledger.sum_window(timedelta(days=1)) == pytest.approx(0.92)


def test_multiple_calls_accumulate_in_ledger(fixed_now):
    ledger = InMemoryCostLedger(_now=lambda: fixed_now)
    cfg = LlmSection(model="gpt-4o-mini", usd_to_eur_rate=1.0)  # easier math
    fake = FakeLlmClient().register_always(_completion(cost_usd=2.0))
    capped = _build(fake=fake, ledger=ledger, cfg=cfg)

    capped.generate(messages=[{"role": "user", "content": "x"}])
    capped.generate(messages=[{"role": "user", "content": "x"}])
    assert ledger.sum_window(timedelta(days=1)) == pytest.approx(4.0)


# ── Cap tripped: pre-flight rejects ───────────────────────────────────


def test_day_cap_exceeded_raises_before_inner_call(fixed_now):
    """Pre-existing rolling-day spend > cap → no LLM call made."""
    ledger = InMemoryCostLedger(_now=lambda: fixed_now)
    # Pre-load with €11 spent today.
    ledger.record(cost_eur=11.0, model="x")
    cfg = LlmSection(model="gpt-4o-mini", cost_cap_per_day_eur=10.0)
    fake = FakeLlmClient()  # no rules registered → would raise if called
    capped = _build(fake=fake, ledger=ledger, cfg=cfg)

    with pytest.raises(LlmCostCapExceeded, match="day"):
        capped.generate(messages=[{"role": "user", "content": "x"}])
    # Inner client never invoked (no calls recorded).
    assert fake.calls == []


def test_month_cap_exceeded_raises_before_inner_call(fixed_now):
    """Spread €120 across 15 days — day total stays at €8 each, but month
    rolling sum exceeds the €100 month cap → call rejected pre-flight."""
    cur = {"t": fixed_now}
    ledger = InMemoryCostLedger(_now=lambda: cur["t"])
    for day_offset in range(15):
        cur["t"] = fixed_now - timedelta(days=day_offset)
        ledger.record(cost_eur=8.0, model="x")  # 15 × 8 = 120 EUR
    cur["t"] = fixed_now

    cfg = LlmSection(
        model="gpt-4o-mini",
        cost_cap_per_day_eur=999.0,  # don't trip the day cap
        cost_cap_per_month_eur=100.0,
    )
    fake = FakeLlmClient()
    capped = _build(fake=fake, ledger=ledger, cfg=cfg)

    with pytest.raises(LlmCostCapExceeded, match="month"):
        capped.generate(messages=[{"role": "user", "content": "x"}])
    assert fake.calls == []


def test_cap_exception_carries_amounts(fixed_now):
    ledger = InMemoryCostLedger(_now=lambda: fixed_now)
    ledger.record(cost_eur=15.0, model="x")
    cfg = LlmSection(model="gpt-4o-mini", cost_cap_per_day_eur=10.0)
    fake = FakeLlmClient()
    capped = _build(fake=fake, ledger=ledger, cfg=cfg)

    with pytest.raises(LlmCostCapExceeded) as exc_info:
        capped.generate(messages=[{"role": "user", "content": "x"}])
    err = exc_info.value
    assert err.spent_usd == pytest.approx(15.0)  # the field is named usd-only legacy
    assert err.cap_usd == pytest.approx(10.0)


def test_cap_logs_warning_with_extra_fields(fixed_now, caplog):
    ledger = InMemoryCostLedger(_now=lambda: fixed_now)
    ledger.record(cost_eur=10.5, model="x")
    cfg = LlmSection(model="gpt-4o-mini", cost_cap_per_day_eur=10.0)
    fake = FakeLlmClient()
    capped = _build(fake=fake, ledger=ledger, cfg=cfg)

    caplog.set_level(logging.WARNING, logger="multi_scenario.adapters.llm.cost_cap")
    with pytest.raises(LlmCostCapExceeded):
        capped.generate(messages=[{"role": "user", "content": "x"}])
    assert any("cost cap reached" in rec.getMessage() for rec in caplog.records)


# ── Currency conversion ───────────────────────────────────────────────


def test_cost_recorded_uses_eur_via_usd_to_eur_rate(fixed_now):
    """LiteLLM-reported USD cost gets converted via cfg.usd_to_eur_rate."""
    ledger = InMemoryCostLedger(_now=lambda: fixed_now)
    cfg = LlmSection(model="gpt-4o-mini", usd_to_eur_rate=0.85)
    fake = FakeLlmClient().register_always(_completion(cost_usd=10.0))
    capped = _build(fake=fake, ledger=ledger, cfg=cfg)
    capped.generate(messages=[{"role": "user", "content": "x"}])
    # 10 USD × 0.85 = 8.5 EUR
    assert ledger.sum_window(timedelta(days=1)) == pytest.approx(8.5)
