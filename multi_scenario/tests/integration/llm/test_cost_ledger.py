"""F9.1 — :class:`InMemoryCostLedger` + :class:`FilesystemCostLedger` contract."""

# pylint: disable=missing-function-docstring,redefined-outer-name

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from multi_scenario.adapters.llm.filesystem_cost_ledger import (
    FilesystemCostLedger,
    InMemoryCostLedger,
)


# ── helpers ───────────────────────────────────────────────────────────


@pytest.fixture
def fixed_now():
    """Fixed UTC timestamp for deterministic rolling-window tests."""
    return datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc)


# ── InMemoryCostLedger ────────────────────────────────────────────────


def test_in_memory_ledger_records_and_sums_within_window(fixed_now):
    """Basic contract: record N entries, sum_window returns their total."""
    cur = {"t": fixed_now}
    ledger = InMemoryCostLedger(_now=lambda: cur["t"])
    ledger.record(cost_eur=0.50, model="gpt-4o-mini")
    ledger.record(cost_eur=0.25, model="gpt-4o-mini")
    ledger.record(cost_eur=1.00, model="gpt-4o-mini")
    assert ledger.sum_window(timedelta(hours=1)) == pytest.approx(1.75)


def test_in_memory_ledger_excludes_entries_outside_window(fixed_now):
    """Entries older than the window must not contribute to the sum."""
    cur = {"t": fixed_now - timedelta(hours=2)}
    ledger = InMemoryCostLedger(_now=lambda: cur["t"])
    ledger.record(cost_eur=5.00, model="x")  # 2h ago

    cur["t"] = fixed_now
    ledger.record(cost_eur=0.50, model="x")  # now

    # 1h window → only "now" entry counts.
    assert ledger.sum_window(timedelta(hours=1)) == pytest.approx(0.50)
    # 24h window → both count.
    assert ledger.sum_window(timedelta(days=1)) == pytest.approx(5.50)


def test_in_memory_ledger_empty_returns_zero():
    ledger = InMemoryCostLedger()
    assert ledger.sum_window(timedelta(days=1)) == 0.0
    assert ledger.sum_window(timedelta(days=30)) == 0.0


# ── FilesystemCostLedger ─────────────────────────────────────────────


def test_fs_ledger_persists_across_instances(tmp_path: Path, fixed_now):
    """The whole point of the FS ledger: restart-survival."""
    ledger_path = tmp_path / "ledger.jsonl"
    cur = {"t": fixed_now}
    a = FilesystemCostLedger(ledger_path, now=lambda: cur["t"])
    a.record(cost_eur=2.00, model="gpt-4o-mini")
    a.record(cost_eur=1.00, model="claude-sonnet")

    # Fresh instance reads the same file → same totals.
    b = FilesystemCostLedger(ledger_path, now=lambda: cur["t"])
    assert b.sum_window(timedelta(days=1)) == pytest.approx(3.00)


def test_fs_ledger_creates_parent_directory(tmp_path: Path):
    """Writing should create ``~/.multi_scenario/`` lazily — no manual mkdir."""
    ledger_path = tmp_path / "deeply" / "nested" / "ledger.jsonl"
    ledger = FilesystemCostLedger(ledger_path)
    ledger.record(cost_eur=0.10, model="x")
    assert ledger_path.is_file()
    assert ledger_path.parent.is_dir()


def test_fs_ledger_env_var_override(tmp_path: Path, monkeypatch):
    """``MULTI_SCENARIO_COST_LEDGER`` overrides the default path."""
    custom = tmp_path / "custom_ledger.jsonl"
    monkeypatch.setenv("MULTI_SCENARIO_COST_LEDGER", str(custom))
    ledger = FilesystemCostLedger()
    ledger.record(cost_eur=0.05, model="x")
    assert custom.is_file()


def test_fs_ledger_sum_window_with_missing_file_returns_zero(tmp_path: Path):
    """No file yet (fresh install) → zero spend, no crash."""
    ledger = FilesystemCostLedger(tmp_path / "nonexistent.jsonl")
    assert ledger.sum_window(timedelta(days=1)) == 0.0


def test_fs_ledger_prunes_entries_older_than_31_days(tmp_path: Path, fixed_now):
    """Prune-on-read keeps the file from growing unboundedly."""
    ledger_path = tmp_path / "ledger.jsonl"
    cur = {"t": fixed_now - timedelta(days=60)}
    ledger = FilesystemCostLedger(ledger_path, now=lambda: cur["t"])
    ledger.record(cost_eur=10.00, model="x")  # 60d ago

    cur["t"] = fixed_now
    ledger.record(cost_eur=1.00, model="x")  # now

    # 30d window: only the recent entry counts.
    assert ledger.sum_window(timedelta(days=30)) == pytest.approx(1.00)
    # The prune-on-read should have rewritten the file without the 60d entry.
    text = ledger_path.read_text("utf-8")
    assert "10.0" not in text  # old entry pruned
    assert "1.0" in text


def test_fs_ledger_skips_corrupt_lines(tmp_path: Path):
    """A half-written line from a crashed prior process must not break reads."""
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(
        '{"ts":"2026-05-10T12:00:00+00:00","cost_eur":0.5,"model":"x","prompt_tokens":0,"completion_tokens":0}\n'
        "this is not valid json\n"
        '{"ts":"2026-05-10T13:00:00+00:00","cost_eur":1.0,"model":"x","prompt_tokens":0,"completion_tokens":0}\n',
        encoding="utf-8",
    )
    ledger = FilesystemCostLedger(ledger_path)
    # The two valid lines sum; the corrupt line is silently skipped.
    assert ledger.sum_window(timedelta(days=365)) == pytest.approx(1.5)


def test_fs_ledger_default_path_is_under_home(monkeypatch):
    """Sanity: when no override is set, the default path is in ``~/``."""
    monkeypatch.delenv("MULTI_SCENARIO_COST_LEDGER", raising=False)
    ledger = FilesystemCostLedger()
    # Don't touch the file — just assert path is under HOME.
    assert str(ledger._path).startswith(  # pylint: disable=protected-access
        os.path.expanduser("~")
    )
