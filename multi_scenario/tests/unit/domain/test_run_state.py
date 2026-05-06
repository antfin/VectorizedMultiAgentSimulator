"""F1.4 tests: RunState enum + persisted RunStateRecord shape."""

from datetime import datetime, timezone

import pytest

from multi_scenario.domain.models import RunState, RunStateRecord


def _ts() -> datetime:
    """Single fixed UTC timestamp used across tests."""
    return datetime(2026, 5, 6, 14, 23, 0, tzinfo=timezone.utc)


def test_initial():
    """Fresh record starts in INITIALIZING with one transition logged."""
    rec = RunStateRecord.initial(_ts())
    assert rec.state == RunState.INITIALIZING
    assert len(rec.transitions) == 1
    assert rec.transitions[0].state == RunState.INITIALIZING
    assert rec.transitions[0].ts == _ts()


def test_happy_path():
    """INITIALIZING → RUNNING → DONE chain succeeds."""
    rec = (
        RunStateRecord.initial(_ts())
        .transition_to(RunState.RUNNING, _ts())
        .transition_to(RunState.DONE, _ts())
    )
    assert rec.state == RunState.DONE
    assert [t.state for t in rec.transitions] == [
        RunState.INITIALIZING,
        RunState.RUNNING,
        RunState.DONE,
    ]


def test_crashed_resumed_path():
    """INITIALIZING → RUNNING → CRASHED → RESUMED → RUNNING → DONE succeeds."""
    rec = (
        RunStateRecord.initial(_ts())
        .transition_to(RunState.RUNNING, _ts())
        .transition_to(RunState.CRASHED, _ts())
        .transition_to(RunState.RESUMED, _ts())
        .transition_to(RunState.RUNNING, _ts())
        .transition_to(RunState.DONE, _ts())
    )
    assert rec.state == RunState.DONE
    assert len(rec.transitions) == 6


def test_done_terminal_rejects_running():
    """The plan-specified case: DONE → RUNNING is rejected."""
    rec = (
        RunStateRecord.initial(_ts())
        .transition_to(RunState.RUNNING, _ts())
        .transition_to(RunState.DONE, _ts())
    )
    with pytest.raises(ValueError):
        rec.transition_to(RunState.RUNNING, _ts())


def test_done_terminal_rejects_all():
    """No transition out of DONE is permitted."""
    rec = (
        RunStateRecord.initial(_ts())
        .transition_to(RunState.RUNNING, _ts())
        .transition_to(RunState.DONE, _ts())
    )
    for target in (RunState.CRASHED, RunState.RESUMED, RunState.INITIALIZING):
        with pytest.raises(ValueError):
            rec.transition_to(target, _ts())


def test_skip_to_done_rejected():
    """Cannot skip from INITIALIZING straight to DONE — must go through RUNNING."""
    rec = RunStateRecord.initial(_ts())
    with pytest.raises(ValueError):
        rec.transition_to(RunState.DONE, _ts())


def test_running_to_resumed_rejected():
    """RUNNING cannot go directly to RESUMED — must crash first."""
    rec = RunStateRecord.initial(_ts()).transition_to(RunState.RUNNING, _ts())
    with pytest.raises(ValueError):
        rec.transition_to(RunState.RESUMED, _ts())


def test_json_roundtrip():
    """model_dump_json followed by model_validate_json preserves state and transitions."""
    rec = (
        RunStateRecord.initial(_ts())
        .transition_to(RunState.RUNNING, _ts())
        .transition_to(RunState.DONE, _ts())
    )
    rec2 = RunStateRecord.model_validate_json(rec.model_dump_json())
    assert rec == rec2
