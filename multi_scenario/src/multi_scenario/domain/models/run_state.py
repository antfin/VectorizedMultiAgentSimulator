"""Run lifecycle state machine — persisted in run_state.json."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel

from ._common import STRICT


class RunState(str, Enum):
    """Lifecycle states a run passes through; persisted in ``run_state.json``."""

    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    CRASHED = "CRASHED"
    RESUMED = "RESUMED"


# Allowed forward transitions in the run-state machine. DONE is terminal
# (empty set). Recovery from CRASHED goes through RESUMED before training
# can continue.
_VALID_TRANSITIONS: dict[RunState, set[RunState]] = {
    RunState.INITIALIZING: {RunState.RUNNING, RunState.CRASHED},
    RunState.RUNNING: {RunState.DONE, RunState.CRASHED},
    RunState.DONE: set(),
    RunState.CRASHED: {RunState.RESUMED},
    RunState.RESUMED: {RunState.RUNNING, RunState.DONE, RunState.CRASHED},
}


class RunStateTransition(BaseModel):
    """One step in the run lifecycle log."""

    model_config = STRICT

    state: RunState
    ts: datetime


class RunStateRecord(BaseModel):
    """Persisted run-state shape: current state plus full transition log."""

    model_config = STRICT

    state: RunState
    transitions: list[RunStateTransition]

    @classmethod
    def initial(cls, ts: datetime) -> "RunStateRecord":
        """Create a fresh record in INITIALIZING with the start transition logged."""
        return cls(
            state=RunState.INITIALIZING,
            transitions=[RunStateTransition(state=RunState.INITIALIZING, ts=ts)],
        )

    def transition_to(self, new_state: RunState, ts: datetime) -> "RunStateRecord":
        """Return a new record with the transition appended; raises if illegal."""
        if new_state not in _VALID_TRANSITIONS[self.state]:
            raise ValueError(f"invalid transition {self.state.value} -> {new_state.value}")
        return RunStateRecord(
            state=new_state,
            transitions=[
                *self.transitions,
                RunStateTransition(state=new_state, ts=ts),
            ],
        )
