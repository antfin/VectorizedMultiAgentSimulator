"""F6.2 tests: OvhClient — subprocess wrapper for the ``ovhai`` CLI (mocked)."""

# Each fake `runner(args, timeout)` matches the real subprocess wrapper's
# signature; the args/timeout aren't always inspected — that's not a smell.
# pylint: disable=unused-argument

import json
import subprocess
from typing import Sequence

import pytest

from multi_scenario.adapters.runners.ovh_cli import (
    OvhClient,
    OvhCliError,
)


def _make_proc(
    stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def test_check_available_true_on_zero_rc() -> None:
    """``ovhai --version`` succeeds → True."""
    captured: list[Sequence[str]] = []

    def runner(args, timeout=60):  # noqa: ARG001
        captured.append(args)
        return _make_proc(stdout="ovhai 1.0.0\n")

    assert OvhClient(runner=runner).check_available() is True
    assert captured == [["ovhai", "--version"]]


def test_check_available_false_when_binary_missing() -> None:
    """FileNotFoundError on the subprocess → False."""

    def runner(args, timeout=60):  # noqa: ARG001
        raise FileNotFoundError("no such file: ovhai")

    assert OvhClient(runner=runner).check_available() is False


def test_submit_returns_job_id_from_plain_text() -> None:
    """Plain-text ``ovhai job run`` output → first whitespace token of first line."""

    def runner(args, timeout=60):  # noqa: ARG001
        return _make_proc(stdout="job_42  CREATED  some other text\n")

    assert OvhClient(runner=runner).submit(["--gpu", "V100S", "image"]) == "job_42"


def test_submit_returns_job_id_from_json() -> None:
    """JSON ``ovhai job run`` output → ``id`` field."""

    def runner(args, timeout=60):  # noqa: ARG001
        return _make_proc(stdout=json.dumps({"id": "job_99", "name": "x"}))

    assert OvhClient(runner=runner).submit(["image"]) == "job_99"


def test_submit_raises_on_nonzero_rc() -> None:
    """Non-zero rc → :class:`OvhCliError` with stderr."""

    def runner(args, timeout=60):  # noqa: ARG001
        return _make_proc(stderr="bucket not found", returncode=2)

    with pytest.raises(OvhCliError, match="bucket not found"):
        OvhClient(runner=runner).submit(["image"])


def test_get_parses_state_from_status_json() -> None:
    """``ovhai job get --output json`` → JobInfo with state from ``status.state``."""
    payload = {
        "id": "job_42",
        "spec": {"name": "demo", "image": "img", "resources": {"gpu": "V100S"}},
        "status": {"state": "RUNNING", "startedAt": "2026-05-07T10:00:00Z"},
    }

    def runner(args, timeout=60):  # noqa: ARG001
        return _make_proc(stdout=json.dumps(payload))

    info = OvhClient(runner=runner).get("job_42")
    assert info.id == "job_42"
    assert info.state == "RUNNING"
    assert info.image == "img"
    assert info.gpu == "V100S"
    assert info.is_terminal is False


def test_get_recognises_terminal_states() -> None:
    """DONE / FAILED / KILLED / ERROR are terminal."""
    for state in ("DONE", "FAILED", "KILLED", "ERROR"):
        # pylint: disable=cell-var-from-loop
        payload_str = json.dumps({"id": "x", "status": {"state": state}})

        def runner(args, timeout=60):
            return _make_proc(stdout=payload_str)

        assert OvhClient(runner=runner).get("x").is_terminal


def test_list_jobs_filters_by_state() -> None:
    """``state_filter`` keeps only matching JobInfos."""
    records = [
        {"id": "a", "status": {"state": "RUNNING"}},
        {"id": "b", "status": {"state": "DONE"}},
        {"id": "c", "status": {"state": "DONE"}},
    ]

    def runner(args, timeout=60):  # noqa: ARG001
        return _make_proc(stdout=json.dumps(records))

    out = OvhClient(runner=runner).list_jobs(state_filter="DONE")
    assert [j.id for j in out] == ["b", "c"]


def test_logs_returns_stdout() -> None:
    """``ovhai job logs`` stdout is returned verbatim."""

    def runner(args, timeout=60):  # noqa: ARG001
        return _make_proc(stdout="line1\nline2\n")

    assert OvhClient(runner=runner).logs("x", tail=2) == "line1\nline2\n"


def test_stop_returns_true_on_zero_rc() -> None:
    """``ovhai job stop`` rc=0 → True."""
    assert OvhClient(runner=lambda args, timeout=60: _make_proc(returncode=0)).stop("x") is True
    assert OvhClient(runner=lambda args, timeout=60: _make_proc(returncode=1)).stop("x") is False
