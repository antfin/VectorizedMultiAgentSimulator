"""F6.2 tests: OvhClient — subprocess wrapper for the ``ovhai`` CLI (mocked)."""

# Each fake `runner(args, timeout)` matches the real subprocess wrapper's
# signature; the args/timeout aren't always inspected — that's not a smell.
# pylint: disable=unused-argument

import json
import subprocess
from typing import Sequence

import pytest

from multi_scenario.adapters.runners.ovh_cli import OvhClient, OvhCliError


def _make_proc(
    stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


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


def test_ensure_available_raises_with_install_hint_when_missing() -> None:
    """``ensure_available`` raises :class:`OvhCliError` whose message includes the install URL."""

    def runner(args, timeout=60):  # noqa: ARG001
        raise FileNotFoundError("no such file: ovhai")

    with pytest.raises(OvhCliError, match="cli.bhs.ai.cloud.ovh.net/install.sh"):
        OvhClient(runner=runner).ensure_available()


def test_ensure_available_silent_when_binary_present() -> None:
    """``ensure_available`` is a no-op when the CLI is on PATH."""

    def runner(args, timeout=60):  # noqa: ARG001
        return _make_proc(stdout="ovhai 3.35.0\n")

    OvhClient(runner=runner).ensure_available()  # must not raise


def test_submit_returns_job_id_from_plain_text_uuid() -> None:
    """Plain-text fallback: scan all lines for a UUID-shaped token."""

    def runner(args, timeout=60):  # noqa: ARG001
        return _make_proc(stdout="49c01e83-faf0-4e1d-a2e7-5534554f8f00  some other text\n")

    out = OvhClient(runner=runner).submit(["--gpu", "V100S", "image"])
    assert out == "49c01e83-faf0-4e1d-a2e7-5534554f8f00"


def test_submit_handles_created_then_uuid_format_smoke_2026_05_09() -> None:
    """Regression: ovhai 3.35 prints ``Created\\n<uuid>``; pre-fix returned 'Created'."""

    def runner(args, timeout=60):  # noqa: ARG001
        return _make_proc(stdout="Created\n49c01e83-faf0-4e1d-a2e7-5534554f8f00\n")

    out = OvhClient(runner=runner).submit(["image"])
    assert out == "49c01e83-faf0-4e1d-a2e7-5534554f8f00"


def test_submit_raises_when_no_uuid_in_text() -> None:
    """Plain-text without any UUID → clean error mentioning the stdout snippet."""

    def runner(args, timeout=60):  # noqa: ARG001
        return _make_proc(stdout="Created\nbut no UUID here\n")

    with pytest.raises(OvhCliError, match="UUID-shaped"):
        OvhClient(runner=runner).submit(["image"])


def test_submit_passes_output_json_flag() -> None:
    """F7.7.A5: ``--output json`` is appended to keep the JSON path active."""
    captured: list = []

    def runner(args, timeout=60):  # noqa: ARG001
        captured.append(args)
        return _make_proc(stdout=json.dumps({"id": "job_99"}))

    OvhClient(runner=runner).submit(["--gpu", "V100S", "image"])
    assert captured[0][-2:] == ["--output", "json"]


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
        payload_str = json.dumps({"id": "x", "status": {"state": state}})

        # Bind ``payload_str`` via default arg so the closure captures the
        # current iteration's value (not the loop variable's last binding).
        def runner(args, timeout=60, _payload=payload_str):  # noqa: ARG001
            return _make_proc(stdout=_payload)

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
    assert (
        OvhClient(runner=lambda args, timeout=60: _make_proc(returncode=0)).stop("x")
        is True
    )
    assert (
        OvhClient(runner=lambda args, timeout=60: _make_proc(returncode=1)).stop("x")
        is False
    )


# ── F7.7.A1: bucket verbs (used by preflight probes after F7.7.A2) ──


def test_bucket_list_parses_json_array() -> None:
    """``ovhai bucket list <region> --output json`` → list of BucketInfo."""
    captured: list[Sequence[str]] = []
    payload = json.dumps(
        [
            {
                "name": "ms-code",
                "bytes": 0,
                "count": 0,
                "last_modified": "2026-01-01T00:00:00",
            },
            {
                "name": "ms-results",
                "bytes": 1024,
                "count": 3,
                "last_modified": "2026-01-02T00:00:00",
            },
        ]
    )

    def runner(args, timeout=60):
        captured.append(args)
        return _make_proc(stdout=payload)

    out = OvhClient(runner=runner).bucket_list("GRA")
    assert [(b.name, b.size_bytes, b.count) for b in out] == [
        ("ms-code", 0, 0),
        ("ms-results", 1024, 3),
    ]
    assert captured == [["ovhai", "bucket", "list", "GRA", "--output", "json"]]


def test_bucket_list_objects_flattens_wrapped_records() -> None:
    """The CLI wraps each entry as {object: {…}, detail: {…}} — we project the inner half."""
    payload = json.dumps(
        [
            {
                "object": {
                    "name": "configs/foo.yaml",
                    "bytes": 42,
                    "hash": "abc",
                    "last_modified": "2026-01-01T00:00:00",
                    "manifest": None,  # extra field — must be tolerated
                },
                "detail": {"container": "ms-code"},
            },
        ]
    )
    runner = lambda args, timeout=60: _make_proc(stdout=payload)  # noqa: E731
    out = OvhClient(runner=runner).bucket_list_objects("GRA", "ms-code")
    assert len(out) == 1
    assert out[0].name == "configs/foo.yaml"
    assert out[0].size_bytes == 42
    assert out[0].hash_ == "abc"


def test_bucket_list_objects_passes_prefix_flag() -> None:
    """``--prefix`` is forwarded to the CLI."""
    captured: list[Sequence[str]] = []

    def runner(args, timeout=60):
        captured.append(args)
        return _make_proc(stdout="[]")

    OvhClient(runner=runner).bucket_list_objects("GRA", "ms-code", prefix="foo/")
    assert captured[0] == [
        "ovhai",
        "bucket",
        "object",
        "list",
        "ms-code@GRA",
        "--output",
        "json",
        "--prefix",
        "foo/",
    ]


def test_bucket_list_objects_max_keys_truncates_client_side() -> None:
    """``max_keys`` is applied after parsing (CLI doesn't expose a native cap)."""
    payload = json.dumps([{"object": {"name": f"k{i}", "bytes": 0}} for i in range(5)])
    runner = lambda args, timeout=60: _make_proc(stdout=payload)  # noqa: E731
    out = OvhClient(runner=runner).bucket_list_objects("GRA", "ms-code", max_keys=2)
    assert [o.name for o in out] == ["k0", "k1"]


def test_bucket_list_objects_empty_stdout_returns_empty_list() -> None:
    """No buckets / no objects → ``[]`` (don't try json.loads on empty)."""
    runner = lambda args, timeout=60: _make_proc(stdout="")  # noqa: E731
    assert OvhClient(runner=runner).bucket_list_objects("GRA", "ms-code") == []


def test_bucket_list_objects_raises_on_nonzero_rc() -> None:
    """rc != 0 → :class:`OvhCliError` carrying stderr."""
    runner = lambda args, timeout=60: _make_proc(  # noqa: E731
        stderr="permission denied", returncode=2
    )
    with pytest.raises(OvhCliError, match="permission denied"):
        OvhClient(runner=runner).bucket_list_objects("GRA", "ms-code")


def test_bucket_object_exists_true_on_exact_match() -> None:
    """Exact-key match wins — ``foo.yaml`` matches itself but not ``foo.yaml.bak.gz``."""
    payload = json.dumps(
        [
            {"object": {"name": "foo.yaml", "bytes": 0}},
            {"object": {"name": "foo.yaml.bak", "bytes": 0}},
        ]
    )
    runner = lambda args, timeout=60: _make_proc(stdout=payload)  # noqa: E731
    client = OvhClient(runner=runner)
    assert client.bucket_object_exists("GRA", "ms-code", "foo.yaml") is True
    assert client.bucket_object_exists("GRA", "ms-code", "foo.yaml.bak") is True
    assert client.bucket_object_exists("GRA", "ms-code", "foo.yaml.bak.gz") is False


def test_bucket_object_exists_false_on_cli_error() -> None:
    """Probe must not raise on auth / connectivity issues — return False instead."""
    runner = lambda args, timeout=60: _make_proc(  # noqa: E731
        stderr="auth failed", returncode=1
    )
    assert OvhClient(runner=runner).bucket_object_exists("GRA", "ms-code", "k") is False


def test_bucket_get_object_reads_downloaded_file() -> None:
    """``bucket_get_object`` downloads to a tempdir, reads bytes, cleans up."""
    from pathlib import Path as _P

    expected = b"hello bytes"
    captured_args: list[Sequence[str]] = []

    def runner(args, timeout=60):
        captured_args.append(args)
        # Materialise the expected file at ``<--output>/<key>``.
        idx = args.index("--output")
        out_dir = _P(args[idx + 1])
        key = args[idx - 1]  # last positional before --output
        target = out_dir / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(expected)
        return _make_proc(returncode=0)

    out = OvhClient(runner=runner).bucket_get_object(
        "GRA", "ms-code", "configs/foo.yaml"
    )
    assert out == expected
    assert captured_args[0][0:5] == [
        "ovhai",
        "bucket",
        "object",
        "download",
        "ms-code@GRA",
    ]


def test_bucket_get_object_raises_on_nonzero_rc() -> None:
    """Download failure → :class:`OvhCliError`."""
    runner = lambda args, timeout=60: _make_proc(  # noqa: E731
        stderr="not found", returncode=1
    )
    with pytest.raises(OvhCliError, match="not found"):
        OvhClient(runner=runner).bucket_get_object("GRA", "ms-code", "missing")


def test_bucket_get_object_raises_when_file_doesnt_materialise() -> None:
    """Defensive: rc=0 but the expected file isn't on disk → loud error."""
    runner = lambda args, timeout=60: _make_proc(returncode=0)  # noqa: E731
    with pytest.raises(OvhCliError, match="missing"):
        OvhClient(runner=runner).bucket_get_object("GRA", "ms-code", "x")
