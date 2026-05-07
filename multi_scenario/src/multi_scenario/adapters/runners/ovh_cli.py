"""OvhClient — thin subprocess wrapper around the ``ovhai`` CLI binary.

Each method shells out to the CLI and parses its JSON / stdout response. All
calls go through a single ``_run`` helper for mock-friendliness; tests patch
``subprocess.run`` (or pass a stub ``runner`` callable to the constructor).

Job state strings tracked: ``QUEUED``, ``PENDING``, ``RUNNING``, ``DONE``,
``FAILED``, ``KILLED``, ``ERROR`` (terminal: DONE / FAILED / KILLED / ERROR).
The OvhRunner polls ``get(job_id).state`` until terminal.
"""

import json
import subprocess
from typing import Any, Callable, Sequence

from pydantic import BaseModel

from multi_scenario.domain.models._common import STRICT

TERMINAL_STATES: frozenset[str] = frozenset({"DONE", "FAILED", "KILLED", "ERROR"})


class JobInfo(BaseModel):
    """Minimal projection of an ``ovhai job`` record."""

    model_config = STRICT

    id: str
    name: str = ""
    state: str = "UNKNOWN"
    image: str = ""
    gpu: str = ""
    time: str = ""

    @property
    def is_terminal(self) -> bool:
        """True when the job is in a terminal state (DONE / FAILED / KILLED / ERROR)."""
        return self.state.upper() in TERMINAL_STATES


class OvhCliError(RuntimeError):
    """Raised when an ovhai subprocess returns non-zero or malformed output."""


# Default subprocess runner; tests can substitute a callable for full mocking.
def _default_runner(args: Sequence[str], timeout: int = 60) -> subprocess.CompletedProcess:
    """Default subprocess invocation used by :class:`OvhClient` (mockable in tests)."""
    return subprocess.run(  # noqa: S603 - args are a list; no shell expansion
        list(args), capture_output=True, text=True, timeout=timeout, check=False
    )


class OvhClient:
    """Wraps the ``ovhai`` CLI; one method per job-management verb."""

    # The class is intentionally a thin facade over many distinct verbs;
    # pylint's "too few public methods" doesn't apply to facades.
    # pylint: disable=too-few-public-methods

    def __init__(
        self,
        runner: Callable[..., subprocess.CompletedProcess] = _default_runner,
        binary: str = "ovhai",
    ) -> None:
        self._runner = runner
        self._binary = binary

    def check_available(self) -> bool:
        """True when the ``ovhai`` binary is on PATH and runs ``--version``."""
        try:
            res = self._run(["--version"], timeout=5)
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return False
        return res.returncode == 0

    def ensure_available(self) -> None:
        """Raise :class:`OvhCliError` with install instructions if ``ovhai`` is missing.

        Called at the top of every entry point that needs the CLI (sweep
        ``--runner ovh``, ``OvhRunner.submit``, etc.) so users get a helpful
        message instead of a bare ``FileNotFoundError`` traceback.
        """
        if self.check_available():
            return
        raise OvhCliError(
            "ovhai CLI not found on PATH.\n\n"
            "Install it via OVH's official shell script:\n"
            "    curl -sSf https://cli.bhs.ai.cloud.ovh.net/install.sh | bash\n\n"
            "Then authenticate:\n"
            "    ovhai login\n\n"
            "See docs/ovh_setup.md for the full one-time setup."
        )

    def submit(self, args: Sequence[str]) -> str:
        """Run ``ovhai job run <args>``; return the job ID parsed from output."""
        res = self._run(["job", "run", *args])
        if res.returncode != 0:
            raise OvhCliError(f"ovhai job run failed (rc={res.returncode}): {res.stderr.strip()}")
        return _parse_job_id(res.stdout)

    def get(self, job_id: str) -> JobInfo:
        """Run ``ovhai job get <id> --output json`` and project to :class:`JobInfo`."""
        res = self._run(["job", "get", job_id, "--output", "json"])
        if res.returncode != 0:
            raise OvhCliError(f"ovhai job get failed (rc={res.returncode}): {res.stderr.strip()}")
        return _job_info_from_json(res.stdout)

    def list_jobs(self, state_filter: str | None = None) -> list[JobInfo]:
        """List jobs, optionally filtered by state."""
        args = ["job", "list", "--output", "json"]
        res = self._run(args)
        if res.returncode != 0:
            raise OvhCliError(f"ovhai job list failed (rc={res.returncode}): {res.stderr.strip()}")
        records = json.loads(res.stdout) if res.stdout.strip() else []
        infos = [_job_info_from_record(r) for r in records]
        if state_filter is not None:
            infos = [j for j in infos if j.state.upper() == state_filter.upper()]
        return infos

    def logs(self, job_id: str, tail: int = 100) -> str:
        """Return the last ``tail`` lines of the job's combined stdout/stderr."""
        res = self._run(["job", "logs", job_id, "--tail", str(tail)])
        if res.returncode != 0:
            raise OvhCliError(f"ovhai job logs failed (rc={res.returncode}): {res.stderr.strip()}")
        return res.stdout

    def stop(self, job_id: str) -> bool:
        """Stop a running job; returns True on rc=0."""
        res = self._run(["job", "stop", job_id])
        return res.returncode == 0

    def _run(self, args: Sequence[str], timeout: int = 60) -> subprocess.CompletedProcess:
        return self._runner([self._binary, *args], timeout=timeout)


def _parse_job_id(stdout: str) -> str:
    """Extract the job ID from ``ovhai job run`` output.

    The CLI prints one line per submitted job in the form ``<id>  ...``.
    Newer versions print JSON when ``--output json`` is set; we accept either.
    """
    text = stdout.strip()
    if not text:
        raise OvhCliError("ovhai job run produced empty stdout — no job id to parse")
    # JSON path: ``{"id": "...", ...}`` or ``[{"id": "..."}]``.
    if text.startswith("{") or text.startswith("["):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise OvhCliError(f"ovhai job run produced unparseable JSON: {text[:200]}") from exc
        record = data[0] if isinstance(data, list) else data
        if isinstance(record, dict) and "id" in record:
            return str(record["id"])
        raise OvhCliError(f"ovhai job run JSON missing 'id' field: {record}")
    # Plain-text path: first whitespace-separated token of the first line.
    return text.splitlines()[0].split()[0]


def _job_info_from_json(stdout: str) -> JobInfo:
    """Parse ``ovhai job get --output json`` stdout into a :class:`JobInfo`."""
    text = stdout.strip()
    if not text:
        raise OvhCliError("ovhai job get produced empty stdout")
    try:
        record = json.loads(text)
    except json.JSONDecodeError as exc:
        raise OvhCliError(f"ovhai job get produced unparseable JSON: {text[:200]}") from exc
    if isinstance(record, list):
        record = record[0]
    return _job_info_from_record(record)


def _job_info_from_record(record: dict[str, Any]) -> JobInfo:
    """Project an ovhai JSON record into a :class:`JobInfo` (best-effort)."""
    status = record.get("status") or {}
    spec = record.get("spec") or {}
    return JobInfo(
        id=str(record.get("id", "")),
        name=str(spec.get("name", "") or record.get("name", "")),
        state=str(status.get("state", "") or record.get("state", "UNKNOWN")),
        image=str(spec.get("image", "") or record.get("image", "")),
        gpu=str(spec.get("resources", {}).get("gpu", "") or record.get("gpu", "")),
        time=str(status.get("startedAt", "") or record.get("time", "")),
    )
