"""OvhClient — thin subprocess wrapper around the ``ovhai`` CLI binary.

Each method shells out to the CLI and parses its JSON / stdout response. All
calls go through a single ``_run`` helper for mock-friendliness; tests patch
``subprocess.run`` (or pass a stub ``runner`` callable to the constructor).

Job state strings tracked: ``QUEUED``, ``PENDING``, ``RUNNING``, ``DONE``,
``FAILED``, ``KILLED``, ``ERROR`` (terminal: DONE / FAILED / KILLED / ERROR).
The OvhRunner polls ``get(job_id).state`` until terminal.

This module also exposes bucket verbs (:meth:`OvhClient.bucket_list`,
:meth:`bucket_list_objects`, :meth:`bucket_object_exists`,
:meth:`bucket_get_object`) used by the F7.7 preflight probes. They reuse
ovhai's OAuth session, so the user never needs separate AWS-style S3 keys.
"""

import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence

from pydantic import BaseModel, Field

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


#: Pydantic config for *projections* of CLI JSON we don't fully own —
#: ignore extra keys (the ``ovhai`` CLI may grow new fields without
#: warning) but still honour ``alias=`` for keyword-clashing names like
#: ``bytes`` / ``hash``. Different from :data:`STRICT` which forbids extras.
_CLI_PROJECTION = {"populate_by_name": True}


class BucketInfo(BaseModel):
    """One bucket entry from ``ovhai bucket list <region> --output json``."""

    model_config = _CLI_PROJECTION

    name: str
    size_bytes: int = Field(default=0, alias="bytes")
    count: int = 0
    last_modified: str | None = None


class BucketObject(BaseModel):
    """One object entry from ``ovhai bucket object list <bucket>@<region>``.

    The CLI wraps each entry as ``{"object": {...}, "detail": {...}}`` —
    :func:`_bucket_object_from_record` flattens both halves into this model.
    Extra keys (``manifest`` / ``slo_etag`` / future CLI additions) are
    accepted-and-ignored so a CLI minor version bump can't break the probe.
    """

    model_config = _CLI_PROJECTION

    name: str
    size_bytes: int = Field(default=0, alias="bytes")
    last_modified: str | None = None
    hash_: str | None = Field(default=None, alias="hash")
    content_type: str | None = None


class OvhCliError(RuntimeError):
    """Raised when an ovhai subprocess returns non-zero or malformed output."""


# Default subprocess runner; tests can substitute a callable for full mocking.
def _default_runner(
    args: Sequence[str], timeout: int = 60
) -> subprocess.CompletedProcess:
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
        except (subprocess.TimeoutExpired, OSError):
            # OSError covers FileNotFoundError + PermissionError + the rest.
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
        """Run ``ovhai job run <args> --output json``; return the job ID.

        Forces ``--output json`` so the parser reads the structured response
        instead of the text-mode "Created\\n<uuid>" stream — the smoke run
        on 2026-05-09 surfaced a parse bug where "Created" was taken as the
        job id (the actual UUID was on the second line). JSON path is
        unambiguous and forward-compatible with future ovhai output tweaks.
        """
        res = self._run(["job", "run", *args, "--output", "json"])
        if res.returncode != 0:
            raise OvhCliError(
                f"ovhai job run failed (rc={res.returncode}): {res.stderr.strip()}"
            )
        return _parse_job_id(res.stdout)

    def get(self, job_id: str) -> JobInfo:
        """Run ``ovhai job get <id> --output json`` and project to :class:`JobInfo`."""
        res = self._run(["job", "get", job_id, "--output", "json"])
        if res.returncode != 0:
            raise OvhCliError(
                f"ovhai job get failed (rc={res.returncode}): {res.stderr.strip()}"
            )
        return _job_info_from_json(res.stdout)

    def list_jobs(self, state_filter: str | None = None) -> list[JobInfo]:
        """List jobs, optionally filtered by state."""
        args = ["job", "list", "--output", "json"]
        res = self._run(args)
        if res.returncode != 0:
            raise OvhCliError(
                f"ovhai job list failed (rc={res.returncode}): {res.stderr.strip()}"
            )
        records = json.loads(res.stdout) if res.stdout.strip() else []
        infos = [_job_info_from_record(r) for r in records]
        if state_filter is not None:
            infos = [j for j in infos if j.state.upper() == state_filter.upper()]
        return infos

    def logs(self, job_id: str, tail: int = 100) -> str:
        """Return the last ``tail`` lines of the job's combined stdout/stderr."""
        res = self._run(["job", "logs", job_id, "--tail", str(tail)])
        if res.returncode != 0:
            raise OvhCliError(
                f"ovhai job logs failed (rc={res.returncode}): {res.stderr.strip()}"
            )
        return res.stdout

    def stop(self, job_id: str) -> bool:
        """Stop a running job; returns True on rc=0."""
        res = self._run(["job", "stop", job_id])
        return res.returncode == 0

    # ── Bucket verbs (used by F7.7 preflight probes) ───────────────────
    #
    # These reuse the ovhai CLI's OAuth session — no separate AWS S3 keys
    # required. The frontend's preflight probes call them instead of going
    # to boto3 directly, so the credential surface stays unified at
    # ``ovhai login`` and the frontend layer never imports boto3.

    def bucket_list(self, region: str) -> list[BucketInfo]:
        """``ovhai bucket list <region> --output json`` → list of buckets.

        Empty stdout (no buckets in the region) returns an empty list.
        """
        res = self._run(["bucket", "list", region, "--output", "json"])
        if res.returncode != 0:
            raise OvhCliError(
                f"ovhai bucket list failed (rc={res.returncode}): {res.stderr.strip()}"
            )
        records = json.loads(res.stdout) if res.stdout.strip() else []
        return [BucketInfo.model_validate(r) for r in records]

    def bucket_list_objects(
        self,
        region: str,
        bucket: str,
        *,
        prefix: str | None = None,
        max_keys: int | None = None,
    ) -> list[BucketObject]:
        """``ovhai bucket object list <bucket>@<region> --output json`` →
        list of objects.

        ``prefix`` passes through as ``--prefix``. ``max_keys`` is applied
        client-side (the CLI doesn't expose a native cap yet) — useful for
        existence checks where pulling the whole listing would be wasteful.
        """
        args = [
            "bucket",
            "object",
            "list",
            f"{bucket}@{region}",
            "--output",
            "json",
        ]
        if prefix is not None:
            args.extend(["--prefix", prefix])
        res = self._run(args)
        if res.returncode != 0:
            raise OvhCliError(
                f"ovhai bucket object list failed (rc={res.returncode}): "
                f"{res.stderr.strip()}"
            )
        records = json.loads(res.stdout) if res.stdout.strip() else []
        objects = [_bucket_object_from_record(r) for r in records]
        if max_keys is not None:
            objects = objects[:max_keys]
        return objects

    def bucket_object_exists(self, region: str, bucket: str, key: str) -> bool:
        """Cheap exact-match existence check via ``bucket_list_objects(prefix=key)``.

        Listing-with-prefix returns every object whose key *starts with*
        ``key``; we filter to exact equality so ``"foo.yaml"`` doesn't
        accidentally match ``"foo.yaml.bak"``.
        """
        try:
            objs = self.bucket_list_objects(region, bucket, prefix=key)
        except OvhCliError:
            return False
        return any(o.name == key for o in objs)

    def bucket_get_object(self, region: str, bucket: str, key: str) -> bytes:
        """Download the bytes of a single object via ``ovhai bucket object download``.

        The CLI writes to a directory (``--output <dir>/``) preserving the
        full key path. We download into a tempdir, read the resulting file,
        and unlink — the caller gets bytes back as if it were a single
        ``GetObject`` call. Raises :class:`OvhCliError` on rc != 0 OR when
        the expected file doesn't materialise (defensive against silent CLI
        regressions).
        """
        with tempfile.TemporaryDirectory(prefix="ovh_get_") as tmpdir:
            # ovhai requires --output to end with "/" (it's an output prefix).
            output_dir = f"{tmpdir}/"
            res = self._run(
                [
                    "bucket",
                    "object",
                    "download",
                    f"{bucket}@{region}",
                    key,
                    "--output",
                    output_dir,
                ]
            )
            if res.returncode != 0:
                raise OvhCliError(
                    f"ovhai bucket object download failed "
                    f"(rc={res.returncode}): {res.stderr.strip()}"
                )
            local = Path(tmpdir) / key
            if not local.is_file():
                raise OvhCliError(
                    f"ovhai bucket object download succeeded but {local} "
                    "is missing — CLI behaviour may have changed."
                )
            return local.read_bytes()

    def bucket_put_object(
        self, region: str, bucket: str, key: str, body: bytes
    ) -> None:
        """Upload one in-memory blob to ``<bucket>/<key>`` via ``ovhai bucket object upload``.

        Mirrors :meth:`bucket_get_object` for the put direction. Used by
        :class:`CodeUploader` (F7.7.A5) so ``multi-scenario upload-code``
        works with only ``ovhai login`` — no AWS-style S3 credentials.

        Implementation: stages ``body`` to a tempfile shaped like ``<key>``
        (so ovhai picks the right object name), then ``--remove-prefix``
        strips the temp dir off the upload path. Per-call subprocess cost
        is ~100ms; ~10s for the full ~100-file upload-code corpus.
        """
        with tempfile.TemporaryDirectory(prefix="ovh_put_") as tmpdir:
            staged = Path(tmpdir) / key
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(body)
            # ``--remove-prefix`` peels off the tmpdir root (with trailing /)
            # so the uploaded object lands at <bucket>/<key>.
            res = self._run(
                [
                    "bucket",
                    "object",
                    "upload",
                    f"{bucket}@{region}",
                    str(staged),
                    "--remove-prefix",
                    f"{tmpdir}/",
                ]
            )
            if res.returncode != 0:
                raise OvhCliError(
                    f"ovhai bucket object upload {key} failed "
                    f"(rc={res.returncode}): {res.stderr.strip()}"
                )

    def _run(
        self, args: Sequence[str], timeout: int = 60
    ) -> subprocess.CompletedProcess:
        return self._runner([self._binary, *args], timeout=timeout)


#: UUID-ish pattern: 8-4-4-4-12 hex with dashes (matches ``ovhai`` job IDs).
_JOB_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


def _parse_job_id(stdout: str) -> str:
    """Extract the job ID from ``ovhai job run`` output.

    Two formats observed:

    - **JSON** (when ``--output json`` is set; the canonical path post-F7.7.A5):
      ``{"id": "<uuid>", ...}`` or ``[{"id": "<uuid>"}]``.
    - **Plain text** (legacy / fallback): the CLI prints ``Created\\n<uuid>\\n``.
      The first line is a status word ("Created"), the second is the UUID.
      The pre-F7.7.A5 parser took the first whitespace token of the first
      line and ended up returning ``"Created"`` (smoke run 2026-05-09
      caught this). Now we scan all lines for a UUID-shaped token.
    """
    text = stdout.strip()
    if not text:
        raise OvhCliError("ovhai job run produced empty stdout — no job id to parse")
    # JSON path: ``{"id": "...", ...}`` or ``[{"id": "..."}]``.
    if text.startswith("{") or text.startswith("["):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise OvhCliError(
                f"ovhai job run produced unparseable JSON: {text[:200]}"
            ) from exc
        record = data[0] if isinstance(data, list) else data
        if isinstance(record, dict) and "id" in record:
            return str(record["id"])
        raise OvhCliError(f"ovhai job run JSON missing 'id' field: {record}")
    # Plain-text path: scan every line for a UUID-shaped token. Survives
    # leading status words ("Created") and trailing blank lines.
    for line in text.splitlines():
        for token in line.split():
            if _JOB_ID_RE.match(token):
                return token
    raise OvhCliError(
        f"ovhai job run produced text without a UUID-shaped job id: {text[:200]}"
    )


def _job_info_from_json(stdout: str) -> JobInfo:
    """Parse ``ovhai job get --output json`` stdout into a :class:`JobInfo`."""
    text = stdout.strip()
    if not text:
        raise OvhCliError("ovhai job get produced empty stdout")
    try:
        record = json.loads(text)
    except json.JSONDecodeError as exc:
        raise OvhCliError(
            f"ovhai job get produced unparseable JSON: {text[:200]}"
        ) from exc
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


def _bucket_object_from_record(record: dict[str, Any]) -> BucketObject:
    """Flatten an ``ovhai bucket object list`` JSON record.

    Records arrive as ``{"object": {name, bytes, hash, last_modified, …},
    "detail": {container, object_type}}``. We project the inner ``object``
    half into :class:`BucketObject` — ``detail`` carries no fields the
    preflight cares about today.
    """
    obj = record.get("object") if isinstance(record, dict) else None
    if not isinstance(obj, dict):
        # Some CLI versions / environments may emit flat records — accept both.
        obj = record if isinstance(record, dict) else {}
    return BucketObject.model_validate(obj)
