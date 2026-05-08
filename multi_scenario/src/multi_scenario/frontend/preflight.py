"""Preflight check abstraction for the Submit page.

A *preflight check* is a single named verification that runs before a job is
submitted — does the storage path exist, is the OVH CLI on PATH, does local
code match what's uploaded to the OVH bucket, etc. Each check has a
:class:`CheckStatus` (idle / checking / pass / fail) which the UI renders
as an LED dot, plus a short ``detail`` string surfaced once the check ran.

Phase B replaces the mocked outcomes with **real probes** for the local
runner (config schema, storage writable). OVH-only checks remain mocked
until Phase C wires the boto3/ovhai-CLI probes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import ValidationError

from multi_scenario.domain.models import ExperimentConfig

#: Section header keys (preserve insertion order for the renderer).
#:
#: The Submit page's preflight panel rolls every probe up into one of three
#: top-level LEDs: Configuration / System / Storage. Each LED's status is
#: ``category_status(rows)`` (🟢 if all pass, 🔴 if any fail, ⚪ otherwise) and
#: failed rows expand inline with the underlying probe's root-cause detail.
CATEGORY_ORDER: tuple[str, ...] = ("config", "system", "storage")
CATEGORY_LABELS: dict[str, str] = {
    "config": "Configuration",
    "system": "System",
    "storage": "Storage",
}


class CheckStatus(Enum):
    """LED states for a preflight check row."""

    IDLE = "idle"  # ⚪ not yet run
    CHECKING = "checking"  # 🟡 in progress
    PASS = "pass"  # 🟢 OK
    FAIL = "fail"  # 🔴 error


_LED_GLYPH = {
    CheckStatus.IDLE: "⚪",
    CheckStatus.CHECKING: "🟡",
    CheckStatus.PASS: "🟢",
    CheckStatus.FAIL: "🔴",
}


def led(status: CheckStatus) -> str:
    """Emoji glyph for a status (used by the UI; here so tests don't import st)."""
    return _LED_GLYPH[status]


@dataclass
class PreflightCheck:
    """One row in the preflight panel."""

    name: str
    runners: tuple[Literal["local", "ovh"], ...]  # which runner choices show this row
    description: str  # what we're verifying (shown as tooltip / placeholder)
    status: CheckStatus = CheckStatus.IDLE
    detail: str = ""  # populated after the check runs
    category: str = "local_env"  # grouping key for the renderer (CATEGORY_ORDER)
    # Callable form_dict → bool — extra precondition on top of ``runners``.
    # Used for runtime-conditional rows (e.g. GPU probe only when device=cuda).
    condition: Callable[[dict[str, Any]], bool] = field(
        default=lambda _f: True, repr=False
    )


# Canonical list of checks. Additional rows appear automatically based on
# the user's runner choice — local-only checks are dropped on OVH and v.v.
def _device_is_cuda(form: dict[str, Any]) -> bool:
    """Precondition for GPU probe: only check when cfg requests cuda."""
    return form.get("training", {}).get("device") == "cuda"


def default_checks() -> list[PreflightCheck]:
    """Fresh list of all checks in IDLE state.

    Categorisation is intentional — each row maps to exactly one of the
    three Submit-page LEDs (config / system / storage). When restructuring
    keep the mapping aligned with the page header labels so a failed LED
    correctly points the user at where the root cause sits.
    """
    return [
        # ── Configuration ────────────────────────────────────────────
        PreflightCheck(
            name="Config schema valid",
            runners=("local", "ovh"),
            description="ExperimentConfig.model_validate(form_dict) succeeds.",
            category="config",
        ),
        PreflightCheck(
            name="OVH config valid",
            runners=("ovh",),
            description=(
                "configs/ovh.yaml parses into a valid OvhJobConfig. "
                "When this fails, every other OVH row stays idle — fix this "
                "first."
            ),
            category="config",
        ),
        # ── System ───────────────────────────────────────────────────
        PreflightCheck(
            name="Required deps importable",
            runners=("local",),
            description=(
                "torch / vmas / benchmarl / imageio import in the active "
                "venv. Missing deps would crash training ~30s in. (OVH "
                "submit doesn't need these locally — the container provides them.)"
            ),
            category="system",
        ),
        PreflightCheck(
            name="GPU available",
            runners=("local",),
            description="torch.cuda.is_available() is True (only checked when device=cuda).",
            category="system",
            condition=_device_is_cuda,
        ),
        PreflightCheck(
            name="OVH CLI installed",
            runners=("ovh",),
            description="`ovhai --version` returns successfully.",
            category="system",
        ),
        PreflightCheck(
            name="No active OVH job with this run_id",
            runners=("ovh",),
            description=(
                "No currently-RUNNING ovhai job already targets "
                "<exp_id>_s<seed> (would FINALIZING-clobber each other)."
            ),
            category="system",
        ),
        PreflightCheck(
            name="Cost cap not exceeded",
            runners=("ovh",),
            description="OvhJobConfig.estimate_cost_eur(...) < cost_cap_eur.",
            category="system",
        ),
        # ── Storage ──────────────────────────────────────────────────
        PreflightCheck(
            name="Storage path writable",
            runners=("local",),
            description="A temp file can be created under runtime.storage.path.",
            category="storage",
        ),
        PreflightCheck(
            name="Run dir does not collide",
            runners=("local",),
            description=(
                "<storage>/<run_id>__<timestamp> is fresh — defensive guard "
                "against two same-second submissions sharing a folder."
            ),
            category="storage",
        ),
        PreflightCheck(
            name="Results bucket reachable",
            runners=("ovh",),
            description="boto3.head_bucket against the OVH results bucket.",
            category="storage",
        ),
        PreflightCheck(
            name="Code matches OVH bucket",
            runners=("ovh",),
            description=(
                "Local source hash matches the .code_hash uploaded by the "
                "last `multi-scenario upload-code` run. Stale code in the "
                "bucket is the most common cloud-run pitfall."
            ),
            category="storage",
        ),
        PreflightCheck(
            name="Submitted YAML present in bucket",
            runners=("ovh",),
            description=(
                "The YAML the OVH container will execute is actually in "
                "the code bucket — catches the case where the user saved "
                "a new variant but hasn't re-run upload-code."
            ),
            category="storage",
        ),
        PreflightCheck(
            name="Per-run prefix not occupied",
            runners=("ovh",),
            description=(
                "<results-bucket>/<exp_id>_s<seed>/ is empty. Reusing a "
                "prefix risks parallel-job FINALIZING overwriting earlier "
                "runs (the rendezvous_comm 2026-04-16 lesson)."
            ),
            category="storage",
        ),
    ]


def applicable_checks(
    runner: str, form: dict[str, Any] | None = None
) -> list[PreflightCheck]:
    """Subset of :func:`default_checks` for the chosen runner + form state.

    Filters by both the static ``runners`` tuple and the ``condition``
    predicate (so the GPU row only appears when ``device == "cuda"``, etc.).
    ``form=None`` is treated as an empty form, which naturally excludes
    rows that need a positive form-state to apply (e.g. GPU).
    """
    form = form or {}
    out = []
    for c in default_checks():
        if runner not in c.runners:
            continue
        if not c.condition(form):
            continue
        out.append(c)
    return out


def group_by_category(checks: list[PreflightCheck]) -> list[tuple[str, list[PreflightCheck]]]:
    """Group ``checks`` by ``category``, preserving :data:`CATEGORY_ORDER`.

    Returns a list of ``(category_label, [checks...])`` tuples — the page
    iterates this to render the section headers and rows.
    """
    by_cat: dict[str, list[PreflightCheck]] = {}
    for c in checks:
        by_cat.setdefault(c.category, []).append(c)
    out = []
    for key in CATEGORY_ORDER:
        if key in by_cat:
            out.append((CATEGORY_LABELS[key], by_cat[key]))
    # Catch any stray categories not in CATEGORY_ORDER (forward-compat).
    for key, rows in by_cat.items():
        if key not in CATEGORY_ORDER:
            out.append((CATEGORY_LABELS.get(key, key), rows))
    return out


def category_status(rows: list[PreflightCheck]) -> CheckStatus:
    """Roll up a section's rows into a single LED for the section header."""
    if not rows:
        return CheckStatus.IDLE
    if any(c.status == CheckStatus.FAIL for c in rows):
        return CheckStatus.FAIL
    if all(c.status == CheckStatus.PASS for c in rows):
        return CheckStatus.PASS
    if any(c.status == CheckStatus.CHECKING for c in rows):
        return CheckStatus.CHECKING
    return CheckStatus.IDLE


# ── Mock outcomes for Phase A ────────────────────────────────────────


@dataclass
class MockOutcome:
    """A scripted result the page applies to checks for UI preview."""

    status: CheckStatus
    detail_template: str = ""  # ``"{name}"`` placeholder substituted at apply time


# Three mockup scenarios the user can flip through in Phase A.
_PASS = CheckStatus.PASS
_FAIL = CheckStatus.FAIL

MOCK_OUTCOMES: dict[str, dict[str, MockOutcome]] = {
    "all_ok": {
        "Config schema valid": MockOutcome(_PASS, "OK"),
        "OVH config valid": MockOutcome(_PASS, "configs/ovh.yaml parsed"),
        "Required deps importable": MockOutcome(_PASS, "torch · vmas · benchmarl · imageio"),
        "Storage path writable": MockOutcome(_PASS, "writable"),
        "Run dir does not collide": MockOutcome(_PASS, "demo_s0__YYYYMMDD_HHMM is fresh"),
        "GPU available": MockOutcome(_PASS, "1 GPU(s) visible"),
        "OVH CLI installed": MockOutcome(_PASS, "ovhai 3.35.0"),
        "Results bucket reachable": MockOutcome(_PASS, "ms-results bucket reachable"),
        "Code matches OVH bucket": MockOutcome(
            _PASS, "local sha a1b2c3 = remote sha a1b2c3 (uploaded 2h ago)"
        ),
        "Per-run prefix not occupied": MockOutcome(_PASS, "ms-results/<run_id>/ clear"),
        "Submitted YAML present in bucket": MockOutcome(
            _PASS, "ms-code/experiments/.../mappo_smoke.yaml present"
        ),
        "No active OVH job with this run_id": MockOutcome(
            _PASS, "no active job collides"
        ),
        "Cost cap not exceeded": MockOutcome(_PASS, "€0.42 < €5.00 cap"),
    },
    "code_drift": {
        "Config schema valid": MockOutcome(_PASS, "OK"),
        "Storage path writable": MockOutcome(_PASS, "writable"),
        "OVH CLI installed": MockOutcome(_PASS, "ovhai 3.35.0"),
        "Results bucket reachable": MockOutcome(_PASS, "ms-results reachable"),
        "Code matches OVH bucket": MockOutcome(
            _FAIL,
            "local sha a1b2c3 ≠ remote sha 9d8e7f (uploaded 12d ago) — "
            "run `multi-scenario upload-code` first",
        ),
        "Per-run prefix not occupied": MockOutcome(_PASS, "ms-results/<run_id>/ clear"),
        "Submitted YAML present in bucket": MockOutcome(
            _PASS, "ms-code/experiments/.../mappo_smoke.yaml present"
        ),
        "No active OVH job with this run_id": MockOutcome(
            _PASS, "no active job collides"
        ),
        "Cost cap not exceeded": MockOutcome(_PASS, "€0.42 < €5.00 cap"),
    },
    "cloud_unreachable": {
        "Config schema valid": MockOutcome(_PASS, "OK"),
        "Storage path writable": MockOutcome(_PASS, "writable"),
        "OVH CLI installed": MockOutcome(_FAIL, "command 'ovhai' not found on PATH"),
        "Results bucket reachable": MockOutcome(_FAIL, "boto3.head_bucket → 403 Forbidden"),
        "Code matches OVH bucket": MockOutcome(_FAIL, "couldn't query bucket"),
        "Per-run prefix not occupied": MockOutcome(_FAIL, "couldn't list bucket"),
        "Submitted YAML present in bucket": MockOutcome(_FAIL, "couldn't head_object"),
        "No active OVH job with this run_id": MockOutcome(
            _FAIL, "couldn't query running jobs"
        ),
        "Cost cap not exceeded": MockOutcome(_PASS, "€0.42 < €5.00 cap"),
    },
}


def apply_mock(checks: list[PreflightCheck], scenario: str) -> list[PreflightCheck]:
    """Update ``checks`` in-place with the named mock outcome (Phase A fallback)."""
    outcomes = MOCK_OUTCOMES.get(scenario, MOCK_OUTCOMES["all_ok"])
    for check in checks:
        m = outcomes.get(check.name)
        if m is None:
            continue
        check.status = m.status
        check.detail = m.detail_template
    return checks


# ── Real probes (Phase B for local; Phase C will add OVH ones) ──────


def _probe_config_schema(form_dict: dict[str, Any]) -> tuple[CheckStatus, str]:
    """Run :meth:`ExperimentConfig.model_validate` against the form output."""
    try:
        ExperimentConfig.model_validate(form_dict)
    except ValidationError as exc:
        first = exc.errors()[0]
        path = ".".join(str(p) for p in first["loc"])
        return CheckStatus.FAIL, f"{path}: {first['msg']}"
    return CheckStatus.PASS, "Pydantic schema OK"


def _probe_storage_writable(form_dict: dict[str, Any]) -> tuple[CheckStatus, str]:
    """Touch a temp file under ``runtime.storage.path`` and clean up.

    Validates that the path's parent exists and the user has write access.
    For S3 storage this is N/A (would land in the OVH suite of checks).
    """
    storage = form_dict.get("runtime", {}).get("storage", {})
    if storage.get("type") != "fs":
        return CheckStatus.PASS, f"backend={storage.get('type')} (filesystem probe N/A)"
    raw_path = storage.get("path")
    if not raw_path:
        return CheckStatus.FAIL, "runtime.storage.path is empty"
    path = Path(raw_path).expanduser().resolve()
    parent = path if path.exists() else path.parent
    if not parent.exists():
        return CheckStatus.FAIL, f"{parent} does not exist"
    if not os.access(parent, os.W_OK):
        return CheckStatus.FAIL, f"{parent} is not writable"
    # Actual touch — proves write works.
    parent.mkdir(parents=True, exist_ok=True)
    probe = parent / ".ms_preflight_probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return CheckStatus.FAIL, f"write probe failed: {exc}"
    return CheckStatus.PASS, f"{parent} writable"


def _probe_required_deps(_form: dict[str, Any]) -> tuple[CheckStatus, str]:
    """Verify torch / vmas / benchmarl / imageio import in the active venv."""
    # pylint: disable=import-outside-toplevel
    import importlib

    required = ("torch", "vmas", "benchmarl", "imageio")
    versions = []
    for mod_name in required:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError as exc:
            return CheckStatus.FAIL, f"missing dep: {mod_name} ({exc})"
        ver = getattr(mod, "__version__", "?")
        versions.append(f"{mod_name} {ver}")
    return CheckStatus.PASS, " · ".join(versions)


def _probe_gpu_available(_form: dict[str, Any]) -> tuple[CheckStatus, str]:
    """Verify ``torch.cuda.is_available()`` matches the cfg's device choice."""
    # pylint: disable=import-outside-toplevel
    try:
        import torch
    except ImportError:
        return CheckStatus.FAIL, "torch not importable (see Required deps)"
    if not torch.cuda.is_available():
        return (
            CheckStatus.FAIL,
            "torch.cuda.is_available() = False — set device=cpu or fix CUDA install",
        )
    return CheckStatus.PASS, f"{torch.cuda.device_count()} GPU(s) visible"


def _probe_run_dir_collision(form: dict[str, Any]) -> tuple[CheckStatus, str]:
    """Verify the computed ``<storage>/<exp_id>_s<seed>__<HHMM>`` is fresh.

    Reuses :class:`RunId` for folder-name parity with the actual runner so
    we're checking the SAME path the runner will use seconds later.
    """
    # pylint: disable=import-outside-toplevel
    from datetime import datetime, timezone

    from multi_scenario.domain.models import RunId

    storage = form.get("runtime", {}).get("storage", {})
    if storage.get("type") != "fs":
        return CheckStatus.PASS, f"backend={storage.get('type')} (collision N/A)"
    raw_path = storage.get("path")
    if not raw_path:
        return CheckStatus.FAIL, "runtime.storage.path is empty"
    exp = form.get("experiment", {})
    try:
        seed = int(exp.get("seed", 0))
    except (TypeError, ValueError):
        # If the seed isn't an int the schema check will already FAIL with a
        # clear message — the collision probe just bails to PASS so we don't
        # double-flag the same root cause.
        return CheckStatus.PASS, "seed is not an int — see Config schema row"
    run_id = RunId(exp_id=exp.get("id", "demo"), seed=seed)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    candidate = Path(raw_path).expanduser() / run_id.folder_name(timestamp)
    if candidate.exists():
        return CheckStatus.FAIL, f"{candidate.name} already exists under {raw_path}"
    return CheckStatus.PASS, f"{candidate.name} is fresh"


def run_real_local_checks(
    checks: list[PreflightCheck], form_dict: dict[str, Any]
) -> list[PreflightCheck]:
    """Phase B: run the real local probes; OVH-only rows stay IDLE / mocked.

    Mutates and returns ``checks`` so the page can re-render the same list
    with updated statuses + details.
    """
    real_probes = {
        "Config schema valid": _probe_config_schema,
        "Storage path writable": _probe_storage_writable,
        "Required deps importable": _probe_required_deps,
        "GPU available": _probe_gpu_available,
        "Run dir does not collide": _probe_run_dir_collision,
    }
    for check in checks:
        probe = real_probes.get(check.name)
        if probe is None:
            # OVH-only rows — leave for Phase C; mark idle so the user sees
            # they're not yet checked instead of confusingly green.
            continue
        status, detail = probe(form_dict)
        check.status = status
        check.detail = detail
    return checks


# ── Real OVH probes (Phase C) ───────────────────────────────────────


def _probe_ovh_cli() -> tuple[CheckStatus, str]:
    """Run :meth:`OvhClient.ensure_available` (which shells out to ``ovhai --version``)."""
    # pylint: disable=import-outside-toplevel
    from multi_scenario.adapters.runners.ovh_cli import OvhClient, OvhCliError

    try:
        OvhClient().ensure_available()
    except OvhCliError as exc:
        return CheckStatus.FAIL, str(exc)
    return CheckStatus.PASS, "ovhai CLI on PATH"


def _probe_results_bucket(ovh_cfg: Any) -> tuple[CheckStatus, str]:
    """``boto3.head_bucket`` against the configured results bucket."""
    # pylint: disable=import-outside-toplevel
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    if ovh_cfg is None:
        return CheckStatus.FAIL, "no OVH config loaded"
    bucket = ovh_cfg.bucket_results
    region = ovh_cfg.region
    try:
        client = boto3.client("s3", region_name=region)
        client.head_bucket(Bucket=bucket)
    except (BotoCoreError, ClientError) as exc:
        return CheckStatus.FAIL, f"head_bucket {bucket}: {exc}"
    return CheckStatus.PASS, f"{bucket}@{region} reachable"


# 17 locals here is intrinsic — boto3, hash compute, blob fetch, age format,
# remote compare. Splitting helpers would just spread the linear flow across
# four functions without simplifying anything.
# pylint: disable-next=too-many-locals
def _probe_code_hash(ovh_cfg: Any, repo_root: Path) -> tuple[CheckStatus, str]:
    """Compare local source hash to the ``.code_hash`` blob in the OVH code bucket.

    On success, the detail string includes a "(uploaded N ago)" suffix derived
    from the blob's ``LastModified`` so the user can spot a stale upload even
    when the hash matches (e.g. they edited code, uploaded, then reverted
    locally — hashes match again but the *bucket version* may still be old).
    """
    # pylint: disable=import-outside-toplevel
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    from multi_scenario.adapters.storage.code_uploader import (
        CODE_HASH_KEY,
        compute_local_code_hash,
    )

    if ovh_cfg is None:
        return CheckStatus.FAIL, "no OVH config loaded"
    bucket = ovh_cfg.bucket_code
    try:
        local = compute_local_code_hash(repo_root)
    except OSError as exc:
        return CheckStatus.FAIL, f"local hash failed: {exc}"
    key = CODE_HASH_KEY  # bucket is shared between configs; prefix not used here
    try:
        client = boto3.client("s3", region_name=ovh_cfg.region)
        obj = client.get_object(Bucket=bucket, Key=key)
        body = obj["Body"].read().decode("utf-8")
        last_modified = obj.get("LastModified")
    except (BotoCoreError, ClientError) as exc:
        return (
            CheckStatus.FAIL,
            f"no .code_hash in {bucket}: {exc} — run `multi-scenario upload-code`",
        )
    remote = body.strip()
    age_suffix = f" (uploaded {_format_age(last_modified)})" if last_modified else ""
    if local != remote:
        return (
            CheckStatus.FAIL,
            f"local {local[:18]}… ≠ remote {remote[:18]}…{age_suffix} "
            "— run `multi-scenario upload-code`",
        )
    return CheckStatus.PASS, f"local hash matches {bucket}/{key}{age_suffix}"


def _format_age(last_modified: Any) -> str:
    """Render a ``LastModified`` datetime as "5m ago" / "3h ago" / "2d ago"."""
    # pylint: disable=import-outside-toplevel
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    delta = now - last_modified
    secs = int(delta.total_seconds())
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"


def _probe_yaml_in_bucket(
    ovh_cfg: Any, yaml_relpath: str | None
) -> tuple[CheckStatus, str]:
    """Verify the to-be-submitted YAML is actually present in the code bucket.

    The OVH job's ``bash -c …`` invokes ``python -m multi_scenario.cli run
    <yaml_path_in_repo>`` against a path *inside* the code bucket — if the
    user picked / saved a YAML that hasn't been ``upload-code``'d yet, the
    container will fail with FileNotFoundError. Catching it here saves a
    minutes-long round-trip.
    """
    # pylint: disable=import-outside-toplevel
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    if ovh_cfg is None:
        return CheckStatus.FAIL, "no OVH config loaded"
    if not yaml_relpath:
        return CheckStatus.FAIL, "no YAML path supplied (active config unset)"
    bucket = ovh_cfg.bucket_code
    try:
        client = boto3.client("s3", region_name=ovh_cfg.region)
        client.head_object(Bucket=bucket, Key=yaml_relpath)
    except (BotoCoreError, ClientError):
        return (
            CheckStatus.FAIL,
            f"{bucket}/{yaml_relpath} not found — `multi-scenario upload-code` "
            "needs to run after you save a new config",
        )
    return CheckStatus.PASS, f"{bucket}/{yaml_relpath} present"


def _probe_no_active_collision(
    exp_id: str, seed: int, *, client: Any = None
) -> tuple[CheckStatus, str]:
    """No currently-RUNNING ``ovhai`` job already targets the same ``run_id``.

    The job's name is the ``run_id`` (``<exp_id>_s<seed>``) per ``OvhRunner``.
    Same-name parallel jobs would land in the same S3 prefix and clobber
    each other's results during FINALIZING (the rendezvous_comm 2026-04-16
    lesson). Better to refuse the second submit than to corrupt the first.
    """
    # pylint: disable=import-outside-toplevel
    from multi_scenario.adapters.runners.ovh_cli import OvhClient, OvhCliError

    run_id = f"{exp_id}_s{seed}"
    cli = client or OvhClient()
    try:
        running = cli.list_jobs(state_filter="RUNNING")
    except OvhCliError as exc:
        return CheckStatus.FAIL, f"could not query running jobs: {exc}"
    matches = [j for j in running if run_id in (j.name, j.id)]
    if matches:
        ids = ", ".join(j.id for j in matches)
        return (
            CheckStatus.FAIL,
            f"{len(matches)} RUNNING job(s) already target {run_id}: {ids}",
        )
    return CheckStatus.PASS, f"no active job collides with {run_id}"


def _probe_prefix_collision(
    ovh_cfg: Any, exp_id: str, seed: int
) -> tuple[CheckStatus, str]:
    """List ``<bucket>/<prefix>/<run_id>/`` keys; must be empty (rendezvous lesson)."""
    # pylint: disable=import-outside-toplevel
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    if ovh_cfg is None:
        return CheckStatus.FAIL, "no OVH config loaded"
    run_id = f"{exp_id}_s{seed}"
    bucket = ovh_cfg.bucket_results
    prefix = f"{run_id}/"
    try:
        client = boto3.client("s3", region_name=ovh_cfg.region)
        resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    except (BotoCoreError, ClientError) as exc:
        return CheckStatus.FAIL, f"list_objects_v2 failed: {exc}"
    if resp.get("KeyCount", 0) > 0:
        return (
            CheckStatus.FAIL,
            f"{bucket}/{prefix} already has data — pick a different exp_id/seed",
        )
    return CheckStatus.PASS, f"{bucket}/{prefix} clear"


def _probe_cost_cap(
    ovh_cfg: Any, seconds_per_run: float | None = None
) -> tuple[CheckStatus, str]:
    """``OvhJobConfig.estimate_cost_eur(...)`` < ``cost_cap_eur``.

    Without a wall-clock estimate we can only sanity-check that ``cost_cap_eur``
    is set; an actual cost projection requires ``seconds_per_run`` from the user
    (matches the ``--seconds-per-run`` CLI flag in F6.7).
    """
    if ovh_cfg is None:
        return CheckStatus.FAIL, "no OVH config loaded"
    if seconds_per_run is None or seconds_per_run <= 0:
        return (
            CheckStatus.PASS,
            f"cost_cap_eur={ovh_cfg.cost_cap_eur:.2f} (no seconds_per_run estimate provided)",
        )
    estimate = ovh_cfg.estimate_cost_eur(seconds_per_run / 3600.0)
    if estimate > ovh_cfg.cost_cap_eur:
        return (
            CheckStatus.FAIL,
            f"€{estimate:.2f} > cap €{ovh_cfg.cost_cap_eur:.2f}",
        )
    return CheckStatus.PASS, f"€{estimate:.2f} < cap €{ovh_cfg.cost_cap_eur:.2f}"


# Many kwargs by design — each carries one external dependency the caller
# can swap (config, repo, YAML, runtime, ovhai client). Bundling into a
# config object would just shift the boilerplate to the caller. The local
# count is also high because we maintain a flat dispatch table inside.
# pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-locals
def run_real_ovh_checks(
    checks: list[PreflightCheck],
    form_dict: dict[str, Any],
    *,
    ovh_cfg: Any,
    repo_root: Path,
    yaml_relpath: str | None = None,
    seconds_per_run: float | None = None,
    ovh_client: Any = None,
) -> list[PreflightCheck]:
    """Phase C: run the local + OVH probes against ``ovh_cfg`` + ``repo_root``.

    Args:
        ovh_client: optional pre-built ``OvhClient`` (for tests / DI). When
            ``None`` a fresh ``OvhClient()`` is created per probe that needs it.
        yaml_relpath: path of the to-be-submitted YAML, relative to ``repo_root``
            (matches how it sits in the code bucket). Only used by the
            "Submitted YAML present in bucket" probe.
    """
    exp_id = form_dict.get("experiment", {}).get("id", "")
    seed = int(form_dict.get("experiment", {}).get("seed", 0))

    # If the OVH config didn't load, "OVH config valid" fails and every
    # downstream OVH probe stays IDLE — the user can't fix anything else
    # until the YAML parses. Local-environment rows still run normally.
    config_blocked = ovh_cfg is None
    skip_ovh_keys = {
        "OVH CLI installed",
        "Results bucket reachable",
        "Code matches OVH bucket",
        "Per-run prefix not occupied",
        "Submitted YAML present in bucket",
        "No active OVH job with this run_id",
        "Cost cap not exceeded",
    }

    real_probes: dict[str, Callable[[dict[str, Any]], tuple[CheckStatus, str]]] = {
        "Config schema valid": lambda _f: _probe_config_schema(form_dict),
        "OVH config valid": lambda _f: (
            (CheckStatus.FAIL, "configs/ovh.yaml missing or invalid")
            if config_blocked
            else (CheckStatus.PASS, "configs/ovh.yaml parsed")
        ),
        "OVH CLI installed": lambda _f: _probe_ovh_cli(),
        "Results bucket reachable": lambda _f: _probe_results_bucket(ovh_cfg),
        "Code matches OVH bucket": lambda _f: _probe_code_hash(ovh_cfg, repo_root),
        "Per-run prefix not occupied": lambda _f: _probe_prefix_collision(
            ovh_cfg, exp_id, seed
        ),
        "Submitted YAML present in bucket": lambda _f: _probe_yaml_in_bucket(
            ovh_cfg, yaml_relpath
        ),
        "No active OVH job with this run_id": lambda _f: _probe_no_active_collision(
            exp_id, seed, client=ovh_client
        ),
        "Cost cap not exceeded": lambda _f: _probe_cost_cap(ovh_cfg, seconds_per_run),
    }
    for check in checks:
        if config_blocked and check.name in skip_ovh_keys:
            # Cascade: keep these IDLE so the user sees "fix OVH config first".
            check.status = CheckStatus.IDLE
            check.detail = "blocked — fix the OVH config row first"
            continue
        probe = real_probes.get(check.name)
        if probe is None:
            continue
        status, detail = probe(form_dict)
        check.status = status
        check.detail = detail
    return checks
