"""Preflight check abstraction for the Submit page.

A *preflight check* is a single named verification that runs before a job is
submitted — does the storage path exist, is the OVH CLI on PATH, does local
code match what's uploaded to the OVH bucket, etc. Each check has a
:class:`CheckStatus` (idle / checking / pass / fail) which the UI renders
as an LED dot, plus a short ``detail`` string surfaced once the check ran.

Each row carries a ``category`` (config / system / storage) — the Submit
page rolls them up into three top-level LED cards. ``run_real_local_checks``
and ``run_real_ovh_checks`` dispatch the active rows to real probes (Pydantic
schema validation, local filesystem checks, and — for the OVH probes —
``ovhai`` CLI bucket verbs via :class:`OvhClient`). The frontend layer
deliberately never imports ``boto3``: every OVH-side operation goes through
the existing :class:`OvhClient` adapter so credentials stay unified at
``ovhai login`` and the hex-architecture port boundary stays clean.
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
        # NOTE: the standalone "GPU available" check (was: torch.cuda.is_available()
        # for local + cuda only) has been folded into the unified "Runner
        # provisioning consistent with device" probe below.
        PreflightCheck(
            name="OVH CLI installed",
            runners=("ovh",),
            description="`ovhai --version` returns successfully.",
            category="system",
        ),
        PreflightCheck(
            name="Runner provisioning consistent with device",
            runners=("local", "ovh"),
            description=(
                "Per-runner provisioning probe (F7.7.A4 architecture): "
                "local+cuda needs torch.cuda available on host; ovh+cuda "
                "needs a GPU flavor in configs/ovh.yaml; ovh+cpu warns "
                "(billed for GPU but unused). Adding new runners = one "
                "entry in application.runner_provisioning.PROVISION_CHECKS."
            ),
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
        # F9.8 — LERO-only: LLM API key present in env. Shown when the
        # YAML carries an ``lero:`` section (loaded YAML's form dict has
        # a non-empty ``lero`` key). Without this, the OVH container's
        # LiteLlmClient raises 401 mid-iter after ~5 min of boot+pull —
        # very expensive to discover. We catch it at preflight instead.
        PreflightCheck(
            name="LLM API key present for cfg.lero",
            runners=("local", "ovh"),
            description=(
                "When the YAML has an ``lero:`` block, the corresponding "
                "``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` / ``OVH_API_KEY`` "
                "(by ``cfg.llm.model`` prefix) must be in the env or ``.env`` "
                "so the LERO loop can place LLM calls. Missing → block submit."
            ),
            category="system",
            condition=lambda f: bool(f.get("lero")),
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
            description=(
                "`ovhai bucket list <region>` reports the configured results "
                "bucket. Reuses ovhai's OAuth — no separate AWS S3 keys needed."
            ),
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


def group_by_category(
    checks: list[PreflightCheck],
) -> list[tuple[str, list[PreflightCheck]]]:
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


# ── Real probes ──────────────────────────────────────────────────────


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
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    candidate = Path(raw_path).expanduser() / run_id.folder_name(timestamp)
    if candidate.exists():
        return CheckStatus.FAIL, f"{candidate.name} already exists under {raw_path}"
    return CheckStatus.PASS, f"{candidate.name} is fresh"


def _probe_lero_api_key(form_dict: dict[str, Any]) -> tuple[CheckStatus, str]:
    """F9.8: bridge :func:`lero_preflight.check_lero_api_key` to the probe shape.

    No-op (PASS) when the YAML has no ``lero:`` block — the row's
    ``condition`` would also exclude it via :func:`applicable_checks`,
    but the probe stays defensive so a caller that passes raw checks
    around can't trip a false-FAIL.
    """
    # pylint: disable=import-outside-toplevel
    from multi_scenario.domain.models import ExperimentConfig
    from multi_scenario.frontend.lero_preflight import check_lero_api_key

    if not form_dict.get("lero"):
        return CheckStatus.PASS, "no lero: section — check skipped"
    try:
        cfg = ExperimentConfig.model_validate(form_dict)
    except Exception as exc:  # pylint: disable=broad-except
        # Schema failure surfaces on its own row ("Config schema valid").
        # We can't run our key check without a parsed cfg — stay IDLE.
        return CheckStatus.IDLE, f"waiting on schema: {exc!s}"[:200]
    result = check_lero_api_key(cfg)
    if result.ok:
        return CheckStatus.PASS, result.detail
    return CheckStatus.FAIL, result.detail


def run_real_local_checks(
    checks: list[PreflightCheck], form_dict: dict[str, Any]
) -> list[PreflightCheck]:
    """Run the local-target probes; rows the dispatch table doesn't know
    are left untouched.

    Mutates and returns ``checks`` so the page can re-render the same list
    with updated statuses + details. OVH-only rows are not in the dispatch
    table — they stay in their incoming state (caller decides cascade).
    """
    real_probes = {
        "Config schema valid": _probe_config_schema,
        "Storage path writable": _probe_storage_writable,
        "Required deps importable": _probe_required_deps,
        "Run dir does not collide": _probe_run_dir_collision,
        "Runner provisioning consistent with device": (
            lambda _f: _provision_for_runner("local", form_dict)
        ),
        "LLM API key present for cfg.lero": _probe_lero_api_key,
    }
    for check in checks:
        probe = real_probes.get(check.name)
        if probe is None:
            continue
        status, detail = probe(form_dict)
        check.status = status
        check.detail = detail
    return checks


# ── Real OVH probes ─────────────────────────────────────────────────


def _probe_ovh_cli() -> tuple[CheckStatus, str]:
    """Run :meth:`OvhClient.ensure_available` (which shells out to ``ovhai --version``)."""
    # pylint: disable=import-outside-toplevel
    from multi_scenario.adapters.runners.ovh_cli import OvhClient, OvhCliError

    try:
        OvhClient().ensure_available()
    except OvhCliError as exc:
        return CheckStatus.FAIL, str(exc)
    return CheckStatus.PASS, "ovhai CLI on PATH"


#: OVH flavors that include a GPU. Sourced from ``ovhai capabilities flavor list``
#: (verified live 2026-05-09); update if OVH adds new flavors.
def _provision_for_runner(
    runner_type: str, form_dict: dict[str, Any], **ctx: Any
) -> tuple[CheckStatus, str]:
    """Adapt ``check_runner_provisioning`` to the probe-dispatch signature.

    Maps the ``(bool, str)`` return shape from the application layer to
    ``(CheckStatus, str)`` the preflight rendering expects. Keeps the
    application module hex-clean (no frontend imports).
    """
    # pylint: disable=import-outside-toplevel
    from multi_scenario.application.runner_provisioning import check_runner_provisioning

    device = form_dict.get("training", {}).get("device", "cpu")
    ok, detail = check_runner_provisioning(runner_type, device, **ctx)
    return (CheckStatus.PASS if ok else CheckStatus.FAIL), detail


def _probe_results_bucket(ovh_cfg: Any, *, client: Any) -> tuple[CheckStatus, str]:
    """Verify the configured results bucket exists in the project's OVH region.

    Uses ``ovhai bucket list <region>`` (via :class:`OvhClient`) instead of
    ``boto3.head_bucket`` so the user doesn't need to maintain separate AWS-
    style S3 credentials — ovhai's OAuth session is sufficient.
    """
    # pylint: disable=import-outside-toplevel
    from multi_scenario.adapters.runners.ovh_cli import OvhCliError

    if ovh_cfg is None:
        return CheckStatus.FAIL, "no OVH config loaded"
    bucket = ovh_cfg.bucket_results
    region = ovh_cfg.region
    try:
        buckets = client.bucket_list(region)
    except OvhCliError as exc:
        return CheckStatus.FAIL, f"ovhai bucket list {region}: {exc}"
    names = {b.name for b in buckets}
    if bucket not in names:
        return (
            CheckStatus.FAIL,
            f"{bucket} not in {region} (found: {sorted(names) or '∅'})",
        )
    return CheckStatus.PASS, f"{bucket}@{region} reachable"


def _probe_code_hash(
    ovh_cfg: Any, repo_root: Path, *, client: Any
) -> tuple[CheckStatus, str]:
    """Compare local source hash to the ``.code_hash`` blob in the OVH code bucket.

    Uses :meth:`OvhClient.bucket_get_object` for the blob and
    :meth:`OvhClient.bucket_list_objects` for the freshness suffix
    (``last_modified`` lives on the listing entry, not the download).
    On success, the detail string carries a "(uploaded N ago)" suffix so
    the user can spot a stale upload even when the hash matches.
    """
    # pylint: disable=import-outside-toplevel
    from multi_scenario.adapters.runners.ovh_cli import OvhCliError
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
    try:
        body = client.bucket_get_object(ovh_cfg.region, bucket, CODE_HASH_KEY)
    except OvhCliError as exc:
        return (
            CheckStatus.FAIL,
            f"no .code_hash in {bucket}: {exc} — run `multi-scenario upload-code`",
        )
    remote = body.decode("utf-8").strip()
    # Best-effort age suffix — list with prefix to find the entry's
    # last_modified. If the listing fails we still succeed on hash match.
    age_suffix = ""
    try:
        objs = client.bucket_list_objects(ovh_cfg.region, bucket, prefix=CODE_HASH_KEY)
        match = next((o for o in objs if o.name == CODE_HASH_KEY), None)
        if match is not None and match.last_modified:
            age_suffix = f" (uploaded {_format_age(match.last_modified)})"
    except OvhCliError:
        pass
    if local != remote:
        return (
            CheckStatus.FAIL,
            f"local {local[:18]}… ≠ remote {remote[:18]}…{age_suffix} "
            "— run `multi-scenario upload-code`",
        )
    return CheckStatus.PASS, f"local hash matches {bucket}/{CODE_HASH_KEY}{age_suffix}"


def _format_age(last_modified: Any) -> str:
    """Render an ISO-string or datetime timestamp as "5m ago" / "3h ago" / "2d ago".

    Accepts ``str`` (the ``ovhai`` CLI returns ISO-format timestamps), an
    aware ``datetime``, or anything ``fromisoformat`` can parse. On parse
    failure returns "?" so the caller's caption still reads cleanly.
    """
    # pylint: disable=import-outside-toplevel
    from datetime import datetime, timezone

    if isinstance(last_modified, str):
        try:
            last_modified = datetime.fromisoformat(last_modified)
        except ValueError:
            return "?"
    if last_modified.tzinfo is None:
        last_modified = last_modified.replace(tzinfo=timezone.utc)
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
    ovh_cfg: Any, yaml_relpath: str | None, *, client: Any
) -> tuple[CheckStatus, str]:
    """Verify the to-be-submitted YAML is actually present in the code bucket.

    The OVH job's ``bash -c …`` invokes ``python -m multi_scenario.cli run
    <yaml_path_in_repo>`` against a path *inside* the code bucket — if the
    user picked / saved a YAML that hasn't been ``upload-code``'d yet, the
    container will fail with FileNotFoundError. Catching it here saves a
    minutes-long round-trip. Uses :meth:`OvhClient.bucket_object_exists`.
    """
    if ovh_cfg is None:
        return CheckStatus.FAIL, "no OVH config loaded"
    if not yaml_relpath:
        return CheckStatus.FAIL, "no YAML path supplied (active config unset)"
    bucket = ovh_cfg.bucket_code
    if not client.bucket_object_exists(ovh_cfg.region, bucket, yaml_relpath):
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
    ovh_cfg: Any, exp_id: str, seed: int, *, client: Any
) -> tuple[CheckStatus, str]:
    """No prior runs of ``<exp_id>_s<seed>`` exist in the results bucket.

    Stage 1 made S3 prefixes per-run (``<run_id>__<timestamp>``), so true
    collisions are impossible. But the user explicitly chose to **hard-block**
    re-runs that would create a duplicate exp_id+seed in the bucket — the
    rationale is "if there's already an experiment with this name, the user
    should explicitly clean up first so they don't end up with multiple
    timestamped copies cluttering analysis".

    Lists ``<bucket>/<run_id>__`` (note double-underscore) so ``demo_s0``
    doesn't match ``demo_s00``. To unblock a re-run, either:

      ovhai bucket object delete <bucket>@<region> --prefix <run_id>__ --yes

    or edit the YAML's ``experiment.id`` / ``experiment.seed`` to a fresh
    value.
    """
    # pylint: disable=import-outside-toplevel
    from multi_scenario.adapters.runners.ovh_cli import OvhCliError

    if ovh_cfg is None:
        return CheckStatus.FAIL, "no OVH config loaded"
    run_id = f"{exp_id}_s{seed}"
    bucket = ovh_cfg.bucket_results
    prefix = f"{run_id}__"
    try:
        objs = client.bucket_list_objects(
            ovh_cfg.region, bucket, prefix=prefix, max_keys=1
        )
    except OvhCliError as exc:
        return CheckStatus.FAIL, f"ovhai bucket object list failed: {exc}"
    if objs:
        return (
            CheckStatus.FAIL,
            f"{bucket}/{prefix}* already has prior run(s) — delete them via "
            f"'ovhai bucket object delete {bucket}@{ovh_cfg.region} "
            f"--prefix {prefix} --yes' or change exp_id/seed",
        )
    return CheckStatus.PASS, f"{bucket}/{prefix}* clear (no prior runs)"


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
    if estimate is None:
        # No ``eur_per_hour`` registered for this gpu_type → can't project
        # a number, but the cap itself is set; treat as PASS and tell the
        # user how to make the projection real.
        return (
            CheckStatus.PASS,
            f"cost_cap_eur={ovh_cfg.cost_cap_eur:.2f} (no eur_per_hour for "
            f"{ovh_cfg.gpu_type!r} in gpu_models — projection skipped)",
        )
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
    """Run the configuration / system / storage probes against the active form.

    Args:
        ovh_client: optional pre-built ``OvhClient`` (for tests / DI). When
            ``None`` a fresh ``OvhClient()`` is created here and threaded into
            every probe that needs it — so probes don't each spin up their own
            subprocess wrapper.
        yaml_relpath: path of the to-be-submitted YAML, relative to ``repo_root``
            (matches how it sits in the code bucket). Only used by the
            "Submitted YAML present in bucket" probe.
    """
    # pylint: disable=import-outside-toplevel
    from multi_scenario.adapters.runners.ovh_cli import OvhClient

    exp_id = form_dict.get("experiment", {}).get("id", "")
    seed = int(form_dict.get("experiment", {}).get("seed", 0))
    # Single OvhClient threaded through every probe that talks to ovhai —
    # keeps the credentials / binary-discovery cost down to one lookup.
    client = ovh_client or OvhClient()

    # If the OVH config didn't load, "OVH config valid" fails and every
    # downstream OVH probe stays IDLE — the user can't fix anything else
    # until the YAML parses. Local-environment rows still run normally.
    config_blocked = ovh_cfg is None
    skip_ovh_keys = {
        "OVH CLI installed",
        "Runner provisioning consistent with device",
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
        "Runner provisioning consistent with device": (
            lambda _f: _provision_for_runner("ovh", form_dict, ovh_cfg=ovh_cfg)
        ),
        "Results bucket reachable": lambda _f: _probe_results_bucket(
            ovh_cfg, client=client
        ),
        "Code matches OVH bucket": lambda _f: _probe_code_hash(
            ovh_cfg, repo_root, client=client
        ),
        "Per-run prefix not occupied": lambda _f: _probe_prefix_collision(
            ovh_cfg, exp_id, seed, client=client
        ),
        "Submitted YAML present in bucket": lambda _f: _probe_yaml_in_bucket(
            ovh_cfg, yaml_relpath, client=client
        ),
        "No active OVH job with this run_id": lambda _f: _probe_no_active_collision(
            exp_id, seed, client=client
        ),
        "Cost cap not exceeded": lambda _f: _probe_cost_cap(ovh_cfg, seconds_per_run),
        "LLM API key present for cfg.lero": _probe_lero_api_key,
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
