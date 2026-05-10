"""Preflight check tests: filtering, dispatch, and category rollup."""

# pylint: disable=missing-function-docstring,redefined-outer-name

import pytest

from multi_scenario.frontend.preflight import (
    applicable_checks,
    CheckStatus,
    default_checks,
    led,
)


def test_default_checks_starts_in_idle():
    checks = default_checks()
    assert all(c.status == CheckStatus.IDLE for c in checks)
    assert all(c.detail == "" for c in checks)


def test_applicable_checks_local_drops_ovh_only_rows():
    """Local runner sees only the runner-agnostic rows."""
    local_only = [c.name for c in applicable_checks("local")]
    assert "Config schema valid" in local_only
    assert "Storage path writable" in local_only
    assert "OVH CLI installed" not in local_only
    assert "Code matches OVH bucket" not in local_only


def test_applicable_checks_ovh_includes_all_relevant_rows():
    ovh_only = {c.name for c in applicable_checks("ovh")}
    assert "OVH CLI installed" in ovh_only
    assert "Results bucket reachable" in ovh_only
    assert "Code matches OVH bucket" in ovh_only
    assert "Cost cap not exceeded" in ovh_only


def test_led_glyphs_are_distinct():
    """Sanity: each status maps to a unique glyph so the UI stays unambiguous."""
    glyphs = {led(s) for s in CheckStatus}
    assert len(glyphs) == len(CheckStatus)


# ── Real local probes (Phase B) ──────────────────────────────────────


def _good_cfg(storage_path: str) -> dict:
    """Minimal cfg dict that passes ExperimentConfig.model_validate."""
    return {
        "experiment": {"id": "demo", "seed": 0},
        "scenario": {"type": "discovery", "params": {}},
        "algorithm": {"type": "mappo", "params": {}},
        "training": {"max_iters": 1},
        "evaluation": {"interval_iters": 1, "episodes": 1},
        "runtime": {
            "runner": {"type": "local", "params": {}},
            "storage": {"type": "fs", "path": storage_path, "params": {}},
        },
    }


def test_run_real_local_checks_passes_for_valid_cfg(tmp_path):
    # pylint: disable=import-outside-toplevel
    from multi_scenario.frontend.preflight import run_real_local_checks

    cfg = _good_cfg(str(tmp_path))
    checks = applicable_checks("local")
    run_real_local_checks(checks, cfg)
    assert all(c.status == CheckStatus.PASS for c in checks), [
        (c.name, c.status, c.detail) for c in checks
    ]


def test_run_real_local_checks_fails_storage_when_path_missing(tmp_path):
    # pylint: disable=import-outside-toplevel
    from multi_scenario.frontend.preflight import run_real_local_checks

    cfg = _good_cfg("/this/path/should/not/exist/anywhere")
    checks = applicable_checks("local")
    run_real_local_checks(checks, cfg)
    by_name = {c.name: c for c in checks}
    assert by_name["Storage path writable"].status == CheckStatus.FAIL


def test_run_real_local_checks_fails_config_with_bad_schema(tmp_path):
    # pylint: disable=import-outside-toplevel
    from multi_scenario.frontend.preflight import run_real_local_checks

    cfg = _good_cfg(str(tmp_path))
    cfg["experiment"]["seed"] = "not_an_int"  # schema violation
    checks = applicable_checks("local")
    run_real_local_checks(checks, cfg)
    by_name = {c.name: c for c in checks}
    assert by_name["Config schema valid"].status == CheckStatus.FAIL
    assert "experiment.seed" in by_name["Config schema valid"].detail


# ── Real OVH probes (F7.7.A2: routed through OvhClient, not boto3) ──


def _ovh_cfg():
    # pylint: disable=import-outside-toplevel
    from multi_scenario.domain.models import OvhJobConfig

    return OvhJobConfig(
        region="GRA",
        image="ovhcom/ai-training-pytorch:latest",
        gpu_type="V100S",
        n_gpu=1,
        bucket_code="ms-test-code",
        bucket_results="ms-test-results",
        poll_interval_sec=0.0,
        timeout_sec=10.0,
    )


def _seed_repo(tmp_path):
    """Minimal repo tree CodeUploader will accept."""
    (tmp_path / "src" / "multi_scenario").mkdir(parents=True)
    (tmp_path / "src" / "multi_scenario" / "demo.py").write_text(
        "x = 1", encoding="utf-8"
    )
    return tmp_path


class FakeOvhClient:
    """In-memory stand-in for :class:`OvhClient` covering every verb the probes use.

    Tracks buckets + objects + running jobs in plain dicts. Each method
    mirrors ``OvhClient``'s signature so tests can swap the fake in via
    ``run_real_ovh_checks(ovh_client=fake)``. Failure modes (e.g. simulating
    an auth error) are toggled via ``self.fail_bucket_list`` etc.
    """

    def __init__(self):
        self.buckets: dict[str, set[str]] = {}  # name → set of region aliases
        self.objects: dict[
            tuple[str, str, str], bytes
        ] = {}  # (region, bucket, key) → body
        self.last_modified: dict[tuple[str, str, str], str] = {}
        self.running_jobs: list[dict] = []
        self.fail_bucket_list = False

    # ── helpers tests use to set up state ──
    def add_bucket(self, region: str, name: str) -> None:
        self.buckets.setdefault(name, set()).add(region)

    def put_object(
        self,
        region: str,
        bucket: str,
        key: str,
        body: bytes,
        last_modified: str | None = None,
    ) -> None:
        self.add_bucket(region, bucket)
        self.objects[(region, bucket, key)] = body
        if last_modified is not None:
            self.last_modified[(region, bucket, key)] = last_modified

    # ── interface mirrored on OvhClient ──
    def bucket_list(self, region: str):
        # pylint: disable=import-outside-toplevel
        from multi_scenario.adapters.runners.ovh_cli import BucketInfo, OvhCliError

        if self.fail_bucket_list:
            raise OvhCliError("simulated bucket_list failure")
        return [
            BucketInfo(name=name)
            for name, regions in self.buckets.items()
            if region in regions
        ]

    def bucket_list_objects(
        self,
        region: str,
        bucket: str,
        *,
        prefix: str | None = None,
        max_keys: int | None = None,
    ):
        # pylint: disable=import-outside-toplevel
        from multi_scenario.adapters.runners.ovh_cli import BucketObject

        out = [
            BucketObject(
                name=key,
                last_modified=self.last_modified.get((region, bucket, key)),
            )
            for (r, b, key) in self.objects
            if r == region
            and b == bucket
            and (prefix is None or key.startswith(prefix))
        ]
        return out[:max_keys] if max_keys is not None else out

    def bucket_object_exists(self, region: str, bucket: str, key: str) -> bool:
        return (region, bucket, key) in self.objects

    def bucket_get_object(self, region: str, bucket: str, key: str) -> bytes:
        # pylint: disable=import-outside-toplevel
        from multi_scenario.adapters.runners.ovh_cli import OvhCliError

        try:
            return self.objects[(region, bucket, key)]
        except KeyError as exc:
            raise OvhCliError(f"key not found: {bucket}/{key}") from exc

    def list_jobs(self, state_filter=None):
        # pylint: disable=import-outside-toplevel
        from multi_scenario.adapters.runners.ovh_cli import JobInfo

        infos = [JobInfo(**j) for j in self.running_jobs]
        if state_filter is not None:
            infos = [j for j in infos if j.state.upper() == state_filter.upper()]
        return infos


@pytest.fixture
def fake_ovh_client():
    """Pre-populated :class:`FakeOvhClient` with both test buckets registered."""
    fake = FakeOvhClient()
    fake.add_bucket("GRA", "ms-test-code")
    fake.add_bucket("GRA", "ms-test-results")
    return fake


def _run_with(checks, cfg, fake, **kw):
    # pylint: disable=import-outside-toplevel
    from multi_scenario.frontend.preflight import run_real_ovh_checks

    return run_real_ovh_checks(
        checks,
        cfg,
        ovh_cfg=_ovh_cfg(),
        repo_root=kw.pop("repo_root"),
        ovh_client=fake,
        **kw,
    )


def test_probe_results_bucket_passes_when_listing_includes_bucket(
    fake_ovh_client, tmp_path
):
    cfg = _good_cfg(str(tmp_path))
    cfg["runtime"]["runner"]["type"] = "ovh"
    checks = applicable_checks("ovh")
    _run_with(checks, cfg, fake_ovh_client, repo_root=_seed_repo(tmp_path))
    by_name = {c.name: c for c in checks}
    assert by_name["Results bucket reachable"].status == CheckStatus.PASS


def test_probe_results_bucket_fails_when_listing_omits_bucket(tmp_path):
    """Bucket missing from ``ovhai bucket list`` → FAIL with a clear hint."""
    fake = FakeOvhClient()  # empty — no buckets registered
    cfg = _good_cfg(str(tmp_path))
    cfg["runtime"]["runner"]["type"] = "ovh"
    checks = applicable_checks("ovh")
    _run_with(checks, cfg, fake, repo_root=_seed_repo(tmp_path))
    by_name = {c.name: c for c in checks}
    assert by_name["Results bucket reachable"].status == CheckStatus.FAIL
    assert "ms-test-results" in by_name["Results bucket reachable"].detail


def test_probe_results_bucket_fails_on_cli_error(fake_ovh_client, tmp_path):
    """OvhCliError surfaces in the detail string."""
    fake_ovh_client.fail_bucket_list = True
    cfg = _good_cfg(str(tmp_path))
    cfg["runtime"]["runner"]["type"] = "ovh"
    checks = applicable_checks("ovh")
    _run_with(checks, cfg, fake_ovh_client, repo_root=_seed_repo(tmp_path))
    by_name = {c.name: c for c in checks}
    assert by_name["Results bucket reachable"].status == CheckStatus.FAIL
    assert "simulated" in by_name["Results bucket reachable"].detail


def test_probe_code_hash_fails_when_no_blob_uploaded(fake_ovh_client, tmp_path):
    cfg = _good_cfg(str(tmp_path))
    cfg["runtime"]["runner"]["type"] = "ovh"
    checks = applicable_checks("ovh")
    _run_with(checks, cfg, fake_ovh_client, repo_root=_seed_repo(tmp_path))
    by_name = {c.name: c for c in checks}
    assert by_name["Code matches OVH bucket"].status == CheckStatus.FAIL
    assert "upload-code" in by_name["Code matches OVH bucket"].detail


def test_probe_code_hash_passes_when_local_matches_uploaded(fake_ovh_client, tmp_path):
    """Plant the local hash blob in the fake bucket; probe should match."""
    # pylint: disable=import-outside-toplevel
    from multi_scenario.adapters.storage.code_uploader import (
        CODE_HASH_KEY,
        compute_local_code_hash,
    )

    repo_root = _seed_repo(tmp_path)
    expected_hash = compute_local_code_hash(repo_root)
    fake_ovh_client.put_object(
        "GRA",
        "ms-test-code",
        CODE_HASH_KEY,
        expected_hash.encode("utf-8"),
        last_modified="2026-05-08T16:00:00",
    )
    cfg = _good_cfg(str(tmp_path))
    cfg["runtime"]["runner"]["type"] = "ovh"
    checks = applicable_checks("ovh")
    _run_with(checks, cfg, fake_ovh_client, repo_root=repo_root)
    by_name = {c.name: c for c in checks}
    detail = by_name["Code matches OVH bucket"].detail
    assert by_name["Code matches OVH bucket"].status == CheckStatus.PASS
    assert "uploaded" in detail and "ago" in detail


def test_probe_code_hash_fails_when_local_differs_from_remote(
    fake_ovh_client, tmp_path
):
    # pylint: disable=import-outside-toplevel
    from multi_scenario.adapters.storage.code_uploader import CODE_HASH_KEY

    repo_root = _seed_repo(tmp_path)
    fake_ovh_client.put_object(
        "GRA",
        "ms-test-code",
        CODE_HASH_KEY,
        b"sha256:DIFFERENT_HASH_VALUE",
    )
    cfg = _good_cfg(str(tmp_path))
    cfg["runtime"]["runner"]["type"] = "ovh"
    checks = applicable_checks("ovh")
    _run_with(checks, cfg, fake_ovh_client, repo_root=repo_root)
    by_name = {c.name: c for c in checks}
    assert by_name["Code matches OVH bucket"].status == CheckStatus.FAIL
    assert "upload-code" in by_name["Code matches OVH bucket"].detail


def test_probe_prefix_collision_fails_when_prior_run_exists(fake_ovh_client, tmp_path):
    """Stage 1: probe lists ``<run_id>__`` (double-underscore) to find PRIOR
    timestamped runs of the same exp_id+seed. With per-run prefixes, no true
    collision is possible — but the user explicitly chose to hard-block
    re-runs that would create a duplicate (so the experiment dir doesn't
    accumulate clutter from accidental re-submissions).
    """
    cfg = _good_cfg(str(tmp_path))
    cfg["runtime"]["runner"]["type"] = "ovh"
    cfg["experiment"]["id"] = "demo"
    cfg["experiment"]["seed"] = 0
    # Plant a prior timestamped run.
    fake_ovh_client.put_object(
        "GRA",
        "ms-test-results",
        "demo_s0__20260507_123000/output/metrics.json",
        b"{}",
    )
    checks = applicable_checks("ovh")
    _run_with(checks, cfg, fake_ovh_client, repo_root=_seed_repo(tmp_path))
    by_name = {c.name: c for c in checks}
    assert by_name["Per-run prefix not occupied"].status == CheckStatus.FAIL
    assert "demo_s0__" in by_name["Per-run prefix not occupied"].detail
    # Detail tells the user how to clean up.
    assert "delete" in by_name["Per-run prefix not occupied"].detail.lower()


def test_probe_prefix_collision_passes_when_listing_empty(fake_ovh_client, tmp_path):
    cfg = _good_cfg(str(tmp_path))
    cfg["runtime"]["runner"]["type"] = "ovh"
    checks = applicable_checks("ovh")
    _run_with(checks, cfg, fake_ovh_client, repo_root=_seed_repo(tmp_path))
    by_name = {c.name: c for c in checks}
    assert by_name["Per-run prefix not occupied"].status == CheckStatus.PASS


def test_probe_cost_cap_passes_without_seconds_estimate(fake_ovh_client, tmp_path):
    cfg = _good_cfg(str(tmp_path))
    cfg["runtime"]["runner"]["type"] = "ovh"
    checks = applicable_checks("ovh")
    _run_with(checks, cfg, fake_ovh_client, repo_root=_seed_repo(tmp_path))
    by_name = {c.name: c for c in checks}
    assert by_name["Cost cap not exceeded"].status == CheckStatus.PASS


def test_probe_yaml_in_bucket_passes_when_present(fake_ovh_client, tmp_path):
    fake_ovh_client.put_object(
        "GRA",
        "ms-test-code",
        "experiments/discovery/baseline/configs/foo.yaml",
        b"x: 1",
    )
    cfg = _good_cfg(str(tmp_path))
    cfg["runtime"]["runner"]["type"] = "ovh"
    checks = applicable_checks("ovh")
    _run_with(
        checks,
        cfg,
        fake_ovh_client,
        repo_root=_seed_repo(tmp_path),
        yaml_relpath="experiments/discovery/baseline/configs/foo.yaml",
    )
    by_name = {c.name: c for c in checks}
    assert by_name["Submitted YAML present in bucket"].status == CheckStatus.PASS


def test_probe_yaml_in_bucket_fails_when_missing(fake_ovh_client, tmp_path):
    cfg = _good_cfg(str(tmp_path))
    cfg["runtime"]["runner"]["type"] = "ovh"
    checks = applicable_checks("ovh")
    _run_with(
        checks,
        cfg,
        fake_ovh_client,
        repo_root=_seed_repo(tmp_path),
        yaml_relpath="experiments/never_uploaded.yaml",
    )
    by_name = {c.name: c for c in checks}
    assert by_name["Submitted YAML present in bucket"].status == CheckStatus.FAIL
    assert "upload-code" in by_name["Submitted YAML present in bucket"].detail


def test_probe_yaml_in_bucket_fails_when_no_path_supplied(fake_ovh_client, tmp_path):
    """Active config outside the repo → no relpath → probe fails explicitly."""
    cfg = _good_cfg(str(tmp_path))
    cfg["runtime"]["runner"]["type"] = "ovh"
    checks = applicable_checks("ovh")
    _run_with(
        checks,
        cfg,
        fake_ovh_client,
        repo_root=_seed_repo(tmp_path),
        yaml_relpath=None,
    )
    by_name = {c.name: c for c in checks}
    assert by_name["Submitted YAML present in bucket"].status == CheckStatus.FAIL


def test_probe_no_active_collision_passes_when_no_overlap(fake_ovh_client, tmp_path):
    fake_ovh_client.running_jobs = [
        {"id": "job_1", "name": "other_run_s0", "state": "RUNNING"},
    ]
    cfg = _good_cfg(str(tmp_path))
    cfg["runtime"]["runner"]["type"] = "ovh"
    cfg["experiment"]["id"] = "demo"
    cfg["experiment"]["seed"] = 0
    checks = applicable_checks("ovh")
    _run_with(checks, cfg, fake_ovh_client, repo_root=_seed_repo(tmp_path))
    by_name = {c.name: c for c in checks}
    assert by_name["No active OVH job with this run_id"].status == CheckStatus.PASS


def test_required_deps_probe_passes_with_torch_vmas_benchmarl_imageio():
    """All four deps are dev-time installed → 🟢."""
    # pylint: disable=import-outside-toplevel
    from multi_scenario.frontend.preflight import _probe_required_deps

    status, detail = _probe_required_deps({})
    assert status == CheckStatus.PASS
    for dep in ("torch", "vmas", "benchmarl", "imageio"):
        assert dep in detail


def test_run_dir_collision_probe_passes_for_fresh_path(tmp_path):
    # pylint: disable=import-outside-toplevel
    from multi_scenario.frontend.preflight import _probe_run_dir_collision

    status, _ = _probe_run_dir_collision(_good_cfg(str(tmp_path)))
    assert status == CheckStatus.PASS


def test_run_dir_collision_probe_fails_when_target_exists(tmp_path):
    """Plant the exact folder the runner would pick → 🔴."""
    # pylint: disable=import-outside-toplevel
    from datetime import datetime, timezone

    from multi_scenario.domain.models import RunId
    from multi_scenario.frontend.preflight import _probe_run_dir_collision

    cfg = _good_cfg(str(tmp_path))
    cfg["experiment"]["id"] = "demo"
    cfg["experiment"]["seed"] = 0
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    folder = RunId(exp_id="demo", seed=0).folder_name(timestamp)
    (tmp_path / folder).mkdir(parents=True)
    status, detail = _probe_run_dir_collision(cfg)
    assert status == CheckStatus.FAIL
    assert "already exists" in detail


def test_runner_provisioning_check_present_for_both_targets(tmp_path):
    """F7.7.A4: 'Runner provisioning consistent with device' is the unified
    check that subsumed the old conditional 'GPU available' row. Present
    for both runner targets; the actual status differs by device + host.
    """
    cfg_cpu = _good_cfg(str(tmp_path))
    names_local = {c.name for c in applicable_checks("local", cfg_cpu)}
    names_ovh = {c.name for c in applicable_checks("ovh", cfg_cpu)}
    assert "Runner provisioning consistent with device" in names_local
    assert "Runner provisioning consistent with device" in names_ovh
    # Old conditional probe is gone.
    assert "GPU available" not in names_local
    assert "GPU available" not in names_ovh


def test_ovh_config_invalid_cascades_idle_to_other_ovh_rows(fake_ovh_client, tmp_path):
    """When ``ovh_cfg`` is None, only the OVH-config row fails; rest stay IDLE."""
    # pylint: disable=import-outside-toplevel
    from multi_scenario.frontend.preflight import run_real_ovh_checks

    cfg = _good_cfg(str(tmp_path))
    cfg["runtime"]["runner"]["type"] = "ovh"
    checks = applicable_checks("ovh")
    run_real_ovh_checks(
        checks,
        cfg,
        ovh_cfg=None,
        repo_root=_seed_repo(tmp_path),
        ovh_client=fake_ovh_client,
    )
    by_name = {c.name: c for c in checks}
    assert by_name["OVH config valid"].status == CheckStatus.FAIL
    # Every cloud-env / resources OVH row must be idle (cascade).
    for cloud_row in (
        "OVH CLI installed",
        "Results bucket reachable",
        "Code matches OVH bucket",
        "Per-run prefix not occupied",
        "Submitted YAML present in bucket",
        "No active OVH job with this run_id",
        "Cost cap not exceeded",
    ):
        assert by_name[cloud_row].status == CheckStatus.IDLE
        assert "fix the OVH config row first" in by_name[cloud_row].detail


# ── Grouping helpers ─────────────────────────────────────────────────


def test_group_by_category_preserves_canonical_order():
    # pylint: disable=import-outside-toplevel
    from multi_scenario.frontend.preflight import (
        CATEGORY_LABELS,
        default_checks,
        group_by_category,
    )

    grouped = group_by_category(default_checks())
    labels = [label for label, _ in grouped]
    # Three top-level LEDs in this exact order.
    assert labels == [
        CATEGORY_LABELS["config"],
        CATEGORY_LABELS["system"],
        CATEGORY_LABELS["storage"],
    ]


def test_three_categories_only():
    """Sanity: the rollup has exactly three buckets, matching the page UI."""
    # pylint: disable=import-outside-toplevel
    from multi_scenario.frontend.preflight import CATEGORY_ORDER

    assert CATEGORY_ORDER == ("config", "system", "storage")


def test_local_target_categorisation():
    """Each row in the local check set lands in a known category."""
    rows = applicable_checks("local")
    by_cat: dict[str, list[str]] = {}
    for c in rows:
        by_cat.setdefault(c.category, []).append(c.name)
    assert "Config schema valid" in by_cat["config"]
    assert "Required deps importable" in by_cat["system"]
    assert "Storage path writable" in by_cat["storage"]
    assert "Run dir does not collide" in by_cat["storage"]
    # Local target should never include OVH-only checks.
    assert all(
        name not in (n for names in by_cat.values() for n in names)
        for name in ("OVH CLI installed", "Code matches OVH bucket")
    )


def test_ovh_target_categorisation():
    """OVH check set carries the expected rows in each category."""
    rows = applicable_checks("ovh")
    by_cat: dict[str, list[str]] = {}
    for c in rows:
        by_cat.setdefault(c.category, []).append(c.name)
    assert {"Config schema valid", "OVH config valid"} <= set(by_cat["config"])
    assert {
        "OVH CLI installed",
        "No active OVH job with this run_id",
        "Cost cap not exceeded",
    } <= set(by_cat["system"])
    assert {
        "Results bucket reachable",
        "Code matches OVH bucket",
        "Submitted YAML present in bucket",
        "Per-run prefix not occupied",
    } <= set(by_cat["storage"])


def test_category_status_rolls_up_to_fail_when_any_row_fails():
    # pylint: disable=import-outside-toplevel
    from multi_scenario.frontend.preflight import category_status, CheckStatus as _S

    rows = applicable_checks("local")
    rows[0].status = _S.PASS
    rows[1].status = _S.FAIL
    assert category_status(rows) == _S.FAIL


def test_category_status_passes_only_when_all_pass():
    # pylint: disable=import-outside-toplevel
    from multi_scenario.frontend.preflight import category_status, CheckStatus as _S

    rows = applicable_checks("local")
    for r in rows:
        r.status = _S.PASS
    assert category_status(rows) == _S.PASS


# ── F7.7.C3: per-probe coverage gap audit ────────────────────────────


def test_probe_ovh_cli_pass_when_binary_available(monkeypatch):
    """``_probe_ovh_cli`` returns PASS when ``OvhClient.ensure_available`` succeeds."""
    # pylint: disable=import-outside-toplevel
    from multi_scenario.frontend.preflight import _probe_ovh_cli

    def _ok(self):  # noqa: ARG001
        return None

    monkeypatch.setattr(
        "multi_scenario.adapters.runners.ovh_cli.OvhClient.ensure_available",
        _ok,
    )
    status, detail = _probe_ovh_cli()
    assert status == CheckStatus.PASS
    assert "ovhai CLI on PATH" in detail


def test_probe_ovh_cli_fail_when_binary_missing(monkeypatch):
    """Missing ovhai → FAIL, install hint surfaced via OvhCliError message."""
    # pylint: disable=import-outside-toplevel
    from multi_scenario.adapters.runners.ovh_cli import OvhCliError
    from multi_scenario.frontend.preflight import _probe_ovh_cli

    def _missing(self):  # noqa: ARG001
        raise OvhCliError("ovhai CLI not found on PATH. Install via …")

    monkeypatch.setattr(
        "multi_scenario.adapters.runners.ovh_cli.OvhClient.ensure_available",
        _missing,
    )
    status, detail = _probe_ovh_cli()
    assert status == CheckStatus.FAIL
    assert "not found" in detail


@pytest.mark.parametrize(
    "delta_secs,expected_unit",
    [
        (10, "s"),  # < 60s
        (5 * 60, "m"),  # 5 min
        (3 * 3600, "h"),  # 3 h
        (2 * 86400, "d"),  # 2 d
    ],
)
def test_format_age_picks_right_unit(delta_secs, expected_unit):
    """``_format_age`` rolls 's' → 'm' → 'h' → 'd' at the right thresholds."""
    # pylint: disable=import-outside-toplevel
    from datetime import datetime, timedelta, timezone

    from multi_scenario.frontend.preflight import _format_age

    ts = datetime.now(timezone.utc) - timedelta(seconds=delta_secs)
    out = _format_age(ts)
    assert out.endswith(f"{expected_unit} ago"), out


def test_format_age_returns_question_mark_on_unparseable():
    """Garbage ISO string → '?' instead of raising — caller's caption stays clean."""
    # pylint: disable=import-outside-toplevel
    from multi_scenario.frontend.preflight import _format_age

    assert _format_age("not-an-iso-string") == "?"


def test_format_age_handles_naive_datetime():
    """Naive datetime is interpreted as UTC (defensive against timezone-less input)."""
    # pylint: disable=import-outside-toplevel
    from datetime import datetime, timedelta, timezone

    from multi_scenario.frontend.preflight import _format_age

    naive = (datetime.now(timezone.utc) - timedelta(seconds=30)).replace(tzinfo=None)
    out = _format_age(naive)
    assert "ago" in out


def test_storage_writable_fails_when_path_is_empty(tmp_path):
    """Empty ``runtime.storage.path`` → FAIL with a clear message."""
    # pylint: disable=import-outside-toplevel
    from multi_scenario.frontend.preflight import _probe_storage_writable

    cfg = _good_cfg(str(tmp_path))
    cfg["runtime"]["storage"]["path"] = ""
    status, detail = _probe_storage_writable(cfg)
    assert status == CheckStatus.FAIL
    assert "empty" in detail


def test_storage_writable_passes_for_non_fs_backend(tmp_path):
    """``runtime.storage.type != 'fs'`` → PASS with N/A note (no probe to run)."""
    # pylint: disable=import-outside-toplevel
    from multi_scenario.frontend.preflight import _probe_storage_writable

    cfg = _good_cfg(str(tmp_path))
    cfg["runtime"]["storage"]["type"] = "s3"  # hypothetical future backend
    status, detail = _probe_storage_writable(cfg)
    assert status == CheckStatus.PASS
    assert "N/A" in detail


def test_run_dir_collision_passes_when_seed_is_not_int(tmp_path):
    """Bad seed → PASS (Config schema row carries the real complaint)."""
    # pylint: disable=import-outside-toplevel
    from multi_scenario.frontend.preflight import _probe_run_dir_collision

    cfg = _good_cfg(str(tmp_path))
    cfg["experiment"]["seed"] = "bogus"
    status, detail = _probe_run_dir_collision(cfg)
    assert status == CheckStatus.PASS
    assert "Config schema" in detail


def test_probe_cost_cap_fails_when_estimate_exceeds_cap():
    """``estimate_cost_eur(...) > cost_cap_eur`` → FAIL with both numbers."""
    # pylint: disable=import-outside-toplevel
    from multi_scenario.domain.models.ovh_job_config import OvhGpuModel
    from multi_scenario.frontend.preflight import _probe_cost_cap

    cfg = _ovh_cfg()
    # Register an eur_per_hour for the V100S so estimate_cost_eur returns a number,
    # then squeeze the cap → guaranteed bust.
    cfg = cfg.model_copy(
        update={
            "cost_cap_eur": 0.01,
            "gpu_models": {"V100S": OvhGpuModel(eur_per_hour=2.5)},
        }
    )
    status, detail = _probe_cost_cap(cfg, seconds_per_run=3600.0)
    assert status == CheckStatus.FAIL
    assert "€" in detail and "cap" in detail


def test_probe_cost_cap_passes_with_unknown_gpu_pricing():
    """No ``eur_per_hour`` for the gpu_type → PASS with explanatory message."""
    # pylint: disable=import-outside-toplevel
    from multi_scenario.frontend.preflight import _probe_cost_cap

    cfg = _ovh_cfg()  # no gpu_models registered → estimate=None
    status, detail = _probe_cost_cap(cfg, seconds_per_run=3600.0)
    assert status == CheckStatus.PASS
    assert "no eur_per_hour" in detail


def test_probe_no_active_collision_fails_when_overlap(fake_ovh_client, tmp_path):
    fake_ovh_client.running_jobs = [
        {"id": "job_99", "name": "demo_s0", "state": "RUNNING"},
    ]
    cfg = _good_cfg(str(tmp_path))
    cfg["runtime"]["runner"]["type"] = "ovh"
    cfg["experiment"]["id"] = "demo"
    cfg["experiment"]["seed"] = 0
    checks = applicable_checks("ovh")
    _run_with(checks, cfg, fake_ovh_client, repo_root=_seed_repo(tmp_path))
    by_name = {c.name: c for c in checks}
    assert by_name["No active OVH job with this run_id"].status == CheckStatus.FAIL
    assert "job_99" in by_name["No active OVH job with this run_id"].detail
