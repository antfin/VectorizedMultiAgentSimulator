"""F7.5 Phase A tests: preflight check filtering + mock-outcome application."""

# pylint: disable=missing-function-docstring,redefined-outer-name

import pytest

from multi_scenario.frontend.preflight import (
    CheckStatus,
    apply_mock,
    applicable_checks,
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


def test_apply_mock_all_ok_passes_every_row():
    checks = applicable_checks("ovh")
    apply_mock(checks, "all_ok")
    assert all(c.status == CheckStatus.PASS for c in checks)


def test_apply_mock_code_drift_fails_only_the_code_check():
    checks = applicable_checks("ovh")
    apply_mock(checks, "code_drift")
    by_name = {c.name: c for c in checks}
    assert by_name["Code matches OVH bucket"].status == CheckStatus.FAIL
    assert by_name["Config schema valid"].status == CheckStatus.PASS
    assert by_name["Cost cap not exceeded"].status == CheckStatus.PASS


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


# ── Real OVH probes (Phase C) ────────────────────────────────────────


def _ovh_cfg():
    # pylint: disable=import-outside-toplevel
    from multi_scenario.domain.models import OvhJobConfig

    return OvhJobConfig(
        region="us-east-1",  # match moto's default
        image="ovhcom/ai-training-pytorch:latest",
        gpu_type="V100S",
        n_gpu=1,
        bucket_code="ms-test-code",
        bucket_results="ms-test-results",
        poll_interval_sec=0.0,
        timeout_sec=10.0,
    )


@pytest.fixture
def mocked_ovh_buckets():
    """moto S3 with both buckets pre-created."""
    # pylint: disable=import-outside-toplevel
    import boto3
    from moto import mock_aws

    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="ms-test-code")
        client.create_bucket(Bucket="ms-test-results")
        yield client


def _seed_repo(tmp_path):
    """Minimal repo tree CodeUploader will accept."""
    (tmp_path / "src" / "multi_scenario").mkdir(parents=True)
    (tmp_path / "src" / "multi_scenario" / "demo.py").write_text("x = 1", encoding="utf-8")
    return tmp_path


def test_probe_results_bucket_passes_when_reachable(mocked_ovh_buckets, tmp_path):
    # pylint: disable=import-outside-toplevel
    from multi_scenario.frontend.preflight import run_real_ovh_checks

    cfg = _good_cfg(str(tmp_path))
    cfg["runtime"]["runner"]["type"] = "ovh"
    checks = applicable_checks("ovh")
    run_real_ovh_checks(
        checks, cfg, ovh_cfg=_ovh_cfg(), repo_root=_seed_repo(tmp_path)
    )
    by_name = {c.name: c for c in checks}
    assert by_name["Results bucket reachable"].status == CheckStatus.PASS


def test_probe_code_hash_fails_when_no_blob_uploaded(mocked_ovh_buckets, tmp_path):
    # pylint: disable=import-outside-toplevel
    from multi_scenario.frontend.preflight import run_real_ovh_checks

    cfg = _good_cfg(str(tmp_path))
    cfg["runtime"]["runner"]["type"] = "ovh"
    checks = applicable_checks("ovh")
    run_real_ovh_checks(
        checks, cfg, ovh_cfg=_ovh_cfg(), repo_root=_seed_repo(tmp_path)
    )
    by_name = {c.name: c for c in checks}
    assert by_name["Code matches OVH bucket"].status == CheckStatus.FAIL
    assert "upload-code" in by_name["Code matches OVH bucket"].detail


def test_probe_code_hash_passes_when_local_matches_uploaded(
    mocked_ovh_buckets, tmp_path
):
    """Upload code via CodeUploader, then probe — should match exactly."""
    # pylint: disable=import-outside-toplevel
    from multi_scenario.adapters.storage.code_uploader import CodeUploader
    from multi_scenario.adapters.storage.s3 import S3StorageAdapter
    from multi_scenario.domain.models import S3StorageConfig
    from multi_scenario.frontend.preflight import run_real_ovh_checks

    repo_root = _seed_repo(tmp_path)
    s3 = S3StorageAdapter(
        S3StorageConfig(bucket="ms-test-code", prefix="", region="us-east-1"),
        client=mocked_ovh_buckets,
    )
    CodeUploader(s3).upload(repo_root)
    cfg = _good_cfg(str(tmp_path))
    cfg["runtime"]["runner"]["type"] = "ovh"
    checks = applicable_checks("ovh")
    run_real_ovh_checks(checks, cfg, ovh_cfg=_ovh_cfg(), repo_root=repo_root)
    by_name = {c.name: c for c in checks}
    assert by_name["Code matches OVH bucket"].status == CheckStatus.PASS


def test_probe_prefix_collision_fails_when_run_id_already_used(
    mocked_ovh_buckets, tmp_path
):
    """If an object exists at <bucket>/<run_id>/ the probe must fail."""
    # pylint: disable=import-outside-toplevel
    from multi_scenario.frontend.preflight import run_real_ovh_checks

    cfg = _good_cfg(str(tmp_path))
    cfg["runtime"]["runner"]["type"] = "ovh"
    cfg["experiment"]["id"] = "demo"
    cfg["experiment"]["seed"] = 0
    # Plant a leftover from a prior run.
    mocked_ovh_buckets.put_object(
        Bucket="ms-test-results", Key="demo_s0/leftover.json", Body=b"{}"
    )
    checks = applicable_checks("ovh")
    run_real_ovh_checks(
        checks, cfg, ovh_cfg=_ovh_cfg(), repo_root=_seed_repo(tmp_path)
    )
    by_name = {c.name: c for c in checks}
    assert by_name["Per-run prefix not occupied"].status == CheckStatus.FAIL
    assert "demo_s0" in by_name["Per-run prefix not occupied"].detail


def test_probe_cost_cap_passes_without_seconds_estimate(mocked_ovh_buckets, tmp_path):
    # pylint: disable=import-outside-toplevel
    from multi_scenario.frontend.preflight import run_real_ovh_checks

    cfg = _good_cfg(str(tmp_path))
    cfg["runtime"]["runner"]["type"] = "ovh"
    checks = applicable_checks("ovh")
    run_real_ovh_checks(
        checks, cfg, ovh_cfg=_ovh_cfg(), repo_root=_seed_repo(tmp_path)
    )
    by_name = {c.name: c for c in checks}
    assert by_name["Cost cap not exceeded"].status == CheckStatus.PASS


def test_probe_code_hash_includes_age_suffix_when_pass(mocked_ovh_buckets, tmp_path):
    """Hash-freshness annotation appears on PASS."""
    # pylint: disable=import-outside-toplevel
    from multi_scenario.adapters.storage.code_uploader import CodeUploader
    from multi_scenario.adapters.storage.s3 import S3StorageAdapter
    from multi_scenario.domain.models import S3StorageConfig
    from multi_scenario.frontend.preflight import run_real_ovh_checks

    repo_root = _seed_repo(tmp_path)
    s3 = S3StorageAdapter(
        S3StorageConfig(bucket="ms-test-code", prefix="", region="us-east-1"),
        client=mocked_ovh_buckets,
    )
    CodeUploader(s3).upload(repo_root)
    cfg = _good_cfg(str(tmp_path))
    cfg["runtime"]["runner"]["type"] = "ovh"
    checks = applicable_checks("ovh")
    run_real_ovh_checks(checks, cfg, ovh_cfg=_ovh_cfg(), repo_root=repo_root)
    by_name = {c.name: c for c in checks}
    detail = by_name["Code matches OVH bucket"].detail
    assert by_name["Code matches OVH bucket"].status == CheckStatus.PASS
    assert "uploaded" in detail and "ago" in detail


def test_probe_yaml_in_bucket_passes_when_present(mocked_ovh_buckets, tmp_path):
    """`head_object` succeeds → 🟢 with the bucket/key in the detail."""
    # pylint: disable=import-outside-toplevel
    from multi_scenario.frontend.preflight import run_real_ovh_checks

    mocked_ovh_buckets.put_object(
        Bucket="ms-test-code",
        Key="experiments/discovery/baseline/configs/foo.yaml",
        Body=b"x: 1",
    )
    cfg = _good_cfg(str(tmp_path))
    cfg["runtime"]["runner"]["type"] = "ovh"
    checks = applicable_checks("ovh")
    run_real_ovh_checks(
        checks,
        cfg,
        ovh_cfg=_ovh_cfg(),
        repo_root=_seed_repo(tmp_path),
        yaml_relpath="experiments/discovery/baseline/configs/foo.yaml",
    )
    by_name = {c.name: c for c in checks}
    assert by_name["Submitted YAML present in bucket"].status == CheckStatus.PASS


def test_probe_yaml_in_bucket_fails_when_missing(mocked_ovh_buckets, tmp_path):
    # pylint: disable=import-outside-toplevel
    from multi_scenario.frontend.preflight import run_real_ovh_checks

    cfg = _good_cfg(str(tmp_path))
    cfg["runtime"]["runner"]["type"] = "ovh"
    checks = applicable_checks("ovh")
    run_real_ovh_checks(
        checks,
        cfg,
        ovh_cfg=_ovh_cfg(),
        repo_root=_seed_repo(tmp_path),
        yaml_relpath="experiments/never_uploaded.yaml",
    )
    by_name = {c.name: c for c in checks}
    assert by_name["Submitted YAML present in bucket"].status == CheckStatus.FAIL
    assert "upload-code" in by_name["Submitted YAML present in bucket"].detail


def test_probe_yaml_in_bucket_fails_when_no_path_supplied(mocked_ovh_buckets, tmp_path):
    """Active config outside the repo → no relpath → probe fails explicitly."""
    # pylint: disable=import-outside-toplevel
    from multi_scenario.frontend.preflight import run_real_ovh_checks

    cfg = _good_cfg(str(tmp_path))
    cfg["runtime"]["runner"]["type"] = "ovh"
    checks = applicable_checks("ovh")
    run_real_ovh_checks(
        checks,
        cfg,
        ovh_cfg=_ovh_cfg(),
        repo_root=_seed_repo(tmp_path),
        yaml_relpath=None,
    )
    by_name = {c.name: c for c in checks}
    assert by_name["Submitted YAML present in bucket"].status == CheckStatus.FAIL


class _FakeOvhClient:
    """Stand-in for OvhClient.list_jobs that returns scripted JobInfo lists."""

    def __init__(self, running):
        self._running = running

    def list_jobs(self, state_filter=None):  # noqa: ARG002
        # pylint: disable=import-outside-toplevel
        from multi_scenario.adapters.runners.ovh_cli import JobInfo
        return [JobInfo(**j) for j in self._running]


def test_probe_no_active_collision_passes_when_no_overlap(mocked_ovh_buckets, tmp_path):
    # pylint: disable=import-outside-toplevel
    from multi_scenario.frontend.preflight import run_real_ovh_checks

    fake = _FakeOvhClient(running=[
        {"id": "job_1", "name": "other_run_s0", "state": "RUNNING"},
    ])
    cfg = _good_cfg(str(tmp_path))
    cfg["runtime"]["runner"]["type"] = "ovh"
    cfg["experiment"]["id"] = "demo"
    cfg["experiment"]["seed"] = 0
    checks = applicable_checks("ovh")
    run_real_ovh_checks(
        checks,
        cfg,
        ovh_cfg=_ovh_cfg(),
        repo_root=_seed_repo(tmp_path),
        ovh_client=fake,
    )
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
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    folder = RunId(exp_id="demo", seed=0).folder_name(timestamp)
    (tmp_path / folder).mkdir(parents=True)
    status, detail = _probe_run_dir_collision(cfg)
    assert status == CheckStatus.FAIL
    assert "already exists" in detail


def test_gpu_probe_only_appears_when_device_is_cuda(tmp_path):
    """The GPU row is conditional — absent when device=cpu, present when cuda."""
    cfg_cpu = _good_cfg(str(tmp_path))
    names_cpu = {c.name for c in applicable_checks("local", cfg_cpu)}
    assert "GPU available" not in names_cpu

    cfg_cuda = _good_cfg(str(tmp_path))
    cfg_cuda["training"]["device"] = "cuda"
    names_cuda = {c.name for c in applicable_checks("local", cfg_cuda)}
    assert "GPU available" in names_cuda


def test_ovh_config_invalid_cascades_idle_to_other_ovh_rows(mocked_ovh_buckets, tmp_path):
    """When ``ovh_cfg`` is None, only the OVH-config row fails; rest stay IDLE."""
    # pylint: disable=import-outside-toplevel
    from multi_scenario.frontend.preflight import run_real_ovh_checks

    cfg = _good_cfg(str(tmp_path))
    cfg["runtime"]["runner"]["type"] = "ovh"
    checks = applicable_checks("ovh")
    run_real_ovh_checks(checks, cfg, ovh_cfg=None, repo_root=_seed_repo(tmp_path))
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
    from multi_scenario.frontend.preflight import (
        CheckStatus as _S,
        category_status,
        default_checks,
    )

    rows = applicable_checks("local")
    rows[0].status = _S.PASS
    rows[1].status = _S.FAIL
    assert category_status(rows) == _S.FAIL


def test_category_status_passes_only_when_all_pass():
    # pylint: disable=import-outside-toplevel
    from multi_scenario.frontend.preflight import (
        CheckStatus as _S,
        category_status,
    )

    rows = applicable_checks("local")
    for r in rows:
        r.status = _S.PASS
    assert category_status(rows) == _S.PASS


def test_probe_no_active_collision_fails_when_overlap(mocked_ovh_buckets, tmp_path):
    # pylint: disable=import-outside-toplevel
    from multi_scenario.frontend.preflight import run_real_ovh_checks

    fake = _FakeOvhClient(running=[
        {"id": "job_99", "name": "demo_s0", "state": "RUNNING"},
    ])
    cfg = _good_cfg(str(tmp_path))
    cfg["runtime"]["runner"]["type"] = "ovh"
    cfg["experiment"]["id"] = "demo"
    cfg["experiment"]["seed"] = 0
    checks = applicable_checks("ovh")
    run_real_ovh_checks(
        checks,
        cfg,
        ovh_cfg=_ovh_cfg(),
        repo_root=_seed_repo(tmp_path),
        ovh_client=fake,
    )
    by_name = {c.name: c for c in checks}
    assert by_name["No active OVH job with this run_id"].status == CheckStatus.FAIL
    assert "job_99" in by_name["No active OVH job with this run_id"].detail
