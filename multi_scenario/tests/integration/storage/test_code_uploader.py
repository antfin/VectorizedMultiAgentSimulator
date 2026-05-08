"""F6.4 tests: CodeUploader — selective S3 upload of source tree (moto-mocked)."""

# Pytest fixture-injection shadow false positive; same disable as test_s3.py.
# pylint: disable=redefined-outer-name

from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from multi_scenario.adapters.storage.code_uploader import (
    CODE_HASH_KEY,
    DEFAULT_EXCLUDE_PATTERNS,
    CodeUploader,
    compute_local_code_hash,
)
from multi_scenario.adapters.storage.s3 import S3StorageAdapter
from multi_scenario.domain.models import S3StorageConfig

_BUCKET = "ms-test-code"
_PREFIX = "code"


def _mk_repo(root: Path) -> None:
    """Lay down a small repo tree mirroring multi_scenario's layout."""
    (root / "src" / "multi_scenario").mkdir(parents=True)
    (root / "src" / "multi_scenario" / "cli.py").write_text("# cli", encoding="utf-8")
    (root / "src" / "multi_scenario" / "__pycache__").mkdir()
    (root / "src" / "multi_scenario" / "__pycache__" / "x.pyc").write_text("c", encoding="utf-8")
    (root / "experiments" / "discovery" / "baseline" / "configs").mkdir(parents=True)
    (root / "experiments" / "discovery" / "baseline" / "configs" / "mappo_smoke.yaml").write_text(
        "x: 1", encoding="utf-8"
    )
    (root / "experiments" / "discovery" / "baseline" / "results").mkdir()
    (root / "experiments" / "discovery" / "baseline" / "results" / "old.json").write_text(
        "{}", encoding="utf-8"
    )
    # Per-run folder (matches the §3.5.2 ``<run_id>__<timestamp>`` pattern).
    run_folder = root / "experiments" / "discovery" / "baseline" / "smoke_s0__20260507_0000"
    run_folder.mkdir()
    (run_folder / "run_state.json").write_text("{}", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname='ms'", encoding="utf-8")
    (root / "README.md").write_text("# ms", encoding="utf-8")


@pytest.fixture
def mocked_s3_with_bucket():
    """Spin up moto S3 with the test bucket pre-created."""
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=_BUCKET)
        yield client


def _make_uploader(client) -> CodeUploader:
    config = S3StorageConfig(bucket=_BUCKET, prefix=_PREFIX, region="us-east-1")
    return CodeUploader(S3StorageAdapter(config, client=client))


def test_upload_includes_curated_subset(tmp_path: Path, mocked_s3_with_bucket) -> None:
    """src/multi_scenario + experiments + configs + pyproject + README all upload."""
    _mk_repo(tmp_path)
    uploader = _make_uploader(mocked_s3_with_bucket)
    uploaded = uploader.upload(tmp_path)

    keys = sorted(
        obj["Key"]
        for obj in mocked_s3_with_bucket.list_objects_v2(Bucket=_BUCKET).get("Contents", [])
    )
    assert f"{_PREFIX}/src/multi_scenario/cli.py" in keys
    assert f"{_PREFIX}/pyproject.toml" in keys
    assert f"{_PREFIX}/README.md" in keys
    assert f"{_PREFIX}/experiments/discovery/baseline/configs/mappo_smoke.yaml" in keys
    # Returned file list contains repo-relative paths.
    assert Path("src/multi_scenario/cli.py") in uploaded


def test_upload_excludes_pycache_and_results(tmp_path: Path, mocked_s3_with_bucket) -> None:
    """__pycache__/*.pyc and results/* never appear in S3."""
    _mk_repo(tmp_path)
    _make_uploader(mocked_s3_with_bucket).upload(tmp_path)

    keys = [
        obj["Key"]
        for obj in mocked_s3_with_bucket.list_objects_v2(Bucket=_BUCKET).get("Contents", [])
    ]
    assert not any("__pycache__" in k for k in keys), keys
    assert not any(k.endswith(".pyc") for k in keys), keys
    assert not any("/results/" in k for k in keys), keys


def test_upload_skips_run_folders(tmp_path: Path, mocked_s3_with_bucket) -> None:
    """Per-run folders (``<run_id>__<timestamp>``) are excluded from code upload."""
    _mk_repo(tmp_path)
    _make_uploader(mocked_s3_with_bucket).upload(tmp_path)

    keys = [
        obj["Key"]
        for obj in mocked_s3_with_bucket.list_objects_v2(Bucket=_BUCKET).get("Contents", [])
    ]
    assert not any("smoke_s0__20260507_0000" in k for k in keys), keys


def test_dry_run_returns_list_without_uploading(tmp_path: Path, mocked_s3_with_bucket) -> None:
    """``dry_run=True`` returns paths but uploads nothing."""
    _mk_repo(tmp_path)
    uploader = _make_uploader(mocked_s3_with_bucket)
    out = uploader.upload(tmp_path, dry_run=True)

    assert len(out) > 0  # non-empty preview
    listed = mocked_s3_with_bucket.list_objects_v2(Bucket=_BUCKET).get("Contents", [])
    assert listed in (None, [])


def test_upload_empty_repo_returns_empty(tmp_path: Path, mocked_s3_with_bucket) -> None:
    """Empty repo (no include dirs/files exist) → empty result, zero uploads."""
    out = _make_uploader(mocked_s3_with_bucket).upload(tmp_path)
    assert out == []


def test_default_exclude_patterns_contain_expected_globs() -> None:
    """Sanity: exclude patterns enumerate the canonical ignores."""
    assert "*/__pycache__/*" in DEFAULT_EXCLUDE_PATTERNS
    assert "*.pyc" in DEFAULT_EXCLUDE_PATTERNS
    assert "*/results/*" in DEFAULT_EXCLUDE_PATTERNS


def test_custom_include_files_only(tmp_path: Path, mocked_s3_with_bucket) -> None:
    """Override includes — only README.md uploaded (plus the .code_hash blob).

    F7.5 Phase C added the .code_hash sidecar to every upload — assert both
    keys land in the bucket.
    """
    _mk_repo(tmp_path)
    out = _make_uploader(mocked_s3_with_bucket).upload(
        tmp_path, include_dirs=(), include_files=("README.md",)
    )
    assert out == [Path("README.md")]
    keys = sorted(
        obj["Key"]
        for obj in mocked_s3_with_bucket.list_objects_v2(Bucket=_BUCKET).get("Contents", [])
    )
    assert keys == [f"{_PREFIX}/.code_hash", f"{_PREFIX}/README.md"]


def test_upload_writes_code_hash_blob_matching_local_compute(
    tmp_path: Path, mocked_s3_with_bucket
) -> None:
    """The ``.code_hash`` blob equals ``compute_local_code_hash`` over the same set."""
    _mk_repo(tmp_path)
    _make_uploader(mocked_s3_with_bucket).upload(tmp_path)
    blob = (
        mocked_s3_with_bucket.get_object(Bucket=_BUCKET, Key=f"{_PREFIX}/{CODE_HASH_KEY}")[
            "Body"
        ]
        .read()
        .decode("utf-8")
    )
    expected = compute_local_code_hash(tmp_path)
    assert blob == expected
    assert blob.startswith("sha256:")


def test_local_hash_changes_when_a_file_changes(tmp_path: Path) -> None:
    """Sanity: editing a tracked file shifts the local hash (so drift is detectable)."""
    _mk_repo(tmp_path)
    h1 = compute_local_code_hash(tmp_path)
    (tmp_path / "src" / "multi_scenario" / "cli.py").write_text("# edited", encoding="utf-8")
    h2 = compute_local_code_hash(tmp_path)
    assert h1 != h2
