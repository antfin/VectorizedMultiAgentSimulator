"""F6.4 tests: ``multi-scenario upload-code <s3-config.yaml>``."""

# Pytest fixture-injection shadow false positive; same disable as test_s3.py.
# pylint: disable=redefined-outer-name

from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from multi_scenario.cli import app
from typer.testing import CliRunner

_BUCKET = "ms-test-code-cli"
_PREFIX = "code"


def _write_s3_yaml(path: Path) -> Path:
    path.write_text(
        f"bucket: {_BUCKET}\nprefix: {_PREFIX}\nregion: us-east-1\n",
        encoding="utf-8",
    )
    return path


def _mk_repo(root: Path) -> None:
    (root / "src" / "multi_scenario").mkdir(parents=True)
    (root / "src" / "multi_scenario" / "cli.py").write_text("# cli", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname='ms'", encoding="utf-8")


@pytest.fixture
def mocked_s3_with_bucket():
    """Spin up moto S3 with the test bucket pre-created."""
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=_BUCKET)
        yield client


def test_upload_code_dry_run_prints_files_without_uploading(
    tmp_path: Path, mocked_s3_with_bucket
) -> None:
    """`--dry-run` exits 0, lists files, doesn't put anything to S3."""
    _mk_repo(tmp_path)
    cfg_path = _write_s3_yaml(tmp_path / "s3.yaml")

    result = CliRunner().invoke(
        app,
        ["upload-code", str(cfg_path), "--repo-root", str(tmp_path), "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "would upload" in result.output
    assert "src/multi_scenario/cli.py" in result.output
    assert "pyproject.toml" in result.output
    listed = mocked_s3_with_bucket.list_objects_v2(Bucket=_BUCKET).get("Contents", [])
    assert listed in (None, [])


def test_upload_code_uploads_files(tmp_path: Path, mocked_s3_with_bucket) -> None:
    """Without `--dry-run`: files appear in S3 under the configured prefix."""
    _mk_repo(tmp_path)
    cfg_path = _write_s3_yaml(tmp_path / "s3.yaml")

    result = CliRunner().invoke(
        app,
        ["upload-code", str(cfg_path), "--repo-root", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert "uploaded" in result.output

    keys = sorted(
        obj["Key"]
        for obj in mocked_s3_with_bucket.list_objects_v2(Bucket=_BUCKET).get(
            "Contents", []
        )
    )
    assert f"{_PREFIX}/pyproject.toml" in keys
    assert f"{_PREFIX}/src/multi_scenario/cli.py" in keys
