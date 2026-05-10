"""Stage 2 — pullback_run_dir contract.

Pin the behavioural surface a Streamlit auto-sync (Stage 3) and a CLI
``sweep --follow`` will both depend on:

- materialises the OVH per-run prefix as a local folder tree,
- is idempotent (skips files already present at matching size),
- no-op-safe when the prefix is empty (just warns),
- uses ``ovhai`` (no AWS keys, same auth as ``upload-code``).
"""

# pylint: disable=missing-function-docstring,redefined-outer-name

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from multi_scenario.adapters.runners.ovh_cli import BucketObject
from multi_scenario.application.ovh_pullback import pullback_run_dir, PullbackResult
from multi_scenario.domain.models import OvhJobConfig


def _ovh_cfg() -> OvhJobConfig:
    return OvhJobConfig(
        region="GRA",
        image="ovhcom/ai-training-pytorch:latest",
        flavor="ai1-1-gpu",
        n_gpu=1,
        bucket_code="ms-test-code",
        bucket_results="ms-test-results",
    )


def _stub_client(objects: dict[str, bytes]):
    """Stub OvhClient: list returns BucketObject for each key, get returns bytes."""
    client = MagicMock()
    client.bucket_list_objects.return_value = [
        BucketObject(name=k, size_bytes=len(v)) for k, v in objects.items()
    ]
    client.bucket_get_object.side_effect = lambda region, bucket, key: objects[key]
    return client


def test_pullback_writes_each_object_under_dest_dir(tmp_path: Path):
    objects = {
        "demo_s0__t/output/metrics.json": b'{"M1": 0.5}',
        "demo_s0__t/input/config.json": b"{}",
        "demo_s0__t/run_state.json": b'{"state": "DONE"}',
    }
    client = _stub_client(objects)
    dest = tmp_path / "demo_s0__t"

    result = pullback_run_dir(
        ovh_cfg=_ovh_cfg(),
        run_dir_name="demo_s0__t",
        dest_dir=dest,
        client=client,
    )

    assert isinstance(result, PullbackResult)
    assert result.n_downloaded == 3
    assert result.n_skipped == 0
    assert (dest / "output" / "metrics.json").read_bytes() == b'{"M1": 0.5}'
    assert (dest / "input" / "config.json").read_bytes() == b"{}"
    assert (dest / "run_state.json").read_bytes() == b'{"state": "DONE"}'


def test_pullback_lists_with_trailing_slash_to_avoid_sibling_runs(tmp_path: Path):
    """Probe must use ``<run_dir_name>/`` so ``demo_s0`` doesn't pull
    ``demo_s0__20260101_...`` (a different run sharing the prefix).
    """
    client = _stub_client({})
    pullback_run_dir(
        ovh_cfg=_ovh_cfg(),
        run_dir_name="demo_s0",
        dest_dir=tmp_path / "out",
        client=client,
    )
    kwargs = client.bucket_list_objects.call_args.kwargs
    assert kwargs["prefix"] == "demo_s0/"


def test_pullback_is_idempotent_skips_files_with_matching_size(tmp_path: Path):
    objects = {"demo_s0__t/output/metrics.json": b'{"M1": 0.5}'}
    client = _stub_client(objects)
    dest = tmp_path / "demo_s0__t"

    # First run: downloads.
    first = pullback_run_dir(
        ovh_cfg=_ovh_cfg(),
        run_dir_name="demo_s0__t",
        dest_dir=dest,
        client=client,
    )
    assert first.n_downloaded == 1 and first.n_skipped == 0

    # Second run with the same client: file is on disk → skip.
    second = pullback_run_dir(
        ovh_cfg=_ovh_cfg(),
        run_dir_name="demo_s0__t",
        dest_dir=dest,
        client=client,
    )
    assert second.n_downloaded == 0 and second.n_skipped == 1
    # bucket_get_object called exactly once across both invocations.
    assert client.bucket_get_object.call_count == 1


def test_pullback_redownloads_when_local_size_mismatches(tmp_path: Path):
    """Defensive: a partial / corrupt local file (wrong size) gets re-fetched."""
    objects = {"demo_s0__t/output/metrics.json": b'{"M1": 0.5}'}
    client = _stub_client(objects)
    dest = tmp_path / "demo_s0__t"
    (dest / "output").mkdir(parents=True)
    (dest / "output" / "metrics.json").write_bytes(b"truncated")  # wrong size

    result = pullback_run_dir(
        ovh_cfg=_ovh_cfg(),
        run_dir_name="demo_s0__t",
        dest_dir=dest,
        client=client,
    )
    assert result.n_downloaded == 1
    assert (dest / "output" / "metrics.json").read_bytes() == b'{"M1": 0.5}'


def test_pullback_empty_prefix_warns_and_returns_zero(tmp_path: Path):
    client = _stub_client({})
    logger = MagicMock()
    result = pullback_run_dir(
        ovh_cfg=_ovh_cfg(),
        run_dir_name="never_ran",
        dest_dir=tmp_path / "out",
        client=client,
        logger=logger,
    )
    assert result.n_downloaded == 0 and result.n_skipped == 0
    logger.warning.assert_called_once()
    msg = logger.warning.call_args.args[0]
    assert "never_ran" in msg


def test_pullback_creates_dest_dir_when_missing(tmp_path: Path):
    client = _stub_client({})
    dest = tmp_path / "deeply" / "nested" / "out"
    pullback_run_dir(
        ovh_cfg=_ovh_cfg(),
        run_dir_name="x",
        dest_dir=dest,
        client=client,
    )
    assert dest.is_dir()


# ── Hex-architecture invariant ──────────────────────────────────────


def test_pullback_module_doesnt_import_cli_or_frontend():
    """Application layer must not depend on caller layers."""
    import multi_scenario.application.ovh_pullback as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "from multi_scenario.cli" not in src
    assert "from multi_scenario.frontend" not in src
    assert "import multi_scenario.cli" not in src
    assert "import multi_scenario.frontend" not in src


def test_pullback_default_logger_is_noop_when_caller_omits(tmp_path: Path):
    """Caller must be able to omit logger without crashing on the warning path."""
    client = _stub_client({})
    pullback_run_dir(  # no logger=
        ovh_cfg=_ovh_cfg(),
        run_dir_name="never_ran",
        dest_dir=tmp_path / "out",
        client=client,
    )  # should not raise


def test_pullback_defaults_client_to_real_OvhClient_when_omitted(monkeypatch):
    """Constructor default must instantiate OvhClient — not require injection."""
    # Patch OvhClient to a no-op stub so we don't actually hit the CLI.
    fake_cls = MagicMock(return_value=_stub_client({}))
    monkeypatch.setattr(
        "multi_scenario.application.ovh_pullback.OvhClient",
        fake_cls,
    )
    pullback_run_dir(
        ovh_cfg=_ovh_cfg(),
        run_dir_name="x",
        dest_dir=Path(__import__("tempfile").mkdtemp()),
        # client kwarg intentionally omitted
    )
    fake_cls.assert_called_once_with()


@pytest.mark.parametrize("region", ["GRA", "DE", "WAW"])
def test_pullback_passes_region_through_to_client(tmp_path: Path, region: str):
    client = _stub_client({"x__t/a.json": b"a"})
    cfg = OvhJobConfig(
        region=region,
        image="ovhcom/ai-training-pytorch:latest",
        flavor="ai1-1-gpu",
        n_gpu=1,
        bucket_code="c",
        bucket_results="r",
    )
    pullback_run_dir(
        ovh_cfg=cfg,
        run_dir_name="x__t",
        dest_dir=tmp_path / "out",
        client=client,
    )
    assert client.bucket_list_objects.call_args.kwargs["region"] == region
    assert client.bucket_get_object.call_args.kwargs["region"] == region
