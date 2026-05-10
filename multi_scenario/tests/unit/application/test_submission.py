"""F7.7.A6 — application-layer submission use cases.

Pin the contract that both ``cli/run.py`` and ``frontend/pages/submit.py``
get the same orchestration: same OvhRunner construction, same run-dir
shape, same OvhSubmission bundle. Differences between callers (error
wrapping, post-submission UX) live OUTSIDE this module.
"""

# pylint: disable=missing-function-docstring,redefined-outer-name

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from multi_scenario.application.submission import (
    build_run_dir,
    LocalSubmission,
    OvhSubmission,
    submit_to_local,
    submit_to_ovh,
)
from multi_scenario.domain.models import ExperimentConfig, OvhJobConfig


def _good_cfg(storage_path: str) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "experiment": {"id": "demo", "seed": 0},
            "scenario": {"type": "discovery", "params": {}},
            "algorithm": {"type": "mappo", "params": {}},
            "training": {"max_iters": 1, "device": "cpu"},
            "evaluation": {"interval_iters": 1, "episodes": 1},
            "runtime": {
                "runner": {"type": "local", "params": {}},
                "storage": {"type": "fs", "path": storage_path, "params": {}},
            },
        }
    )


def _good_ovh_cfg() -> OvhJobConfig:
    return OvhJobConfig(
        region="GRA",
        image="ovhcom/ai-training-pytorch:latest",
        flavor="ai1-1-gpu",
        n_gpu=1,
        bucket_code="ms-test-code",
        bucket_results="ms-test-results",
    )


# ── build_run_dir ────────────────────────────────────────────────────


def test_build_run_dir_default_does_not_mkdir(tmp_path: Path):
    """Default (``mkdir=False``) is pure-data — safe for OVH-target YAMLs
    whose storage.path is a container mount that doesn't exist locally.
    """
    cfg = _good_cfg(str(tmp_path / "nonexistent"))
    run_id, run_dir = build_run_dir(cfg)
    assert run_id.exp_id == "demo"
    assert run_id.seed == 0
    assert run_dir.name.startswith("demo_s0__")
    assert not run_dir.exists(), "default must not mkdir"


def test_build_run_dir_mkdir_true_creates_dir(tmp_path: Path):
    """``mkdir=True`` creates parents — used by the local dispatch path."""
    cfg = _good_cfg(str(tmp_path))
    _, run_dir = build_run_dir(cfg, mkdir=True)
    assert run_dir.is_dir()


def test_build_run_dir_falls_back_to_experiments_when_runtime_missing(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    cfg_dict = {
        "experiment": {"id": "demo", "seed": 0},
        "scenario": {"type": "discovery", "params": {}},
        "algorithm": {"type": "mappo", "params": {}},
        "training": {"max_iters": 1, "device": "cpu"},
        "evaluation": {"interval_iters": 1, "episodes": 1},
        # no runtime section
    }
    cfg = ExperimentConfig.model_validate(cfg_dict)
    _, run_dir = build_run_dir(cfg)
    # storage_root falls back to "experiments/" (per the function's contract).
    assert run_dir.parent == Path("experiments")


def test_build_run_dir_uses_storage_root_directly_when_env_var_set(
    tmp_path: Path, monkeypatch
):
    """OVH container path: env var → no nested timestamp segment.

    Pre-fix: container created ``/workspace/results/<exp_id>_s<seed>__<ts>/``
    INSIDE the host's per-run S3 prefix → nested folders post-pullback,
    Run Detail couldn't find the run. Fix: when env var set, return
    ``storage_root`` itself as the run-dir so input/output/... land directly.
    """
    monkeypatch.setenv("MULTI_SCENARIO_USE_STORAGE_ROOT_AS_RUN_DIR", "1")
    cfg = _good_cfg(str(tmp_path))
    _, run_dir = build_run_dir(cfg)
    assert run_dir == tmp_path  # storage_root, no <run_id>__<ts>/ suffix


def test_build_run_dir_default_safe_for_ovh_container_paths(tmp_path: Path):
    """Regression: pre-fix this raised OSError on macOS for /workspace paths.

    Default ``mkdir=False`` means we never touch the FS for OVH-target YAMLs
    whose ``storage.path`` is the container mount (e.g. /workspace/results).
    """
    cfg = _good_cfg("/workspace/results")  # macOS: read-only, would crash mkdir
    _, run_dir = build_run_dir(cfg)  # default mkdir=False — must NOT raise
    assert str(run_dir).startswith("/workspace/results/demo_s0__")


# ── submit_to_ovh ────────────────────────────────────────────────────


def test_submit_to_ovh_returns_ovh_submission_with_all_fields(tmp_path: Path):
    """Mock OvhClient via the runner; assert the returned bundle is complete."""
    cfg = _good_cfg("/workspace/results")
    ovh_cfg = _good_ovh_cfg()
    fake_client = MagicMock()
    fake_client.submit.return_value = "abc-12345-uuid"

    # Inject a fake OvhClient via the dispatch — OvhRunner.submit calls
    # client.submit(). We patch the OvhRunner constructor's client kwarg.
    run_dir = tmp_path / "demo_s0__20260510_000000"
    submission = submit_to_ovh(
        cfg,
        ovh_cfg=ovh_cfg,
        yaml_path_in_repo="experiments/discovery/baseline/configs/baseline.yaml",
        run_dir=run_dir,
        logger=MagicMock(),
        client=fake_client,
        secrets=MagicMock(),
    )
    assert isinstance(submission, OvhSubmission)
    assert submission.job_id == "abc-12345-uuid"
    assert submission.run_id.exp_id == "demo"
    assert submission.run_id.seed == 0
    # Stage 1: per-run S3 prefix uses run_dir.name (timestamped) — not just run_id.
    assert submission.s3_prefix == "ms-test-results@GRA/demo_s0__20260510_000000"
    assert "abc-12345-uuid" in submission.dashboard_url
    assert "ovh.com" in submission.dashboard_url
    fake_client.submit.assert_called_once()


def test_submit_to_ovh_propagates_runner_exceptions(tmp_path: Path):
    """OvhCliError / RuntimeError surface to the caller — caller-specific UX."""
    # pylint: disable=import-outside-toplevel
    from multi_scenario.adapters.runners.ovh_cli import OvhCliError

    cfg = _good_cfg("/workspace/results")
    fake_client = MagicMock()
    fake_client.submit.side_effect = OvhCliError("simulated failure")
    with pytest.raises(OvhCliError, match="simulated failure"):
        submit_to_ovh(
            cfg,
            ovh_cfg=_good_ovh_cfg(),
            yaml_path_in_repo="x.yaml",
            run_dir=tmp_path,
            logger=MagicMock(),
            client=fake_client,
            secrets=MagicMock(),
        )


# ── submit_to_local ─────────────────────────────────────────────────


def test_submit_to_local_returns_local_submission(tmp_path: Path, monkeypatch):
    """LocalRunner.run is mocked; submit_to_local returns the wrapped result."""
    fake_result = MagicMock()
    fake_result.run_id = "demo_s0"

    def _fake_run(self, cfg, run_dir, resume_from=None):  # noqa: ARG001
        return fake_result

    monkeypatch.setattr(
        "multi_scenario.adapters.runners.local.LocalRunner.run",
        _fake_run,
    )
    submission = submit_to_local(
        _good_cfg(str(tmp_path)),
        run_dir=tmp_path / "demo_s0__t",
        logger=MagicMock(),
    )
    assert isinstance(submission, LocalSubmission)
    assert submission.run_id == "demo_s0"
    assert submission.experiment_result is fake_result


# ── Hex-architecture invariant ──────────────────────────────────────


def test_submission_module_doesnt_import_cli_or_frontend():
    """Application layer must not depend on caller layers (F7.7.A6 invariant).

    A failing test here is a smoking gun: someone wired application/submission
    to cli or frontend by mistake, breaking the dependency direction.
    """
    import multi_scenario.application.submission as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "from multi_scenario.cli" not in src
    assert "from multi_scenario.frontend" not in src
    assert "import multi_scenario.cli" not in src
    assert "import multi_scenario.frontend" not in src
