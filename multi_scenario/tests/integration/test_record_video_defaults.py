"""F8.2.A — record_video default invariants across shipped YAMLs.

Pin the contract that every shipped YAML's *effective* ``record_video`` value
matches its purpose: smoke YAMLs must produce no videos (CI must stay fast),
non-smoke (production / research) YAMLs must produce them by default.

The detection runs `_should_record_video(cfg, run_dir)` against a synthetic
``run_dir`` so the function evaluates the same gating it would in production.
"""

# pylint: disable=missing-function-docstring,redefined-outer-name

from pathlib import Path

import pytest
import yaml

from multi_scenario.adapters.algorithms.benchmarl_base import _should_record_video
from multi_scenario.domain.models import ExperimentConfig


_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXPERIMENTS_ROOT = _REPO_ROOT / "experiments"


def _all_yamls() -> list[Path]:
    """Every shipped YAML under experiments/<scenario>/<folder>/configs/."""
    return sorted(_EXPERIMENTS_ROOT.rglob("configs/*.yaml"))


def _is_smoke(path: Path) -> bool:
    """Smoke YAMLs are identified by filename suffix ``*_smoke*.yaml``."""
    stem = path.stem.lower()
    return "_smoke" in stem or stem.endswith("smoke")


def _load_cfg(path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(yaml.safe_load(path.read_text()))


@pytest.fixture
def fake_run_dir(tmp_path: Path) -> Path:
    """A synthetic run_dir so ``_should_record_video`` doesn't no-op on missing dir."""
    return tmp_path / "fake_run_s0__t"


@pytest.mark.parametrize(
    "yaml_path", _all_yamls(), ids=lambda p: str(p.relative_to(_REPO_ROOT))
)
def test_smoke_yamls_default_to_no_videos(yaml_path: Path, fake_run_dir: Path):
    """F8.2.A: every ``*_smoke*.yaml`` must resolve to record_video=False.

    Catches a silent regression where a future smoke YAML lands without
    setting record_video false AND without a smoke-suffixed exp_id — the
    smoke would suddenly start recording in CI and slow the suite.
    """
    if not _is_smoke(yaml_path):
        pytest.skip("not a smoke yaml")
    cfg = _load_cfg(yaml_path)
    assert _should_record_video(cfg, fake_run_dir) is False, (
        f"{yaml_path.relative_to(_REPO_ROOT)}: smoke YAML resolves to "
        "record_video=True. Either suffix experiment.id with '_smoke' OR set "
        "runtime.runner.params.record_video: false explicitly."
    )


@pytest.mark.parametrize(
    "yaml_path", _all_yamls(), ids=lambda p: str(p.relative_to(_REPO_ROOT))
)
def test_non_smoke_yamls_default_to_videos_on(yaml_path: Path, fake_run_dir: Path):
    """F8.2.A: every non-smoke YAML must resolve to record_video=True by default.

    The whole point of F8.2.A is that production / research / reproducibility
    runs (baseline.yaml et al.) generate the before/after pair without
    per-YAML opt-in. Catches a regression where a future YAML lands with
    record_video: false AND a non-smoke exp_id — ER1-style runs would
    suddenly stop producing videos.
    """
    if _is_smoke(yaml_path):
        pytest.skip("smoke yaml; covered by the inverse test")
    cfg = _load_cfg(yaml_path)
    assert _should_record_video(cfg, fake_run_dir) is True, (
        f"{yaml_path.relative_to(_REPO_ROOT)}: non-smoke YAML resolves to "
        "record_video=False. Drop the explicit override OR rename the file "
        "to *_smoke.yaml if it really is a smoke."
    )
