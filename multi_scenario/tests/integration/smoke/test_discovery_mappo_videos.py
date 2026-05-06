"""F2.11 — videos-enabled smoke. End-to-end with ``record_video: true``.

Mirrors ``test_discovery_mappo.py`` but flips the runner flag so both
``before_training.mp4`` and ``after_training.mp4`` get written and the
report links resolve.
"""

from pathlib import Path

import pytest
import yaml

from multi_scenario.adapters.logging.file_logger import FileLogger
from multi_scenario.adapters.runners.local import LocalRunner
from multi_scenario.domain.models import ExperimentConfig, RunId, RunReport

_SMOKE_YAML = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "discovery"
    / "baseline"
    / "configs"
    / "mappo_smoke.yaml"
)


@pytest.mark.slow
def test_smoke_with_record_video_writes_both_mp4s(tmp_path: Path) -> None:
    """Both before/after MP4s land under output/videos and report links resolve."""
    raw = yaml.safe_load(_SMOKE_YAML.read_text(encoding="utf-8"))
    raw["runtime"]["runner"]["params"]["record_video"] = True
    raw["runtime"]["storage"]["path"] = str(tmp_path)
    cfg = ExperimentConfig.model_validate(raw)

    run_id = RunId(exp_id=cfg.experiment.id, seed=cfg.experiment.seed)
    run_dir = tmp_path / f"{run_id}__20260506_0000"
    run_dir.mkdir(parents=True)

    logger = FileLogger(run_dir / "logs" / "run.log")
    runner = LocalRunner(logger=logger)
    runner.run(cfg, run_dir=run_dir)

    before = run_dir / "output" / "videos" / "before_training.mp4"
    after = run_dir / "output" / "videos" / "after_training.mp4"
    assert before.is_file() and before.stat().st_size > 0
    assert after.is_file() and after.stat().st_size > 0

    report = RunReport.model_validate_json(
        (run_dir / "output" / "report.json").read_text(encoding="utf-8")
    )
    assert report.links.videos.before_training is not None
    assert report.links.videos.after_training is not None
    assert (run_dir / report.links.videos.before_training).is_file()
    assert (run_dir / report.links.videos.after_training).is_file()
