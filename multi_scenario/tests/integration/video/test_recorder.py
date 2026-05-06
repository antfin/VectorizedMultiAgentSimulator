"""F2.11 tests: VideoRecorder — rolls out a policy with rendering and writes MP4."""

from pathlib import Path

import pytest

from multi_scenario.adapters.algorithms.mappo import MappoAdapter
from multi_scenario.adapters.video.recorder import VideoRecorder
from multi_scenario.domain.models import ExperimentConfig

_SMOKE_YAML = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "discovery"
    / "baseline"
    / "configs"
    / "mappo_smoke.yaml"
)


@pytest.mark.slow
def test_record_writes_nonempty_mp4(tmp_path: Path) -> None:
    """Recording one episode through the smoke config produces a non-empty MP4."""
    cfg = ExperimentConfig.from_yaml(_SMOKE_YAML)
    adapter = MappoAdapter()
    # Build only — no training; the random-init policy is sufficient for the
    # recorder to step through the env and write frames. ``run_dir=tmp_path``
    # so BenchMARL's native output stays inside the auto-cleaned tmp folder.
    experiment = adapter.build_experiment(cfg, run_dir=tmp_path)

    out = tmp_path / "video.mp4"
    VideoRecorder().record(
        test_env=experiment.test_env,
        policy=experiment.policy,
        max_steps=experiment.max_steps,
        output_path=out,
    )

    assert out.is_file()
    assert out.stat().st_size > 0
