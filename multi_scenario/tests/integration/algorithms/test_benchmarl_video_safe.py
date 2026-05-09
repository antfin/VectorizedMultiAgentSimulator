"""F6.6 tests: ``_record_video_safe`` wraps VideoRecorder calls fail-soft."""

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from multi_scenario.adapters.algorithms.benchmarl_base import _record_video_safe


def test_record_video_safe_swallows_exception_and_warns(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """Recorder error → no exception propagates; warning includes label + cause."""

    def boom(**_kwargs):
        raise RuntimeError("pyglet.gl import failed: no OpenGL")

    with patch(
        "multi_scenario.adapters.algorithms.benchmarl_base.VideoRecorder"
    ) as fake:
        fake.return_value.record.side_effect = boom
        with caplog.at_level(logging.WARNING):
            _record_video_safe(
                "before_training",
                test_env=None,
                policy=None,
                max_steps=10,
                output_path=tmp_path / "before_training.mp4",
            )

    assert any(
        "before_training" in rec.message and "pyglet" in rec.message
        for rec in caplog.records
    ), [r.message for r in caplog.records]


def test_record_video_safe_no_warning_on_success(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """Recorder succeeds → no warning logged; the recorder was actually invoked."""
    with patch(
        "multi_scenario.adapters.algorithms.benchmarl_base.VideoRecorder"
    ) as fake:
        fake.return_value.record.return_value = None
        with caplog.at_level(logging.WARNING):
            _record_video_safe(
                "after_training",
                test_env="env",
                policy="pol",
                max_steps=10,
                output_path=tmp_path / "after_training.mp4",
            )

    assert fake.return_value.record.called
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
