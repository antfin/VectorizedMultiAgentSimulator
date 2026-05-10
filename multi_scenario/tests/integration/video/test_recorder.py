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


# ── F8.2.C — per-recorder safety cap ───────────────────────────────────


def test_video_cap_skips_after_max(tmp_path: Path, monkeypatch, caplog) -> None:
    """11th record() call on the same recorder is a no-op + warning.

    Cap is per-instance, so a regen invocation that constructs a fresh
    VideoRecorder isn't affected — only a multi-video mode that reuses
    one recorder would ever trip this. We don't have such a mode today;
    the test pins the contract for future features.
    """
    # pylint: disable=import-outside-toplevel
    import logging

    from multi_scenario.adapters.video.recorder import _MAX_VIDEOS_PER_RUN

    # Stub the heavy parts of record() so the test stays fast — we only
    # care about the counter / cap behaviour, not the real rollout.
    captured_outputs: list[Path] = []

    def _fake_mimsave(path, frames, **kwargs):  # noqa: ARG001
        captured_outputs.append(Path(path))
        Path(path).write_bytes(b"fake mp4 contents")

    monkeypatch.setattr(
        "multi_scenario.adapters.video.recorder.imageio.mimsave", _fake_mimsave
    )

    # Stub out the policy / env path so record() doesn't need real BenchMARL.
    class _FakeTd(dict):
        def __getitem__(self, k):
            if k == "next":
                return _FakeTd({"done": __import__("torch").zeros(1)})
            return super().__getitem__(k)

        def keys(self):
            return ["done"]

    fake_env = type(
        "E",
        (),
        {
            "base_env": type(
                "B",
                (),
                {"_env": type("V", (), {"render": staticmethod(lambda **kw: None)})()},
            )(),
            "reset": staticmethod(lambda: _FakeTd()),
            "step": staticmethod(lambda td: _FakeTd()),
        },
    )()
    fake_policy = staticmethod(lambda td: _FakeTd())

    rec = VideoRecorder()
    caplog.set_level(logging.WARNING, logger="multi_scenario.adapters.video.recorder")

    # First _MAX_VIDEOS_PER_RUN calls succeed.
    for i in range(_MAX_VIDEOS_PER_RUN):
        rec.record(
            test_env=fake_env,
            policy=fake_policy,
            max_steps=1,
            output_path=tmp_path / f"v{i}.mp4",
        )
    assert len(captured_outputs) == _MAX_VIDEOS_PER_RUN
    assert rec._n_recorded == _MAX_VIDEOS_PER_RUN  # pylint: disable=protected-access

    # 11th: no MP4 written, warning logged.
    rec.record(
        test_env=fake_env,
        policy=fake_policy,
        max_steps=1,
        output_path=tmp_path / "overflow.mp4",
    )
    assert len(captured_outputs) == _MAX_VIDEOS_PER_RUN  # unchanged
    assert not (tmp_path / "overflow.mp4").exists()
    assert any(
        "video cap reached" in rec_msg.getMessage()
        for rec_msg in caplog.records
        if rec_msg.levelname == "WARNING"
    )


def test_video_cap_is_per_instance(tmp_path: Path, monkeypatch) -> None:
    """A fresh VideoRecorder() resets the counter — regen still works after train.

    Critical: BenchmarlBaseAdapter and the regen CLI each construct their
    own VideoRecorder. If the cap were class- or process-level, regen would
    silently no-op after a long training run that already wrote 10 videos.
    """
    # pylint: disable=import-outside-toplevel
    from multi_scenario.adapters.video.recorder import _MAX_VIDEOS_PER_RUN

    monkeypatch.setattr(
        "multi_scenario.adapters.video.recorder.imageio.mimsave",
        lambda path, frames, **kw: Path(path).write_bytes(b"x"),
    )

    fake_env = type(
        "E",
        (),
        {
            "base_env": type(
                "B",
                (),
                {"_env": type("V", (), {"render": staticmethod(lambda **kw: None)})()},
            )(),
            "reset": staticmethod(
                lambda: {"next": {"done": __import__("torch").zeros(1)}}
            ),
            "step": staticmethod(
                lambda td: {"next": {"done": __import__("torch").zeros(1)}}
            ),
        },
    )()

    # First recorder hits the cap.
    rec1 = VideoRecorder()
    for i in range(_MAX_VIDEOS_PER_RUN + 2):
        rec1.record(
            test_env=fake_env,
            policy=lambda td: td,
            max_steps=1,
            output_path=tmp_path / f"a{i}.mp4",
        )
    assert rec1._n_recorded == _MAX_VIDEOS_PER_RUN  # pylint: disable=protected-access

    # New recorder starts fresh — can record again.
    rec2 = VideoRecorder()
    rec2.record(
        test_env=fake_env,
        policy=lambda td: td,
        max_steps=1,
        output_path=tmp_path / "b0.mp4",
    )
    assert rec2._n_recorded == 1  # pylint: disable=protected-access
