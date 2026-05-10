"""F8.2.B — cross-device CUDA→CPU loading in regenerate-videos.

Strategy (a) per F8.2.B's TDD note: stub ``torch.cuda.is_available()`` to
False and verify the cfg-mutation + monkey-patch logic fires correctly. We
can't manufacture CUDA tensors on a CPU-only test runner, so the integration
test for CUDA-checkpoint loading happens out-of-band on OVH (manual smoke).
"""

# pylint: disable=missing-function-docstring,redefined-outer-name

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from multi_scenario.application.regenerate_videos import _force_cpu_load_when_no_cuda


# ── _force_cpu_load_when_no_cuda context manager ──────────────────────


def test_force_cpu_load_no_op_when_cuda_available(monkeypatch):
    """Guard: with CUDA available, the manager doesn't touch torch.load.

    A CUDA-equipped host loads checkpoints natively on GPU. Patching
    map_location to 'cpu' would silently move tensors off-GPU and break
    BenchMARL's downstream device assumptions.
    """
    # pylint: disable=import-outside-toplevel
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    original_load = torch.load
    with _force_cpu_load_when_no_cuda():
        assert torch.load is original_load
    assert torch.load is original_load


def test_force_cpu_load_patches_when_no_cuda(monkeypatch):
    """Without CUDA, torch.load gets a wrapper that injects map_location='cpu'."""
    # pylint: disable=import-outside-toplevel
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    original_load = torch.load
    captured_kwargs = []

    def _spy_original(*args, **kwargs):
        captured_kwargs.append(dict(kwargs))
        return MagicMock()

    monkeypatch.setattr(torch, "load", _spy_original)
    with _force_cpu_load_when_no_cuda():
        torch.load("some/path.pt")
        torch.load("other.pt", weights_only=True)

    assert torch.load is _spy_original  # restored to the spy after exit
    # Both calls should have been routed through the wrapper that injected
    # map_location='cpu' before delegating to the underlying torch.load.
    assert all(kw.get("map_location") == "cpu" for kw in captured_kwargs)
    # First call: only map_location injected.
    assert captured_kwargs[0] == {"map_location": "cpu"}
    # Second call: map_location injected alongside user's weights_only.
    assert captured_kwargs[1]["map_location"] == "cpu"
    assert captured_kwargs[1]["weights_only"] is True
    # Restore for safety; monkeypatch handles it but explicit doesn't hurt.
    torch.load = original_load  # noqa: E501  (assignment intentional for clarity)


def test_force_cpu_load_doesnt_override_explicit_map_location(monkeypatch):
    """If caller already set map_location, respect it (don't clobber)."""
    # pylint: disable=import-outside-toplevel
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    captured = []

    def _spy(*args, **kwargs):
        captured.append(dict(kwargs))

    monkeypatch.setattr(torch, "load", _spy)
    with _force_cpu_load_when_no_cuda():
        torch.load("x.pt", map_location="cuda:0")
    # Explicit map_location must survive — the wrapper uses setdefault, not
    # overwrite, so we don't accidentally break a CUDA-host caller that's
    # routed through this code path.
    assert captured[0]["map_location"] == "cuda:0"


def test_force_cpu_load_restores_on_exception(monkeypatch):
    """Exception inside the with-block must still restore torch.load."""
    # pylint: disable=import-outside-toplevel
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    original_load = torch.load
    with pytest.raises(RuntimeError, match="boom"):
        with _force_cpu_load_when_no_cuda():
            assert torch.load is not original_load  # patched
            raise RuntimeError("boom")
    assert torch.load is original_load  # restored despite exception


# ── End-to-end cfg device-fallback logic ──────────────────────────────


def test_regenerate_videos_flips_cuda_cfg_to_cpu_on_cpu_host(
    tmp_path: Path, monkeypatch
):
    """Cfg with ``device='cuda'`` gets remapped to ``'cpu'`` on a CUDA-less host.

    This is the BEFORE-video path's fix: build_experiment uses cfg.training.device
    directly, so we mutate the cfg before it reaches the adapter. The check
    is the same as the AFTER path's monkey-patch but on a different layer.
    """
    # pylint: disable=import-outside-toplevel
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    # Plant minimal run-dir artefacts so regenerate_videos gets past its
    # validation gates (config + checkpoint).
    storage_dir = tmp_path / "run"
    (storage_dir / "input").mkdir(parents=True)
    (storage_dir / "output" / "benchmarl" / "x" / "checkpoints").mkdir(parents=True)
    (
        storage_dir / "output" / "benchmarl" / "x" / "checkpoints" / "checkpoint_1.pt"
    ).write_bytes(b"")

    # A real-shaped cfg with device=cuda, written via the storage adapter
    # so the loader round-trips it correctly.
    from multi_scenario.adapters.storage.local import LocalStorageAdapter
    from multi_scenario.domain.models import ExperimentConfig

    cfg = ExperimentConfig.model_validate(
        {
            "experiment": {"id": "demo", "seed": 0},
            "scenario": {
                "type": "discovery",
                "params": {"n_agents": 2, "n_targets": 2, "max_steps": 5},
            },
            "algorithm": {"type": "mappo", "params": {}},
            "training": {
                "max_iters": 1,
                "num_envs": 1,
                "device": "cuda",  # ← what OVH wrote
                "frames_per_batch": 50,
                "minibatch_size": 25,
                "n_minibatch_iters": 1,
            },
            "evaluation": {"interval_iters": 1, "episodes": 1},
            "runtime": {
                "runner": {"type": "ovh", "params": {}},
                "storage": {"type": "fs", "path": str(storage_dir), "params": {}},
            },
        }
    )
    LocalStorageAdapter().save_config(storage_dir, cfg)

    # Capture the cfg passed to build_experiment so we can assert device.
    captured_cfgs = []

    fake_experiment = MagicMock()
    fake_experiment.test_env = MagicMock()
    fake_experiment.policy = MagicMock()
    fake_experiment.max_steps = 5

    def _fake_build_experiment(self, cfg_in, run_dir):  # noqa: ARG001
        captured_cfgs.append(cfg_in)
        return fake_experiment

    fake_recorder = MagicMock()

    monkeypatch.setattr(
        "multi_scenario.adapters.algorithms.benchmarl_base."
        "BenchmarlBaseAdapter.build_experiment",
        _fake_build_experiment,
    )
    monkeypatch.setattr(
        "multi_scenario.adapters.video.recorder.VideoRecorder",
        lambda: fake_recorder,
    )
    # Stub Experiment.reload_from_file so we don't try to load the empty .pt.
    monkeypatch.setattr(
        "benchmarl.experiment.Experiment.reload_from_file",
        lambda _path, experiment_patch=None: fake_experiment,
    )
    # Stub the report builder's downstream calls (load_result / load_run_state /
    # save_report) so we don't need a real result on disk for this unit test.
    fake_storage_adapter = MagicMock()
    fake_storage_adapter.load_config.return_value = cfg
    fake_storage_adapter.load_result.return_value = MagicMock()
    fake_storage_adapter.load_run_state.return_value = MagicMock()
    monkeypatch.setattr(
        "multi_scenario.application.regenerate_videos.LocalStorageAdapter",
        lambda: fake_storage_adapter,
    )
    monkeypatch.setattr(
        "multi_scenario.adapters.storage.report_builder.ReportBuilder.build",
        lambda self, *a, **kw: MagicMock(),
    )

    from multi_scenario.application.regenerate_videos import regenerate_videos

    regenerate_videos(storage_dir)
    assert captured_cfgs, "build_experiment was never called"
    # The cfg passed in must have device='cpu' even though the saved cfg was 'cuda'.
    assert captured_cfgs[0].training.device == "cpu", (
        "regenerate_videos must downgrade device='cuda' → 'cpu' when "
        "torch.cuda.is_available() is False (F8.2.B regression)"
    )


def test_regenerate_videos_keeps_cuda_cfg_on_cuda_host(tmp_path: Path, monkeypatch):
    """Inverse: when CUDA IS available, cfg.training.device='cuda' stays untouched.

    Critical invariant — accidentally downgrading on a CUDA host would slow
    OVH-side video generation (where the regen runs on the same GPU as the
    train job) and break LERO's GPU-bound LLM scenarios.
    """
    # pylint: disable=import-outside-toplevel
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    storage_dir = tmp_path / "run"
    (storage_dir / "input").mkdir(parents=True)
    (storage_dir / "output" / "benchmarl" / "x" / "checkpoints").mkdir(parents=True)
    (
        storage_dir / "output" / "benchmarl" / "x" / "checkpoints" / "checkpoint_1.pt"
    ).write_bytes(b"")

    from multi_scenario.adapters.storage.local import LocalStorageAdapter
    from multi_scenario.domain.models import ExperimentConfig

    cfg = ExperimentConfig.model_validate(
        {
            "experiment": {"id": "demo", "seed": 0},
            "scenario": {
                "type": "discovery",
                "params": {"n_agents": 2, "n_targets": 2, "max_steps": 5},
            },
            "algorithm": {"type": "mappo", "params": {}},
            "training": {
                "max_iters": 1,
                "num_envs": 1,
                "device": "cuda",
                "frames_per_batch": 50,
                "minibatch_size": 25,
                "n_minibatch_iters": 1,
            },
            "evaluation": {"interval_iters": 1, "episodes": 1},
            "runtime": {
                "runner": {"type": "ovh", "params": {}},
                "storage": {"type": "fs", "path": str(storage_dir), "params": {}},
            },
        }
    )
    LocalStorageAdapter().save_config(storage_dir, cfg)

    captured_cfgs = []
    fake_experiment = MagicMock(test_env=MagicMock(), policy=MagicMock(), max_steps=5)

    def _fake_build(self, cfg_in, run_dir):  # noqa: ARG001
        captured_cfgs.append(cfg_in)
        return fake_experiment

    monkeypatch.setattr(
        "multi_scenario.adapters.algorithms.benchmarl_base."
        "BenchmarlBaseAdapter.build_experiment",
        _fake_build,
    )
    monkeypatch.setattr(
        "multi_scenario.adapters.video.recorder.VideoRecorder", lambda: MagicMock()
    )
    monkeypatch.setattr(
        "benchmarl.experiment.Experiment.reload_from_file",
        lambda _path, experiment_patch=None: fake_experiment,
    )
    fake_storage = MagicMock()
    fake_storage.load_config.return_value = cfg
    fake_storage.load_result.return_value = MagicMock()
    fake_storage.load_run_state.return_value = MagicMock()
    monkeypatch.setattr(
        "multi_scenario.application.regenerate_videos.LocalStorageAdapter",
        lambda: fake_storage,
    )
    monkeypatch.setattr(
        "multi_scenario.adapters.storage.report_builder.ReportBuilder.build",
        lambda self, *a, **kw: MagicMock(),
    )

    from multi_scenario.application.regenerate_videos import regenerate_videos

    regenerate_videos(storage_dir)
    assert (
        captured_cfgs[0].training.device == "cuda"
    ), "regenerate_videos must NOT downgrade device on a CUDA-equipped host"
