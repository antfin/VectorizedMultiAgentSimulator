"""F7.7.A4 — runner-provisioning dispatch tests.

The dispatch is the runtime half of the runner-device matrix:
``application.factories.RunnerSpec.supported_devices`` is the static
"can this runner KIND accept this device?" check (validated at config
parse time); :func:`check_runner_provisioning` is the "is the actual
host capable of this combo right now?" check (a runtime probe).

Adding a new runner means:
    1. ``_RUNNERS["new_runner"] = RunnerSpec(...)`` in factories.py
    2. ``PROVISION_CHECKS["new_runner"] = _check_new_runner_provision`` here
No preflight code edits, no Submit-page edits.
"""

# pylint: disable=missing-function-docstring,redefined-outer-name

from types import SimpleNamespace

import pytest

from multi_scenario.application.runner_provisioning import (
    check_runner_provisioning,
    PROVISION_CHECKS,
)


# ── Local runner provisioning ────────────────────────────────────────


def test_local_cpu_passes_without_torch_check():
    """``local + cpu`` is unconditional PASS — no per-host probe needed."""
    status, detail = check_runner_provisioning("local", "cpu")
    assert status is True
    assert "cpu" in detail


def test_local_cuda_pass_when_cuda_available(monkeypatch):
    """When ``torch.cuda.is_available()=True``, local+cuda PASSES."""
    import torch  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    status, detail = check_runner_provisioning("local", "cuda")
    assert status is True
    assert "2 CUDA device" in detail


def test_local_cuda_fail_when_cuda_missing(monkeypatch):
    """``local + cuda`` on a CUDA-less host → FAIL with actionable message."""
    import torch  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    status, detail = check_runner_provisioning("local", "cuda")
    assert status is False
    assert "torch.cuda.is_available()=False" in detail
    assert "set device=cpu" in detail or "submit to OVH" in detail


# ── OVH runner provisioning ──────────────────────────────────────────


def _ovh_cfg(flavor: str = "ai1-1-gpu", n_gpu: int = 1):
    """Minimal duck-typed OVH config; the probe only reads .flavor and .n_gpu."""
    return SimpleNamespace(flavor=flavor, n_gpu=n_gpu)


def test_ovh_no_config_fails():
    status, detail = check_runner_provisioning("ovh", "cuda", ovh_cfg=None)
    assert status is False
    assert "no OVH config" in detail


@pytest.mark.parametrize(
    "flavor",
    ["ai1-1-gpu", "a100-1-gpu", "h100-1-gpu", "l40s-1-gpu", "l4-1-gpu", "a10-1-gpu"],
)
def test_ovh_cuda_pass_with_gpu_flavors(flavor):
    """All known GPU-flavor prefixes are accepted with device=cuda."""
    status, detail = check_runner_provisioning(
        "ovh", "cuda", ovh_cfg=_ovh_cfg(flavor=flavor)
    )
    assert status is True
    assert flavor in detail


def test_ovh_cuda_fail_on_cpu_only_flavor():
    status, detail = check_runner_provisioning(
        "ovh",
        "cuda",
        ovh_cfg=_ovh_cfg(flavor="ai1-1-cpu"),
    )
    assert status is False
    assert "ai1-1-cpu" in detail
    assert "GPU flavor" in detail


def test_ovh_cuda_fail_when_n_gpu_zero():
    status, detail = check_runner_provisioning(
        "ovh",
        "cuda",
        ovh_cfg=_ovh_cfg(flavor="ai1-1-gpu", n_gpu=0),
    )
    assert status is False
    assert "n_gpu=0" in detail


def test_ovh_cpu_on_gpu_flavor_passes_with_warning():
    """``ovh + cpu + gpu_flavor`` is allowed but the detail carries a ⚠ warning."""
    status, detail = check_runner_provisioning(
        "ovh",
        "cpu",
        ovh_cfg=_ovh_cfg(flavor="ai1-1-gpu"),
    )
    assert status is True
    assert "⚠" in detail
    assert "GPU but won't use it" in detail


def test_ovh_cpu_on_cpu_flavor_passes_cleanly():
    """No warning when both runner and device agree on CPU."""
    status, detail = check_runner_provisioning(
        "ovh",
        "cpu",
        ovh_cfg=_ovh_cfg(flavor="ai1-1-cpu"),
    )
    assert status is True
    assert "⚠" not in detail


# ── Extension contract — adding a new runner is one entry ────────────


def test_unknown_runner_raises_keyerror():
    """``check_runner_provisioning`` raises KeyError for unregistered runners.

    That's a programming error (the schema would have caught it earlier);
    the test pins the contract so future code can't silently no-op.
    """
    with pytest.raises(KeyError):
        check_runner_provisioning("slurm", "cpu")


def test_provision_checks_registry_matches_runners():
    """Every entry in PROVISION_CHECKS must have a matching ``RunnerSpec``."""
    # pylint: disable=import-outside-toplevel
    from multi_scenario.application.factories import available_runners

    for runner_name in available_runners():
        assert (
            runner_name in PROVISION_CHECKS
        ), f"runner {runner_name!r} is registered but has no provisioning probe"
    for probe_name in PROVISION_CHECKS:
        assert (
            probe_name in available_runners()
        ), f"provisioning probe {probe_name!r} is registered but no RunnerSpec exists"
