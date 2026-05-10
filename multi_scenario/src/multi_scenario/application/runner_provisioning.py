"""Per-runner provisioning checks (F7.7.A4 architecture).

Static device-runner compatibility (e.g. "ovh runner accepts cpu/cuda")
is enforced at config-validation time via :data:`RunnerSpec.supported_devices`
in ``application.factories``.

This module covers the **runtime** half: "is the actual host capable of the
device the user picked?" Each runner registers a probe via
:data:`PROVISION_CHECKS`. Preflight calls :func:`check_runner_provisioning`
once and gets back a ``(ok, detail)`` tuple; new runners plug in by adding
one entry to the dict — no preflight code changes.

Examples of provisioning concerns by runner:

- ``local + cuda`` → ``torch.cuda.is_available()`` on this machine?
- ``ovh + cuda`` → ``OvhJobConfig.flavor`` is a GPU flavor + ``n_gpu ≥ 1``?
- ``ovh + cpu`` → harmless mismatch (ok, but detail carries a ⚠ warning;
  user is paying for a GPU node and not using it).

Return shape is intentionally plain ``(bool, str)`` rather than a domain
enum so this module stays a pure-application leaf — the frontend's preflight
maps ``True`` → ``True`` (with the detail text potentially
carrying a ⚠ marker) and ``False`` → ``False``. Keeps the hex
arrow one-way (application doesn't depend on frontend).
"""

from typing import Any, Callable

#: Probe signature: ``(device, **ctx) -> (ok, detail)``.
#: ``ctx`` carries runner-specific extras (e.g. ``ovh_cfg`` for the ovh runner).
#: ``ok=True`` means the combo is valid (may still carry a ⚠ warning in detail);
#: ``ok=False`` means the combo would crash the run.
ProvisionCheck = Callable[..., tuple[bool, str]]


#: ``ovhai capabilities flavor list`` GPU-flavor prefixes (verified live 2026-05-09).
_OVH_GPU_FLAVOR_PREFIXES: tuple[str, ...] = (
    "ai1-1-gpu",
    "a100-",
    "a10-",
    "h100-",
    "l40s-",
    "l4-",
)


def _is_gpu_flavor(flavor: str) -> bool:
    """True when ``flavor`` matches one of the OVH GPU-bearing flavor families."""
    return any(flavor.startswith(prefix) for prefix in _OVH_GPU_FLAVOR_PREFIXES)


def _check_local_provision(device: str, **_ctx: Any) -> tuple[bool, str]:
    """``local`` provisioning: when ``device=cuda``, host must have CUDA."""
    if device != "cuda":
        return True, f"local + {device} (no provisioning needed)"
    # pylint: disable=import-outside-toplevel
    import torch

    if not torch.cuda.is_available():
        return (
            False,
            "device=cuda but torch.cuda.is_available()=False on this host. "
            "Run on a CUDA-capable machine, set device=cpu, or submit to OVH.",
        )
    return True, f"{torch.cuda.device_count()} CUDA device(s) on host"


def _check_ovh_provision(device: str, **ctx: Any) -> tuple[bool, str]:
    """``ovh`` provisioning: device must match the OvhJobConfig flavor + n_gpu."""
    ovh_cfg = ctx.get("ovh_cfg")
    if ovh_cfg is None:
        return False, "no OVH config loaded"
    flavor_is_gpu = _is_gpu_flavor(ovh_cfg.flavor)
    if device == "cuda":
        if not flavor_is_gpu:
            return (
                False,
                f"device=cuda but configs/ovh.yaml flavor={ovh_cfg.flavor!r} "
                "is CPU-only — set flavor to ai1-1-gpu (or another GPU flavor).",
            )
        if ovh_cfg.n_gpu < 1:
            return (
                False,
                f"device=cuda but n_gpu={ovh_cfg.n_gpu}; set n_gpu ≥ 1.",
            )
        return True, f"flavor={ovh_cfg.flavor}, n_gpu={ovh_cfg.n_gpu}"
    # device=cpu on OVH — harmless but wasteful when flavor is GPU.
    if flavor_is_gpu:
        return (
            True,
            f"⚠ device=cpu on GPU flavor {ovh_cfg.flavor} — billed for GPU "
            "but won't use it. Consider device=cuda or a CPU flavor.",
        )
    return True, f"flavor={ovh_cfg.flavor} (CPU) matches device=cpu"


#: Registry — extend with one entry per new runner. The signature is loose
#: (``**ctx``) so each runner can declare what kwargs it needs without a
#: shared base type. Callers know which kwargs to pass based on the runner.
PROVISION_CHECKS: dict[str, ProvisionCheck] = {
    "local": _check_local_provision,
    "ovh": _check_ovh_provision,
}


def check_runner_provisioning(
    runner_type: str, device: str, **ctx: Any
) -> tuple[bool, str]:
    """Dispatch to the registered provisioning probe for ``runner_type``.

    Raises ``KeyError`` if the runner isn't registered — that's a programming
    error, not a user error; the schema would have caught it earlier via
    ``validate_known_types``.
    """
    return PROVISION_CHECKS[runner_type](device, **ctx)
