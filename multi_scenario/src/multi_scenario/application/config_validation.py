"""Registry-aware ExperimentConfig validation (F7.7.D1).

Lives in the application layer because the ``available_*`` registries are
defined in :mod:`multi_scenario.application.factories`. Importing them from
``domain.models`` would invert the hex dependency arrow (domain depends on
application) and create a static-analysis cycle even with lazy imports.

Schema-only validation stays in :class:`ExperimentConfig`; registry checks
land here so callers that need both invoke them in sequence (or use
:func:`multi_scenario.application.config_loader.load_experiment_config`).
"""

from multi_scenario.application.factories import (
    available_algorithms,
    available_runners,
    available_scenarios,
    available_storages,
    runner_spec,
)
from multi_scenario.domain.models import ExperimentConfig


def validate_known_types(cfg: ExperimentConfig) -> None:
    """Raise ``ValueError`` if any ``type`` field doesn't resolve to a registered adapter.

    Specifically checks ``scenario.type``, ``algorithm.type``, and (when
    ``cfg.runtime`` is set) ``runtime.runner.type`` + ``runtime.storage.type``,
    plus the cross-field invariant that ``cfg.training.device`` is one of
    ``runner_spec(runtime.runner.type).supported_devices``. Each error message
    lists the registered names so the user can spot a typo in one glance.
    """
    if cfg.scenario.type not in available_scenarios():
        raise ValueError(
            f"scenario.type {cfg.scenario.type!r} is not registered; "
            f"known: {available_scenarios()}"
        )
    if cfg.algorithm.type not in available_algorithms():
        raise ValueError(
            f"algorithm.type {cfg.algorithm.type!r} is not registered; "
            f"known: {available_algorithms()}"
        )
    if cfg.runtime is not None:
        if cfg.runtime.runner.type not in available_runners():
            raise ValueError(
                f"runtime.runner.type {cfg.runtime.runner.type!r} is not "
                f"registered; known: {available_runners()}"
            )
        if cfg.runtime.storage.type not in available_storages():
            raise ValueError(
                f"runtime.storage.type {cfg.runtime.storage.type!r} is not "
                f"registered; known: {available_storages()}"
            )
        # Static device-runner compatibility check (F7.7.A4 architecture).
        # Per-host provisioning (CUDA actually available on this Mac, GPU
        # flavor on configs/ovh.yaml, …) is a separate runtime concern —
        # see the preflight probes that consult runner_provisioning.
        spec = runner_spec(cfg.runtime.runner.type)
        if cfg.training.device not in spec.supported_devices:
            raise ValueError(
                f"training.device {cfg.training.device!r} is not supported by "
                f"runtime.runner.type={cfg.runtime.runner.type!r}; "
                f"supported: {sorted(spec.supported_devices)}"
            )
