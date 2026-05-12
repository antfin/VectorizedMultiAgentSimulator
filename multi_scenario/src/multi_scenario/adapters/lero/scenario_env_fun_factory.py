"""F9.6.e — Picklable env-factory that builds a fresh patched scenario per worker.

BenchMARL's task object exposes ``get_env_fun`` — a callable the runtime
asks for an env-builder when spinning up workers. The default reads the
scenario by name (``"discovery"`` → ``vmas.scenarios.discovery``);
LERO needs to swap that for our ``PatchedDiscoveryScenario`` class so
the LLM-generated reward / observation methods participate in training.

This factory replaces ``task.get_env_fun`` and constructs a ``VmasEnv``
backed by ``scenario_class()`` per worker. Each env gets a NEW
scenario instance (BenchMARL's normal pattern) so concurrent workers
don't share mutable scenario state.

Pickle-safety: BenchMARL serialises the task object to disk (for the
``config.pkl`` checkpoint), so any captured state has to survive
pickling. We stash only the class + config dict and provide a minimal
``__getstate__`` / ``__setstate__`` pair — same pattern rendezvous_comm
shipped.
"""

from typing import Any, Callable


class ScenarioEnvFunFactory:
    """Replaces ``task.get_env_fun`` to inject the patched scenario class.

    Pickle-safety (Phase 2 fix): ``make_patched_discovery_class`` returns
    a LOCAL class (defined inside a function), which Python's pickle
    protocol cannot serialise by reference. Storing the source strings
    + reconstruction kwargs and rebuilding the class on unpickle
    sidesteps this — the pickled bytes are all primitives.

    ``patched_kwargs`` carries the arguments to ``make_patched_discovery_class``
    so a worker that unpickles this factory can rebuild the same class.
    When None (factory built for a non-patched scenario), the unpickle
    path falls back to whatever scenario_class was supplied to ``__init__``
    — useful for tests that mock the factory.
    """

    def __init__(
        self,
        scenario_class: type,
        config: dict[str, Any],
        patched_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.scenario_class = scenario_class
        self.config = config
        # Reconstruction kwargs; only set by ``_build_patched_experiment``.
        self._patched_kwargs = patched_kwargs

    def __call__(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: int,
        device: Any,
    ) -> Callable[[], Any]:
        scenario_class = self.scenario_class
        config = self.config

        def make_env() -> Any:
            # pylint: disable=import-outside-toplevel
            from torchrl.envs.libs.vmas import VmasEnv

            return VmasEnv(
                scenario=scenario_class(),
                num_envs=num_envs,
                continuous_actions=continuous_actions,
                seed=seed,
                device=device,
                categorical_actions=True,
                clamp_actions=True,
                **config,
            )

        return make_env

    # BenchMARL pickles the task object in two paths:
    #
    # 1. **Checkpoint save** (``config.pkl`` next to ``checkpoints/``) —
    #    needed so ``Experiment.reload_from_file`` can reconstruct the
    #    full env on the SAME machine for ``multi-scenario eval`` or
    #    post-hoc rollouts.
    # 2. **Worker fork on Linux** — fork inherits memory so no pickle,
    #    but **spawn on macOS / Windows** does pickle. A future BenchMARL
    #    change to use spawn-by-default would break inner-loop training
    #    silently if the factory doesn't survive a round-trip.
    #
    # The earlier dummy ``__getstate__`` survived only because Phase 5a
    # ran on Linux (fork) and never exercised the reload path. Phase 5a
    # post-hoc eval needed a monkey-patch (Phase 5a issue #5). This
    # implementation persists the state so both paths work.
    def __getstate__(self) -> dict[str, Any]:
        """Persist ``config`` + ``patched_kwargs`` (not the class itself).

        Python's pickle protocol cannot serialise local classes, and
        :func:`make_patched_discovery_class` returns one. We store the
        primitive args required to rebuild it; ``__setstate__`` does the
        rebuild. ``scenario_class`` is intentionally omitted from the
        state dict.

        Defensive ``getattr`` on ``_patched_kwargs`` covers legacy
        factory instances (pre-Phase-2 pickle) that ``__setstate__``-
        monkey-patched their way to a usable scenario_class without
        setting the kwargs attribute.
        """
        return {
            "config": self.config,
            "patched_kwargs": getattr(self, "_patched_kwargs", None),
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the factory; rebuild ``scenario_class`` from kwargs."""
        self.config = state["config"]
        self._patched_kwargs = state.get("patched_kwargs")
        if self._patched_kwargs is not None:
            # pylint: disable=import-outside-toplevel
            from multi_scenario.adapters.scenarios.patched_discovery import (
                make_patched_discovery_class,
            )

            self.scenario_class = make_patched_discovery_class(**self._patched_kwargs)
        # else: scenario_class stays unset; CLI-side hot-patches (e.g.
        # Phase 7 eval CLI's _install_patched_factory_for_lero_reload)
        # may set it externally for legacy runs.
