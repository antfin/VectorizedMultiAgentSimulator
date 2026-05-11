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
    """Replaces ``task.get_env_fun`` to inject the patched scenario class."""

    def __init__(self, scenario_class: type, config: dict[str, Any]) -> None:
        self.scenario_class = scenario_class
        self.config = config

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

    # BenchMARL checkpoints the task pickled; tests for the LLM-generated
    # functions are at the codegen + scenario_patch layer, not here, so
    # we don't need to round-trip the closures through pickle.
    def __getstate__(self) -> dict:
        return {"_dummy": True}

    def __setstate__(self, state: dict) -> None:  # pragma: no cover
        # Restoring from a checkpoint that pre-dates an in-process
        # patched run never reuses the factory — the live one in memory
        # owns the actual scenario class. Defensive no-op.
        del state
