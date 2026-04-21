"""Runtime fairness enforcement for LLM-generated observation functions.

The LLM's ``enhance_observation`` receives a state dict. In obs-mode
``local`` (the CTDE-fair default for LERO-MP), that dict must only
expose keys an agent could actually observe from its own sensors.

Historically we relied on the prompt's state-schema block to tell the
LLM which keys are available. This is not enforcement — the LLM can
ignore the prompt. ``AllowedKeysDict`` converts silent cheating into a
loud ``FairnessViolation`` the first time a forbidden key is read, so
the outer loop's fail-mode taxonomy (meta.failmode) can detect and
respond to it.

See docs/lero_metaprompt_plan.md §4 for the full policy.
"""

from typing import Any, Iterable, Mapping, Optional, Set


# Kept intentionally small and explicit. Changing either set requires a
# PR-visible edit and a matching update in prompts/*/fairness.txt.

LOCAL_ALLOWED_KEYS: frozenset = frozenset({
    # own sensors / own state
    "agent_pos",
    "agent_vel",
    "agent_idx",
    "lidar_targets",
    "lidar_agents",
    "messages",
    # static scenario constants (fine — same for every agent)
    "n_agents",
    "n_targets",
    "covering_range",
    "agents_per_target_required",
})

# Forbidden in local mode. Listed so violations name the key explicitly.
LOCAL_FORBIDDEN_KEYS: frozenset = frozenset({
    "agents_pos",
    "targets_pos",
    "agents_targets_dists",
    "covered_targets",
    "agents_per_target",
    "all_time_covered",
    "collision_rew",
    "collision_penalty",
    "time_penalty",
})


class FairnessViolation(KeyError):
    """Raised when an LLM-generated observation function reads a key
    that is not locally observable by the agent.

    Subclasses ``KeyError`` so existing ``state[key]`` patterns still
    get a KeyError-shaped exception, but downstream code can ``except
    FairnessViolation`` to tag this distinctly from a missing-key bug.
    """


class AllowedKeysDict(Mapping):
    """Read-only mapping that gates key access by a whitelist.

    Lookups for keys in ``allowed`` return the wrapped value. Lookups
    for keys in ``forbidden`` raise ``FairnessViolation``. Lookups for
    any other key raise a normal ``KeyError`` (e.g. LLM typo).
    """

    def __init__(
        self,
        state: Mapping[str, Any],
        allowed: Iterable[str] = LOCAL_ALLOWED_KEYS,
        forbidden: Iterable[str] = LOCAL_FORBIDDEN_KEYS,
        label: str = "enhance_observation (local)",
    ) -> None:
        self._state = state
        self._allowed: Set[str] = set(allowed)
        self._forbidden: Set[str] = set(forbidden)
        self._label = label

    def __getitem__(self, key: str) -> Any:
        if key in self._forbidden:
            raise FairnessViolation(
                f"{self._label}: forbidden key '{key}'. Local observation "
                f"functions may only read locally-sensed keys; see "
                f"prompts/*/fairness.txt for the full list."
            )
        if key in self._allowed:
            if key not in self._state:
                raise KeyError(
                    f"{self._label}: key '{key}' is allowed but not present "
                    f"in the current state dict (scenario may lack it — e.g. "
                    f"'lidar_agents' when use_agent_lidar=False)."
                )
            return self._state[key]
        raise KeyError(
            f"{self._label}: unknown key '{key}'. Allowed: "
            f"{sorted(self._allowed)}."
        )

    def __iter__(self):
        return iter(k for k in self._state if k in self._allowed)

    def __len__(self):
        return sum(1 for k in self._state if k in self._allowed)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._allowed and key in self._state

    def get(self, key: str, default: Optional[Any] = None) -> Any:  # type: ignore[override]
        try:
            return self[key]
        except FairnessViolation:
            raise
        except KeyError:
            return default
