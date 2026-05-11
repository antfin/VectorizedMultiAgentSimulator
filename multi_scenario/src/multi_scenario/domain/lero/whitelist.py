"""F9.5 — :class:`AllowedKeysDict` + the local/global key whitelists.

Pure-domain enforcement of CTDE fairness: when LERO is configured for
``local`` observation mode, the dict handed to the LLM-generated
``enhance_observation`` function is wrapped in :class:`AllowedKeysDict`,
which raises :class:`FairnessViolation` on any forbidden-key lookup.

Why this lives in the domain layer (not adapters): the whitelist is the
fairness *contract* between the experiment design and the LLM-generated
code. Adapters compile the LLM source and call the function; the domain
layer says "if the function fishes for global state, fail loudly". No
torch / no VMAS — pure Python so the contract stays test-isolated.

Whitelist contents ported verbatim from
``rendezvous_comm/src/lero/meta/fairness.py``. Editing either set
requires a PR-visible change AND a matching update to the prompts'
fairness slot (we don't have a slot yet — the rendezvous_comm-ported
``v2_fewshot_k2_local`` advertises only local keys in its
``initial_user.txt`` body).
"""

from collections.abc import Iterable, Mapping
from typing import Any

from multi_scenario.domain.lero.exceptions import FairnessViolation


#: Keys the LLM is allowed to read in ``local`` observation mode.
#: Mirrors the local sensor surface — own pos/vel/idx, own LiDAR rays,
#: incoming comm messages, and static scenario constants that every
#: agent shares (n_agents, n_targets, etc.).
LOCAL_ALLOWED_KEYS: frozenset[str] = frozenset(
    {
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
    }
)

#: Keys explicitly named in the forbidden list — the typical "I want
#: global oracle" lookups the LLM might attempt. Listed so violations
#: name the offending key cleanly in the exception message.
LOCAL_FORBIDDEN_KEYS: frozenset[str] = frozenset(
    {
        "agents_pos",
        "targets_pos",
        "agents_targets_dists",
        "covered_targets",
        "agents_per_target",
        "all_time_covered",
        "collision_rew",
        "collision_penalty",
        "time_penalty",
    }
)


class AllowedKeysDict(Mapping[str, Any]):
    """Read-only mapping that gates key access by a whitelist.

    Lookups for keys in ``allowed`` return the wrapped value; lookups
    for keys in ``forbidden`` raise :class:`FairnessViolation`; lookups
    for any other key raise a plain :class:`KeyError` (typo case —
    we don't escalate to FairnessViolation so the orchestrator's
    fail-mode taxonomy can distinguish "LLM cheated" from "LLM typo'd").

    The wrapped state dict isn't copied (this is a view), so the
    enforcement runs on every lookup with zero extra memory overhead.
    """

    def __init__(
        self,
        state: Mapping[str, Any],
        *,
        allowed: Iterable[str] = LOCAL_ALLOWED_KEYS,
        forbidden: Iterable[str] = LOCAL_FORBIDDEN_KEYS,
        label: str = "enhance_observation (local)",
    ) -> None:
        self._state = state
        self._allowed: frozenset[str] = frozenset(allowed)
        self._forbidden: frozenset[str] = frozenset(forbidden)
        self._label = label

    def __getitem__(self, key: str) -> Any:
        if key in self._forbidden:
            raise FairnessViolation(
                f"{self._label}: forbidden key {key!r}. Local observation "
                "functions may only read locally-sensed keys; allowed: "
                f"{sorted(self._allowed)}"
            )
        if key in self._allowed:
            if key not in self._state:
                raise KeyError(
                    f"{self._label}: key {key!r} is allowed but not present "
                    "in the current state dict (scenario may lack it — "
                    "e.g. 'lidar_agents' when use_agent_lidar=False)."
                )
            return self._state[key]
        raise KeyError(
            f"{self._label}: unknown key {key!r}. Allowed: " f"{sorted(self._allowed)}."
        )

    def __iter__(self):
        return iter(k for k in self._state if k in self._allowed)

    def __len__(self):
        return sum(1 for k in self._state if k in self._allowed)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._allowed and key in self._state

    def get(self, key: str, default: Any | None = None) -> Any:  # type: ignore[override]
        """``dict.get`` semantics: raise on fairness violations even with default."""
        try:
            return self[key]
        except FairnessViolation:
            raise
        except KeyError:
            return default
