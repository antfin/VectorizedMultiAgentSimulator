"""VMAS navigation scenario adapter — implements the Scenario port.

Success (M1) is the binary "all agents reached their goals during the
episode" — read from BenchMARL's universal ``episode_terminated`` rollout
key (populated by ``BenchmarlBaseAdapter._extract_terminated``). Mirrors
discovery's all-or-nothing success semantics.

M6 coverage progress is stubbed for now; sharpening it (e.g. per-agent
on-goal fraction at episode end) needs per-agent position/goal extraction
into the rollout dict — out of scope for F4.1.

Rollout-shape contract: relies on the universal ``episode_terminated``
key from the F4.1 base adapter extension; no scenario-specific rollout
fields needed.
"""

from typing import Any

import torch
import vmas

from multi_scenario.domain.models import ScenarioSection


class VmasNavigationAdapter:
    """Scenario adapter wrapping ``vmas.make_env(scenario='navigation', ...)``."""

    # M6/M8 stub-return None for F4.1; the unused args are required by the
    # Scenario Protocol. Drop this disable when M6 lands.
    # pylint: disable=unused-argument

    name = "navigation"

    def make_env(self, cfg: ScenarioSection, num_envs: int, seed: int) -> Any:
        """Build the VMAS navigation env; cfg.params override default_params."""
        params = {**self.default_params(), **cfg.params}
        return vmas.make_env(
            scenario="navigation",
            num_envs=num_envs,
            seed=seed,
            **params,
        )

    def default_params(self) -> dict[str, Any]:
        """Baseline navigation params — small env, partial-observation goals."""
        return {
            "n_agents": 4,
            "agents_with_same_goal": 1,
            "observe_all_goals": False,
            "shared_rew": True,
        }

    def has_comm(self) -> bool:
        """Navigation has no comm channel; M5 is N/A."""
        return False

    def success_predicate(self, rollout: Any) -> Any:
        """M1: True per episode if the env terminated naturally (= all on goal)."""
        terminated = rollout.get("episode_terminated")
        if terminated is None:
            return None
        return torch.as_tensor(terminated).bool()

    def coverage_progress(self, rollout: Any) -> Any | None:
        """M6: stub — sharper navigation coverage lands in a later feature."""
        return None

    def utilization_predicate(self, state: Any) -> Any:
        """M8: per-(env, agent) utilization — stub, lands in a later feature."""
        return None
