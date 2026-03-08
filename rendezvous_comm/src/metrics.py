"""Metric computation for K-N convergence experiments.

Metric IDs (from experiment plan):
  M1  - Success Rate: fraction of episodes where all targets covered
  M2  - Average Return: mean cumulative reward per episode
  M3  - Steps to Completion: avg steps until done (or max_steps if not done)
  M4  - Collisions per Episode: avg agent-agent collisions
  M5  - Tokens per Episode: total comm tokens used (0 for no-comm baselines)
  M6  - Success vs Budget: success rate at different token budget levels
  M7  - Transfer Score: performance ratio when transferred to unseen config
  M9  - ATE (Ablation Treatment Effect): delta when masking/shuffling comms
  M18 - Delta Return per Message: marginal return improvement per message sent
  M19 - Trigger Calibration: precision/recall of event triggers
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch import Tensor


@dataclass
class EpisodeMetrics:
    """Metrics collected during a single episode across all envs.

    M1 (success_rate) works in two modes:
      - targets_respawn=False: done() fires when all targets covered → binary
      - targets_respawn=True: done() never fires, so we track the cumulative
        number of targets covered per step and report the coverage rate.
    """

    # Per-step accumulators (call update_step each step)
    cumulative_reward: Optional[Tensor] = None   # (n_envs,)
    step_count: Optional[Tensor] = None          # (n_envs,)
    collision_count: Optional[Tensor] = None     # (n_envs,)
    done_at_step: Optional[Tensor] = None        # (n_envs,) step when done
    total_tokens: Optional[Tensor] = None        # (n_envs,)
    is_done: Optional[Tensor] = None             # (n_envs,) bool
    total_targets_covered: Optional[Tensor] = None  # (n_envs,) cumulative

    def init(self, n_envs: int, device: str = "cpu"):
        self.cumulative_reward = torch.zeros(n_envs, device=device)
        self.step_count = torch.zeros(n_envs, device=device, dtype=torch.long)
        self.collision_count = torch.zeros(n_envs, device=device)
        self.done_at_step = torch.full(
            (n_envs,), -1, device=device, dtype=torch.long
        )
        self.total_tokens = torch.zeros(n_envs, device=device)
        self.is_done = torch.zeros(n_envs, device=device, dtype=torch.bool)
        self.total_targets_covered = torch.zeros(n_envs, device=device)
        return self

    def update_step(
        self,
        rewards: List[Tensor],
        dones: Tensor,
        info: List[Dict[str, Tensor]],
        step: int,
        tokens_this_step: int = 0,
    ):
        """Update metrics from one env.step() result.

        Args:
            rewards: list of (n_envs,) tensors, one per agent
            dones: (n_envs,) bool tensor
            info: list of info dicts, one per agent
            step: current step number
            tokens_this_step: communication tokens used this step
        """
        # Sum rewards across agents
        total_rew = torch.stack(rewards, dim=-1).sum(dim=-1)
        self.cumulative_reward += total_rew
        self.step_count += 1

        # Collisions: sum across agents
        for agent_info in info:
            if "collision_rew" in agent_info:
                # collision_rew is negative when collision occurs
                self.collision_count += (agent_info["collision_rew"] < 0).float()

        # Targets covered this step (from any agent's info — all share same value)
        if info and "targets_covered" in info[0]:
            self.total_targets_covered += info[0]["targets_covered"]

        # Track completion step
        newly_done = dones & ~self.is_done
        self.done_at_step[newly_done] = step
        self.is_done |= dones

        # Tokens
        self.total_tokens += tokens_this_step

    def compute(self, max_steps: int) -> Dict[str, float]:
        """Compute final scalar metrics across all envs."""
        n = self.cumulative_reward.shape[0]

        # M1: Success Rate
        # If done() ever fired (targets_respawn=False), use that.
        # Otherwise, report coverage rate (targets covered per step).
        success_rate = self.is_done.float().mean().item()

        # M1b: Average targets covered per step (meaningful when respawn=True)
        steps = self.step_count.float().clamp(min=1)
        avg_targets_covered_per_step = (
            self.total_targets_covered / steps
        ).mean().item()

        # M2: Average Return
        avg_return = self.cumulative_reward.mean().item()

        # M3: Steps to Completion (use max_steps for incomplete episodes)
        completion_steps = self.done_at_step.clone().float()
        completion_steps[completion_steps < 0] = max_steps
        avg_steps = completion_steps.mean().item()

        # M4: Collisions per Episode
        avg_collisions = self.collision_count.mean().item()

        # M5: Tokens per Episode
        avg_tokens = self.total_tokens.mean().item()

        return {
            "M1_success_rate": success_rate,
            "M1b_avg_targets_covered_per_step": avg_targets_covered_per_step,
            "M2_avg_return": avg_return,
            "M3_avg_steps": avg_steps,
            "M4_avg_collisions": avg_collisions,
            "M5_avg_tokens": avg_tokens,
            "n_envs": n,
        }


def compute_m6_budget_frontier(
    results_at_budgets: Dict[float, Dict[str, float]],
) -> Dict[float, float]:
    """M6: Success rate at each token budget level.

    Args:
        results_at_budgets: {budget_fraction: metrics_dict}
    Returns:
        {budget_fraction: success_rate}
    """
    return {
        budget: metrics["M1_success_rate"]
        for budget, metrics in sorted(results_at_budgets.items())
    }


def compute_m7_transfer(
    source_metrics: Dict[str, float],
    target_metrics: Dict[str, float],
) -> float:
    """M7: Transfer Score = target_success / source_success."""
    src = source_metrics["M1_success_rate"]
    if src == 0:
        return 0.0
    return target_metrics["M1_success_rate"] / src


def compute_m9_ate(
    baseline_metrics: Dict[str, float],
    ablated_metrics: Dict[str, float],
) -> Dict[str, float]:
    """M9: Ablation Treatment Effect (delta when masking/shuffling comms)."""
    return {
        "ate_success": (
            baseline_metrics["M1_success_rate"]
            - ablated_metrics["M1_success_rate"]
        ),
        "ate_return": (
            baseline_metrics["M2_avg_return"]
            - ablated_metrics["M2_avg_return"]
        ),
    }


def compute_m18_delta_return_per_msg(
    metrics_with_comm: Dict[str, float],
    metrics_no_comm: Dict[str, float],
) -> float:
    """M18: Marginal return improvement per message sent."""
    tokens = metrics_with_comm["M5_avg_tokens"]
    if tokens == 0:
        return 0.0
    delta_return = (
        metrics_with_comm["M2_avg_return"] - metrics_no_comm["M2_avg_return"]
    )
    return delta_return / tokens
