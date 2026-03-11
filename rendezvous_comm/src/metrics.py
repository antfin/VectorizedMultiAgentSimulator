"""Metric computation for K-N convergence experiments.

Core metrics (computed during evaluation):
  M1  - Success Rate: fraction of episodes where all targets covered
  M2  - Avg Episodic Return: total team reward per episode
  M3  - Steps to Completion: avg timesteps to cover all targets
  M4  - Collisions / Episode: mean inter-robot collisions
  M5  - Tokens / Episode: total comm tokens exchanged (0 for no-comm)
  M6  - Coverage Progress: fraction of targets covered by episode end
  M7  - Sample Efficiency: frames to reach 80% of final eval reward
  M8  - Agent Utilization: balance of per-agent covering contributions
  M9  - Spatial Spread: mean pairwise agent distance (exploration vs clumping)

Field geometry:
  World is 2.0 x 2.0 (x_semidim=1.0, y_semidim=1.0, coords from -1 to +1).
  Agent radius = 0.05, target radius = 0.05, covering_range = 0.25.
  Diagonal ~ 2.83. Agent terminal velocity ~ 0.4 units/step (drag=0.25).

Cross-experiment analysis utilities (unnumbered, used for ER2+):
  compute_budget_frontier  - Success rate at each token budget level
  compute_transfer_score   - Normalised success across transfer conditions
  compute_ate              - Ablation Treatment Effect (mask/shuffle comms)
  compute_delta_return_per_msg - Marginal return improvement per message
"""
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import Tensor


@dataclass
class EpisodeMetrics:
    """Metrics collected during a single episode across all envs.

    Tracks M1-M6, M8-M9 per step. M7 is computed post-hoc from CSV logs.
    Call init() first, then update_step() each step, then compute().
    """

    # Per-step accumulators
    cumulative_reward: Optional[Tensor] = None   # (n_envs,)
    step_count: Optional[Tensor] = None          # (n_envs,)
    collision_count: Optional[Tensor] = None     # (n_envs,)
    done_at_step: Optional[Tensor] = None        # (n_envs,) step when done
    total_tokens: Optional[Tensor] = None        # (n_envs,)
    is_done: Optional[Tensor] = None             # (n_envs,) bool
    targets_covered_total: Optional[Tensor] = None  # (n_envs,) cumulative
    per_agent_covering: Optional[Tensor] = None  # (n_envs, n_agents)
    pairwise_dist_sum: Optional[Tensor] = None   # (n_envs,)
    n_targets: int = 7
    n_agents: int = 4

    def init(
        self, n_envs: int, device: str = "cpu",
        n_targets: int = 7, n_agents: int = 4,
    ):
        self.cumulative_reward = torch.zeros(n_envs, device=device)
        self.step_count = torch.zeros(n_envs, device=device, dtype=torch.long)
        self.collision_count = torch.zeros(n_envs, device=device)
        self.done_at_step = torch.full(
            (n_envs,), -1, device=device, dtype=torch.long
        )
        self.total_tokens = torch.zeros(n_envs, device=device)
        self.is_done = torch.zeros(n_envs, device=device, dtype=torch.bool)
        self.targets_covered_total = torch.zeros(n_envs, device=device)
        self.per_agent_covering = torch.zeros(
            n_envs, n_agents, device=device
        )
        self.pairwise_dist_sum = torch.zeros(n_envs, device=device)
        self.n_targets = n_targets
        self.n_agents = n_agents
        return self

    def update_step(
        self,
        rewards: List[Tensor],
        dones: Tensor,
        info: List[Dict[str, Tensor]],
        step: int,
        tokens_this_step: int = 0,
        agent_positions: Optional[Tensor] = None,
    ):
        """Update metrics from one env.step() result.

        Args:
            rewards: list of (n_envs,) tensors, one per agent
            dones: (n_envs,) bool tensor
            info: list of info dicts, one per agent
            step: current step number
            tokens_this_step: communication tokens used this step
            agent_positions: (n_envs, n_agents, 2) tensor of agent positions
        """
        # Sum rewards across agents
        total_rew = torch.stack(rewards, dim=-1).sum(dim=-1)
        self.cumulative_reward += total_rew
        self.step_count += 1

        # M4: Collisions — sum across agents
        for agent_info in info:
            if "collision_rew" in agent_info:
                self.collision_count += (
                    agent_info["collision_rew"] < 0
                ).float()

        # M6: Track cumulative targets covered.
        # With targets_respawn=False, each target is covered exactly once
        # (then moved off-screen), so summing gives total unique targets.
        if info and "targets_covered" in info[0]:
            self.targets_covered_total += info[0]["targets_covered"].float()

        # M8: Per-agent covering contribution
        for i, agent_info in enumerate(info):
            if "covering_reward" in agent_info and i < self.n_agents:
                self.per_agent_covering[:, i] += (
                    agent_info["covering_reward"] > 0
                ).float()

        # M9: Spatial spread — mean pairwise distance between agents
        if agent_positions is not None and agent_positions.shape[1] >= 2:
            # agent_positions: (n_envs, n_agents, 2)
            dists = torch.cdist(agent_positions, agent_positions)
            # Upper triangle mean (exclude diagonal)
            n_ag = agent_positions.shape[1]
            mask = torch.triu(torch.ones(n_ag, n_ag, device=dists.device), diagonal=1).bool()
            n_pairs = mask.sum().item()
            if n_pairs > 0:
                pairwise_mean = dists[:, mask].mean(dim=-1)
                self.pairwise_dist_sum += pairwise_mean

        # Track completion step.
        # Use cumulative target coverage to determine true task completion,
        # NOT the done signal from VMAS which includes time-limit truncation.
        task_done = self.targets_covered_total >= self.n_targets
        newly_done = task_done & ~self.is_done
        self.done_at_step[newly_done] = step
        self.is_done = task_done

        # M5: Tokens
        self.total_tokens += tokens_this_step

    def compute(self, max_steps: int) -> Dict[str, float]:
        """Compute final scalar metrics across all envs."""
        n = self.cumulative_reward.shape[0]

        # M1: Success Rate
        success_rate = self.is_done.float().mean().item()

        # M2: Average Return
        avg_return = self.cumulative_reward.mean().item()

        # M3: Steps to Completion (max_steps for incomplete episodes)
        completion_steps = self.done_at_step.clone().float()
        completion_steps[completion_steps < 0] = max_steps
        avg_steps = completion_steps.mean().item()

        # M4: Collisions per Episode
        avg_collisions = self.collision_count.mean().item()

        # M5: Tokens per Episode
        avg_tokens = self.total_tokens.mean().item()

        # M6: Coverage Progress
        targets_covered = self.targets_covered_total.clamp(
            max=self.n_targets
        )
        coverage_progress = (targets_covered / self.n_targets).mean().item()

        # M8: Agent Utilization (coefficient of variation of covering counts)
        # CV = std/mean; lower = more balanced; 0 = perfectly equal
        mean_cov = self.per_agent_covering.mean(dim=-1)
        std_cov = self.per_agent_covering.std(dim=-1)
        cv = torch.where(
            mean_cov > 0, std_cov / mean_cov,
            torch.zeros_like(mean_cov),
        )
        agent_utilization = cv.mean().item()

        # M9: Spatial Spread (mean pairwise distance, averaged over steps)
        steps = self.step_count.float().clamp(min=1)
        spatial_spread = (self.pairwise_dist_sum / steps).mean().item()

        return {
            "M1_success_rate": success_rate,
            "M2_avg_return": avg_return,
            "M3_avg_steps": avg_steps,
            "M4_avg_collisions": avg_collisions,
            "M5_avg_tokens": avg_tokens,
            "M6_coverage_progress": coverage_progress,
            "M8_agent_utilization": agent_utilization,
            "M9_spatial_spread": spatial_spread,
            "n_envs": n,
        }


def compute_m7_sample_efficiency(
    run_dir: Path,
    threshold_fraction: float = 0.80,
) -> Optional[float]:
    """M7: Frames to reach threshold_fraction of final eval reward.

    Reads BenchMARL's eval_reward_episode_reward_mean.csv and
    counters_total_frames.csv from the run's benchmarl output directory.
    Returns the frame count at which eval reward first exceeds
    threshold_fraction * final_reward, or None if data unavailable.
    """
    benchmarl_dir = run_dir / "output" / "benchmarl"
    if not benchmarl_dir.exists():
        return None

    # Find scalars directory (name has a hash)
    scalars_dirs = list(benchmarl_dir.glob("**/scalars"))
    if not scalars_dirs:
        return None
    scalars_dir = scalars_dirs[0]

    eval_csv = scalars_dir / "eval_reward_episode_reward_mean.csv"
    frames_csv = scalars_dir / "counters_total_frames.csv"

    if not eval_csv.exists() or not frames_csv.exists():
        return None

    # Parse CSVs: format is "step,value" (TensorBoard CSV export)
    def read_csv_values(path):
        rows = []
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    try:
                        rows.append((int(row[0]), float(row[1])))
                    except (ValueError, IndexError):
                        continue
        return rows

    eval_rows = read_csv_values(eval_csv)
    frames_rows = read_csv_values(frames_csv)

    if not eval_rows or not frames_rows:
        return None

    # Build iteration → total_frames mapping
    iter_to_frames = {step: val for step, val in frames_rows}

    # Get final reward and threshold
    final_reward = eval_rows[-1][1]
    if final_reward <= 0:
        return None  # Can't compute meaningful threshold for negative reward

    threshold = threshold_fraction * final_reward

    # Find first iteration exceeding threshold
    for step, reward in eval_rows:
        if reward >= threshold:
            frames = iter_to_frames.get(step)
            if frames is not None:
                return frames
            # Fall back to linear interpolation from step index
            if frames_rows:
                # Estimate frames from step proportion
                max_step = frames_rows[-1][0]
                max_frames = frames_rows[-1][1]
                if max_step > 0:
                    return (step / max_step) * max_frames
            break

    return None


# ── Cross-experiment analysis utilities ─────────────────────────────

def compute_budget_frontier(
    results_at_budgets: Dict[float, Dict[str, float]],
) -> Dict[float, float]:
    """Success rate at each token budget level (for Pareto analysis)."""
    return {
        budget: metrics["M1_success_rate"]
        for budget, metrics in sorted(results_at_budgets.items())
    }


def compute_transfer_score(
    source_metrics: Dict[str, float],
    target_metrics: Dict[str, float],
) -> float:
    """Transfer Score = target_success / source_success."""
    src = source_metrics["M1_success_rate"]
    if src == 0:
        return 0.0
    return target_metrics["M1_success_rate"] / src


def compute_ate(
    baseline_metrics: Dict[str, float],
    ablated_metrics: Dict[str, float],
) -> Dict[str, float]:
    """Ablation Treatment Effect (delta when masking/shuffling comms)."""
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


def compute_delta_return_per_msg(
    metrics_with_comm: Dict[str, float],
    metrics_no_comm: Dict[str, float],
) -> float:
    """Marginal return improvement per message sent."""
    tokens = metrics_with_comm["M5_avg_tokens"]
    if tokens == 0:
        return 0.0
    delta_return = (
        metrics_with_comm["M2_avg_return"] - metrics_no_comm["M2_avg_return"]
    )
    return delta_return / tokens
