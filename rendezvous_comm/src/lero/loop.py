"""LERO evolutionary loop: LLM-driven reward & observation optimization.

Orchestrates:
  1. LLM generates candidate (reward, obs) function pairs
  2. Each candidate is evaluated via short MARL training
  3. Top candidates + metrics fed back to LLM
  4. Repeat for N iterations
  5. Final full training with the best candidate
"""

import copy
import inspect
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..config import ExperimentSpec, TaskConfig, TrainConfig
from ..storage import ExperimentStorage
from .codegen import CandidateCode, build_feedback, extract_candidates
from .config import LeroConfig, LLMConfig
from .llm_client import LLMClient
from .prompts.loader import PromptLoader
from .scenario_patch import patch_scenario_class, unpatch_scenario_class

_log = logging.getLogger("rendezvous.lero")


def _build_experiment_context(
    spec: ExperimentSpec, task_params: Dict[str, Any],
) -> Dict[str, str]:
    """Build dynamic context strings for prompt templates.

    Returns a dict of context blocks that describe the current experiment
    setup. These are injected into prompt templates so the LLM understands
    what it's optimizing for.
    """
    dim_c = task_params.get("dim_c", 0)
    n_agents = task_params.get("n_agents", 4)
    n_targets = task_params.get("n_targets", 4)
    k = task_params.get("agents_per_target", 2)
    lidar_range = task_params.get("lidar_range", 0.35)
    shared_reward = task_params.get("shared_reward", False)
    collision_penalty = task_params.get("agent_collision_penalty", -0.1)

    # ── Experiment-level context (name + description from YAML) ──
    exp_lines = [
        f"**{spec.name}** (`{spec.exp_id}`)",
        "",
        spec.description.strip() if spec.description else "",
        "",
        f"Configuration: {n_agents} agents, {n_targets} targets, "
        f"k={k} agents required per target.",
    ]

    if dim_c == 0:
        exp_lines.append(
            "This is a **no-communication** experiment. Agents must "
            "coordinate purely through spatial observation (LiDAR). "
            "The challenge is implicit coordination — agents need to "
            "spread out and avoid redundantly covering the same targets."
        )
    else:
        exp_lines.append(
            f"This is a **communication** experiment. Agents have "
            f"{dim_c}-dimensional continuous message channels. Each step, "
            f"an agent broadcasts a {dim_c}-float vector that nearby "
            f"agents (within {lidar_range} range) receive in their "
            f"observation. The challenge is learning *what* to communicate "
            f"to improve coordination."
        )

    if k >= 2:
        exp_lines.append(
            f"With k={k}, agents must learn to **rendezvous** — multiple "
            f"agents must converge on the same target simultaneously. "
            f"This is the hardest coordination challenge in this project."
        )

    experiment_context = "\n".join(exp_lines)

    # ── Agent LiDAR description ──
    if task_params.get("use_agent_lidar", False):
        agent_lidar_description = (
            "Agent LiDAR is ENABLED — agents can detect nearby agents."
        )
    else:
        agent_lidar_description = (
            "Agent LiDAR is DISABLED — agents cannot directly sense "
            "each other's positions."
        )

    # ── Communication description ──
    if dim_c > 0:
        comm_proximity = task_params.get("comm_proximity", True)
        prox_note = (
            f"Messages are proximity-gated: only agents within "
            f"{lidar_range} range receive them."
            if comm_proximity
            else "Messages are broadcast globally to all agents."
        )
        comm_description = (
            f"- **Communication**: {dim_c}-dimensional continuous channel. "
            f"Each agent outputs {dim_c} floats per step as part of its "
            f"action. These floats are received by other agents in their "
            f"next observation. {prox_note}\n"
            f"  The policy must learn both *what to say* (communication "
            f"action) and *how to use* received messages (observation)."
        )
    else:
        comm_description = (
            "- **Communication**: NONE. Agents are silent. Coordination "
            "must emerge from movement patterns and LiDAR observations alone."
        )

    # ── Reward description ──
    reward_parts = [
        f"- **Covering reward**: +{task_params.get('covering_rew_coeff', 1.0)} "
        f"per target covered by this agent.",
    ]
    if shared_reward:
        reward_parts.append(
            "- **Shared reward**: ON — covering reward is averaged across "
            "all agents (team reward). This helps credit assignment but "
            "reduces individual incentive."
        )
    else:
        reward_parts.append(
            "- **Shared reward**: OFF — each agent receives reward only "
            "for targets it personally helps cover."
        )
    reward_parts.append(
        f"- **Collision penalty**: {collision_penalty} per agent-agent collision."
    )
    reward_parts.append(
        f"- **Time penalty**: {task_params.get('time_penalty', -0.01)} per step "
        f"(encourages faster completion)."
    )
    reward_description = "\n".join(reward_parts)

    # ── Communication state in scenario_state dict ──
    if dim_c > 0:
        comm_state_description = (
            f'"messages":        '
            f"# [batch, {n_agents - 1}, {dim_c}] — messages from other agents"
        )
    else:
        comm_state_description = "# No communication channels available"

    # ── Observation enhancement guidance for comm ──
    if dim_c > 0:
        comm_obs_guidance = (
            "- Summary statistics of received messages "
            "(mean, max, variance)\n"
            "    - Whether communication is informative "
            "(message entropy/diversity)"
        )
    else:
        comm_obs_guidance = ""

    return {
        "experiment_context": experiment_context,
        "agent_lidar_description": agent_lidar_description,
        "comm_description": comm_description,
        "reward_description": reward_description,
        "comm_state_description": comm_state_description,
        "comm_obs_guidance": comm_obs_guidance,
    }


def _get_scenario_source_snippets() -> Tuple[str, str]:
    """Extract reward and observation source code from Discovery scenario."""
    from vmas.scenarios.discovery import Scenario

    reward_src = inspect.getsource(Scenario.reward)
    obs_src = inspect.getsource(Scenario.observation)
    return reward_src, obs_src


def _get_scenario_from_experiment(experiment):
    """Extract the Discovery scenario instance from a BenchMARL experiment.

    BenchMARL wraps the scenario deeply. This navigates the chain:
    Experiment -> task -> env_func -> ... -> scenario
    """
    # The test_env contains the actual VMAS environments
    test_env = experiment.test_env
    # Navigate to the base env which holds the scenario
    env = test_env
    while hasattr(env, "base_env"):
        env = env.base_env
    if hasattr(env, "_env"):
        env = env._env
    # VMAS vectorized env -> scenario
    if hasattr(env, "scenario"):
        return env.scenario
    # Try the parallel env wrapper
    if hasattr(env, "par_env"):
        return env.par_env.scenario
    raise AttributeError(
        "Could not find scenario instance in experiment. "
        f"Final env type: {type(env)}"
    )


class LeroLoop:
    """Main LERO evolutionary loop."""

    def __init__(
        self,
        spec: ExperimentSpec,
        lero_config: LeroConfig,
        llm_config: LLMConfig,
        output_dir: Optional[Path] = None,
    ):
        self.spec = spec
        self.lero = lero_config
        self.llm_config = llm_config
        self.prompt_loader = PromptLoader(version=llm_config.prompt_version)
        self.llm = LLMClient(llm_config)

        # Output directory for LERO artifacts
        if output_dir is None:
            storage = ExperimentStorage(spec.exp_id)
            ts = time.strftime("%Y%m%d_%H%M")
            output_dir = storage.base_dir / "lero" / ts
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track all iterations
        self.history: List[Dict] = []

    def run(
        self,
        task_overrides: Optional[Dict[str, Any]] = None,
        algorithm: str = "mappo",
        seed: int = 0,
    ) -> Dict[str, Any]:
        """Run the full LERO evolutionary loop + final training.

        Returns:
            Dict with final metrics and best candidate info.
        """
        _log.info(
            "=== LERO START === %d iterations, %d candidates/iter",
            self.lero.n_iterations, self.lero.n_candidates,
        )

        # Build initial LLM conversation
        messages = self._build_initial_messages(task_overrides)

        # Save initial messages
        self._save_json("messages_initial.json", messages)

        best_candidate = None
        best_metrics = None

        for iteration in range(self.lero.n_iterations):
            iter_dir = self.output_dir / f"iter_{iteration}"
            iter_dir.mkdir(exist_ok=True)

            _log.info(
                "--- Iteration %d/%d ---",
                iteration + 1, self.lero.n_iterations,
            )

            # 1. Generate candidates from LLM
            _log.info("Generating %d candidates...", self.lero.n_candidates)
            responses = self.llm.generate(
                messages, n=self.lero.n_candidates,
            )

            candidates = extract_candidates(
                responses,
                evolve_reward=self.lero.evolve_reward,
                evolve_observation=self.lero.evolve_observation,
            )

            if not candidates:
                _log.warning(
                    "No valid candidates extracted in iteration %d. "
                    "Retrying with same messages.",
                    iteration,
                )
                continue

            _log.info(
                "Extracted %d valid candidates from %d responses",
                len(candidates), len(responses),
            )

            # Save candidate source code
            for j, cand in enumerate(candidates):
                if cand.reward_source:
                    (iter_dir / f"candidate_{j}_reward.py").write_text(
                        cand.reward_source
                    )
                if cand.obs_source:
                    (iter_dir / f"candidate_{j}_obs.py").write_text(
                        cand.obs_source
                    )
                (iter_dir / f"candidate_{j}_response.txt").write_text(
                    cand.raw_response
                )

            # 2. Evaluate each candidate (short training)
            results: List[Tuple[CandidateCode, Dict]] = []
            for j, cand in enumerate(candidates):
                _log.info(
                    "  Evaluating candidate %d/%d ...",
                    j + 1, len(candidates),
                )
                try:
                    metrics = self._evaluate_candidate(
                        cand, task_overrides, algorithm, seed,
                    )
                    results.append((cand, metrics))
                    # Save metrics
                    self._save_json(
                        str(iter_dir / f"candidate_{j}_metrics.json"),
                        metrics, absolute=True,
                    )
                    _log.info(
                        "    M1=%.3f  M2=%.2f  M6=%.3f",
                        metrics.get("M1_success_rate", 0),
                        metrics.get("M2_avg_return", 0),
                        metrics.get("M6_coverage_progress", 0),
                    )
                except Exception as e:
                    _log.warning(
                        "  Candidate %d failed: %s", j, e,
                    )
                    results.append((cand, {"_error": str(e)}))

            if not any(
                "_error" not in r[1] for r in results
            ):
                _log.warning("All candidates failed in iteration %d", iteration)
                continue

            # 3. Rank by fitness (M1 primary, M2 secondary)
            valid_results = [
                (c, m) for c, m in results if "_error" not in m
            ]
            valid_results.sort(
                key=lambda x: (
                    x[1].get("M1_success_rate", 0),
                    x[1].get("M2_avg_return", -1e9),
                ),
                reverse=True,
            )

            best_candidate = valid_results[0][0]
            best_metrics = valid_results[0][1]

            _log.info(
                "  Best: M1=%.3f, M2=%.2f, M6=%.3f",
                best_metrics.get("M1_success_rate", 0),
                best_metrics.get("M2_avg_return", 0),
                best_metrics.get("M6_coverage_progress", 0),
            )

            # 4. Build feedback for next iteration
            task_params = self._effective_task_params(task_overrides)
            feedback = build_feedback(
                [c for c, _ in valid_results],
                [m for _, m in valid_results],
                self.lero.top_k,
                self.prompt_loader,
                task_params,
            )

            # Update conversation for next iteration
            messages.append({
                "role": "assistant",
                "content": best_candidate.source,
            })
            messages.append({
                "role": "user",
                "content": feedback,
            })

            # Save feedback
            (iter_dir / "feedback.txt").write_text(feedback)

            # Record iteration history
            self.history.append({
                "iteration": iteration,
                "n_candidates": len(candidates),
                "n_valid": len(valid_results),
                "best_M1": best_metrics.get("M1_success_rate", 0),
                "best_M2": best_metrics.get("M2_avg_return", 0),
                "best_M6": best_metrics.get("M6_coverage_progress", 0),
            })

        # Save full conversation history
        self._save_json("messages_final.json", messages)
        self._save_json("evolution_history.json", self.history)

        if best_candidate is None:
            _log.error("No valid candidates found across all iterations")
            return {"_error": "no valid candidates"}

        # 5. Full training with best candidate
        _log.info("=== FULL TRAINING with best candidate ===")
        final_result = self._full_training(
            best_candidate, task_overrides, algorithm, seed,
        )

        # Save best candidate source
        if best_candidate.reward_source:
            (self.output_dir / "best_reward.py").write_text(
                best_candidate.reward_source
            )
        if best_candidate.obs_source:
            (self.output_dir / "best_obs.py").write_text(
                best_candidate.obs_source
            )

        final_result["lero_iterations"] = self.lero.n_iterations
        final_result["lero_candidates_per_iter"] = self.lero.n_candidates
        final_result["llm_model"] = self.llm_config.model
        final_result["prompt_version"] = self.llm_config.prompt_version

        self._save_json("final_metrics.json", final_result)
        _log.info("=== LERO COMPLETE ===")

        return final_result

    # ── internal methods ─────────────────────────────────────────

    def _build_initial_messages(
        self, task_overrides: Optional[Dict] = None,
    ) -> List[Dict[str, str]]:
        """Build the initial LLM conversation with full experiment context."""
        task_params = self._effective_task_params(task_overrides)
        reward_src, obs_src = _get_scenario_source_snippets()
        ctx = _build_experiment_context(self.spec, task_params)

        system_prompt = self.prompt_loader.render(
            "system.txt",
            experiment_context=ctx["experiment_context"],
        )

        user_prompt = self.prompt_loader.render(
            "initial_user.txt",
            # Task params
            n_agents=task_params.get("n_agents", 4),
            n_targets=task_params.get("n_targets", 4),
            agents_per_target=task_params.get("agents_per_target", 2),
            covering_range=task_params.get("covering_range", 0.25),
            lidar_range=task_params.get("lidar_range", 0.35),
            max_steps=task_params.get("max_steps", 200),
            collision_penalty=task_params.get("agent_collision_penalty", -0.1),
            time_penalty=task_params.get("time_penalty", -0.01),
            n_lidar_rays_entities=task_params.get("n_lidar_rays_entities", 15),
            n_lidar_rays_agents=task_params.get("n_lidar_rays_agents", 12),
            # Dynamic context blocks
            agent_lidar_description=ctx["agent_lidar_description"],
            comm_description=ctx["comm_description"],
            reward_description=ctx["reward_description"],
            comm_state_description=ctx["comm_state_description"],
            comm_obs_guidance=ctx["comm_obs_guidance"],
            # Source code
            scenario_reward_code=reward_src,
            scenario_observation_code=obs_src,
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _effective_task_params(
        self, task_overrides: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Get effective task parameters with overrides applied."""
        params = self.spec.task.to_dict()
        if task_overrides:
            params.update(task_overrides)
        return params

    def _evaluate_candidate(
        self,
        candidate: CandidateCode,
        task_overrides: Optional[Dict],
        algorithm: str,
        seed: int,
    ) -> Dict[str, float]:
        """Short training run with patched scenario to evaluate a candidate."""
        from ..runner import build_experiment, evaluate_trained

        short_train = copy.copy(self.spec.train)
        short_train.max_n_frames = self.lero.eval_frames
        short_train.evaluation_episodes = self.lero.eval_episodes

        # Patch class BEFORE building experiment so BenchMARL sees
        # the correct observation size when probing the env.
        originals = patch_scenario_class(
            reward_source=candidate.reward_source,
            obs_source=candidate.obs_source,
        )

        try:
            experiment = build_experiment(
                self.spec.task, short_train, algorithm, seed,
                task_overrides,
            )
            experiment.run()
            metrics = evaluate_trained(
                self.spec, experiment, task_overrides,
                n_eval_episodes=self.lero.eval_episodes,
            )
        finally:
            unpatch_scenario_class(originals)

        return metrics

    def _full_training(
        self,
        candidate: CandidateCode,
        task_overrides: Optional[Dict],
        algorithm: str,
        seed: int,
    ) -> Dict[str, float]:
        """Full training run with the best candidate."""
        from ..runner import build_experiment, evaluate_trained

        full_train = copy.copy(self.spec.train)
        full_train.max_n_frames = self.lero.full_frames

        # Patch class BEFORE building experiment
        originals = patch_scenario_class(
            reward_source=candidate.reward_source,
            obs_source=candidate.obs_source,
        )

        try:
            experiment = build_experiment(
                self.spec.task, full_train, algorithm, seed,
                task_overrides,
            )

            t0 = time.monotonic()
            experiment.run()
            elapsed = time.monotonic() - t0

            metrics = evaluate_trained(self.spec, experiment, task_overrides)
            metrics["training_seconds"] = elapsed

            # Save policy
            policy_path = self.output_dir / "best_policy.pt"
            torch.save(experiment.policy.state_dict(), policy_path)
            _log.info("Saved best policy to %s", policy_path)
        finally:
            unpatch_scenario_class(originals)

        return metrics

    # ── helpers ───────────────────────────────────────────────────

    def _save_json(
        self, filename: str, data: Any, absolute: bool = False,
    ):
        path = Path(filename) if absolute else self.output_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        def _default(o):
            if isinstance(o, Path):
                return str(o)
            if isinstance(o, torch.Tensor):
                return o.tolist()
            raise TypeError(f"Not serializable: {type(o)}")

        path.write_text(json.dumps(data, indent=2, default=_default))
