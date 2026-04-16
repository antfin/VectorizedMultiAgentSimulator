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
from .scenario_patch import make_patched_scenario_class

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

    # ── Obs-local: agent LiDAR and comm lines for obs_state ──
    if task_params.get("use_agent_lidar", False):
        obs_lidar_agents = (
            f'"lidar_agents":    '
            f"# [batch, {task_params.get('n_lidar_rays_agents', 12)}] "
            f"— agent LiDAR readings"
        )
    else:
        obs_lidar_agents = "# agent LiDAR not enabled"

    # ── Task-specific coordination guidance for reward design ──
    if k == 1 and dim_c == 0:
        coordination_guidance = (
            "COORDINATION STRATEGY for k=1 (no communication):\n"
            f"Each target needs only 1 agent. With {n_agents} agents and "
            f"{n_targets} targets, agents should SPREAD OUT to different "
            "targets. The key challenge is avoiding redundancy — two agents "
            "going to the same target wastes effort.\n\n"
            "Your reward bonus should:\n"
            "- Reward approaching uncovered targets\n"
            "- Penalize multiple agents converging on the same target "
            "when other targets remain uncovered\n"
            "- Reward agents that go to targets with fewer nearby agents\n"
            "- NOT penalize convergence on the LAST remaining target"
        )
    elif k >= 2 and dim_c == 0:
        coordination_guidance = (
            f"COORDINATION STRATEGY for k={k} (no communication):\n"
            f"Each target needs EXACTLY {k} agents within covering_range "
            f"SIMULTANEOUSLY. This is a RENDEZVOUS task — agents must "
            f"CONVERGE in groups of {k} on the same target.\n\n"
            f"With {n_agents} agents and {n_targets} targets, the optimal "
            f"strategy is to ASSIGN agent pairs to targets. For example:\n"
            f"  - Agents 0,1 → Target A (nearest uncovered)\n"
            f"  - Agents 2,3 → Target B (next nearest uncovered)\n"
            f"Then once A and B are covered, reassign to remaining targets.\n\n"
            "CRITICAL RULES:\n"
            "- Do NOT penalize agents for being near each other! "
            f"Co-location at an uncovered target is the GOAL.\n"
            f"- STRONGLY reward agents that are near a target that already "
            f"has {k-1} other agent(s) nearby — they are completing a team.\n"
            "- Use agent_idx to create IMPLICIT ASSIGNMENT: e.g., "
            "even-indexed agents (0,2) prefer the nearest uncovered target, "
            "odd-indexed agents (1,3) prefer the target that already has "
            "a partner approaching.\n\n"
            "Your reward bonus should:\n"
            f"1. TEAM COMPLETION (strongest signal, weight ~2-5): "
            f"Big bonus when this agent is near a target that has "
            f"{k-1} other agents also within covering_range\n"
            f"2. PARTNER FOLLOWING (weight ~1-3): Reward approaching a "
            f"target that already has 1+ agents nearby (join the team)\n"
            f"3. APPROACH (weight ~0.5-1): Reward getting closer to any "
            f"uncovered target (basic shaping)\n"
            f"4. LONE PENALTY: Small penalty for being "
            f"the only agent near a target when other agents are also "
            f"alone elsewhere (should join forces instead)\n\n"
            "Focus on relative weights between components. The absolute "
            "magnitude will be automatically normalized."
        )
    elif k == 1 and dim_c > 0:
        coordination_guidance = (
            f"COORDINATION STRATEGY for k=1 (with {dim_c}-dim communication):\n"
            f"Each target needs 1 agent. Agents can communicate via "
            f"{dim_c}-float messages each step.\n\n"
            "Your reward bonus should:\n"
            "- Reward approaching uncovered targets\n"
            "- Penalize redundant coverage (two agents at same target)\n"
            "- Encourage informative communication — agents should "
            "broadcast which target they are heading to so others avoid it\n"
            "- Reward message diversity (different agents sending different "
            "messages suggests they are communicating distinct intentions)"
        )
    else:  # k >= 2 and dim_c > 0
        coordination_guidance = (
            f"COORDINATION STRATEGY for k={k} (with {dim_c}-dim communication):\n"
            f"Each target needs EXACTLY {k} agents simultaneously. This is "
            f"a RENDEZVOUS task with communication channels.\n\n"
            f"With {n_agents} agents and {n_targets} targets, agents must "
            f"form {k}-agent teams. Communication should enable EXPLICIT "
            f"coordination — agents can signal their target choice.\n\n"
            "CRITICAL RULES:\n"
            "- Co-location at an uncovered target is the GOAL, not a problem.\n"
            f"- STRONGLY reward agents completing a team of {k} at a target.\n"
            "- Use communication to avoid conflicts — reward message diversity "
            "(agents signaling different target choices).\n\n"
            "Your reward bonus should:\n"
            f"1. TEAM COMPLETION (weight ~2-5): Big bonus when this agent "
            f"is at a target with {k-1} other agents nearby\n"
            f"2. PARTNER FOLLOWING (weight ~1-3): Reward joining a target "
            f"that already has 1+ agents\n"
            f"3. COMM INCENTIVE (weight ~0.5-1): Reward sending non-zero "
            f"messages and reward when received messages correlate with "
            f"the agent's target choice\n"
            f"4. APPROACH (weight ~0.5-1): Basic approach shaping\n\n"
            "Focus on relative weights between components. The absolute "
            "magnitude will be automatically normalized."
        )

    if dim_c > 0:
        obs_comm_state = (
            f'"messages":        '
            f"# [batch, {n_agents - 1}, {dim_c}] — received messages "
            f"(proximity-masked if comm_proximity=True)"
        )
        comm_obs_guidance = (
            "- Summary statistics of received messages "
            "(mean, max, variance)\n"
            "    - Whether communication is informative "
            "(message entropy/diversity)"
        )
    else:
        obs_comm_state = "# No communication channels available"
        comm_obs_guidance = ""

    return {
        "experiment_context": experiment_context,
        "agent_lidar_description": agent_lidar_description,
        "comm_description": comm_description,
        "reward_description": reward_description,
        "coordination_guidance": coordination_guidance,
        "comm_state_description": comm_state_description,
        "obs_lidar_agents": obs_lidar_agents,
        "obs_comm_state": obs_comm_state,
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


class _ScenarioEnvFunFactory:
    """Picklable env factory that creates VmasEnv with a scenario CLASS.

    Each env gets a NEW instance of the patched scenario class.
    BenchMARL pickles this for naming, so __getstate__ is minimal.
    """

    def __init__(self, scenario_class, config):
        self.scenario_class = scenario_class
        self.config = config

    def __call__(self, num_envs, continuous_actions, seed, device):
        ScenarioClass = self.scenario_class
        config = self.config

        def make_env():
            from torchrl.envs.libs.vmas import VmasEnv
            # Each env creates a FRESH scenario instance
            return VmasEnv(
                scenario=ScenarioClass(),
                num_envs=num_envs,
                continuous_actions=continuous_actions,
                seed=seed,
                device=device,
                categorical_actions=True,
                clamp_actions=True,
                **config,
            )

        return make_env

    def __getstate__(self):
        return {"_dummy": True}

    def __setstate__(self, state):
        pass


def _build_patched_experiment(
    task_config,
    train_config,
    algorithm: str,
    seed: int,
    task_overrides: Optional[Dict] = None,
    save_folder: Optional[str] = None,
    reward_source: Optional[str] = None,
    obs_source: Optional[str] = None,
    reward_mode: str = "replace",
    obs_state_mode: str = "global",
    bonus_scale: float = 0.5,
    reward_clip: Optional[float] = 50.0,
):
    """Build a BenchMARL Experiment with LLM-patched scenario.

    Creates a Scenario subclass with the LLM-generated reward/observation
    and passes it as a scenario instance to BenchMARL. This way TorchRL
    sees the patched observation size when computing specs during init.
    """
    from ..runner import get_algorithm_config

    from benchmarl.environments import VmasTask
    from benchmarl.experiment import Experiment, ExperimentConfig
    from benchmarl.models.mlp import MlpConfig

    # Create patched scenario class
    PatchedScenario = make_patched_scenario_class(
        reward_source, obs_source,
        reward_mode=reward_mode,
        obs_state_mode=obs_state_mode,
        bonus_scale=bonus_scale,
        reward_clip=reward_clip,
    )
    # Configure the task
    task = VmasTask.DISCOVERY.get_from_yaml()
    config = task_config.to_dict()
    if task_overrides:
        config.update(task_overrides)
    if (train_config.model_type == "gnn"
            and train_config.gnn_topology == "from_pos"):
        config["dict_obs"] = True
    task.config.update(config)

    # Override the env factory to use our patched scenario CLASS
    # Each env creates a fresh instance of the patched class
    task.get_env_fun = _ScenarioEnvFunFactory(
        PatchedScenario, config,
    )

    # Algorithm
    algo_config = get_algorithm_config(algorithm)
    if train_config.entropy_coef is not None:
        algo_config.entropy_coef = train_config.entropy_coef
    if train_config.lmbda is not None:
        algo_config.lmbda = train_config.lmbda
    if train_config.clip_epsilon is not None:
        algo_config.clip_epsilon = train_config.clip_epsilon

    # Model
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()
    if train_config.hidden_layers is not None:
        model_config.num_cells = train_config.hidden_layers
        critic_model_config.num_cells = train_config.hidden_layers
    if train_config.activation is not None:
        import torch.nn as nn
        act_map = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}
        act_cls = act_map.get(train_config.activation.lower())
        if act_cls is not None:
            model_config.activation_class = act_cls
            critic_model_config.activation_class = act_cls

    # Experiment config
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.max_n_frames = train_config.max_n_frames
    experiment_config.gamma = train_config.gamma
    experiment_config.on_policy_collected_frames_per_batch = (
        train_config.on_policy_collected_frames_per_batch
    )
    experiment_config.on_policy_n_envs_per_worker = (
        train_config.on_policy_n_envs_per_worker
    )
    experiment_config.on_policy_n_minibatch_iters = (
        train_config.on_policy_n_minibatch_iters
    )
    experiment_config.on_policy_minibatch_size = (
        train_config.on_policy_minibatch_size
    )
    experiment_config.train_device = train_config.train_device
    experiment_config.sampling_device = train_config.sampling_device
    experiment_config.share_policy_params = train_config.share_policy_params
    experiment_config.evaluation = True

    eval_interval = train_config.evaluation_interval
    batch_size = train_config.on_policy_collected_frames_per_batch
    if eval_interval % batch_size != 0:
        eval_interval = max(
            batch_size, (eval_interval // batch_size) * batch_size,
        )
    experiment_config.evaluation_interval = eval_interval
    experiment_config.evaluation_episodes = train_config.evaluation_episodes
    experiment_config.loggers = ["csv"]
    experiment_config.render = False

    if save_folder is not None:
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        experiment_config.save_folder = save_folder

    experiment = Experiment(
        task=task,
        algorithm_config=algo_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=seed,
        config=experiment_config,
    )
    return experiment


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
            output_dir = storage.results_dir / "lero" / ts
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
        # Track ALL valid candidates across all iterations so we can
        # fall back to second/third best if full training crashes.
        all_valid: List[Tuple[CandidateCode, Dict, int]] = []  # (cand, metrics, iter)

        for iteration in range(self.lero.n_iterations):
            iter_dir = self.output_dir / f"iter_{iteration}"
            iter_dir.mkdir(exist_ok=True)

            _log.info(
                "--- Iteration %d/%d ---",
                iteration + 1, self.lero.n_iterations,
            )

            # 1. Generate candidates from LLM
            _log.info("Generating %d candidates...", self.lero.n_candidates)
            try:
                responses = self.llm.generate(
                    messages, n=self.lero.n_candidates,
                )
            except Exception as e:
                _log.warning(
                    "LLM call failed in iteration %d: %s. "
                    "Skipping iteration.",
                    iteration, e,
                )
                continue

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
                        iter_dir=iter_dir, candidate_idx=j,
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
            # Accumulate cross-iteration list for fallback ranking
            for c, m in valid_results:
                all_valid.append((c, m, iteration))
            # Fitness: M1 primary, M6 secondary (fallback when all M1=0)
            # This handles hard tasks (k>=2) where no candidate achieves
            # full coverage at short eval, but M6 still differentiates.
            valid_results.sort(
                key=lambda x: (
                    x[1].get("M1_success_rate", 0),
                    x[1].get("M6_coverage_progress", 0),
                    x[1].get("M2_avg_return", -1e9),
                ),
                reverse=True,
            )

            iter_best_candidate = valid_results[0][0]
            iter_best_metrics = valid_results[0][1]

            # Track global best across ALL iterations
            # Use (M1, M6, M2) as composite fitness — M6 breaks ties
            # when all candidates have M1=0 (common for hard k>=2 tasks)
            iter_fitness = (
                iter_best_metrics.get("M1_success_rate", 0),
                iter_best_metrics.get("M6_coverage_progress", 0),
                iter_best_metrics.get("M2_avg_return", -1e9),
            )
            global_fitness = (
                (best_metrics or {}).get("M1_success_rate", -1),
                (best_metrics or {}).get("M6_coverage_progress", -1),
                (best_metrics or {}).get("M2_avg_return", -1e9),
            ) if best_metrics else (-1, -1, -1e9)
            iter_m1 = iter_fitness[0]
            global_m1 = global_fitness[0]
            if iter_fitness > global_fitness:
                best_candidate = iter_best_candidate
                best_metrics = iter_best_metrics
                _log.info(
                    "  NEW GLOBAL BEST (iter %d): M1=%.3f, M2=%.2f, M6=%.3f",
                    iteration,
                    best_metrics.get("M1_success_rate", 0),
                    best_metrics.get("M2_avg_return", 0),
                    best_metrics.get("M6_coverage_progress", 0),
                )
            else:
                _log.info(
                    "  Iter best: M1=%.3f, M2=%.2f, M6=%.3f "
                    "(global best M1=%.3f from earlier iter)",
                    iter_m1,
                    iter_best_metrics.get("M2_avg_return", 0),
                    iter_best_metrics.get("M6_coverage_progress", 0),
                    global_m1,
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

            # Update conversation for next iteration.
            # Use a sliding window: keep system + initial user + only
            # the LAST assistant/feedback pair. This prevents the
            # conversation from growing beyond the model's context
            # window (OVH/vLLM returns null content when prompt is
            # too long — see vllm-project/vllm#18006).
            messages = [
                messages[0],  # system
                messages[1],  # initial user prompt
                {
                    "role": "assistant",
                    "content": iter_best_candidate.source,
                },
                {
                    "role": "user",
                    "content": feedback,
                },
            ]

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

        # 5. Full training with best candidate, with fallback chain.
        # If the chosen reward causes training to crash (e.g. NaN actions
        # from PPO gradient explosion on large-magnitude rewards), fall
        # back to the next-best candidate ranked by (M1, M6, M2).
        all_valid.sort(
            key=lambda x: (
                x[1].get("M1_success_rate", 0),
                x[1].get("M6_coverage_progress", 0),
                x[1].get("M2_avg_return", -1e9),
            ),
            reverse=True,
        )
        fallback_chain = []
        final_result = None
        chosen_candidate = None
        for rank, (cand, eval_metrics, src_iter) in enumerate(all_valid):
            label = (
                f"rank {rank} (iter {src_iter}, "
                f"eval M1={eval_metrics.get('M1_success_rate', 0):.3f})"
            )
            _log.info(
                "=== FULL TRAINING with candidate %s ===", label,
            )
            try:
                final_result = self._full_training(
                    cand, task_overrides, algorithm, seed,
                )
                chosen_candidate = cand
                fallback_chain.append({
                    "rank": rank, "iter": src_iter,
                    "eval_M1": eval_metrics.get("M1_success_rate", 0),
                    "eval_M2": eval_metrics.get("M2_avg_return", 0),
                    "eval_M6": eval_metrics.get("M6_coverage_progress", 0),
                    "outcome": "success",
                })
                break
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                _log.warning(
                    "Full training failed for %s: %s. Trying next candidate.",
                    label, err[:200],
                )
                fallback_chain.append({
                    "rank": rank, "iter": src_iter,
                    "eval_M1": eval_metrics.get("M1_success_rate", 0),
                    "eval_M2": eval_metrics.get("M2_avg_return", 0),
                    "eval_M6": eval_metrics.get("M6_coverage_progress", 0),
                    "outcome": "crashed",
                    "error": err[:200],
                })
                continue

        if final_result is None:
            _log.error(
                "ALL %d candidates crashed during full training",
                len(all_valid),
            )
            self._save_json("fallback_chain.json", fallback_chain)
            return {
                "_error": "all candidates crashed at full training",
                "fallback_chain": fallback_chain,
            }

        # Save the candidate that actually completed training
        best_candidate = chosen_candidate

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
        final_result["fallback_chain"] = fallback_chain
        self._save_json("fallback_chain.json", fallback_chain)

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
            covering_range=task_params.get("covering_range", 0.25),
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
            coordination_guidance=ctx["coordination_guidance"],
            comm_state_description=ctx["comm_state_description"],
            obs_lidar_agents=ctx["obs_lidar_agents"],
            obs_comm_state=ctx["obs_comm_state"],
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
        iter_dir: Optional[Path] = None,
        candidate_idx: int = 0,
    ) -> Dict[str, float]:
        """Short training run with patched scenario to evaluate a candidate."""
        from ..runner import build_experiment, evaluate_trained
        from benchmarl.environments import VmasTask

        short_train = copy.copy(self.spec.train)
        short_train.max_n_frames = self.lero.eval_frames
        short_train.evaluation_episodes = self.lero.eval_episodes

        save_folder = None
        if iter_dir is not None:
            save_folder = str(iter_dir / f"benchmarl_c{candidate_idx}")

        # build_experiment creates its own task internally, but we need
        # to patch the task's env factory BEFORE the Experiment is built.
        # So we replicate the task setup here and pass the patched task.
        experiment = _build_patched_experiment(
            self.spec.task, short_train, algorithm, seed,
            task_overrides, save_folder,
            reward_source=candidate.reward_source,
            obs_source=candidate.obs_source,
            reward_mode=self.lero.reward_mode,
            obs_state_mode=self.lero.obs_state_mode,
            bonus_scale=self.lero.bonus_scale,
            reward_clip=self.lero.reward_clip,
        )

        experiment.run()
        metrics = evaluate_trained(
            self.spec, experiment, task_overrides,
            n_eval_episodes=self.lero.eval_episodes,
        )
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

        save_folder = str(self.output_dir / "benchmarl_final")

        experiment = _build_patched_experiment(
            self.spec.task, full_train, algorithm, seed,
            task_overrides, save_folder,
            reward_source=candidate.reward_source,
            obs_source=candidate.obs_source,
            reward_mode=self.lero.reward_mode,
            obs_state_mode=self.lero.obs_state_mode,
            bonus_scale=self.lero.bonus_scale,
            reward_clip=self.lero.reward_clip,
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
