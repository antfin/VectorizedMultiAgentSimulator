"""LERO-MP v4 outer loop — replaces v3's LeroMpOuterLoop.

Pipeline:
  Phase 0 (once): bootstrap_from_description → BootstrapCard + initial
    composable prompt directory + thoughts.md
  Phase 1 (×n_rounds): for round in 0..n_rounds-1:
    a. emit_strategies → StrategyBundle of N candidates
    b. for each strategy s:
         - compose_prompt_for_strategy → strategy-specific prompt dir
         - run inner LeroLoop at eval_frames (e.g. 200k)
         - analyze_candidate_trajectory → CandidateAnalysis
    c. pick best by stability_score → mid-train at mid_frames (e.g. 2M)
    d. record RoundResult; aggregate analysis feeds next round
  Phase 2 (once): pick global best across rounds → final at full_frames
    (e.g. 10M)

Coexists with the inner LeroLoop (`src/lero/loop.py`); the v3 outer
loop / v3 strategist / v3 mutation builder are removed (replaced by
this file + v4_strategist + v4_composer + v4_bootstrap).
"""

from __future__ import annotations

import copy
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...config import ExperimentSpec
from ..config import LeroConfig as InnerLeroConfig, LLMConfig
from ..llm_client import LLMClient
from ..loop import LeroLoop
from ..prompts import loader as _loader_mod
from . import provenance as _prov_mod
from .v4_analyzer import aggregate_round_analysis, analyze_candidate_trajectory
from .v4_bootstrap import bootstrap_from_description
from .v4_composer import compose_prompt_for_strategy
from .v4_schemas import (
    BootstrapCard,
    CandidateAnalysis,
    FitnessWeights,
    RoundResult,
    StrategyV4,
    V4Result,
)
from .v4_strategist import emit_strategies

_log = logging.getLogger("rendezvous.lero.mp.v4")


# ── Helpers ────────────────────────────────────────────────────


def _make_inner_lero_config(
    base: InnerLeroConfig,
    *,
    eval_frames: int,
    full_frames: int,
    skip_full_training: bool,
    n_candidates: int = 1,
    n_iterations: int = 1,
    evolve_reward: bool,
    evolve_observation: bool = True,
) -> InnerLeroConfig:
    """Build a per-call LeroConfig matching the v4 stage's budget."""
    cfg = copy.copy(base)
    cfg.n_iterations = n_iterations
    cfg.n_candidates = n_candidates
    cfg.eval_frames = eval_frames
    cfg.full_frames = full_frames
    cfg.skip_full_training = skip_full_training
    cfg.evolve_reward = evolve_reward
    cfg.evolve_observation = evolve_observation
    return cfg


def _trajectory_from_inner_result(
    inner_result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Extract metrics_history compatible with analyze_candidate_trajectory.

    LeroLoop's _full_training stores a peak_m1_trajectory in the
    returned final_metrics dict. When skip_full_training is True, we
    fall back to the candidate's own M1/M6/M2 final values as a
    single-point trajectory.
    """
    traj = inner_result.get("peak_m1_trajectory")
    if traj:
        # peak_m1_trajectory entries are
        # {"iteration": N, "frame": F, "M1": v}
        # but lack M6. Synthesize using the final M6 across the run as a
        # constant — better than nothing for analyzer slope/noise calcs.
        m6_final = float(inner_result.get("M6_coverage_progress", 0.0))
        out = []
        for p in traj:
            out.append(
                {
                    "M1": float(p.get("M1", 0.0)),
                    "M6": m6_final,
                    "frame": int(p.get("frame", 0)),
                }
            )
        return out
    # Fallback: single point from the eval-only run
    return [
        {
            "M1": float(inner_result.get("M1_success_rate", 0.0)),
            "M6": float(inner_result.get("M6_coverage_progress", 0.0)),
            "frame": int(inner_result.get("eval_frames", 0))
            or int(inner_result.get("max_n_frames", 0)),
        }
    ]


def _redirect_prompt_loader(prompts_root: Path) -> None:
    """Repoint PromptLoader + provenance at a writable prompts root.

    Used so v4 can materialize composed prompt directories under the
    run's output folder and have ``PromptLoader(version=<dirname>)``
    find them.
    """
    prompts_root = Path(prompts_root)
    prompts_root.mkdir(parents=True, exist_ok=True)
    _loader_mod._PROMPTS_DIR = prompts_root
    _prov_mod._PROMPTS_DIR = prompts_root


# ── Result container ──────────────────────────────────────────


@dataclass
class _PerSeedRoundOutput:
    """Internal: what we pass between rounds within one seed."""

    candidates: List[CandidateAnalysis]
    chosen_strategy: StrategyV4
    mid_analysis: CandidateAnalysis
    aggregate_text: str


# ── The v4 outer loop ────────────────────────────────────────


class LeroMpV4OuterLoop:
    def __init__(
        self,
        spec: ExperimentSpec,
        v4_config,  # LeroMPv4Config
        llm_config: LLMConfig,  # inner-LLM config (not meta)
        output_dir: Path,
        base_prompt_dir: Optional[Path] = None,
    ):
        self.spec = spec
        self.cfg = v4_config
        self.llm_cfg = llm_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Where the bootstrap copies the modular template from. By default
        # use v2_fewshot_modular_v2 (the one with the slot decomposition).
        if base_prompt_dir is None:
            # Find the prompts dir relative to the package
            from ..prompts import loader as _l

            base_prompt_dir = _l._PROMPTS_DIR / "v2_fewshot_modular_v2"
        self.base_prompt_dir = Path(base_prompt_dir)

        # Where we materialize composed strategy-specific prompts so
        # PromptLoader can find them.
        self.prompts_root = self.output_dir / "prompts"
        _redirect_prompt_loader(self.prompts_root)

        # If the base prompt isn't already in our writable prompts root,
        # copy it there so PromptLoader can find it from the bootstrap
        # path too.
        target_base = self.prompts_root / self.base_prompt_dir.name
        if not target_base.exists():
            shutil.copytree(self.base_prompt_dir, target_base)
        self._writable_base = target_base

        # Build the meta-LLM client (separate from inner)
        meta_cfg = LLMConfig(
            model=self.cfg.meta_model,
            temperature=self.cfg.meta_temperature,
            api_base=self.cfg.meta_api_base,
        )
        self.meta_llm = LLMClient(meta_cfg)

        self.fitness_weights = FitnessWeights(
            peak=self.cfg.fitness_peak,
            final=self.cfg.fitness_final,
            stability_penalty=self.cfg.fitness_stability_penalty,
            m6_bonus=self.cfg.fitness_m6_bonus,
        )

    # ── Public API ────────────────────────────────────────────

    def run(
        self,
        task_overrides: Optional[Dict[str, Any]] = None,
        algorithm: str = "mappo",
        seed: int = 0,
    ) -> V4Result:
        t0 = time.monotonic()
        _log.info("=== LERO-MP v4 START === seed=%d", seed)

        # Phase 0: bootstrap
        bootstrap_result = bootstrap_from_description(
            description_path=Path(self.cfg.description_path),
            meta_llm=self.meta_llm,
            output_dir=self.output_dir,
            base_prompt_dir=self._writable_base,
            cache_dir=(
                Path(self.cfg.bootstrap_cache_dir)
                if self.cfg.bootstrap_cache_dir
                else None
            ),
        )
        bootstrap = bootstrap_result.card
        _log.info(
            "Phase 0 done: %s (%d obs features, %d reward components)",
            "cached" if bootstrap_result.cache_hit else "generated",
            len(bootstrap.proposed_initial_obs_features),
            len(bootstrap.proposed_initial_reward_components),
        )

        # Phase 1: rounds
        rounds: List[RoundResult] = []
        for round_idx in range(self.cfg.n_rounds):
            _log.info(
                "--- Round %d/%d ---",
                round_idx + 1,
                self.cfg.n_rounds,
            )
            round_output = self._run_round(
                round_idx=round_idx,
                bootstrap=bootstrap,
                bootstrap_dir=bootstrap_result.bootstrap_dir,
                history=rounds,
                seed=seed,
                task_overrides=task_overrides,
                algorithm=algorithm,
            )
            rounds.append(
                RoundResult(
                    round_idx=round_idx,
                    bundle=round_output["bundle"],
                    candidates=round_output["candidates"],
                    best_strategy_id=round_output["best_strategy_id"],
                    best_candidate_2M=round_output["mid_analysis"],
                    cross_round_summary=round_output["aggregate_text"],
                )
            )

        # Phase 2: pick global best, deep-train at full_frames
        global_best_round = max(
            rounds,
            key=lambda r: r.best_candidate_2M.stability_score,
        )
        global_best_strategy = next(
            s
            for s in global_best_round.bundle.strategies
            if s.strategy_id == global_best_round.best_strategy_id
        )
        _log.info(
            "=== Phase 2: global best is round %d strategy %s "
            "(score=%+.3f) — training at %d frames ===",
            global_best_round.round_idx,
            global_best_strategy.strategy_id,
            global_best_round.best_candidate_2M.stability_score,
            self.cfg.final_full_frames,
        )
        # Final deep-train uses the exact strategy that won the global
        # best round (with same evolve_reward gating + prior winner code
        # for inner-LLM context).
        final_round_evolve_reward = (
            self.cfg.evolve_reward
            and global_best_round.round_idx >= self.cfg.evolve_reward_from_round
        )
        final_prior_winner = self._extract_prior_winner_code(rounds)
        final_analysis = self._train_strategy(
            strategy=global_best_strategy,
            bootstrap_dir=bootstrap_result.bootstrap_dir,
            output_subdir=self.output_dir / "final",
            n_candidates=2,
            n_iterations=1,
            eval_frames=self.cfg.eval_frames,
            full_frames=self.cfg.final_full_frames,
            skip_full_training=False,
            seed=seed,
            task_overrides=task_overrides,
            algorithm=algorithm,
            stage_label="final",
            evolve_reward_override=final_round_evolve_reward,
            prior_winner_code=final_prior_winner,
        )

        elapsed = time.monotonic() - t0
        result = V4Result(
            bootstrap=bootstrap,
            rounds=rounds,
            global_best_round_idx=global_best_round.round_idx,
            final_M1=final_analysis.final_M1,
            final_M6=final_analysis.final_M6,
            peak_M1=final_analysis.peak_M1,
            peak_at_frame=final_analysis.peak_at_frame,
            final_stability_score=final_analysis.stability_score,
            elapsed_seconds=elapsed,
        )
        (self.output_dir / "v4_result.json").write_text(
            result.model_dump_json(indent=2),
        )
        _log.info(
            "=== LERO-MP v4 DONE === elapsed=%.0fs final_M1=%.3f "
            "peak_M1=%.3f score=%+.3f",
            elapsed,
            final_analysis.final_M1,
            final_analysis.peak_M1,
            final_analysis.stability_score,
        )
        return result

    # ── Private: one round ────────────────────────────────────

    def _run_round(
        self,
        round_idx: int,
        bootstrap: BootstrapCard,
        bootstrap_dir: Path,
        history: List[RoundResult],
        seed: int,
        task_overrides: Optional[Dict[str, Any]],
        algorithm: str,
    ) -> Dict[str, Any]:
        round_dir = self.output_dir / f"round_{round_idx:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        # 1. Strategist emits N strategies
        bundle = emit_strategies(
            bootstrap=bootstrap,
            round_history=history,
            meta_llm=self.meta_llm,
            round_idx=round_idx,
            n_strategies=self.cfg.n_strategies_per_round,
            fairness_excerpt=self._fairness_excerpt(),
        )
        (round_dir / "bundle.json").write_text(
            bundle.model_dump_json(indent=2),
        )

        # v4.1 Change E — capture the winning code from the immediately
        # prior round (if any) so the composer can inject it as
        # cross-round context for the inner LLM.
        prior_winner_code = self._extract_prior_winner_code(history)

        # 2. Run each strategy at eval_frames
        # v4.1 Change B — observation-only for rounds before
        # evolve_reward_from_round; afterwards strategist's
        # target_domain decides per strategy.
        round_evolve_reward = (
            self.cfg.evolve_reward and round_idx >= self.cfg.evolve_reward_from_round
        )
        candidates: List[CandidateAnalysis] = []
        for strategy in bundle.strategies:
            analysis = self._train_strategy(
                strategy=strategy,
                bootstrap_dir=bootstrap_dir,
                output_subdir=round_dir / strategy.strategy_id,
                # v4.1 Change C — multi-iteration inner loop
                n_candidates=self.cfg.inner_n_candidates_per_iter,
                n_iterations=self.cfg.inner_n_iterations,
                eval_frames=self.cfg.eval_frames,
                full_frames=self.cfg.eval_frames,  # same; skipped below
                skip_full_training=True,
                seed=seed,
                task_overrides=task_overrides,
                algorithm=algorithm,
                stage_label=f"r{round_idx}_{strategy.strategy_id}_eval",
                evolve_reward_override=round_evolve_reward,
                prior_winner_code=prior_winner_code,
            )
            candidates.append(analysis)

        # 3. Aggregate
        aggregate_text = aggregate_round_analysis(round_idx, candidates)
        (round_dir / "aggregate.txt").write_text(aggregate_text)
        _log.info("\n%s", aggregate_text)

        # 4. Pick best, mid-train at mid_frames
        ranked = sorted(
            candidates,
            key=lambda c: c.stability_score,
            reverse=True,
        )
        best_id = ranked[0].strategy_id
        best_strategy = next(s for s in bundle.strategies if s.strategy_id == best_id)
        _log.info(
            "Round %d winner: %s (score=%+.3f) — mid-training at %d frames",
            round_idx,
            best_id,
            ranked[0].stability_score,
            self.cfg.mid_frames,
        )
        mid_analysis = self._train_strategy(
            strategy=best_strategy,
            bootstrap_dir=bootstrap_dir,
            output_subdir=round_dir / f"mid_{best_id}",
            n_candidates=1,
            n_iterations=1,
            eval_frames=self.cfg.eval_frames,
            full_frames=self.cfg.mid_frames,
            skip_full_training=False,
            seed=seed,
            task_overrides=task_overrides,
            algorithm=algorithm,
            stage_label=f"r{round_idx}_{best_id}_mid",
            evolve_reward_override=round_evolve_reward,
            prior_winner_code=prior_winner_code,
        )
        return {
            "bundle": bundle,
            "candidates": candidates,
            "best_strategy_id": best_id,
            "mid_analysis": mid_analysis,
            "aggregate_text": aggregate_text,
        }

    # ── Private: train one strategy ──────────────────────────

    def _train_strategy(
        self,
        strategy: StrategyV4,
        bootstrap_dir: Path,
        output_subdir: Path,
        n_candidates: int,
        eval_frames: int,
        full_frames: int,
        skip_full_training: bool,
        seed: int,
        task_overrides: Optional[Dict[str, Any]],
        algorithm: str,
        stage_label: str,
        n_iterations: int = 1,
        evolve_reward_override: Optional[bool] = None,
        prior_winner_code: Optional[str] = None,
    ) -> CandidateAnalysis:
        """Compose the strategy's prompt, run an inner LeroLoop, return
        a CandidateAnalysis."""
        prompt_dir = compose_prompt_for_strategy(
            base_prompt_dir=bootstrap_dir,
            strategy=strategy,
            output_root=self.prompts_root,
            candidate_id=stage_label,
            prior_winner_code=prior_winner_code,
        )
        prompt_version = prompt_dir.name  # PromptLoader uses dirname

        # v4.1 Change B — round-gated evolve_reward (override) plus the
        # strategy's own revert_to_baseline_reward flag. revert always
        # wins — if the strategist explicitly asks for revert, honor it.
        if evolve_reward_override is not None:
            evolve_reward = evolve_reward_override
        else:
            evolve_reward = self.cfg.evolve_reward
        if strategy.revert_to_baseline_reward:
            evolve_reward = False

        inner_cfg = _make_inner_lero_config(
            base=self.spec.lero,
            eval_frames=eval_frames,
            full_frames=full_frames,
            skip_full_training=skip_full_training,
            n_candidates=n_candidates,
            n_iterations=n_iterations,
            evolve_reward=evolve_reward,
            evolve_observation=self.cfg.evolve_observation,
        )
        # Override prompt_version on a copy of the inner LLMConfig
        inner_llm_cfg = copy.copy(self.llm_cfg)
        inner_llm_cfg.prompt_version = prompt_version

        output_subdir = Path(output_subdir)
        output_subdir.mkdir(parents=True, exist_ok=True)
        loop = LeroLoop(
            spec=self.spec,
            lero_config=inner_cfg,
            llm_config=inner_llm_cfg,
            output_dir=output_subdir,
        )
        inner_result = loop.run(
            task_overrides=task_overrides,
            algorithm=algorithm,
            seed=seed,
        )
        traj = _trajectory_from_inner_result(inner_result)
        analysis = analyze_candidate_trajectory(
            candidate_id=stage_label,
            strategy_id=strategy.strategy_id,
            metrics_history=traj,
            weights=self.fitness_weights,
        )
        # Persist analysis for downstream inspection
        (output_subdir / "analysis.json").write_text(
            analysis.model_dump_json(indent=2),
        )
        return analysis

    # ── Helpers ────────────────────────────────────────────────

    def _fairness_excerpt(self) -> str:
        """Read the base prompt's fairness slot for the strategist."""
        f = self._writable_base / "fairness.txt"
        if f.exists():
            return f.read_text()
        return ""

    def _extract_prior_winner_code(
        self,
        history: List[RoundResult],
    ) -> Optional[str]:
        """v4.1 Change E — return the actual best obs/reward code from
        the most recent completed round so the composer can inject it
        as cross-round context for the inner LLM.

        Returns None if no prior round exists. Returns a markdown block
        with code excerpts otherwise.
        """
        if not history:
            return None
        last = history[-1]
        # Find the mid-train output dir; it has best_obs.py / best_reward.py
        mid_dir = (
            self.output_dir
            / f"round_{last.round_idx:02d}"
            / f"mid_{last.best_strategy_id}"
        )
        if not mid_dir.exists():
            return None

        chunks: List[str] = []
        score = last.best_candidate_2M.stability_score
        peak = last.best_candidate_2M.peak_M1
        final_m1 = last.best_candidate_2M.final_M1
        chunks.append(
            f"# Prior round {last.round_idx} winning code "
            f"(strategy {last.best_strategy_id}). "
            f"Mid-train@2M: stability_score={score:+.3f}, "
            f"peak_M1={peak:.3f}, final_M1={final_m1:.3f}, "
            f"shape={last.best_candidate_2M.shape_tag}."
        )
        chunks.append(
            "# Use this as a STARTING POINT, not as ground truth — "
            "your task is to refine or surpass it, not copy it."
        )
        for fname in ("best_obs.py", "best_reward.py"):
            p = mid_dir / fname
            if p.exists() and p.stat().st_size > 0:
                txt = p.read_text()
                # Truncate aggressively if huge
                if len(txt) > 4000:
                    txt = txt[:4000] + "\n# … (truncated)\n"
                chunks.append(f"```python\n# {fname}\n{txt}\n```")
        if len(chunks) == 2:  # only the headers, no code found
            return None
        return "\n".join(chunks)
