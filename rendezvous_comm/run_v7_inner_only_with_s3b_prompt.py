#!/usr/bin/env python3
"""Bug-check experiment: run v5/v7 inner_loop with S3b-local's prompt,
no meta-LLM intervention.

If this produces M1>0 like S3b-local does (replicated mean 0.845 at
10M deep-train, M1=0.05 at iter-1 1M-eval), it confirms our inner
loop code path is bug-free and v7's M1=0 outcome was purely a
prompt-content limitation. If this produces M1=0, there's a real
code bug somewhere in v5/v6/v7's inner_loop wrapper.

Same task params as S3b-local. Same eval budget (4 iters × 3 cands
× 1M = 12M frames inner). NO meta-LLM, NO bundle, NO reflection.
Just: load S3b-local prompt → call v5_inner_loop → log results.

Usage: python run_v7_inner_only_with_s3b_prompt.py --seed 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path


os.environ.setdefault("MPLBACKEND", "Agg")
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--algorithm", type=str, default="mappo")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--log-level", type=str, default="INFO")
    p.add_argument(
        "--config",
        type=str,
        default="configs/lero/s3b_local.yaml",
        help="S3b-local config (used as-is, no mutation)",
    )
    p.add_argument(
        "--n-iterations",
        type=int,
        default=4,
        help="inner LERO iterations (S3b-local default: 4)",
    )
    p.add_argument(
        "--n-candidates",
        type=int,
        default=3,
        help="candidates per iter (S3b-local default: 3)",
    )
    args = p.parse_args(argv)

    _configure_logging(args.log_level)
    log = logging.getLogger("rendezvous.lero.v7.bugcheck")

    if os.environ.get("LERO_ENCRYPTED"):
        from src.secrets_util import decrypt_and_load_env

        decrypt_and_load_env()

    import random as _random
    import numpy as _np
    import torch as _torch

    _random.seed(args.seed)
    _np.random.seed(args.seed)
    _torch.manual_seed(args.seed)
    log.info("RNG seeds locked at seed=%d", args.seed)

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        log.error("Config not found: %s", cfg_path)
        return 2

    from src.config import load_experiment

    spec = load_experiment(cfg_path)
    if spec.lero is None or spec.llm is None:
        log.error("Config missing 'lero:' or 'llm:' block")
        return 2

    # Verify the S3b-local config bits we care about
    log.info("S3b-local config check:")
    log.info(
        "  prompt_version=%s (must be v2_fewshot_k2_local)", spec.llm.prompt_version
    )
    log.info("  evolve_reward=%s (must be False)", spec.lero.evolve_reward)
    log.info("  evolve_observation=%s (must be True)", spec.lero.evolve_observation)
    log.info("  eval_frames=%d", spec.lero.eval_frames)
    log.info("  obs_state_mode=%s", spec.lero.obs_state_mode)
    if spec.llm.prompt_version != "v2_fewshot_k2_local":
        log.error("Config does not use S3b-local prompt — abort")
        return 2

    if args.output_dir:
        output_root = Path(args.output_dir)
    else:
        base = Path(os.environ.get("RESULTS_DIR", str(_ROOT / "results")))
        run_id = f"{time.strftime('%Y%m%d_%H%M')}_s{args.seed}"
        output_root = base / "v7_bugcheck_s3b_inner_only" / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    (output_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "purpose": "bug-check: v7 inner_loop + s3b-local prompt, no meta",
                "config_source": str(cfg_path),
                "seed": args.seed,
                "algorithm": args.algorithm,
                "n_iterations": args.n_iterations,
                "n_candidates": args.n_candidates,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            indent=2,
        )
    )

    log.info("output: %s", output_root)

    # Build the LeroLoop (used by inner_loop for _evaluate_candidate +
    # _build_initial_messages). Critical: prompt_version stays as
    # configured (v2_fewshot_k2_local).
    from src.lero.loop import LeroLoop

    base_loop = LeroLoop(
        spec=spec,
        lero_config=spec.lero,
        llm_config=spec.llm,
        output_dir=output_root / "_inner_legacy_outdir",
    )

    # Call run_inner_loop with the S3b-local prompt version. NO meta
    # interference, NO bundle, NO slot_edits.
    log.info("=== STARTING INNER LOOP (no meta) ===")
    log.info("  prompt_version: %s", spec.llm.prompt_version)
    log.info(
        "  iters × cands × frames = %d × %d × %d",
        args.n_iterations,
        args.n_candidates,
        spec.lero.eval_frames,
    )
    t0 = time.monotonic()

    from src.lero.v5.inner_loop import run_inner_loop

    result = run_inner_loop(
        base_loop=base_loop,
        metaprompt_version=spec.llm.prompt_version,
        n_iterations=args.n_iterations,
        n_candidates_per_iter=args.n_candidates,
        seed=args.seed,
        output_dir=output_root / "inner",
        task_overrides=None,
        algorithm=args.algorithm,
    )
    elapsed = time.monotonic() - t0

    # Summarize
    log.info("=== INNER LOOP DONE in %.0fs ===", elapsed)
    if result.best is None:
        log.error("No valid candidates produced — there IS a code bug")
        return 1

    summary = {
        "elapsed_s": elapsed,
        "best_M1": float(result.best.metrics.get("M1_success_rate", 0.0)),
        "best_M6": float(result.best.metrics.get("M6_coverage_progress", 0.0)),
        "best_shape": result.best.shape,
        "best_fitness": result.best.fitness,
        "n_iters_run": result.n_iters_run,
        "n_total_candidates": len(result.all_outcomes),
        "did_stagnate": result.did_stagnate,
        "fitness_trajectory": list(result.registry.fitness_trajectory),
        "all_M1_per_candidate": [
            float(o.metrics.get("M1_success_rate", 0.0)) for o in result.all_outcomes
        ],
        "all_M6_per_candidate": [
            float(o.metrics.get("M6_coverage_progress", 0.0))
            for o in result.all_outcomes
        ],
    }
    (output_root / "bugcheck_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    log.info(
        "best M1=%.3f best M6=%.3f shape=%s fitness=%+.3f",
        summary["best_M1"],
        summary["best_M6"],
        summary["best_shape"],
        summary["best_fitness"],
    )
    log.info("fitness trajectory: %s", summary["fitness_trajectory"])

    if summary["best_M1"] >= 0.02:
        log.info("✓ INNER LOOP IS BUG-FREE (M1 ≥ 0.02 matches S3b-local)")
    elif any(m1 >= 0.02 for m1 in summary["all_M1_per_candidate"]):
        log.info("✓ INNER LOOP IS BUG-FREE (some candidate hit M1 ≥ 0.02)")
    else:
        log.warning(
            "⚠ All candidates M1 < 0.02 — investigate (S3b-local "
            "typically gets at least one candidate at M1 ≥ 0.02 by iter 1)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
