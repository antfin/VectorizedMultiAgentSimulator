#!/usr/bin/env python3
"""Entry point for LERO v5 (focused-depth, two-level textual gradient).

Usage:
    python run_lero_v5.py configs/lero_v5/rendezvous_k2_smoke.yaml --seed 0

The config YAML must contain a top-level ``v5:`` block (see
configs/lero_v5/*.yaml for the schema).

Outputs land under ``results/lero_v5/<exp_id>/<timestamp>_s<seed>/``:
    prompts/                       # writable PromptLoader root
        v5_outer_0_seed<S>/        # initial metaprompt copy
        v5_outer_1_seed<S>/        # refined after outer iter 0
        v5_outer_2_seed<S>/        # refined after outer iter 1
        ...
    outer_00_inner/                # inner-loop artifacts per outer iter
        iter_0/, iter_1/, ...      # per-inner-iter candidate code+feedback
    outer_01_inner/, ...
    deep_train/                    # final 10M training of global best
        benchmarl_final/
        best_policy.pt
        final_metrics.json
    v5_summary.json                # top-level snapshot
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import yaml

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--algorithm", type=str, default="mappo")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from _v5_checkpoint.pkl in output_dir if present. "
        "Requires --output-dir to point at the prior run's directory.",
    )
    args = parser.parse_args(argv)

    _configure_logging(args.log_level)
    log = logging.getLogger("rendezvous.lero.v5.cli")

    if os.environ.get("LERO_ENCRYPTED"):
        from src.secrets_util import decrypt_and_load_env

        n_keys = len(decrypt_and_load_env())
        log.info("Decrypted %d LLM key(s)", n_keys)

    import random as _random

    import numpy as _np
    import torch as _torch

    _random.seed(args.seed)
    _np.random.seed(args.seed)
    _torch.manual_seed(args.seed)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(args.seed)
    log.info("RNG seeds locked at seed=%d", args.seed)

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        log.error("Config not found: %s", cfg_path)
        return 2

    raw = yaml.safe_load(cfg_path.read_text())
    if "v5" not in raw:
        log.error("Config has no top-level 'v5:' block")
        return 2

    from src.config import load_experiment

    spec = load_experiment(cfg_path)
    if spec.lero is None or spec.llm is None:
        log.error("Config missing 'lero:' or 'llm:' block")
        return 2

    v5_raw = raw["v5"]
    from src.lero.v5.outer_loop import V5OuterConfig

    v5_cfg = V5OuterConfig(
        n_outer=int(v5_raw["n_outer"]),
        n_inner_iter=int(v5_raw["n_inner_iter"]),
        n_inner_candidates_per_iter=int(v5_raw["n_inner_candidates_per_iter"]),
        eval_frames=int(v5_raw["eval_frames"]),
        full_frames=int(v5_raw["full_frames"]),
        pivot_eps=float(v5_raw.get("pivot_eps", 0.05)),
        base_prompt_version=str(
            v5_raw.get("base_prompt_version", "v2_fewshot_modular_v2")
        ),
        meta_model=str(v5_raw.get("meta_model", spec.llm.model)),
        meta_temperature=float(
            v5_raw.get(
                "meta_temperature",
                spec.llm.temperature,
            )
        ),
        task_summary=str(v5_raw.get("task_summary", spec.description or "")),
    )

    if args.output_dir:
        output_root = Path(args.output_dir)
    else:
        base = Path(os.environ.get("RESULTS_DIR", str(_ROOT / "results")))
        run_id = f"{time.strftime('%Y%m%d_%H%M')}_s{args.seed}"
        output_root = base / "lero_v5" / spec.exp_id.lower() / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    (output_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "exp_id": spec.exp_id,
                "name": spec.name,
                "config_source": str(cfg_path),
                "seed": args.seed,
                "algorithm": args.algorithm,
                "v5_config": v5_raw,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            indent=2,
        )
    )

    log.info("v5 ready: exp=%s out=%s", spec.exp_id, output_root)

    if args.dry_run:
        log.info("Dry run — exiting before any LLM calls.")
        return 0

    # Override eval_frames + full_frames in spec.lero so LeroLoop's
    # _evaluate_candidate / _full_training pick them up.
    spec.lero.eval_frames = v5_cfg.eval_frames
    spec.lero.full_frames = v5_cfg.full_frames

    from src.lero.loop import LeroLoop

    base_loop = LeroLoop(
        spec=spec,
        lero_config=spec.lero,
        llm_config=spec.llm,
        output_dir=output_root / "_inner_legacy_outdir",
    )

    # Build meta-LLM client
    from src.lero.config import LLMConfig
    from src.lero.llm_client import LLMClient

    meta_cfg = LLMConfig(
        model=v5_cfg.meta_model,
        temperature=v5_cfg.meta_temperature,
        max_retries=spec.llm.max_retries,
        prompt_version=spec.llm.prompt_version,
    )
    meta_llm = LLMClient(meta_cfg)

    from src.lero.v5.outer_loop import run_v5_outer_loop

    if args.resume and not args.output_dir:
        log.error("--resume requires --output-dir pointing at the prior run.")
        return 2
    result = run_v5_outer_loop(
        spec=spec,
        base_loop=base_loop,
        cfg=v5_cfg,
        meta_llm=meta_llm,
        output_dir=output_root,
        seed=args.seed,
        algorithm=args.algorithm,
        resume=args.resume,
    )

    log.info(
        "=== DONE === final_M1=%.3f outer_traj=%s elapsed=%.0fs",
        result.get("M1_success_rate", 0.0),
        result.get("_v5_outer_fitness_trajectory", []),
        result.get("_v5_elapsed_s", 0.0),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
