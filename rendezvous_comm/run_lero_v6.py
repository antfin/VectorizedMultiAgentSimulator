#!/usr/bin/env python3
"""Entry point for LERO v6 — simplicity-first meta-strategy, inner-only.

Usage:
    python run_lero_v6.py configs/lero_v6/rendezvous_k2_smoke.yaml --seed 0

The config YAML must contain a top-level ``v6:`` block. See
``docs/v6_plan.md`` for design rationale.

Outputs land under ``results/lero_v6/<exp_id>/<timestamp>_s<seed>/``:
    prompts/                       # writable PromptLoader root
        v6_outer_0_seed<S>/        # initial metaprompt copy
        v6_outer_1_seed<S>/        # one-strategy refinement after outer 0
        ...
    outer_NN_inner/                # per-outer-iter inner-loop artifacts
        iter_{0..3}/feedback.txt   # inner feedback
        _decision.json             # enforced V6MetaDecision
        _meta_response.txt         # raw meta-LLM response
    _v6_checkpoint.pkl
    v6_summary.json
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
        help="Resume from _v6_checkpoint.pkl in output_dir if present.",
    )
    args = parser.parse_args(argv)

    _configure_logging(args.log_level)
    log = logging.getLogger("rendezvous.lero.v6.cli")

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
    if "v6" not in raw:
        log.error("Config has no top-level 'v6:' block")
        return 2

    from src.config import load_experiment

    spec = load_experiment(cfg_path)
    if spec.lero is None or spec.llm is None:
        log.error("Config missing 'lero:' or 'llm:' block")
        return 2

    v6_raw = raw["v6"]
    from src.lero.v6.outer_loop import V6OuterConfig

    v6_cfg = V6OuterConfig(
        max_outer=int(v6_raw.get("max_outer", 5)),
        n_inner_iter=int(v6_raw.get("n_inner_iter", 4)),
        n_inner_candidates_per_iter=int(v6_raw.get("n_inner_candidates_per_iter", 3)),
        eval_frames=int(v6_raw.get("eval_frames", 1_000_000)),
        base_prompt_version=str(
            v6_raw.get("base_prompt_version", "v2_fewshot_modular_v2_local")
        ),
        meta_model=str(v6_raw.get("meta_model", spec.llm.model)),
        meta_temperature=float(v6_raw.get("meta_temperature", spec.llm.temperature)),
        task_summary=str(v6_raw.get("task_summary", spec.description or "")),
    )

    if args.output_dir:
        output_root = Path(args.output_dir)
    else:
        base = Path(os.environ.get("RESULTS_DIR", str(_ROOT / "results")))
        run_id = f"{time.strftime('%Y%m%d_%H%M')}_s{args.seed}"
        output_root = base / "lero_v6" / spec.exp_id.lower() / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    (output_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "exp_id": spec.exp_id,
                "name": spec.name,
                "config_source": str(cfg_path),
                "seed": args.seed,
                "algorithm": args.algorithm,
                "v6_config": v6_raw,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            indent=2,
        )
    )

    log.info("v6 ready: exp=%s out=%s", spec.exp_id, output_root)

    if args.dry_run:
        log.info("Dry run — exiting before any LLM calls.")
        return 0

    # Set inner LERO eval budget from v6 config
    spec.lero.eval_frames = v6_cfg.eval_frames

    from src.lero.loop import LeroLoop

    base_loop = LeroLoop(
        spec=spec,
        lero_config=spec.lero,
        llm_config=spec.llm,
        output_dir=output_root / "_inner_legacy_outdir",
    )

    from src.lero.config import LLMConfig
    from src.lero.llm_client import LLMClient

    meta_cfg = LLMConfig(
        model=v6_cfg.meta_model,
        temperature=v6_cfg.meta_temperature,
        max_retries=spec.llm.max_retries,
        prompt_version=spec.llm.prompt_version,
    )
    meta_llm = LLMClient(meta_cfg)

    if args.resume and not args.output_dir:
        log.error("--resume requires --output-dir pointing at the prior run.")
        return 2

    from src.lero.v6.outer_loop import run_v6_outer_loop

    summary = run_v6_outer_loop(
        spec=spec,
        base_loop=base_loop,
        cfg=v6_cfg,
        meta_llm=meta_llm,
        output_dir=output_root,
        seed=args.seed,
        algorithm=args.algorithm,
        resume=args.resume,
    )

    log.info(
        "=== DONE === early_stopped=%s outers=%d best_M1=%.3f shape=%s",
        summary.get("early_stopped", False),
        summary.get("outer_iters_completed", 0),
        summary.get("global_best_M1", 0.0),
        summary.get("global_best_shape", "n/a"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
