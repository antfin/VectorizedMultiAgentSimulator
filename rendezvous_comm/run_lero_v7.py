#!/usr/bin/env python3
"""LERO v7 entry point — strategy bundle + grounded reflection."""

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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=str)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--algorithm", type=str, default="mappo")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--log-level", type=str, default="INFO")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)

    _configure_logging(args.log_level)
    log = logging.getLogger("rendezvous.lero.v7.cli")

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
    if "v7" not in raw:
        log.error("Config has no top-level 'v7:' block")
        return 2

    from src.config import load_experiment

    spec = load_experiment(cfg_path)
    if spec.lero is None or spec.llm is None:
        log.error("Config missing 'lero:' or 'llm:' block")
        return 2

    v7_raw = raw["v7"]
    from src.lero.v7.outer_loop import V7OuterConfig

    v7_cfg = V7OuterConfig(
        max_outer=int(v7_raw.get("max_outer", 5)),
        n_inner_iter=int(v7_raw.get("n_inner_iter", 4)),
        n_inner_candidates_per_iter=int(v7_raw.get("n_inner_candidates_per_iter", 3)),
        eval_frames=int(v7_raw.get("eval_frames", 1_000_000)),
        base_prompt_version=str(
            v7_raw.get(
                "base_prompt_version",
                "v2_fewshot_modular_v2_local",
            )
        ),
        meta_model=str(v7_raw.get("meta_model", spec.llm.model)),
        meta_temperature=float(
            v7_raw.get("meta_temperature", spec.llm.temperature),
        ),
        task_summary=str(v7_raw.get("task_summary", spec.description or "")),
    )

    if args.output_dir:
        output_root = Path(args.output_dir)
    else:
        base = Path(os.environ.get("RESULTS_DIR", str(_ROOT / "results")))
        run_id = f"{time.strftime('%Y%m%d_%H%M')}_s{args.seed}"
        output_root = base / "lero_v7" / spec.exp_id.lower() / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    (output_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "exp_id": spec.exp_id,
                "name": spec.name,
                "config_source": str(cfg_path),
                "seed": args.seed,
                "algorithm": args.algorithm,
                "v7_config": v7_raw,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            indent=2,
        )
    )

    log.info("v7 ready: exp=%s out=%s", spec.exp_id, output_root)

    if args.dry_run:
        log.info("Dry run — exiting before LLM calls.")
        return 0

    spec.lero.eval_frames = v7_cfg.eval_frames

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
        model=v7_cfg.meta_model,
        temperature=v7_cfg.meta_temperature,
        max_retries=spec.llm.max_retries,
        prompt_version=spec.llm.prompt_version,
    )
    meta_llm = LLMClient(meta_cfg)

    if args.resume and not args.output_dir:
        log.error("--resume requires --output-dir")
        return 2

    from src.lero.v7.outer_loop import run_v7_outer_loop

    summary = run_v7_outer_loop(
        spec=spec,
        base_loop=base_loop,
        cfg=v7_cfg,
        meta_llm=meta_llm,
        output_dir=output_root,
        seed=args.seed,
        algorithm=args.algorithm,
        resume=args.resume,
    )
    log.info(
        "=== DONE === early_stopped=%s outers=%d final_strategy=%s",
        summary.get("early_stopped"),
        summary.get("outer_iters_completed"),
        summary.get("bundle_chosen_at_end"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
