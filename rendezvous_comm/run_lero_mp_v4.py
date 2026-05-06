#!/usr/bin/env python3
"""Entry point for LERO-MP v4 (description-driven, multi-strategy).

Usage:
    python rendezvous_comm/run_lero_mp_v4.py \\
        rendezvous_comm/configs/lero_mp/v4/rendezvous_k2.yaml --seed 0

The config YAML must contain a ``v4:`` block (LeroMPv4Config) that
includes ``description_path:`` pointing at the human-written task .md.

Outputs land under ``results/lero_mp_v4/<exp_id>/<timestamp>_s<seed>/``:
    bootstrap/                     # Phase 0: card.json + thoughts.md
    bootstrap_cache/               # description-keyed cache
    prompts/                       # composed strategy-specific prompts
    round_00/                      # Phase 1 rounds
        bundle.json                # the StrategyBundle
        S1/, S2/, S3/              # per-strategy candidate runs
        aggregate.txt              # cross-strategy summary
        mid_S2/                    # mid-train (best of round)
    round_01/, round_02/, ...
    final/                         # Phase 2: 10M deep-train of global best
    v4_result.json                 # V4Result snapshot
"""

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

from src.config import load_experiment  # noqa: E402
from src.lero.meta.v4_outer_loop import LeroMpV4OuterLoop  # noqa: E402


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
    args = parser.parse_args(argv)

    _configure_logging(args.log_level)
    log = logging.getLogger("rendezvous.lero.mp.v4.cli")

    # OVH-encrypted env decrypt (no-op locally)
    if os.environ.get("LERO_ENCRYPTED"):
        from src.secrets_util import decrypt_and_load_env

        n_keys = len(decrypt_and_load_env())
        log.info("Decrypted %d LLM key(s) from LERO_ENCRYPTED", n_keys)

    # RNG seed-lock
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

    spec = load_experiment(cfg_path)
    if spec.lero is None:
        log.error("Config has no 'lero:' block")
        return 2
    if not hasattr(spec, "v4") or spec.v4 is None:
        log.error("Config has no 'v4:' block — use run_lero_mp.py for v3")
        return 2

    # Resolve description_path relative to the config file when relative
    desc = Path(spec.v4.description_path)
    if not desc.is_absolute():
        # Try relative to config file dir, then repo root
        candidates = [
            cfg_path.parent / spec.v4.description_path,
            _ROOT / spec.v4.description_path,
            _ROOT.parent / spec.v4.description_path,
        ]
        for c in candidates:
            if c.exists():
                spec.v4.description_path = str(c)
                break
        else:
            log.error(
                "description_path not found at any of: %s",
                [str(c) for c in candidates],
            )
            return 2

    # Output layout
    if args.output_dir:
        output_root = Path(args.output_dir)
    else:
        base = Path(os.environ.get("RESULTS_DIR", str(_ROOT / "results")))
        run_id = f"{time.strftime('%Y%m%d_%H%M')}_s{args.seed}"
        output_root = base / "lero_mp_v4" / spec.exp_id.lower() / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    # Mirror prompts to writable area on OVH (read-only code mount)
    if os.environ.get("RESULTS_DIR"):
        import shutil

        src_prompts = _ROOT / "src" / "lero" / "prompts"
        dst_prompts = output_root / "_base_prompts"
        if src_prompts.exists() and not dst_prompts.exists():
            shutil.copytree(src_prompts, dst_prompts)
            log.info("Mirrored read-only prompts %s → %s", src_prompts, dst_prompts)

    # Manifest
    (output_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "exp_id": spec.exp_id,
                "name": spec.name,
                "config_source": str(cfg_path),
                "seed": args.seed,
                "algorithm": args.algorithm,
                "description_path": spec.v4.description_path,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            indent=2,
        )
    )

    log.info(
        "v4 ready: exp=%s, description=%s, out=%s",
        spec.exp_id,
        spec.v4.description_path,
        output_root,
    )

    if args.dry_run:
        log.info("Dry run — exiting before bootstrap.")
        return 0

    inner_llm_cfg = spec.llm
    outer = LeroMpV4OuterLoop(
        spec=spec,
        v4_config=spec.v4,
        llm_config=inner_llm_cfg,
        output_dir=output_root,
    )
    result = outer.run(
        algorithm=args.algorithm,
        seed=args.seed,
    )

    log.info(
        "=== DONE === final_M1=%.3f peak_M1=%.3f score=%+.3f elapsed=%.0fs",
        result.final_M1,
        result.peak_M1,
        result.final_stability_score,
        result.elapsed_seconds,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
