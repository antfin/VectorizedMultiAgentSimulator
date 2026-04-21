#!/usr/bin/env python3
"""Entry point for LERO-MP (LERO + meta-prompting outer loop).

Usage:
    python rendezvous_comm/run_lero_mp.py \
        rendezvous_comm/configs/lero_mp/mp_dryrun.yaml

    python rendezvous_comm/run_lero_mp.py \
        rendezvous_comm/configs/lero_mp/mp_k2_obsonly_cr035.yaml --seed 0

The config YAML must contain a top-level ``lero:`` block and a
``meta_prompt:`` block (see configs/lero_mp/*.yaml). When
``meta_prompt.enabled: false`` the outer loop runs exactly one inner
LeroLoop and exits — useful for smoke tests.

Outputs land under ``results/lero_mp/<exp_id>/<timestamp>/``:
    outer_000_<parent_version>/   # inner LeroLoop artifacts
    outer_001_<mutated_version>/  # inner LeroLoop artifacts (next iter)
    history.json                  # rolling TemplateRecord list
    final_result.json             # OuterLoopResult snapshot at exit

Environment:
    OPENAI_API_KEY / ANTHROPIC_API_KEY etc. are read by LiteLLM for
    both the inner-loop LLM (from ``llm:``) and the meta-LLM (from
    ``meta_prompt.meta_model``).
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
from src.lero.meta.outer_loop import LeroMpOuterLoop  # noqa: E402


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a LERO-MP experiment (meta-prompt outer loop).",
    )
    parser.add_argument(
        "config", type=str,
        help="Path to a YAML config with lero: + meta_prompt: blocks.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Base seed for the inner LeroLoop runs.",
    )
    parser.add_argument(
        "--algorithm", type=str, default="mappo",
        help="MARL algorithm for inner-loop training.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory (default: results/lero_mp/<exp_id>/<ts>).",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        help="DEBUG / INFO / WARNING / ERROR.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Load config and print the resolved plan; do not train.",
    )
    args = parser.parse_args(argv)

    _configure_logging(args.log_level)
    log = logging.getLogger("rendezvous.lero.mp.cli")

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        log.error("Config not found: %s", cfg_path)
        return 2

    spec = load_experiment(cfg_path)

    if spec.lero is None:
        log.error("Config has no 'lero:' block — use train.py for non-LERO runs.")
        return 2
    if spec.meta_prompt is None:
        log.error(
            "Config has no 'meta_prompt:' block — this is a LERO-MP runner. "
            "Add `meta_prompt: {enabled: false}` for a single-template run.",
        )
        return 2

    # Output layout
    output_root = (
        Path(args.output_dir)
        if args.output_dir
        else _ROOT / "results" / "lero_mp" / spec.exp_id.lower() / time.strftime("%Y%m%d_%H%M")
    )
    output_root.mkdir(parents=True, exist_ok=True)

    # Write a small manifest so post-hoc analysis can find everything.
    (output_root / "run_manifest.json").write_text(json.dumps({
        "exp_id": spec.exp_id,
        "name": spec.name,
        "config_source": str(cfg_path),
        "seed": args.seed,
        "algorithm": args.algorithm,
        "prompt_version": spec.llm.prompt_version,
        "meta_prompt_enabled": spec.meta_prompt.enabled,
        "whitelist_strict": spec.lero.whitelist_strict,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))

    log.info(
        "LERO-MP ready: exp=%s, prompt=%s, meta_enabled=%s, out=%s",
        spec.exp_id, spec.llm.prompt_version,
        spec.meta_prompt.enabled, output_root,
    )

    if args.dry_run:
        log.info("Dry run — exiting before any training.")
        return 0

    outer = LeroMpOuterLoop(
        spec=spec,
        lero_config=spec.lero,
        llm_config=spec.llm,
        meta_config=spec.meta_prompt,
        output_dir=output_root,
    )
    result = outer.run(
        algorithm=args.algorithm,
        base_seed=args.seed,
    )

    # Persist the full result snapshot for downstream analysis.
    (output_root / "final_result.json").write_text(
        json.dumps(result.to_dict(), indent=2)
    )
    log.info(
        "=== DONE === stop_reason=%s  elapsed=%.1fs  final_version=%s",
        result.stop_reason.value, result.elapsed_seconds, result.final_version,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
