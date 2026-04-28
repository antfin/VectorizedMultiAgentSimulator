#!/usr/bin/env python3
"""Parse a LERO-MP v3 dry-run results directory and emit pass/fail per
the §9 success criteria in docs/lero_metaprompt_v3_plan.md.

Usage:
    python rendezvous_comm/scripts/review_dry_run.py \\
        results/lero_mp/<exp_id>/<run_stamp>

Exits 0 if ≥3 of 5 success criteria fire, 1 otherwise. Prints a
human-readable report either way.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List


RATIONALE_SIGNAL_RE = re.compile(r"M(?:1|2|3|4|6|8|9)\s*=\s*[-+]?\d")


def _load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    entries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def check_retry_salvage(run_dir: Path) -> Dict:
    """Criterion 1: ≥20% of successful candidates required retries."""
    total = 0
    retried = 0
    for attempts_file in run_dir.rglob("candidate_*_attempts.json"):
        data = _load_json(attempts_file)
        if not data:
            continue
        total += 1
        if data.get("attempts", 1) > 1:
            retried += 1
    pct = (retried / total * 100) if total else 0.0
    return {
        "name": "retry_salvage",
        "total_candidates": total,
        "retried": retried,
        "pct_retried": pct,
        "passed": pct >= 20.0 and retried >= 1,
    }


def check_strategist_cites_signal(run_dir: Path) -> Dict:
    """Criterion 2: Strategist rationale cites a behavioral signal by value."""
    log_entries = []
    for mlog in run_dir.rglob("mutation_log.jsonl"):
        log_entries.extend(_load_jsonl(mlog))
    cites = 0
    non_default_include = 0
    for e in log_entries:
        sc = e.get("strategy_card") or {}
        rationale = sc.get("rationale", "") or ""
        if RATIONALE_SIGNAL_RE.search(rationale):
            cites += 1
        include = sc.get("include_signals") or ["scalar"]
        if set(include) != {"scalar"}:
            non_default_include += 1
    return {
        "name": "strategist_cites_signal",
        "total_strategies": len(log_entries),
        "rationale_cites_metric": cites,
        "include_signals_non_default": non_default_include,
        "passed": cites >= 1 and non_default_include >= 1,
    }


def check_critic_revision(run_dir: Path) -> Dict:
    """Criterion 3: Editor Critic triggered a revision at least once."""
    # Look for 'Editor critique: revisions=N' lines in outer-loop log,
    # if captured. Otherwise look for mutation_log entries that flag
    # critique_revisions > 0.
    hits = 0
    total = 0
    for mlog in run_dir.rglob("mutation_log.jsonl"):
        for e in _load_jsonl(mlog):
            total += 1
            revs = e.get("critique_revisions", 0) or 0
            if revs > 0:
                hits += 1
    # Fallback to log file scan if mutation_log doesn't track it yet
    log_hits = 0
    for log_file in run_dir.rglob("*.log"):
        txt = log_file.read_text(errors="ignore")
        log_hits += len(
            re.findall(r"Editor critique:\s+revisions=([1-9])", txt)
        )
    return {
        "name": "critic_revision",
        "log_revisions": log_hits,
        "mutation_log_revisions": hits,
        "passed": (hits + log_hits) >= 1,
    }


def check_marginal_improvement(run_dir: Path) -> Dict:
    """Criterion 4: ≥1 mutation scored marginal or strong improvement."""
    marginal = 0
    strong = 0
    total = 0
    for mlog in run_dir.rglob("mutation_log.jsonl"):
        for e in _load_jsonl(mlog):
            verdict = e.get("verdict")
            if verdict is None:
                continue
            total += 1
            if verdict == "marginal_improvement":
                marginal += 1
            elif verdict == "strong_improvement":
                strong += 1
    return {
        "name": "marginal_improvement",
        "total_verdicts": total,
        "marginal": marginal,
        "strong": strong,
        "passed": (marginal + strong) >= 1,
    }


def check_cache_replay(run_dir: Path) -> Dict:
    """Criterion 5: read_only cache replay matches bit-exact (offline only)."""
    # This criterion requires a *second* run in read_only mode; we
    # can only check whether a cache directory exists + was populated.
    cache_root = Path.home() / ".cache" / "lero_llm"
    n = 0
    if cache_root.exists():
        n = len(list(cache_root.glob("*.txt")))
    return {
        "name": "cache_replay_ready",
        "cache_entries": n,
        "passed": n >= 1,
        "note": (
            "Full bit-exact replay check must be run manually: set "
            "LERO_LLM_CACHE_MODE=read_only and rerun seed 0; compare "
            "strategy_card + mutation_log entries."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir", type=str,
        help="Path to the LERO-MP run directory "
             "(e.g. results/lero_mp/<exp>/<timestamp_sN>)",
    )
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"ERROR: {run_dir} does not exist", file=sys.stderr)
        return 2

    checks = [
        check_retry_salvage(run_dir),
        check_strategist_cites_signal(run_dir),
        check_critic_revision(run_dir),
        check_marginal_improvement(run_dir),
        check_cache_replay(run_dir),
    ]
    print(f"LERO-MP v3 Dry-Run Review — {run_dir}")
    print("=" * 72)
    passed = 0
    for c in checks:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"[{status}] {c['name']}")
        for k, v in c.items():
            if k in ("name", "passed"):
                continue
            print(f"        {k}: {v}")
        if c["passed"]:
            passed += 1
    print("-" * 72)
    print(f"Score: {passed}/5 (need ≥3 to clear the bar)")
    return 0 if passed >= 3 else 1


if __name__ == "__main__":
    sys.exit(main())
