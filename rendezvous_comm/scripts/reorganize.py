#!/usr/bin/env python3
"""Results reorganization script.

Consolidates scattered experiment dirs into er1/runs/, er2/runs/, er3/runs/
with renamed variant tags, master CSV, and changelog.

Usage:
    python reorganize.py --dry-run     # preview all actions
    python reorganize.py --execute     # perform copies + build CSV
    python reorganize.py --delete      # remove originals after verification
"""
import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

RESULTS = Path(__file__).parent.parent / "results"
DOCS = Path(__file__).parent.parent / "docs"

# ── Mapping: (source_exp_dir, run_dir_pattern) → (family, variant_tag_func) ──

def _extract_seed(run_dir_name: str) -> str:
    """Extract seed from run dir name like ...._s0."""
    parts = run_dir_name.split("_s")
    return parts[-1] if len(parts) > 1 else "0"


def _extract_k_l(run_dir_name: str) -> Tuple[str, str]:
    """Extract k and l values from run dir name."""
    k = "2"
    l = "035"
    if "_k1_" in run_dir_name:
        k = "1"
    elif "_k2_" in run_dir_name:
        k = "2"
    if "_l025_" in run_dir_name:
        l = "025"
    elif "_l035_" in run_dir_name:
        l = "035"
    return k, l


def build_variant_tag(exp_dir: str, run_dir_name: str, config: dict) -> str:
    """Build new variant tag from old dir name and config."""
    seed = _extract_seed(run_dir_name)
    k, l = _extract_k_l(run_dir_name)
    task = config.get("task", {})
    train = config.get("train", {})

    parts = []

    # Agent lidar
    if task.get("use_agent_lidar", False):
        parts.append("al")

    # Reward shaping
    if task.get("shared_reward", False) and task.get("agent_collision_penalty", -0.1) > -0.05:
        parts.append("lp_sr")
    elif task.get("shared_reward", False):
        parts.append("sr")
    elif task.get("agent_collision_penalty", -0.1) > -0.05:
        parts.append("lp")

    # Ablation-specific — check both exp_dir and run_dir_name
    # to handle runs nested under er1_ablation/
    ident = f"{exp_dir} {run_dir_name}"
    is_ablation = False
    if "abl_a2" in ident:
        parts = ["abl_ent001"]
        is_ablation = True
    elif "abl_a" in ident and "abl_a2" not in ident:
        parts = ["abl_ent01"]
        is_ablation = True
    elif "abl_b" in ident:
        parts = ["abl_lr1e4"]
        is_ablation = True
    elif "abl_c" in ident:
        parts = ["abl_lam095"]
        is_ablation = True
    elif "abl_g" in ident:
        parts = ["abl_20m"]
        is_ablation = True
    elif "abl_h" in ident:
        parts = ["abl_net512_relu"]
        is_ablation = True
    elif "abl_i" in ident:
        parts = ["abl_k1_sanity"]
        is_ablation = True

    # Comm params (ER2)
    dim_c = task.get("dim_c", 0)
    if dim_c > 0:
        comm_prox = task.get("comm_proximity", True)
        if comm_prox:
            parts.append(f"prox_dc{dim_c}")
        else:
            parts.append(f"bc_dc{dim_c}")

    # GNN model (ER3)
    model_type = train.get("model_type", "mlp")
    if model_type == "gnn":
        gnn_class = train.get("gnn_class", "GATv2Conv")
        if "GATv2" in gnn_class:
            parts.append("gatv2")
        elif "GraphConv" in gnn_class:
            parts.append("graphconv")
        else:
            parts.append(gnn_class.lower())

    # Extended training (skip if already encoded by ablation tag)
    max_frames = train.get("max_n_frames", 10_000_000)
    if not is_ablation:
        if max_frames == 20_000_000:
            parts.append("20m")
        elif max_frames == 30_000_000:
            parts.append("30m")

    # max_steps — all pre-fix runs actually used 100
    # Tag runs whose config intended non-default steps
    cfg_max_steps = task.get("max_steps", 200)
    if cfg_max_steps != 200:
        parts.append("ms100")

    # Non-standard agent/target counts
    n_agents = task.get("n_agents", 4)
    n_targets = task.get("n_targets", 4)
    if n_agents != 4 or n_targets != 4:
        parts.append(f"n{n_agents}_t{n_targets}")

    # Non-standard algorithm
    algorithm = config.get("algorithm",
                           train.get("algorithm", "mappo"))
    if algorithm != "mappo":
        parts.append(algorithm)

    # If no modifiers, it's default
    if not parts:
        parts.append("default")

    # Add k and l
    parts.append(f"k{k}_l{l}")
    parts.append(f"s{seed}")

    return "_".join(parts)


def determine_family(exp_dir: str, config: dict) -> str:
    """Determine er1/er2/er3 family."""
    task = config.get("task", {})
    train = config.get("train", {})

    if train.get("model_type") == "gnn":
        return "er3"
    if task.get("dim_c", 0) > 0:
        return "er2"
    return "er1"


def experiment_label(variant: str, family: str, config: dict) -> str:
    """Human-readable experiment label."""
    task = config.get("task", {})
    train = config.get("train", {})
    parts = []

    if family == "er3":
        parts.append("ER3")
    elif family == "er2":
        parts.append("ER2")
    else:
        parts.append("ER1")

    if "abl_ent01_" in variant and "ent001" not in variant:
        return f"{parts[0]} abl: entropy=0.01"
    if "abl_ent001" in variant:
        return f"{parts[0]} abl: entropy=0.001"
    if "abl_lr1e4" in variant:
        return f"{parts[0]} abl: lr=1e-4"
    if "abl_lam095" in variant:
        return f"{parts[0]} abl: lmbda=0.95"
    if "abl_20m" in variant:
        return f"{parts[0]} abl: 20M frames"
    if "abl_net512" in variant:
        return f"{parts[0]} abl: net=[512,256]+ReLU"
    if "abl_k1_sanity" in variant:
        return f"{parts[0]} abl: k=1 sanity"

    if task.get("use_agent_lidar"):
        parts.append("+ AL")
    if task.get("shared_reward") and task.get("agent_collision_penalty", -0.1) > -0.05:
        parts.append("+ LP+SR")
    elif task.get("shared_reward"):
        parts.append("+ SR")
    elif task.get("agent_collision_penalty", -0.1) > -0.05:
        parts.append("+ LP")

    dim_c = task.get("dim_c", 0)
    if dim_c > 0:
        prox = "prox" if task.get("comm_proximity", True) else "bc"
        parts.append(f"{prox} dc={dim_c}")

    if train.get("model_type") == "gnn":
        gnn = train.get("gnn_class", "GATv2Conv")
        parts.append(f"GNN:{gnn}")

    max_frames = train.get("max_n_frames", 10_000_000)
    if max_frames > 10_000_000:
        parts.append(f"({max_frames//1_000_000}M)")

    cfg_ms = task.get("max_steps", 200)
    if cfg_ms != 200:
        parts.append("(ms100)")

    return " ".join(parts)


def compute_notes(variant: str, family: str, config: dict, metrics: dict) -> str:
    """Compute notes for the run."""
    notes = []
    task = config.get("task", {})

    # All pre-fix runs used max_steps=100 regardless of config
    cfg_ms = task.get("max_steps", 200)
    if cfg_ms != 100:
        notes.append(f"max_steps_actual=100_cfg_intended={cfg_ms}")

    # comm_ignored (ER2 bc without LP+SR = identical to ER1)
    if family == "er2" and not task.get("shared_reward", False):
        dim_c = task.get("dim_c", 0)
        if dim_c > 0 and not task.get("comm_proximity", True):
            notes.append("comm_ignored")

    return "; ".join(notes) if notes else ""


    # Subdirs under er1_ablation/ that are confirmed duplicates
    # of standalone dirs. Only er1_abl_a/ and er1_old/ are unique.
ABLATION_DUPES = {"er1_abl_a2", "er1_abl_b", "er1_abl_c",
                  "er1_abl_g", "er1_abl_h", "er1_abl_i"}


def collect_all_runs() -> List[dict]:
    """Collect all valid runs with config and metrics."""
    runs = []
    for exp_dir in sorted(RESULTS.iterdir()):
        if not exp_dir.is_dir():
            continue
        # Skip target dirs if they already exist
        if exp_dir.name in ("er1", "er2", "er3", "er4") and (exp_dir / "runs").exists():
            continue

        # Find run directories
        run_dirs = []
        for item in sorted(exp_dir.rglob("input/config.yaml")):
            run_dir = item.parent.parent
            if "__" not in run_dir.name:
                continue

            # Skip known duplicate subdirs under er1_ablation/
            # (only skip if under er1_ablation/, not standalone)
            rel = run_dir.relative_to(RESULTS)
            rel_str = str(rel)
            skip = False
            if rel_str.startswith("er1_ablation/"):
                for dupe in ABLATION_DUPES:
                    if dupe in rel_str:
                        skip = True
                        break
                # Skip er1_old/ runs (archived separately)
                if "er1_old" in rel_str:
                    skip = True
            if skip:
                continue

            run_dirs.append(run_dir)

        for run_dir in run_dirs:
            config_path = run_dir / "input" / "config.yaml"
            metrics_path = run_dir / "output" / "metrics.json"

            if not config_path.exists():
                continue

            with open(config_path) as f:
                config = yaml.safe_load(f) or {}

            metrics = {}
            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)

            if not metrics:
                # Skip incomplete runs (will be archived)
                continue

            # Determine family and variant
            family = determine_family(exp_dir.name, config)
            variant = build_variant_tag(exp_dir.name, run_dir.name, config)

            runs.append({
                "old_path": str(run_dir.relative_to(RESULTS)),
                "old_abs": str(run_dir),
                "exp_dir": exp_dir.name,
                "family": family,
                "variant": variant,
                "config": config,
                "metrics": metrics,
                "run_dir_name": run_dir.name,
            })

    return runs


def deduplicate_runs(runs: List[dict]) -> List[dict]:
    """Remove duplicate runs (er1_ablation copies)."""
    seen = {}
    unique = []
    for r in runs:
        # Use metrics hash as identity
        met_str = json.dumps(r["metrics"], sort_keys=True)
        met_hash = hashlib.md5(met_str.encode()).hexdigest()
        key = (r["variant"], met_hash)
        if key in seen:
            # Keep the one NOT in er1_ablation
            if "er1_ablation" in r["old_path"]:
                continue
            else:
                # Replace the er1_ablation one
                unique = [x for x in unique if x["variant"] != r["variant"] or
                          hashlib.md5(json.dumps(x["metrics"], sort_keys=True).encode()).hexdigest() != met_hash]
                unique.append(r)
                seen[key] = r["old_path"]
        else:
            seen[key] = r["old_path"]
            unique.append(r)
    return unique


def build_csv_row(run: dict) -> dict:
    """Build a master CSV row from a run."""
    config = run["config"]
    metrics = run["metrics"]
    task = config.get("task", {})
    train = config.get("train", {})

    return {
        "variant": run["variant"],
        "experiment": experiment_label(run["variant"], run["family"], config),
        "family": run["family"],
        "old_run_id": config.get("run_id", ""),
        "old_dir": run["old_path"],
        # Experimental dimensions
        "agent_lidar": task.get("use_agent_lidar", False),
        "dim_c": task.get("dim_c", 0),
        "comm_mode": "none" if task.get("dim_c", 0) == 0 else
                     ("proximity" if task.get("comm_proximity", True) else "broadcast"),
        "shared_reward": task.get("shared_reward", False),
        "collision_penalty": task.get("agent_collision_penalty", -0.1),
        "agents_per_target": task.get("agents_per_target", 2),
        "lidar_range": task.get("lidar_range", 0.35),
        "n_agents": task.get("n_agents", 4),
        "n_targets": task.get("n_targets", 4),
        "covering_range": task.get("covering_range", 0.25),
        "algorithm": config.get("algorithm", train.get("algorithm", "mappo")),
        "seed": config.get("seed", 0),
        "max_frames": train.get("max_n_frames", 10_000_000),
        "max_steps": 100,  # All pre-fix runs used 100
        "configured_max_steps": task.get("max_steps", 200),
        "lr": train.get("lr", 5e-5),
        "lmbda": train.get("lmbda", None),
        "entropy_coef": train.get("entropy_coef", None),
        "hidden_layers": str(train.get("hidden_layers")) if train.get("hidden_layers") else "",
        "activation": train.get("activation", ""),
        "model_type": train.get("model_type", "mlp"),
        "gnn_class": train.get("gnn_class", "") if train.get("model_type") == "gnn" else "",
        # Metrics
        "M1_success_rate": metrics.get("M1_success_rate", ""),
        "M2_avg_return": metrics.get("M2_avg_return", ""),
        "M3_avg_steps": metrics.get("M3_avg_steps", ""),
        "M4_avg_collisions": metrics.get("M4_avg_collisions", ""),
        "M5_avg_tokens": metrics.get("M5_avg_tokens", ""),
        "M6_coverage_progress": metrics.get("M6_coverage_progress", ""),
        "M7_sample_efficiency": metrics.get("M7_sample_efficiency", ""),
        "M8_agent_utilization": metrics.get("M8_agent_utilization", ""),
        "M9_spatial_spread": metrics.get("M9_spatial_spread", ""),
        # Execution
        "training_seconds": metrics.get("training_seconds", ""),
        "device": train.get("train_device", ""),
        "source": "ovh" if train.get("train_device") == "cuda" else "local",
        "notes": compute_notes(run["variant"], run["family"], config, metrics),
    }


def add_csv_only_rows() -> List[dict]:
    """Add ablation D, E, F rows from consolidated CSV (no run dirs)."""
    csv_path = RESULTS / "er1_ablation" / "ablation_consolidated.csv"
    if not csv_path.exists():
        return []

    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            abl = row.get("ablation", "")
            if abl not in ("D", "E", "F"):
                continue
            seed = row.get("seed", "0")
            k = row.get("agents_per_target", "2")
            l = str(row.get("lidar_range", "0.35")).replace(".", "")

            if abl == "D":
                variant = f"abl_lr1e4_lam095_k{k}_l{l}_s{seed}"
                experiment = "ER1 abl: lr=1e-4 + lmbda=0.95"
            elif abl == "E":
                variant = f"abl_lp_nolid_k{k}_l{l}_s{seed}"
                experiment = "ER1 abl: low penalty (no AL)"
            else:  # F
                variant = f"abl_sr_nolid_k{k}_l{l}_s{seed}"
                experiment = "ER1 abl: shared reward (no AL)"

            rows.append({
                "variant": variant,
                "experiment": experiment,
                "family": "er1",
                "old_run_id": "",
                "old_dir": f"er1_ablation/ablation_consolidated.csv (row {abl}_s{seed})",
                "agent_lidar": False,
                "dim_c": 0,
                "comm_mode": "none",
                "shared_reward": abl == "F",
                "collision_penalty": -0.01 if abl == "E" else (-0.1 if abl == "F" else -0.1),
                "agents_per_target": int(k),
                "lidar_range": float(row.get("lidar_range", 0.35)),
                "n_agents": int(row.get("n_agents", 4)),
                "n_targets": int(row.get("n_targets", 4)),
                "covering_range": 0.25,
                "algorithm": "mappo",
                "seed": int(seed),
                "max_frames": int(row.get("max_n_frames", 10_000_000)),
                "max_steps": 100,
                "configured_max_steps": 200,
                "lr": float(row.get("lr", 5e-5)),
                "lmbda": float(row.get("lmbda", "")) if row.get("lmbda") else None,
                "entropy_coef": float(row.get("entropy_coef", "")) if row.get("entropy_coef") else None,
                "hidden_layers": "",
                "activation": "",
                "model_type": "mlp",
                "gnn_class": "",
                "M1_success_rate": float(row.get("M1_success_rate", 0)),
                "M2_avg_return": float(row.get("M2_avg_return", 0)),
                "M3_avg_steps": float(row.get("M3_avg_steps", 0)),
                "M4_avg_collisions": float(row.get("M4_avg_collisions", 0)),
                "M5_avg_tokens": 0,
                "M6_coverage_progress": float(row.get("M6_coverage_progress", 0)),
                "M7_sample_efficiency": "",
                "M8_agent_utilization": float(row.get("M8_agent_utilization", 0)),
                "M9_spatial_spread": float(row.get("M9_spatial_spread", 0)),
                "training_seconds": row.get("training_seconds", ""),
                "device": "cuda",
                "source": "ovh",
                "notes": "csv_only; max_steps_actual=100_cfg_intended=200",
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Reorganize experiment results")
    parser.add_argument("--dry-run", action="store_true", help="Preview actions")
    parser.add_argument("--execute", action="store_true", help="Perform reorganization")
    parser.add_argument("--delete", action="store_true", help="Delete originals")
    args = parser.parse_args()

    if not any([args.dry_run, args.execute, args.delete]):
        args.dry_run = True

    # Collect all runs
    print("Collecting runs...")
    all_runs = collect_all_runs()
    print(f"Found {len(all_runs)} runs with config+metrics")

    # Deduplicate
    unique_runs = deduplicate_runs(all_runs)
    removed = len(all_runs) - len(unique_runs)
    print(f"After dedup: {len(unique_runs)} unique ({removed} duplicates removed)")

    # Check for variant collisions
    variants = {}
    for r in unique_runs:
        key = (r["family"], r["variant"])
        if key in variants:
            print(f"  *** COLLISION: {key} maps to both {variants[key]} and {r['old_path']}")
        variants[key] = r["old_path"]

    # Group by family
    by_family = {"er1": [], "er2": [], "er3": []}
    for r in unique_runs:
        by_family.setdefault(r["family"], []).append(r)

    for fam, runs in sorted(by_family.items()):
        print(f"\n{fam}/: {len(runs)} runs")
        for r in sorted(runs, key=lambda x: x["variant"]):
            print(f"  {r['variant']:<45s} ← {r['old_path']}")

    if args.dry_run:
        print("\n[DRY RUN] No changes made. Use --execute to proceed.")
        return

    if args.execute:
        changelog = []

        # Phase 1: Create structure
        for fam in ["er1", "er2", "er3"]:
            for sub in ["runs", "archive"]:
                d = RESULTS / fam / sub
                d.mkdir(parents=True, exist_ok=True)
                print(f"Created {d}")
        (DOCS / "archive").mkdir(exist_ok=True)

        # Phase 2: Copy runs
        print("\nCopying runs...")
        for r in unique_runs:
            src = Path(r["old_abs"])
            dst = RESULTS / r["family"] / "runs" / r["variant"]
            if dst.exists():
                print(f"  SKIP (exists): {r['variant']}")
                continue
            shutil.copytree(src, dst)
            changelog.append(f"COPY | {r['old_path']} | {r['family']}/runs/{r['variant']}")
            print(f"  COPIED: {r['variant']}")

        # Phase 3: Verify copies
        print("\nVerifying copies...")
        errors = 0
        for r in unique_runs:
            src = Path(r["old_abs"])
            dst = RESULTS / r["family"] / "runs" / r["variant"]
            # Check metrics match
            src_met = src / "output" / "metrics.json"
            dst_met = dst / "output" / "metrics.json"
            if src_met.exists() and dst_met.exists():
                if src_met.read_bytes() != dst_met.read_bytes():
                    print(f"  *** MISMATCH: {r['variant']} metrics.json")
                    errors += 1
            # Check config match
            src_cfg = src / "input" / "config.yaml"
            dst_cfg = dst / "input" / "config.yaml"
            if src_cfg.exists() and dst_cfg.exists():
                if src_cfg.read_bytes() != dst_cfg.read_bytes():
                    print(f"  *** MISMATCH: {r['variant']} config.yaml")
                    errors += 1
        if errors:
            print(f"\n*** {errors} VERIFICATION ERRORS — aborting ***")
            sys.exit(1)
        print(f"  All {len(unique_runs)} copies verified OK")

        # Phase 4: Build master CSV
        print("\nBuilding master_results.csv...")
        csv_rows = [build_csv_row(r) for r in unique_runs]
        csv_rows.extend(add_csv_only_rows())
        csv_rows.sort(key=lambda x: (x["family"], x["variant"]))

        fieldnames = list(csv_rows[0].keys())
        csv_path = RESULTS / "master_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"  Written {len(csv_rows)} rows to {csv_path}")
        changelog.append(f"CREATE | master_results.csv | {len(csv_rows)} rows")

        # Phase 5: Archive
        print("\nArchiving...")
        archives = [
            (RESULTS / "e1", RESULTS / "er1" / "archive" / "e1"),
            (RESULTS / "er1_ablation" / "er1_old",
             RESULTS / "er1" / "archive" / "er1_old"),
        ]
        for src, dst in archives:
            if src.exists() and not dst.exists():
                shutil.copytree(src, dst)
                changelog.append(f"ARCHIVE | {src.relative_to(RESULTS)} | {dst.relative_to(RESULTS)}")
                print(f"  Archived: {src.name} → {dst.relative_to(RESULTS)}")

        # Copy consolidated CSV
        con_src = RESULTS / "er1_ablation" / "ablation_consolidated.csv"
        con_dst = RESULTS / "er1" / "er1_ablation_consolidated.csv"
        if con_src.exists() and not con_dst.exists():
            shutil.copy2(con_src, con_dst)
            changelog.append(f"COPY | er1_ablation/ablation_consolidated.csv | er1/er1_ablation_consolidated.csv")
            print(f"  Copied consolidated CSV")

        # Archive docs
        for doc_name in ["er1_first_findings_2026-03-15.md", "hyperparameter_analysis.md"]:
            src = DOCS / doc_name
            dst = DOCS / "archive" / doc_name
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
                changelog.append(f"ARCHIVE | docs/{doc_name} | docs/archive/{doc_name}")
                print(f"  Archived doc: {doc_name}")

        # Phase 6: Write changelog
        print("\nWriting changelog...")
        changelog_path = RESULTS / "changelog.md"
        with open(changelog_path, "w") as f:
            f.write("# Results Reorganization Changelog\n\n")
            f.write(f"**Date**: 2026-03-22\n\n")
            f.write("| Action | Source | Destination |\n")
            f.write("|--------|--------|-------------|\n")
            for entry in changelog:
                parts = entry.split(" | ")
                f.write(f"| {' | '.join(parts)} |\n")
        print(f"  Written {len(changelog)} entries")

        # Phase 7: Final counts
        print("\n" + "=" * 60)
        print("FINAL VERIFICATION")
        print("=" * 60)
        for fam in ["er1", "er2", "er3"]:
            runs_dir = RESULTS / fam / "runs"
            if runs_dir.exists():
                count = len([d for d in runs_dir.iterdir() if d.is_dir()])
                print(f"  {fam}/runs/: {count} directories")
        print(f"  master_results.csv: {len(csv_rows)} rows")
        print("\nDone! Review results, then use --delete to clean up originals.")

    if args.delete:
        print("Delete not implemented yet — manual review first.")


if __name__ == "__main__":
    main()
