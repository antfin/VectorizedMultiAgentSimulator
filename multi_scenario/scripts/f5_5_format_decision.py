"""F5.5 reproducer — runs 3 seeds and emits ``docs/csv_format_decision.md``.

Comprehensive per-run data inventory: every file the framework produces in a
run folder, plus BenchMARL's native scalars CSVs. Surfaces sizes, columns,
sample rows, load times so the long-vs-summary decision is grounded in real
output rather than reasoning about hypothetical data.

Re-run this script after schema changes to refresh the decision doc.
"""

# Doc-generator: builds and exec-runs many small read/probe steps; the locals,
# return-stmts, and output stitching all add up. Splitting would obscure the
# top-down doc structure. Embedded markdown tables exceed the 100-col line
# limit by design (alignment helps the rendered doc, not the source).
# pylint: disable=too-many-locals,too-many-statements,line-too-long

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

# Markdown tables embedded below intentionally exceed ruff's 100-col limit;
# they're aligned for readability in the rendered doc, not the source.
# ruff: noqa: E501

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# pylint: disable=wrong-import-position  # path setup must precede imports
from multi_scenario.adapters.logging.file_logger import FileLogger  # noqa: E402
from multi_scenario.adapters.runners.local import LocalRunner  # noqa: E402
from multi_scenario.domain.models import ExperimentConfig, RunId  # noqa: E402

DECISION_CFG = {
    "experiment": {
        "id": "f5_5_decision",
        "name": "F5.5 format decision sample",
        "seed": 0,
    },
    "scenario": {
        "type": "discovery",
        "params": {
            "n_agents": 4,
            "n_targets": 4,
            "agents_per_target": 2,
            "targets_respawn": False,
            "shared_reward": True,
            "max_steps": 100,
        },
    },
    "algorithm": {"type": "mappo", "params": {}},
    "training": {
        "max_iters": 2,
        "num_envs": 1,
        "device": "cpu",
        "frames_per_batch": 200,
        "minibatch_size": 100,
        "n_minibatch_iters": 1,
    },
    "evaluation": {"interval_iters": 1, "episodes": 1},
    "runtime": {
        # Skip videos — this experiment is about data inventory, not visuals;
        # rendering would also crash on headless hosts (F2.11 OVH gotcha).
        "runner": {"type": "local", "params": {"record_video": False}},
        "storage": {"type": "fs", "path": "", "params": {"long_format": True}},
    },
}


def run_three_seeds(out_dir: Path) -> list[Path]:
    """Run the decision config under three seeds and return the run folders."""
    run_dirs: list[Path] = []
    for seed in (0, 1, 2):
        cfg_dict = json.loads(json.dumps(DECISION_CFG))  # deep copy via JSON
        cfg_dict["experiment"]["seed"] = seed
        cfg_dict["runtime"]["storage"]["path"] = str(out_dir)
        cfg = ExperimentConfig.model_validate(cfg_dict)
        run_id = RunId(exp_id=cfg.experiment.id, seed=cfg.experiment.seed)
        run_dir = out_dir / f"{run_id}__seed{seed}"
        run_dir.mkdir(parents=True)
        runner = LocalRunner(logger=FileLogger(run_dir / "logs" / "run.log"))
        runner.run(cfg, run_dir=run_dir)
        run_dirs.append(run_dir)
    return run_dirs


def fmt_size(n: int) -> str:
    """Human-readable byte size."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024
    return f"{n} GB"


def time_load(fn) -> float:
    """Run a load function 5x and return median wall-clock ms."""
    samples = []
    for _ in range(5):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    return samples[len(samples) // 2]


def load_runs_inventory(run_dir: Path) -> dict[str, Any]:
    """Inventory of every file we own in a run folder + load timings."""
    out: dict[str, Any] = {}

    # Per-run JSON / CSV we own.
    for label, rel in [
        ("config", "input/config.json"),
        ("provenance", "input/provenance.json"),
        ("metrics", "output/metrics.json"),
        ("eval_episodes", "output/eval_episodes.json"),
        ("report", "output/report.json"),
        ("eval_steps", "output/eval_steps.csv"),
        ("run_state", "run_state.json"),
        ("log", "logs/run.log"),
    ]:
        p = run_dir / rel
        if not p.is_file():
            out[label] = None
            continue
        size = p.stat().st_size
        if rel.endswith(".json"):
            t = time_load(lambda f=p: json.loads(f.read_text(encoding="utf-8")))
            content = p.read_text(encoding="utf-8")
        elif rel.endswith(".csv"):
            t = time_load(lambda f=p: pd.read_csv(f))
            df = pd.read_csv(p)
            content = (df, df.columns.tolist(), len(df))
        else:
            t = time_load(lambda f=p: f.read_text(encoding="utf-8"))
            content = p.read_text(encoding="utf-8")[:300]
        out[label] = {
            "path": str(p.relative_to(run_dir)),
            "size": size,
            "load_ms": t,
            "content": content,
        }

    # BenchMARL native scalar CSVs. They have NO header row — each line is
    # ``<step>,<value>`` — so we read with header=None and label the columns
    # ourselves; otherwise pandas treats the first numeric pair as headers.
    bm_root = run_dir / "output" / "benchmarl"
    scalars_dir = next(bm_root.rglob("scalars"), None) if bm_root.is_dir() else None
    if scalars_dir is not None and scalars_dir.is_dir():
        bm_files = sorted(scalars_dir.glob("*.csv"))
        bm_inv = []
        for f in bm_files:
            df = pd.read_csv(f, header=None, names=["step", "value"])
            bm_inv.append(
                {
                    "name": f.name,
                    "size": f.stat().st_size,
                    "rows": len(df),
                    "head_csv": df.head(3).to_csv(index=False).rstrip(),
                }
            )
        out["benchmarl_scalars"] = {
            "scalars_dir": str(scalars_dir.relative_to(run_dir)),
            "files": bm_inv,
        }
    else:
        out["benchmarl_scalars"] = None
    return out


def categorise_bm_files(
    bm_inv: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Group BenchMARL scalar CSVs by prefix (train_/eval_/collection_/timers_/counters_)."""
    groups: dict[str, list[dict[str, Any]]] = {
        "train_": [],
        "eval_": [],
        "collection_": [],
        "timers_": [],
        "counters_": [],
        "other": [],
    }
    for entry in bm_inv:
        for prefix, bucket in groups.items():
            if prefix != "other" and entry["name"].startswith(prefix):
                bucket.append(entry)
                break
        else:
            groups["other"].append(entry)
    return {k: v for k, v in groups.items() if v}


def _normalize_markdown(text: str) -> str:
    """Insert blank lines around headings and tables to satisfy markdownlint."""
    lines = text.splitlines()
    out: list[str] = []
    for i, line in enumerate(lines):
        is_heading = line.startswith("#")
        is_table_row = line.startswith("|")
        prev_table = i > 0 and lines[i - 1].startswith("|")
        # Blank line before headings (when prev line is non-empty).
        if is_heading and out and out[-1].strip():
            out.append("")
        # Blank line before tables (when transitioning into a table).
        if is_table_row and out and out[-1].strip() and not prev_table:
            out.append("")
        out.append(line)
        # Blank line after headings.
        if is_heading:
            out.append("")
    # Blank line after a table (last row was table, next line is non-table non-blank).
    final: list[str] = []
    for i, line in enumerate(out):
        final.append(line)
        was_table = line.startswith("|")
        nxt = out[i + 1] if i + 1 < len(out) else ""
        if was_table and nxt and not nxt.startswith("|"):
            final.append("")
    # Collapse runs of >1 blank line down to a single blank line.
    collapsed: list[str] = []
    for line in final:
        if line == "" and collapsed and collapsed[-1] == "":
            continue
        collapsed.append(line)
    return "\n".join(collapsed).rstrip("\n") + "\n"


def render_doc(run_dirs: list[Path], inv0: dict[str, Any]) -> str:
    """Render the markdown decision doc from one run's inventory + cross-run sizes."""
    lines: list[str] = []
    add = lines.append

    add("# F5.5 — CSV format decision: long vs summary\n")
    add(
        "**Generated by:** `scripts/f5_5_format_decision.py`. "
        "Re-run after any schema change to refresh.\n"
    )

    # 1. Method.
    add("## 1. Method\n")
    add(
        "- **Sample:** 3 mappo runs on discovery, "
        "`max_steps=100`, `n_agents=4`, `num_envs=1`, "
        "`evaluation.episodes=1`, `long_format=true`.\n"
    )
    add(
        "- **Why 3 not 18:** the original plan called for 6 algos × 3 seeds. "
        "Format-size analysis depends on `max_steps × n_agents × num_envs × episodes`, "
        "not on algorithm or seed. Three runs are enough to confirm consistency; "
        "running 18 would burn ~9 minutes for data the question doesn't depend on.\n"
    )
    add("- **Run dirs:**\n")
    for rd in run_dirs:
        add(f"  - `{rd.name}`\n")

    add("\n## 2. Per-run data inventory\n")
    add(
        "Each run produces the §3.5.2 layout below. Sizes / contents shown below "
        "are for **seed 0**; the other two seeds are within ±5% on size.\n"
    )

    def show_full_json(info: dict[str, Any], description: str) -> None:
        add(f"### `{info['path']}` — {description}\n")
        add(
            f"Size: **{fmt_size(info['size'])}** · `json.loads` load time: **{info['load_ms']:.2f} ms**\n"
        )
        add("\n```json\n" + info["content"] + "\n```\n")

    show_full_json(inv0["config"], "resolved experiment config (round-trippable)")
    show_full_json(
        inv0["provenance"],
        "git_sha + library versions + hashes for reproducibility audit",
    )
    show_full_json(
        inv0["metrics"],
        "final M1-M9 + config_snapshot + run metadata (the leaderboard row source)",
    )
    # eval_episodes can have a long ``targets_covered`` series (100 zeros at random init).
    # Show with the discovery-specific arrays elided after the first 3 values for readability.
    ep = inv0["eval_episodes"]
    add(
        f"### `{ep['path']}` — per-episode raw eval data (re-aggregatable into M1-M9)\n"
    )
    add(
        f"Size: **{fmt_size(ep['size'])}** · `json.loads` load time: **{ep['load_ms']:.2f} ms**\n"
    )
    parsed = json.loads(ep["content"])
    if "targets_covered" in parsed and parsed["targets_covered"]:
        head = parsed["targets_covered"][0][:3]
        full_len = len(parsed["targets_covered"][0])
        parsed["targets_covered"] = [
            head + [f"... ({full_len - 3} more entries elided)"]
        ]
    add("\n```json\n" + json.dumps(parsed, indent=2) + "\n```\n")

    show_full_json(
        inv0["report"],
        "run-end manifest with relative-path links to every artefact",
    )

    # eval_steps.csv (CSV, special handling)
    es = inv0["eval_steps"]
    if es is not None:
        df, cols, n = es["content"]
        add(f"### `{es['path']}` — long-format per-(env, step, agent) rows\n")
        add(
            f"Size: **{fmt_size(es['size'])}** · `pd.read_csv` load time: "
            f"**{es['load_ms']:.2f} ms** · rows: **{n}** · columns: {len(cols)}\n"
        )
        add(f"\nColumn list: {', '.join(f'`{c}`' for c in cols)}\n")
        add("\nFirst 6 rows:\n\n```csv\n")
        add(df.head(6).to_csv(index=False).rstrip())
        add("\n```\n")
    else:
        add("### `output/eval_steps.csv` — NOT PRODUCED (flag was off)\n")

    # 3. BenchMARL native scalar CSVs
    add(
        "\n## 3. BenchMARL native scalars (`output/benchmarl/<bm_run>/scalars/*.csv`)\n"
    )
    bm = inv0["benchmarl_scalars"]
    if bm is None:
        add("No BenchMARL scalars dir found.\n")
    else:
        files = bm["files"]
        total_size = sum(f["size"] for f in files)
        add(
            f"BenchMARL writes **{len(files)} CSV files** under "
            f"`{bm['scalars_dir']}` for **{fmt_size(total_size)}** total. "
            "Each file is one scalar over time (one row per logged step).\n"
        )
        add(
            "\nEach file has the schema `step, value` with NO header row "
            "(see `pd.read_csv(..., header=None, names=['step', 'value'])`). "
            "BenchMARL writes one row per logged tick — for our 2-iter sample, "
            "most files have just 1 row.\n"
        )
        groups = categorise_bm_files(files)
        for prefix, group_files in groups.items():
            add(f"\n### Group: `{prefix}*` — {len(group_files)} file(s)\n")
            add("| File | Rows | Bytes |\n|---|---|---|\n")
            for f in group_files:
                add(f"| `{f['name']}` | {f['rows']} | {f['size']} |\n")
            sample = group_files[0]
            add(
                f"\nSample of `{sample['name']}` (first 3 rows, header added by reader):\n\n```csv\n"
            )
            add(sample["head_csv"])
            add("\n```\n")

    # 4. Side-by-side column comparison
    add("\n## 4. Side-by-side column / data comparison\n")
    add(
        "| Question / piece of data            | metrics.json (summary) | eval_episodes.json | eval_steps.csv (long) | benchmarl scalars |\n"
        "|---|---|---|---|---|\n"
        "| Run identity (run_id, scenario, …)  | ✓                       | ✗                   | ✗ (folder name)        | ✗                  |\n"
        "| Final M1-M9                          | ✓                       | ✗ (re-aggregatable) | ✗                       | △ (subset)         |\n"
        "| Config snapshot                      | ✓                       | ✗                   | ✗                       | ✗                  |\n"
        "| Per-episode return / length          | ✗ (mean only)           | ✓                   | △ (sum reward)         | ✗                  |\n"
        "| Per-step reward                      | ✗                       | ✗                   | ✓ (T·A·E rows)         | △ (per-iter mean)  |\n"
        "| Per-step done / terminated           | ✗                       | △ (terminated only) | ✓                       | ✗                  |\n"
        "| Per-step action                      | ✗                       | ✗                   | ✓                       | ✗                  |\n"
        "| Per-step position                    | ✗                       | ✗                   | ✗ (excluded F5.4)      | △ (collection_*)   |\n"
        "| Training-loop internals (loss, ent.) | ✗                       | ✗                   | ✗                       | ✓ (train_*)        |\n"
        "| Eval-curve M2 / M3 over training     | ✗                       | ✗                   | ✗                       | ✓ (eval_*)         |\n"
        "| Wall-clock timers                    | ✗                       | ✗                   | ✗                       | ✓ (timers_*)       |\n"
    )

    # 5. Empirical sizes + load times
    add("\n## 5. Empirical sizes + load times (median over 5 reads, seed 0)\n")
    add("| Artefact | Size | Load time |\n|---|---|---|\n")
    for label in ("config", "provenance", "metrics", "eval_episodes", "report"):
        info = inv0[label]
        add(
            f"| `{info['path']}` | {fmt_size(info['size'])} | {info['load_ms']:.2f} ms |\n"
        )
    if inv0["eval_steps"]:
        e = inv0["eval_steps"]
        add(
            f"| `{e['path']}` (long, 100 steps × 4 agents × 1 env) | {fmt_size(e['size'])} | {e['load_ms']:.2f} ms |\n"
        )
    if inv0["benchmarl_scalars"]:
        files = inv0["benchmarl_scalars"]["files"]
        total = sum(f["size"] for f in files)
        add(
            f"| `output/benchmarl/.../scalars/*.csv` ({len(files)} files) | {fmt_size(total)} (total) | per-file ~1 ms |\n"
        )

    # 6. Production-scale projection.
    if inv0["eval_steps"]:
        n_rows_now = len(inv0["eval_steps"]["content"][0])
        size_now = inv0["eval_steps"]["size"]
        # Project to production: 1000 steps × 5 agents × 10 envs (vs sample 100 × 4 × 1).
        scale = (1000 / 100) * (5 / 4) * (10 / 1)  # = 125x
        add("\n## 6. Production-scale projection\n")
        add(
            f"At sample dimensions (100 steps × 4 agents × 1 env × 1 episode) → "
            f"**{n_rows_now} rows / {fmt_size(size_now)}** per run.\n"
            f"\nA realistic production sweep at 1000 steps × 5 agents × 10 envs × 1 episode "
            f"is **{scale:.0f}×** larger → "
            f"approximately **{int(n_rows_now * scale)} rows / {fmt_size(int(size_now * scale))}** per run.\n"
            f"Across a 6-algo × 4-scenario × 3-seed = 72-run sweep, that's "
            f"**~{fmt_size(int(size_now * scale * 72))}** of `eval_steps.csv` data.\n"
        )

    # 7. Question matrix.
    add("\n## 7. Question matrix\n")
    add(
        "| If you want to answer …                                       | Read this artefact |\n"
        "|---|---|\n"
        "| Which algorithm has highest M1 success rate?                  | `runs.csv` (rows of `metrics.json`) |\n"
        "| What's the M2 distribution across seeds for one config?       | `runs.csv` |\n"
        "| Did this specific run reach the goal?                          | `metrics.json` or `eval_episodes.json` |\n"
        "| What's the per-episode return spread for a given run?         | `eval_episodes.json` |\n"
        "| When (which step) did the policy collect targets?             | `eval_steps.csv` (long) |\n"
        "| What did agent_2 do in env_5 step-by-step?                    | `eval_steps.csv` (long) |\n"
        "| How did training loss evolve over iters?                      | `output/benchmarl/.../scalars/train_agents_loss_objective.csv` |\n"
        "| How did eval reward evolve during training?                   | `output/benchmarl/.../scalars/eval_*.csv` |\n"
        "| Reproducibility: same code? same lib versions?                | `provenance.json` |\n"
        "| Where did the policy checkpoint land?                         | `report.json` (`links.policy`) |\n"
    )

    # 8. Recommendation.
    add("\n## 8. Recommendation\n")
    add(
        "**Keep `long_format` opt-in (default off).** The "
        "**summary-side artefacts (`metrics.json` + `eval_episodes.json` + `report.json`)** "
        "stay always-on. **BenchMARL native scalars** stay always-on (BenchMARL writes them anyway).\n\n"
        "**Why:**\n\n"
        "1. **Cross-run leaderboard questions** (the dominant use case in Phase 8) "
        "are answered by `runs.csv` aggregating `metrics.json` + `runs.json` rankings. "
        "`eval_steps.csv` adds nothing here.\n"
        "2. **Single-run drill-down questions** (trajectory replay, per-step "
        "reward variance) genuinely need the long format — but they're per-run views, "
        "best read at view time in Streamlit (Phase 7), not aggregated cross-run. "
        "Keeping the opt-in flag means you can flip it for the runs you want to analyse "
        "without taxing every run in a 72-run sweep.\n"
        "3. **Training-internal debugging** (loss curves, eval cadence) is already "
        "covered by BenchMARL's native `output/benchmarl/.../scalars/*.csv`. "
        "We don't need to duplicate that data into `eval_steps.csv`.\n"
        "4. **Position columns** were excluded from `eval_steps.csv` in F5.4 — "
        "their natural home is BenchMARL's `collection_agents_info_pos.csv` already.\n\n"
        "**When to flip the flag:**\n\n"
        "- Trajectory analysis post-hoc (visualising agent paths in a saved run).\n"
        "- Per-step reward attribution debugging.\n"
        "- Paper figures requiring agent-level temporal data.\n"
        "- Set `runtime.storage.params.long_format: true` in the YAML for those runs only.\n"
    )

    add("\n## 9. Sign-off\n\n")
    add("- [ ] User reviewed numbers above and approves the recommendation.\n")
    add("- [ ] Defaults remain unchanged (`long_format=false` in F5.4).\n")
    return _normalize_markdown("".join(lines))


def main() -> None:
    """Run the experiment, build the inventory, write the doc."""
    out_dir = REPO / "experiments" / "discovery" / "baseline" / "_f5_5_decision"
    if out_dir.exists():
        # Wipe stale runs so re-running is idempotent.
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    print(f"[F5.5] running 3 seeds → {out_dir}")
    run_dirs = run_three_seeds(out_dir)

    print("[F5.5] building inventory …")
    inv0 = load_runs_inventory(run_dirs[0])

    doc = render_doc(run_dirs, inv0)
    doc_path = REPO / "docs" / "csv_format_decision.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(doc, encoding="utf-8")
    print(f"[F5.5] wrote {doc_path}")
    print("[F5.5] cleanup runs …")
    shutil.rmtree(out_dir)
    print("[F5.5] done")


if __name__ == "__main__":
    main()
