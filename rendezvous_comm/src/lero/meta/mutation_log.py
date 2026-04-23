"""Evolutionary memory for LERO-MP v2.

Appends one JSONL entry per mutation so that future Strategist calls
can read "what was tried before and how did it score". This is the
mechanism that lets the meta-LLM avoid re-proposing patterns that
already failed.

File layout (one entry per line, JSON):

    {
      "ts": ISO-8601 UTC,
      "run_id": "<exp_id>_<timestamp>_s<seed>",
      "task_id": "<exp_id>",
      "seed": 0,
      "outer_iter": 2,
      "parent_version": "v2_fewshot_modular_v2",
      "new_version": "v2_fewshot_modular_v2_mp_001",
      "strategy_card": {...},           # serialized StrategyCard
      "slot_name": "guidance_observation",
      "slot_content_sha256": "...",
      "slot_content_excerpt": "first 400 chars",
      "pre_mutation_peak_M1": 0.010,
      "pre_mutation_best_M6": 0.18,
      "post_mutation_peak_M1": 0.025,   # null until next outer iter completes
      "post_mutation_best_M6": 0.17,
      "delta_peak_M1": 0.015,
      "delta_M6": -0.01,
      "verdict": "marginal_improvement",
      "fail_modes_during_next_iter": ["reward_magnitude_inflation"]
    }

Two kinds of writes:

  1. At mutation time — "pre" fields filled, "post" fields null.
  2. At the start of the NEXT outer iter — update the last unresolved
     entry with the post-mutation metrics + verdict.

Design goals:

  - Pure JSONL (append-only) so concurrent seed jobs never step on
    each other's writes.
  - No DB, no lock — the "post" update reads and rewrites the ONE
    trailing line belonging to this run. Different runs write to
    different paths (per-run mutation_log under $RESULTS_DIR).
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


_log = logging.getLogger("rendezvous.lero.mp")


# ── verdict classifier ─────────────────────────────────────────

# Thresholds intentionally lenient. At 1M full training peak_M1 is
# noise-dominated, so a 0.01 move IS informative. Tighten (5× or 10×)
# when promoting to 10M.

STRONG_IMPROVEMENT_M1 = 0.10
MARGINAL_IMPROVEMENT_M1 = 0.01
MARGINAL_REGRESSION_M1 = -0.01
COLLAPSE_M1 = -0.10
M6_NEUTRAL_BAND = 0.05


def classify_verdict(
    pre_m1: float, post_m1: float,
    pre_m6: float = 0.0, post_m6: float = 0.0,
    scale: float = 1.0,
) -> str:
    """Deterministic verdict from metric deltas. No LLM involved.

    ``scale`` multiplies the base thresholds so the same classifier
    can be used at 1M frames (scale=1.0, tight) and 10M frames
    (scale=10.0, 10× looser thresholds so training-noise drifts don't
    get misclassified as "regressions"). Callers should compute
    ``scale = max(1.0, full_frames / 1_000_000)`` or similar.
    """
    dm1 = post_m1 - pre_m1
    dm6 = post_m6 - pre_m6
    collapse_bound = COLLAPSE_M1 * scale
    strong_bound = STRONG_IMPROVEMENT_M1 * scale
    marginal_bound = MARGINAL_IMPROVEMENT_M1 * scale
    regression_bound = MARGINAL_REGRESSION_M1 * scale
    m6_band = M6_NEUTRAL_BAND * scale
    if dm1 <= collapse_bound:
        return "collapse"
    if post_m1 == 0.0 and pre_m1 > 0.0:
        return "collapse"
    if dm1 >= strong_bound:
        return "strong_improvement"
    if dm1 >= marginal_bound and dm6 >= 0:
        return "marginal_improvement"
    if dm1 <= regression_bound or dm6 < -m6_band:
        return "regression"
    return "neutral"


# ── entry schema + persistence ─────────────────────────────────

@dataclass
class MutationLogEntry:
    """In-memory representation of a single log line."""

    ts: str
    run_id: str
    task_id: str
    seed: int
    outer_iter: int
    parent_version: str
    new_version: str
    strategy_card: Dict[str, Any]
    slot_name: str
    slot_content_sha256: str
    slot_content_excerpt: str
    pre_mutation_peak_M1: float
    pre_mutation_best_M6: float
    post_mutation_peak_M1: Optional[float] = None
    post_mutation_best_M6: Optional[float] = None
    delta_peak_M1: Optional[float] = None
    delta_M6: Optional[float] = None
    verdict: Optional[str] = None
    fail_modes_during_next_iter: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        out = dict(self.__dict__)
        return out

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MutationLogEntry":
        # Tolerant of missing optional fields from older log formats.
        return cls(
            ts=d["ts"], run_id=d["run_id"], task_id=d.get("task_id", ""),
            seed=d["seed"], outer_iter=d["outer_iter"],
            parent_version=d["parent_version"],
            new_version=d["new_version"],
            strategy_card=d.get("strategy_card", {}),
            slot_name=d["slot_name"],
            slot_content_sha256=d["slot_content_sha256"],
            slot_content_excerpt=d.get("slot_content_excerpt", ""),
            pre_mutation_peak_M1=d["pre_mutation_peak_M1"],
            pre_mutation_best_M6=d.get("pre_mutation_best_M6", 0.0),
            post_mutation_peak_M1=d.get("post_mutation_peak_M1"),
            post_mutation_best_M6=d.get("post_mutation_best_M6"),
            delta_peak_M1=d.get("delta_peak_M1"),
            delta_M6=d.get("delta_M6"),
            verdict=d.get("verdict"),
            fail_modes_during_next_iter=d.get(
                "fail_modes_during_next_iter", [],
            ),
        )


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def new_entry(
    run_id: str,
    task_id: str,
    seed: int,
    outer_iter: int,
    parent_version: str,
    new_version: str,
    strategy_card: Dict[str, Any],
    slot_name: str,
    slot_content: str,
    pre_peak_M1: float,
    pre_M6: float,
    excerpt_chars: int = 400,
) -> MutationLogEntry:
    """Build a ``pre``-filled entry at mutation time."""
    h = hashlib.sha256(slot_content.encode("utf-8")).hexdigest()
    excerpt = slot_content[:excerpt_chars]
    if len(slot_content) > excerpt_chars:
        excerpt = excerpt + " …"
    return MutationLogEntry(
        ts=_now_iso(),
        run_id=run_id,
        task_id=task_id,
        seed=seed,
        outer_iter=outer_iter,
        parent_version=parent_version,
        new_version=new_version,
        strategy_card=strategy_card,
        slot_name=slot_name,
        slot_content_sha256=h,
        slot_content_excerpt=excerpt,
        pre_mutation_peak_M1=float(pre_peak_M1),
        pre_mutation_best_M6=float(pre_M6),
    )


def append_entry(path: Path, entry: MutationLogEntry) -> None:
    """Append one JSONL line. Creates parents if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry.to_dict()) + "\n")
    _log.info(
        "mutation_log: appended entry for %s (slot=%s, pre_M1=%.3f)",
        entry.new_version, entry.slot_name, entry.pre_mutation_peak_M1,
    )


def update_last_entry_with_post(
    path: Path,
    post_peak_M1: float,
    post_M6: float,
    fail_modes: Optional[List[str]] = None,
    verdict_scale: float = 1.0,
) -> Optional[MutationLogEntry]:
    """Fill the ``post`` fields on the most recent entry in ``path``.

    Idempotent: if the last entry already has ``verdict`` set, this is
    a no-op. Returns the updated entry (or ``None`` if file is absent
    or empty).
    """
    path = Path(path)
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return None
    last = MutationLogEntry.from_dict(json.loads(lines[-1]))
    if last.verdict is not None:
        return last  # already resolved; don't re-update
    last.post_mutation_peak_M1 = float(post_peak_M1)
    last.post_mutation_best_M6 = float(post_M6)
    last.delta_peak_M1 = (
        last.post_mutation_peak_M1 - last.pre_mutation_peak_M1
    )
    last.delta_M6 = last.post_mutation_best_M6 - last.pre_mutation_best_M6
    last.verdict = classify_verdict(
        last.pre_mutation_peak_M1, last.post_mutation_peak_M1,
        last.pre_mutation_best_M6, last.post_mutation_best_M6,
        scale=verdict_scale,
    )
    if fail_modes:
        last.fail_modes_during_next_iter = list(fail_modes)
    # Rewrite: everything except the last line, then the updated last.
    with open(path, "w", encoding="utf-8") as f:
        for prev in lines[:-1]:
            f.write(prev + "\n")
        f.write(json.dumps(last.to_dict()) + "\n")
    return last


def read_recent(
    path,
    n: int = 10,
    task_id: Optional[str] = None,
) -> List[MutationLogEntry]:
    """Return the last ``n`` entries (optionally filtered by task_id),
    oldest first.

    ``path`` may be a single Path/str OR an iterable of Path/str —
    when multiple paths are supplied, all entries are merged and
    sorted by timestamp. This is the extension point for future
    cross-run / cross-seed memory: mount prior sweeps' result
    buckets read-only and pass their mutation_log paths here.
    """
    paths: List[Path]
    if isinstance(path, (str, Path)):
        paths = [Path(path)]
    else:
        paths = [Path(p) for p in path]

    entries: List[MutationLogEntry] = []
    for p in paths:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                e = MutationLogEntry.from_dict(json.loads(line))
            except Exception as ex:  # pragma: no cover
                _log.warning(
                    "Skipping malformed mutation_log entry in %s: %s",
                    p, ex,
                )
                continue
            if task_id is not None and e.task_id != task_id:
                continue
            entries.append(e)
    # Stable sort by ISO-8601 timestamp so merged entries from
    # multiple files end up in chronological order.
    entries.sort(key=lambda e: e.ts)
    return entries[-n:]


def read_prior_slot_versions(
    path,
    slot_name: str,
    task_id: Optional[str] = None,
    n: int = 5,
) -> List[MutationLogEntry]:
    """Return the last ``n`` mutation_log entries that edited the
    specified ``slot_name`` on the given ``task_id``. Used by the
    Editor to show prior attempts on the same slot and explicitly
    encourage diversification."""
    recent = read_recent(path, n=10_000, task_id=task_id)
    slot_hits = [e for e in recent if e.slot_name == slot_name]
    return slot_hits[-n:]


def summarize_for_prompt(entries: List[MutationLogEntry]) -> str:
    """Compact, prompt-friendly summary of prior mutations.

    Collapses each entry into 2 lines — what was tried, what was the
    outcome — so the Strategist can read the full history in ~30 lines
    even with a large ``n``.
    """
    if not entries:
        return "(no prior mutations on this task)"
    parts = []
    for e in entries:
        verdict = e.verdict or "pending"
        delta = (
            f"Δpeak_M1={e.delta_peak_M1:+.3f} ΔM6={e.delta_M6:+.3f}"
            if e.delta_peak_M1 is not None else "outcome not yet measured"
        )
        dom = e.strategy_card.get("target_domain", "?") if e.strategy_card else "?"
        focus = ""
        if e.strategy_card and e.strategy_card.get("focus"):
            focus = "; focus=" + "; ".join(e.strategy_card["focus"][:2])
        parts.append(
            f"- {e.new_version}  slot={e.slot_name}  domain={dom}{focus}"
        )
        parts.append(f"    verdict={verdict}  {delta}")
    return "\n".join(parts)
