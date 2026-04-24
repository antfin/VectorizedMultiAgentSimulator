"""Behavioral signals for meta-prompting (LERO-MP v3 §3.4/§4.1).

Three signal tiers, computed once per inner-loop run and passed to
BOTH the inner-LLM's between-iteration feedback AND the meta-LLM's
Strategist + Editor prompts.

- **Tier 1 — scalar**: M3/M4/M8/M9 per-candidate extras. Cheap, always on.
- **Tier 2 — fingerprint**: coverage-over-time shape from BenchMARL
  scalar CSVs. Derived for the top-1 candidate only (budget).
- **Tier 3 — curve_shape**: tag the learning curve as
  ``monotonic_rise | plateau_then_collapse | oscillating | flat_zero |
  flat_nonzero | reward_hack_shape``. Also per-candidate.

Two-gate noise control: the caller asks for a subset of tiers via the
``include_signals`` argument (driven by the Strategist's decision);
tiers that aren't requested return empty strings so the Editor /
inner-LLM prompt stays lean. A separate outlier gate (``outlier_flags``)
flags which tiers are *potentially* informative; the LLM still has to
request them.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_log = logging.getLogger("rendezvous.lero.behavioral")


# ── Tier 1: per-candidate scalars ────────────────────────────────

def format_tier1(metrics: Dict) -> str:
    """Compact one-line scalar summary covering M1/M2/M3/M4/M6/M8/M9."""
    m1 = metrics.get("M1_success_rate", 0.0)
    m2 = metrics.get("M2_avg_return", 0.0)
    m3 = metrics.get("M3_avg_steps", 0.0)
    m4 = metrics.get("M4_avg_collisions", 0.0)
    m6 = metrics.get("M6_coverage_progress", 0.0)
    m8 = metrics.get("M8_agent_utilization_cv", 0.0)
    m9 = metrics.get("M9_spatial_spread", 0.0)
    return (
        f"M1={m1:.3f}  M2={m2:.2f}  M3={m3:.1f}  M4={m4:.2f}  "
        f"M6={m6:.3f}  M8={m8:.3f}  M9={m9:.3f}"
    )


# ── Outlier gate (threshold-based) ────────────────────────────────

_OUTLIER_RULES = {
    "M4_high_collisions":     lambda m: m.get("M4_avg_collisions", 0) > 50,
    "M9_clustered":           lambda m: m.get("M9_spatial_spread", 0.5) < 0.2,
    "M9_scattered":           lambda m: m.get("M9_spatial_spread", 0.5) > 0.8,
    "M8_role_imbalance":      lambda m: m.get("M8_agent_utilization_cv", 0) > 0.4,
}


def outlier_flags(metrics: Dict) -> List[str]:
    """List of outlier tags that fired for this candidate.

    Empty list = all-normal-range, no point pushing Tier 2/3 context.
    """
    return [name for name, rule in _OUTLIER_RULES.items() if rule(metrics)]


# ── Tier 2: behavioral fingerprint ────────────────────────────────

def fingerprint_from_csv(csv_path: Path) -> Optional[str]:
    """Read BenchMARL scalar CSV and return a coverage-over-time summary.

    Returns a short multi-line string like:
        start M1=0.00, peak M1=0.18 @ 240k frames, end M1=0.02
        (peak→end drop: 0.16 → reward-hack shape)
    or None if the file is missing / empty.
    """
    if not csv_path.exists():
        return None
    rows = _read_scalar_rows(csv_path)
    if not rows:
        return None
    # BenchMARL CSVs typically have columns ``frames``/``step`` and
    # per-metric columns. We accept a few column-name variants.
    frames_col = _pick_col(rows[0], ("frames", "step", "global_step", "total_frames"))
    m1_col = _pick_col(rows[0], (
        "M1_success_rate", "eval/M1_success_rate",
        "eval/success_rate", "success_rate",
    ))
    if not frames_col or not m1_col:
        return None
    series = [
        (float(r[frames_col]), float(r[m1_col]))
        for r in rows
        if r.get(frames_col) and r.get(m1_col)
    ]
    if not series:
        return None
    start = series[0][1]
    end = series[-1][1]
    peak_frames, peak = max(series, key=lambda x: x[1])
    peak_at = peak_frames
    drop = peak - end
    tag = (
        "reward_hack_shape" if drop > 0.20
        else "monotonic_rise" if (end >= peak * 0.95 and peak > 0.05)
        else "plateau_then_collapse" if drop > 0.10
        else "flat_zero" if peak < 0.02
        else "flat_nonzero" if peak < 0.05
        else "oscillating"
    )
    return (
        f"start_M1={start:.3f}  peak_M1={peak:.3f} @ {int(peak_at):,} frames  "
        f"end_M1={end:.3f}  drop={drop:+.3f}  shape={tag}"
    )


def classify_learning_curve(trajectory: Sequence[float]) -> str:
    """Tag a raw M1-over-time trajectory with a shape label.

    Used when the BenchMARL CSV isn't available (e.g. short eval runs).
    """
    if not trajectory:
        return "flat_zero"
    peak = max(trajectory)
    end = trajectory[-1]
    start = trajectory[0]
    if peak < 0.02:
        return "flat_zero"
    drop = peak - end
    if drop > 0.20:
        return "reward_hack_shape"
    if drop > 0.10:
        return "plateau_then_collapse"
    if peak < 0.05:
        return "flat_nonzero"
    if end >= start and end >= peak * 0.9:
        return "monotonic_rise"
    return "oscillating"


# ── Prompt rendering with include_signals gating ──────────────────

def format_behavioral_block(
    metrics: Dict,
    include_signals: Sequence[str],
    csv_path: Optional[Path] = None,
    trajectory: Optional[Sequence[float]] = None,
) -> str:
    """Render the behavioral-signals block for inclusion in a prompt.

    Honors the caller's ``include_signals`` filter (LERO-MP v3 §4.1).
    Only tiers in the list are rendered; others return empty strings
    so they take no prompt tokens.

    Example output (all tiers on):
        Tier 1 (scalars): M1=0.12  M2=-3.20  M3=180  M4=68.00  M6=0.44  M8=0.52  M9=0.19
        Tier 2 (fingerprint): start_M1=0.00  peak_M1=0.18 @ 240,000 frames
          end_M1=0.02  drop=+0.16  shape=reward_hack_shape
        Tier 3 (curve): shape=reward_hack_shape
        Outliers: M9_clustered, M8_role_imbalance
    """
    want = set(include_signals or [])
    lines: List[str] = []
    if "scalar" in want:
        lines.append(f"Tier 1 (scalars): {format_tier1(metrics)}")
    if "fingerprint" in want and csv_path is not None:
        fp = fingerprint_from_csv(csv_path)
        if fp:
            lines.append(f"Tier 2 (fingerprint): {fp}")
    if "curve_shape" in want:
        if trajectory is not None:
            tag = classify_learning_curve(trajectory)
            lines.append(f"Tier 3 (curve): shape={tag}")
        elif csv_path is not None and csv_path.exists():
            fp = fingerprint_from_csv(csv_path)
            if fp and "shape=" in fp:
                tag = fp.split("shape=")[-1].strip()
                lines.append(f"Tier 3 (curve): shape={tag}")
    flags = outlier_flags(metrics)
    if flags:
        lines.append(f"Outliers: {', '.join(flags)}")
    return "\n".join(lines) if lines else "(no behavioral signals included)"


# ── internals ────────────────────────────────────────────────────

def _read_scalar_rows(path: Path) -> List[Dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _pick_col(row: Dict[str, str], candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in row:
            return c
    # Case-insensitive fallback
    lower_map = {k.lower(): k for k in row.keys()}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None
