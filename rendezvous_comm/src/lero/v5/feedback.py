"""v5 inner-loop feedback builder.

Differences vs `codegen.build_feedback`:
  1. Shows BEST and WORST candidate code (not just top_k best)
  2. Embeds the cumulative tried-and-failed registry
  3. Injects a stagnation/pivot warning when fitness trajectory plateaus
  4. Frames the LLM task explicitly: "explain why X failed before
     proposing N+1, do not retry approaches in the failed list"
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .registry import Registry


_PIVOT_INSTRUCTION = """
⚠️ STAGNATION DETECTED — last 2 iterations did not improve.

Before generating new candidates, you MUST:
  1. State an explicit hypothesis about WHY current approaches stall
     (cite specific feature names from the failed registry).
  2. Either (a) sharpen the current direction with a falsifiable
     prediction about what should change M1, or (b) propose a
     FUNDAMENTALLY different feature family / reward structure.

DO NOT regenerate minor variations of approaches already in the
failed registry. If you find yourself proposing similar features,
PIVOT instead.
""".strip()


def _truncate_code(code: str, max_lines: int = 30) -> str:
    if not code:
        return ""
    lines = code.splitlines()
    if len(lines) <= max_lines:
        return code
    return "\n".join(lines[:max_lines]) + "\n# ... (truncated)"


def build_v5_inner_feedback(
    candidates_with_metrics: List[Tuple[object, Dict, float, str]],
    registry: Registry,
    iter_idx: int,
    n_next_candidates: int,
    pivot: bool,
) -> str:
    """Build feedback string for the next inner iteration.

    `candidates_with_metrics` is a list of
        (CandidateCode, metrics_dict, fitness, shape_tag)
    sorted in DECREASING fitness order. `registry` carries the
    cumulative tried-and-failed entries from prior iters.
    """
    if not candidates_with_metrics:
        return "(no valid candidates this iteration — try again)"

    best_cand, best_metrics, best_fit, best_shape = candidates_with_metrics[0]
    worst_cand, worst_metrics, worst_fit, worst_shape = candidates_with_metrics[-1]

    parts: List[str] = []

    parts.append(f"=== ITERATION {iter_idx} RESULTS ===")
    parts.append("")
    parts.append("Per-candidate scores (sorted by fitness):")
    for rank, (cand, m, fit, sh) in enumerate(candidates_with_metrics, 1):
        parts.append(
            f"  #{rank}  M1={m.get('M1_success_rate', 0):.3f}  "
            f"M6={m.get('M6_coverage_progress', 0):.3f}  "
            f"shape={sh}  fitness={fit:+.3f}"
        )

    parts.append("")
    parts.append(f"✅ BEST: fitness={best_fit:+.3f}  shape={best_shape}  "
                 f"M1={best_metrics.get('M1_success_rate', 0):.3f}")
    if best_cand.obs_source:
        parts.append("Best candidate's observation code:")
        parts.append(f"```python\n{_truncate_code(best_cand.obs_source)}\n```")
    if best_cand.reward_source:
        parts.append("Best candidate's reward code:")
        parts.append(f"```python\n{_truncate_code(best_cand.reward_source)}\n```")

    if worst_cand is not best_cand:
        parts.append("")
        parts.append(f"❌ WORST: fitness={worst_fit:+.3f}  shape={worst_shape}  "
                     f"M1={worst_metrics.get('M1_success_rate', 0):.3f}")
        parts.append("Worst candidate's code (DO NOT recreate this approach):")
        if worst_cand.obs_source:
            parts.append("```python")
            parts.append(_truncate_code(worst_cand.obs_source))
            parts.append("```")
        if worst_cand.reward_source:
            parts.append("```python")
            parts.append(_truncate_code(worst_cand.reward_source))
            parts.append("```")

    parts.append("")
    parts.append("=== CUMULATIVE REGISTRY (across all iterations) ===")
    parts.append(registry.format_for_prompt())

    parts.append("")
    if pivot:
        parts.append(_PIVOT_INSTRUCTION)
    else:
        parts.append(
            f"Generate {n_next_candidates} improved candidates. "
            f"Keep what worked, fix what didn't, and AVOID any approach "
            f"already in the failed registry."
        )

    return "\n".join(parts)
