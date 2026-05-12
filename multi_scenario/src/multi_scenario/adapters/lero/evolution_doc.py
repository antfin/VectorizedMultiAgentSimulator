"""Phase 5 — markdown renderer for the LERO evolution doc.

Pure-Python transform from a :class:`LeroRunSummary` + per-candidate
:class:`CandidateResult` list → a human-readable markdown narrative
that summarises the run, with relative links into the
``output/lero/prompts/`` folder structure.

The renderer is intentionally side-effect-free (returns a string). The
filesystem writer composes the renderer's output with disk I/O. This
split keeps the renderer trivially testable (no tmp_path needed) and
the I/O layer minimal.

Folder layout the rendered links target (created by the trace writer)::

    <run_dir>/output/lero/
    ├── evolution_doc.md                  ← rendered here
    └── prompts/
        ├── iter_0/
        │   ├── system.md                 ← composed prompt's "system" message
        │   ├── user_initial.md           ← composed prompt's first "user" message
        │   ├── cand_0/
        │   │   ├── response.md           ← raw LLM output
        │   │   ├── obs_source.py         ← extracted observation code (when present)
        │   │   └── reward_source.py      ← extracted reward code (when present)
        │   ├── cand_1/...
        ├── iter_1/
        │   ├── system.md
        │   ├── user_initial.md
        │   ├── user_feedback.md          ← only on iter > 0
        │   ├── cand_0/...
"""

from multi_scenario.domain.lero import CandidateResult, LeroRunSummary


def render_evolution_doc(
    *,
    summary: LeroRunSummary,
    history: list[CandidateResult],
) -> str:
    """Render the LERO run as a markdown narrative.

    Returns one self-contained markdown string. All cross-document links
    are RELATIVE to the markdown file's location (``output/lero/``), so
    the doc remains readable when the run dir is moved or browsed
    locally via Streamlit / GitHub / a plain file browser.

    Sections:

    1. **Headline** — exp_id, seed, verdict, the two M1s (inner + full),
       full-training success bit, total cost.
    2. **Per-iteration tables** — one row per candidate with metrics +
       links to its response.md and extracted code files.
    3. **Selected winner** — which iter/cand was picked for full
       training, with direct links to the winning code.
    4. **Fallback chain** — verbose ranking of every attempted full
       training, useful when rank 0 crashed.
    """
    lines: list[str] = []
    lines.append(f"# LERO run: {summary.exp_id}_s{summary.seed}")
    lines.append("")
    lines.extend(_render_headline(summary))
    lines.append("")
    lines.extend(_render_iterations(summary, history))
    lines.append("")
    lines.extend(_render_selected_winner(summary, history))
    lines.append("")
    lines.extend(_render_fallback_chain(summary))
    return "\n".join(lines) + "\n"


# ── section renderers ───────────────────────────────────────────────


def _render_headline(summary: LeroRunSummary) -> list[str]:
    """Top-of-doc summary table — the numbers a reader looks for first."""
    inner = summary.best_candidate_metrics
    full = summary.best_candidate_full_metrics
    rows = [
        "## Headline",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| iterations completed | {summary.n_iterations_completed} |",
        f"| candidates evaluated | {summary.n_candidates_total} |",
        f"| best inner-loop verdict | `{summary.best_candidate_verdict}` |",
        f"| inner-loop M1 (1M frames, screening) | {_fmt(inner.M1_success_rate)} |",
        f"| **post-full-train M1 (10M frames, science result)** | {_fmt(full.M1_success_rate if full else None)} |",
        f"| inner-loop M2 (avg return) | {_fmt(inner.M2_avg_return)} |",
        f"| post-full-train M2 | {_fmt(full.M2_avg_return if full else None)} |",
        f"| full training succeeded | {summary.full_training_succeeded} |",
        f"| total LLM cost (USD) | {summary.total_cost_usd:.4f} |",
    ]
    return rows


def _render_iterations(
    summary: LeroRunSummary,
    history: list[CandidateResult],
) -> list[str]:
    """Per-iter tables: candidates × metrics × code links."""
    lines: list[str] = ["## Iterations", ""]
    # Group history by iteration.
    iters: dict[int, list[CandidateResult]] = {}
    for r in history:
        iters.setdefault(r.candidate.iteration, []).append(r)
    for it in sorted(iters):
        lines.append(f"### Iteration {it}")
        lines.append("")
        # Prompt files (relative links). These are written by the
        # filesystem trace writer alongside the markdown doc.
        prompt_links = [
            f"[system](prompts/iter_{it}/system.md)",
            f"[user_initial](prompts/iter_{it}/user_initial.md)",
        ]
        if it > 0:
            prompt_links.append(
                f"[user_feedback](prompts/iter_{it}/user_feedback.md)"
            )
        lines.append("- Prompts: " + " / ".join(prompt_links))
        lines.append("")
        # Per-candidate table.
        lines.append("| cand | verdict | M1 (inner) | M2 (inner) | response | code |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for r in iters[it]:
            cand_dir = f"prompts/iter_{it}/cand_{r.candidate.candidate_idx}"
            code_links = []
            if r.candidate.code.reward_source:
                code_links.append(f"[reward]({cand_dir}/reward_source.py)")
            if r.candidate.code.obs_source:
                code_links.append(f"[obs]({cand_dir}/obs_source.py)")
            code_cell = " · ".join(code_links) if code_links else "—"
            lines.append(
                f"| {r.candidate.candidate_idx} "
                f"| `{r.verdict}` "
                f"| {_fmt(r.metrics.M1_success_rate)} "
                f"| {_fmt(r.metrics.M2_avg_return)} "
                f"| [response]({cand_dir}/response.md) "
                f"| {code_cell} |"
            )
        lines.append("")
    return lines


def _render_selected_winner(
    summary: LeroRunSummary, history: list[CandidateResult]
) -> list[str]:
    """Which candidate ended up full-trained — direct link to its code.

    Only emits code links for the source types the winning candidate
    actually has (obs and/or reward). Avoids broken links to files that
    don't exist — reward-only runs don't write ``obs_source.py``.
    """
    winner = next(
        (e for e in summary.fallback_chain if e.outcome == "success"),
        None,
    )
    lines = ["## Selected winner", ""]
    if winner is None:
        lines.append(
            "_No full-training succeeded; every rank in the fallback chain crashed._"
        )
        return lines
    cand_dir = f"prompts/iter_{winner.iteration}/cand_{winner.candidate_idx}"
    lines.append(
        f"- **iter {winner.iteration} cand {winner.candidate_idx}** "
        f"(fallback rank {winner.rank})"
    )
    lines.append(f"- Response: [response.md]({cand_dir}/response.md)")
    # Walk the history to find the matching candidate and check which
    # code sources it actually carries.
    winner_entry = next(
        (
            r
            for r in history
            if r.candidate.iteration == winner.iteration
            and r.candidate.candidate_idx == winner.candidate_idx
        ),
        None,
    )
    if winner_entry is not None:
        code_links: list[str] = []
        if winner_entry.candidate.code.reward_source:
            code_links.append(f"[reward_source.py]({cand_dir}/reward_source.py)")
        if winner_entry.candidate.code.obs_source:
            code_links.append(f"[obs_source.py]({cand_dir}/obs_source.py)")
        if code_links:
            lines.append("- Code: " + " · ".join(code_links))
    if summary.best_candidate_full_metrics is not None:
        full = summary.best_candidate_full_metrics
        lines.extend(
            [
                "",
                "### Post-full-train metrics (200-episode eval)",
                "",
                "| metric | value |",
                "| --- | --- |",
                f"| M1_success_rate | {_fmt(full.M1_success_rate)} |",
                f"| M2_avg_return   | {_fmt(full.M2_avg_return)} |",
                f"| M3_steps        | {_fmt(full.M3_steps)} |",
                f"| M4_collisions   | {_fmt(full.M4_collisions)} |",
                f"| M6_coverage_progress | {_fmt(full.M6_coverage_progress)} |",
            ]
        )
    return lines


def _render_fallback_chain(summary: LeroRunSummary) -> list[str]:
    """The ranked list of full-training attempts (success + crashes + skips)."""
    lines = ["## Fallback chain", ""]
    if not summary.fallback_chain:
        lines.append("_No fallback chain — every candidate was invalid._")
        return lines
    lines.append("| rank | iter | cand | outcome | error |")
    lines.append("| --- | --- | --- | --- | --- |")
    for e in summary.fallback_chain:
        err = (e.error or "").replace("|", "\\|").replace("\n", " ") if e.error else ""
        if len(err) > 80:
            err = err[:77] + "…"
        lines.append(
            f"| {e.rank} | {e.iteration} | {e.candidate_idx} "
            f"| `{e.outcome}` | {err} |"
        )
    return lines


# ── small helpers ──────────────────────────────────────────────────


def _fmt(v: float | None) -> str:
    """Format a metric value (or `—` for None)."""
    if v is None:
        return "—"
    return f"{v:.3f}"
