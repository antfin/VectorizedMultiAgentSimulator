"""F9.6.a — :class:`InitialAndFeedbackComposer` — default composer.

Mirrors rendezvous_comm's two-phase pattern:

- **Iteration 0**: send ``[system, initial_user]`` to ask for the first
  N candidates.
- **Iteration ≥ 1**: send ``[system, initial_user, feedback]`` where
  the feedback message includes the accumulated history (sorted by
  M1, then M2) and the index of the best candidate so far.

The Strategist hook (``strategy_card``) is accepted but ignored —
F9.7.A's stub composer wraps this one and injects mutations; F9.7.B's
full round-table replaces the stub with real Strategist/Editor/Critic
logic.
"""

from typing import Any

from multi_scenario.domain.lero import CandidateResult, StrategyCard
from multi_scenario.domain.ports import ComposedPrompt, PromptRenderer


_VERDICT_RANK: dict[str, int] = {
    "progress": 3,
    "flat": 2,
    "regression": 1,
    "invalid": 0,
}


def _rank_key(r: CandidateResult) -> tuple[int, float, float, float]:
    """Sort key for the feedback's candidate ranking.

    Mirrors rendezvous_comm: (verdict descending, M1 descending, M2
    descending, M6 descending). Replaces the rendezvous_comm raw-
    metrics-only sort by also threading ``verdict`` so the LLM sees
    "progress" candidates before "regression" / "invalid" ones — fixes
    the inversion the rendezvous_comm 2026-04-16 sort had when M1 was
    high but the run had crashed.
    """
    m = r.metrics
    return (
        _VERDICT_RANK.get(r.verdict, 0),
        m.M1_success_rate if m.M1_success_rate is not None else -1e9,
        m.M2_avg_return if m.M2_avg_return is not None else -1e9,
        m.M6_coverage_progress if m.M6_coverage_progress is not None else -1e9,
    )


def _format_candidate_block(
    rank: int, idx: int, r: CandidateResult, include_code: bool
) -> str:
    """One block in the feedback's candidates_results string."""
    m = r.metrics
    lines = [
        f"--- Candidate #{idx + 1} (rank {rank + 1}, verdict={r.verdict}) ---",
        f"M1 Success Rate:    {(m.M1_success_rate or 0.0):.3f}",
        f"M2 Avg Return:      {(m.M2_avg_return or 0.0):.2f}",
        f"M4 Avg Collisions:  {(m.M4_collisions or 0.0):.2f}",
        f"M6 Coverage:        {(m.M6_coverage_progress or 0.0):.3f}",
    ]
    if m.M5_tokens is not None and m.M5_tokens > 0:
        lines.append(f"M5 Avg Tokens:      {m.M5_tokens:.1f}")
    lines.append("")
    if include_code:
        if r.candidate.code.reward_source:
            lines.append("Reward function:")
            lines.append(f"```python\n{r.candidate.code.reward_source}\n```")
        if r.candidate.code.obs_source:
            lines.append("Observation enhancement:")
            lines.append(f"```python\n{r.candidate.code.obs_source}\n```")
    lines.append("")
    return "\n".join(lines)


class InitialAndFeedbackComposer:
    """Default :class:`PromptComposer` — initial → feedback pattern.

    Top-k history-with-code is configurable so a future tuning sweep
    can dial how much token budget the feedback section consumes
    without orchestrator-side surgery.
    """

    def __init__(
        self,
        *,
        renderer: PromptRenderer,
        prompt_version: str,
        n_candidates: int,
        top_k_with_code: int = 3,
    ) -> None:
        self._renderer = renderer
        self._prompt_version = prompt_version
        self._n_candidates = n_candidates
        self._top_k_with_code = top_k_with_code

    def compose(
        self,
        *,
        iteration: int,
        history: list[CandidateResult],
        task_params: dict[str, Any],
        strategy_card: StrategyCard | None = None,
    ) -> ComposedPrompt:
        # Strategist hook unused by the default composer — F9.7.A's
        # MetaPromptComposer is the consumer when meta-prompting is on.
        del strategy_card
        # Build a single render context with task params + the
        # candidates-results / best-idx pair feedback.j2 needs.
        ctx: dict[str, Any] = dict(task_params)
        ctx.setdefault("n_candidates", self._n_candidates)

        # System + initial_user are always sent (the LLM rebuilds context
        # from scratch each iteration; rendezvous_comm pattern).
        system = self._renderer.render(
            version=self._prompt_version, template="system", context=ctx
        )
        initial_user = self._renderer.render(
            version=self._prompt_version, template="initial_user", context=ctx
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": initial_user},
        ]

        if iteration > 0:
            ranked = sorted(history, key=_rank_key, reverse=True)
            best_idx = (
                history.index(ranked[0]) if ranked else 0
            )  # 0-indexed; the prompt adds +1 for display
            blocks = []
            for rank, r in enumerate(ranked):
                # Original index in history so the LLM sees stable IDs.
                idx = history.index(r)
                blocks.append(
                    _format_candidate_block(
                        rank=rank,
                        idx=idx,
                        r=r,
                        include_code=rank < self._top_k_with_code,
                    )
                )
            ctx_fb: dict[str, Any] = {
                **ctx,
                "candidates_results": "\n".join(blocks),
                "best_idx": best_idx + 1,  # 1-indexed for human-readable display
                # Comm-aware feedback (rendezvous_comm pattern): emit
                # the comm-metrics line only when the history contains
                # at least one candidate with non-zero M5_tokens.
                "comm_metrics": _comm_metrics_line(history),
            }
            feedback = self._renderer.render(
                version=self._prompt_version, template="feedback", context=ctx_fb
            )
            messages.append({"role": "user", "content": feedback})
            ctx = ctx_fb

        return ComposedPrompt(
            messages=messages,
            prompt_version=self._prompt_version,
            render_context=ctx,
            signal_tier="scalar",
        )


def _comm_metrics_line(history: list[CandidateResult]) -> str:
    """Emit the comm-feedback hint when ANY candidate used comm tokens."""
    if any((r.metrics.M5_tokens or 0.0) > 0 for r in history):
        return (
            "- M5 (Avg Tokens): communication messages per episode. "
            "Balance informativeness vs bandwidth cost."
        )
    return ""
