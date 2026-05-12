"""Phase 5: pure-Python render_evolution_doc tests (no I/O).

The renderer takes summary + history and emits a markdown string.
Side-effect-free, so we can hammer it with edge cases without disk
fixtures.
"""

# pylint: disable=missing-function-docstring

from multi_scenario.adapters.lero.evolution_doc import render_evolution_doc
from multi_scenario.domain.lero import (
    Candidate,
    CandidateCode,
    CandidateMetrics,
    CandidateResult,
    FallbackEntry,
    LeroRunSummary,
)


def _candidate(it: int, ci: int, m1: float) -> CandidateResult:
    return CandidateResult(
        candidate=Candidate(
            iteration=it,
            candidate_idx=ci,
            code=CandidateCode(obs_source="def enhance_observation(s): pass\n"),
        ),
        metrics=CandidateMetrics(M1_success_rate=m1, M2_avg_return=m1 * 10),
        verdict="progress",
    )


def _summary(
    *,
    full_metrics: CandidateMetrics | None = None,
    fallback: list[FallbackEntry] | None = None,
) -> LeroRunSummary:
    return LeroRunSummary(
        exp_id="test",
        seed=0,
        n_iterations_completed=1,
        n_candidates_total=2,
        total_cost_usd=0.01,
        best_candidate_metrics=CandidateMetrics(M1_success_rate=0.5),
        best_candidate_verdict="progress",
        best_candidate_full_metrics=full_metrics,
        fallback_chain=fallback or [],
        full_training_succeeded=full_metrics is not None,
    )


def test_renders_headline_with_both_m1_when_full_metrics_present():
    """Headline shows inner-M1 + post-full-train-M1 distinctly."""
    summary = _summary(
        full_metrics=CandidateMetrics(M1_success_rate=0.85, M2_avg_return=15.0)
    )
    doc = render_evolution_doc(summary=summary, history=[_candidate(0, 0, 0.5)])
    # Inner = 0.500 (from best_candidate_metrics); Full = 0.850.
    assert "0.500" in doc
    assert "0.850" in doc
    assert "Headline" in doc


def test_renders_em_dash_when_full_metrics_missing():
    """No full metrics → headline still renders; full M1 cell shows em-dash."""
    summary = _summary(full_metrics=None)
    doc = render_evolution_doc(summary=summary, history=[_candidate(0, 0, 0.3)])
    # The inner M1 (0.500) still renders; the full M1 cell is em-dash.
    assert "post-full-train M1" in doc
    assert "| —" in doc  # em-dash placeholder for missing full metrics


def test_per_iter_table_lists_all_candidates_with_links():
    """Each candidate gets a row with response + obs/reward code links."""
    history = [_candidate(0, 0, 0.1), _candidate(0, 1, 0.4), _candidate(1, 0, 0.6)]
    summary = _summary()
    doc = render_evolution_doc(summary=summary, history=history)
    # 2 iters → 2 ### sections.
    assert "### Iteration 0" in doc
    assert "### Iteration 1" in doc
    # Links use relative paths to prompts/.
    assert "prompts/iter_0/cand_0/response.md" in doc
    assert "prompts/iter_0/cand_1/response.md" in doc
    assert "prompts/iter_1/cand_0/obs_source.py" in doc


def test_feedback_prompt_linked_only_for_iter_gt_zero():
    """Iter 0 has no feedback prompt; iter > 0 does."""
    history = [_candidate(0, 0, 0.1), _candidate(1, 0, 0.4)]
    summary = _summary()
    doc = render_evolution_doc(summary=summary, history=history)
    # Both iters have system + user_initial links.
    assert "prompts/iter_0/system.md" in doc
    assert "prompts/iter_1/system.md" in doc
    # Only iter 1 carries a user_feedback link.
    assert "prompts/iter_0/user_feedback.md" not in doc
    assert "prompts/iter_1/user_feedback.md" in doc


def test_selected_winner_section_shows_winning_candidate():
    """Winner section names the iter/cand from the successful fallback entry."""
    summary = _summary(
        full_metrics=CandidateMetrics(M1_success_rate=0.9),
        fallback=[
            FallbackEntry(
                rank=0,
                iteration=2,
                candidate_idx=1,
                eval_metrics=CandidateMetrics(M1_success_rate=0.4),
                outcome="success",
                full_train_metrics=CandidateMetrics(M1_success_rate=0.9),
            )
        ],
    )
    doc = render_evolution_doc(summary=summary, history=[_candidate(2, 1, 0.4)])
    assert "Selected winner" in doc
    assert "iter 2 cand 1" in doc
    assert "prompts/iter_2/cand_1/obs_source.py" in doc


def test_selected_winner_handles_no_success_case():
    """Every fallback rank crashed → winner section says so without raising."""
    summary = _summary(
        full_metrics=None,
        fallback=[
            FallbackEntry(
                rank=0,
                iteration=0,
                candidate_idx=0,
                eval_metrics=CandidateMetrics(),
                outcome="crashed",
                error="kaboom",
            )
        ],
    )
    doc = render_evolution_doc(summary=summary, history=[_candidate(0, 0, 0.0)])
    assert "No full-training succeeded" in doc


def test_fallback_chain_table_lists_every_attempt():
    """Each fallback entry becomes one row, outcomes shown verbatim."""
    summary = _summary(
        fallback=[
            FallbackEntry(
                rank=0, iteration=0, candidate_idx=0,
                eval_metrics=CandidateMetrics(), outcome="crashed",
                error="OOM at iter 5",
            ),
            FallbackEntry(
                rank=1, iteration=0, candidate_idx=1,
                eval_metrics=CandidateMetrics(), outcome="success",
                full_train_metrics=CandidateMetrics(M1_success_rate=0.7),
            ),
        ],
    )
    doc = render_evolution_doc(summary=summary, history=[])
    assert "Fallback chain" in doc
    assert "`crashed`" in doc
    assert "`success`" in doc
    assert "OOM at iter 5" in doc


def test_no_absolute_paths_in_doc():
    """All cross-document links are relative to output/lero/.

    If an absolute path or anything containing ``/output/lero/`` leaks
    into the doc, moving the run dir breaks the rendered links.
    """
    summary = _summary(
        full_metrics=CandidateMetrics(M1_success_rate=0.5),
    )
    doc = render_evolution_doc(summary=summary, history=[_candidate(0, 0, 0.2)])
    assert "/output/lero/" not in doc
    # Spot-check that links don't start with ``/``.
    for line in doc.splitlines():
        for token in line.split("]("):
            if token.startswith("/"):
                raise AssertionError(f"absolute link leaked: {line!r}")
