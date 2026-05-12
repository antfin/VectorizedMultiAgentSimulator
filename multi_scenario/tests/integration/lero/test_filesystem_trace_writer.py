"""F9.3 — :class:`FilesystemTraceWriter` contract."""

# pylint: disable=missing-function-docstring,protected-access

import json
from pathlib import Path

import pytest

from multi_scenario.adapters.lero import FilesystemTraceWriter
from multi_scenario.domain.lero import (
    Candidate,
    CandidateCode,
    CandidateMetrics,
    CandidateResult,
    LeroRunSummary,
    LlmUsage,
    PromptTrace,
    ReasoningTrace,
    ResponseTrace,
)


@pytest.fixture
def writer():
    return FilesystemTraceWriter()


def _result(iteration: int = 0, candidate_idx: int = 0) -> CandidateResult:
    return CandidateResult(
        candidate=Candidate(
            iteration=iteration,
            candidate_idx=candidate_idx,
            code=CandidateCode(
                reward_source="def compute_reward(s): return s['agent_pos']"
            ),
        ),
        metrics=CandidateMetrics(M1_success_rate=0.42),
        verdict="progress",
        note="example note",
    )


# ── Per-trace writes ──────────────────────────────────────────────────


def test_write_prompt_lands_at_canonical_path(tmp_path: Path, writer):
    trace = PromptTrace(
        prompt_version="v2_fewshot_k2_local",
        messages=[{"role": "user", "content": "hi"}],
    )
    writer.write_prompt(
        run_dir=tmp_path, iteration=2, candidate_idx=1, attempt=0, trace=trace
    )
    expected = (
        tmp_path / "output" / "lero" / "iter_2" / "cand_1" / "attempt_0" / "prompt.json"
    )
    assert expected.is_file()
    parsed = PromptTrace.model_validate_json(expected.read_text("utf-8"))
    assert parsed == trace


def test_write_response_alongside_prompt(tmp_path: Path, writer):
    p = PromptTrace(
        prompt_version="v2",
        messages=[{"role": "user", "content": "x"}],
    )
    r = ResponseTrace(
        text="```python\ndef compute_reward(s): ...\n```",
        finish_reason="stop",
        usage=LlmUsage(prompt_tokens=10, completion_tokens=5, estimated_cost_usd=0.01),
    )
    writer.write_prompt(
        run_dir=tmp_path, iteration=0, candidate_idx=0, attempt=0, trace=p
    )
    writer.write_response(
        run_dir=tmp_path, iteration=0, candidate_idx=0, attempt=0, trace=r
    )
    attempt_dir = tmp_path / "output" / "lero" / "iter_0" / "cand_0" / "attempt_0"
    assert (attempt_dir / "prompt.json").is_file()
    assert (attempt_dir / "response.json").is_file()


def test_write_reasoning_emits_separate_file(tmp_path: Path, writer):
    trace = ReasoningTrace(text="step 1: …\nstep 2: …")
    writer.write_reasoning(
        run_dir=tmp_path, iteration=0, candidate_idx=0, attempt=0, trace=trace
    )
    path = (
        tmp_path
        / "output"
        / "lero"
        / "iter_0"
        / "cand_0"
        / "attempt_0"
        / "reasoning.json"
    )
    assert path.is_file()
    parsed = ReasoningTrace.model_validate_json(path.read_text("utf-8"))
    assert parsed.text == trace.text


def test_write_candidate_result_uses_iter_and_idx_from_payload(tmp_path: Path, writer):
    """Path is derived from result.candidate; orchestrator doesn't pass it twice."""
    res = _result(iteration=3, candidate_idx=2)
    writer.write_candidate_result(run_dir=tmp_path, result=res)
    path = tmp_path / "output" / "lero" / "iter_3" / "cand_2" / "result.json"
    assert path.is_file()
    parsed = CandidateResult.model_validate_json(path.read_text("utf-8"))
    assert parsed.metrics.M1_success_rate == pytest.approx(0.42)


# ── Aggregate writes ──────────────────────────────────────────────────


def test_write_evolution_history_creates_json_array(tmp_path: Path, writer):
    results = [_result(iteration=i) for i in range(3)]
    writer.write_evolution_history(run_dir=tmp_path, results=results)
    path = tmp_path / "output" / "lero" / "evolution_history.json"
    assert path.is_file()
    parsed = json.loads(path.read_text("utf-8"))
    assert isinstance(parsed, list)
    assert len(parsed) == 3
    assert parsed[2]["candidate"]["iteration"] == 2


def test_write_summary_lands_at_final_summary_json(tmp_path: Path, writer):
    summary = LeroRunSummary(
        exp_id="er1_lero",
        seed=0,
        n_iterations_completed=4,
        n_candidates_total=12,
        total_cost_usd=2.50,
        best_candidate_metrics=CandidateMetrics(M1_success_rate=0.88),
        best_candidate_verdict="progress",
        full_training_succeeded=True,
    )
    writer.write_summary(run_dir=tmp_path, summary=summary)
    path = tmp_path / "output" / "lero" / "final_summary.json"
    assert path.is_file()
    parsed = LeroRunSummary.model_validate_json(path.read_text("utf-8"))
    assert parsed == summary


# ── Atomic / idempotent properties ────────────────────────────────────


def test_atomic_write_no_partial_files_left_behind(tmp_path: Path, writer):
    """After a successful write, no .tmp files lurking."""
    res = _result()
    writer.write_candidate_result(run_dir=tmp_path, result=res)
    cand_dir = tmp_path / "output" / "lero" / "iter_0" / "cand_0"
    tmp_files = [p for p in cand_dir.iterdir() if p.suffix == ".tmp"]
    assert tmp_files == []


def test_idempotent_rewrite_is_a_noop(tmp_path: Path, writer):
    """Same payload → no rename triggered (saves syscalls on resume).

    We assert this via mtime: a no-op write shouldn't bump it.
    """
    res = _result()
    writer.write_candidate_result(run_dir=tmp_path, result=res)
    path = tmp_path / "output" / "lero" / "iter_0" / "cand_0" / "result.json"
    mtime_before = path.stat().st_mtime_ns

    writer.write_candidate_result(run_dir=tmp_path, result=res)
    mtime_after = path.stat().st_mtime_ns
    assert mtime_after == mtime_before


def test_overwrite_with_different_payload_replaces_atomically(tmp_path: Path, writer):
    """A re-write with new content lands the new content, not corruption."""
    res_v1 = _result()
    writer.write_candidate_result(run_dir=tmp_path, result=res_v1)
    res_v2 = res_v1.model_copy(update={"note": "updated note"})
    writer.write_candidate_result(run_dir=tmp_path, result=res_v2)

    path = tmp_path / "output" / "lero" / "iter_0" / "cand_0" / "result.json"
    parsed = CandidateResult.model_validate_json(path.read_text("utf-8"))
    assert parsed.note == "updated note"


# ── Hex-arch + path helper invariants ────────────────────────────────


def test_path_helpers_stable_layout():
    """Pin the exact layout so post-hoc tools (and rendezvous_comm
    comparisons) can hardcode the path shape."""
    rd = Path("/tmp/run_dir")
    assert FilesystemTraceWriter._lero_root(rd) == Path("/tmp/run_dir/output/lero")
    assert FilesystemTraceWriter._iter_dir(rd, 5) == Path(
        "/tmp/run_dir/output/lero/iter_5"
    )
    assert FilesystemTraceWriter._cand_dir(rd, 5, 2) == Path(
        "/tmp/run_dir/output/lero/iter_5/cand_2"
    )
    assert FilesystemTraceWriter._attempt_dir(rd, 5, 2, 0) == Path(
        "/tmp/run_dir/output/lero/iter_5/cand_2/attempt_0"
    )


# ── Phase 5: prompts/ markdown side-files + evolution_doc.md ────────


def test_write_prompt_emits_system_and_user_initial_markdown(tmp_path: Path, writer):
    """Phase 5: prompts/iter_N/{system.md, user_initial.md} populated."""
    trace = PromptTrace(
        prompt_version="v2_fewshot_k2_local",
        messages=[
            {"role": "system", "content": "you are a reward designer"},
            {"role": "user", "content": "task is rendezvous with 4 agents"},
        ],
    )
    writer.write_prompt(
        run_dir=tmp_path, iteration=0, candidate_idx=0, attempt=0, trace=trace
    )
    prompts_iter_0 = tmp_path / "output" / "lero" / "prompts" / "iter_0"
    assert (prompts_iter_0 / "system.md").read_text() == "you are a reward designer"
    assert (
        prompts_iter_0 / "user_initial.md"
    ).read_text() == "task is rendezvous with 4 agents"
    # user_feedback.md only appears on iter > 0.
    assert not (prompts_iter_0 / "user_feedback.md").exists()


def test_write_prompt_emits_user_feedback_on_iter_gt_zero(tmp_path: Path, writer):
    """Phase 5: feedback user message goes to prompts/iter_N/user_feedback.md."""
    trace = PromptTrace(
        prompt_version="v2_fewshot_k2_local",
        messages=[
            {"role": "system", "content": "system text"},
            {"role": "user", "content": "initial user text"},
            {"role": "user", "content": "feedback referencing prior candidates"},
        ],
    )
    writer.write_prompt(
        run_dir=tmp_path, iteration=1, candidate_idx=0, attempt=0, trace=trace
    )
    feedback_path = (
        tmp_path / "output" / "lero" / "prompts" / "iter_1" / "user_feedback.md"
    )
    assert feedback_path.read_text() == "feedback referencing prior candidates"


def test_write_response_emits_response_md(tmp_path: Path, writer):
    """Phase 5: ResponseTrace.text → prompts/iter_N/cand_M/response.md."""
    trace = ResponseTrace(text="```python\ndef enhance_observation(s): return s\n```")
    writer.write_response(
        run_dir=tmp_path, iteration=2, candidate_idx=3, attempt=0, trace=trace
    )
    md_path = (
        tmp_path / "output" / "lero" / "prompts" / "iter_2" / "cand_3" / "response.md"
    )
    assert md_path.is_file()
    assert "enhance_observation" in md_path.read_text()


def test_write_candidate_result_extracts_obs_and_reward_source(tmp_path: Path, writer):
    """Phase 5: obs_source + reward_source → separate .py files under prompts/."""
    result = CandidateResult(
        candidate=Candidate(
            iteration=0,
            candidate_idx=1,
            code=CandidateCode(
                reward_source="def compute_reward(s): return s['agent_pos']\n",
                obs_source="def enhance_observation(s): return s['lidar_targets']\n",
            ),
        ),
        metrics=CandidateMetrics(M1_success_rate=0.5),
        verdict="progress",
    )
    writer.write_candidate_result(run_dir=tmp_path, result=result)
    cand_prompts = tmp_path / "output" / "lero" / "prompts" / "iter_0" / "cand_1"
    assert "compute_reward" in (cand_prompts / "reward_source.py").read_text()
    assert "enhance_observation" in (cand_prompts / "obs_source.py").read_text()


def test_write_candidate_result_skips_code_files_when_source_empty(
    tmp_path: Path, writer
):
    """Empty code → no placeholder file (keeps prompts/ tree noise-free)."""
    result = CandidateResult(
        candidate=Candidate(
            iteration=0,
            candidate_idx=0,
            code=CandidateCode(reward_source=None, obs_source=None),
        ),
        metrics=CandidateMetrics(),
        verdict="invalid",
    )
    writer.write_candidate_result(run_dir=tmp_path, result=result)
    cand_prompts = tmp_path / "output" / "lero" / "prompts" / "iter_0" / "cand_0"
    assert not (cand_prompts / "reward_source.py").exists()
    assert not (cand_prompts / "obs_source.py").exists()


def test_write_summary_renders_evolution_doc_md(tmp_path: Path, writer):
    """Phase 5: evolution_doc.md appears next to final_summary.json with
    headline, per-iter tables, and relative links to prompts/."""
    # Seed history + summary.
    history = [
        CandidateResult(
            candidate=Candidate(
                iteration=0,
                candidate_idx=0,
                code=CandidateCode(obs_source="def enhance_observation(s): pass\n"),
            ),
            metrics=CandidateMetrics(M1_success_rate=0.1, M2_avg_return=-2.0),
            verdict="progress",
        ),
        CandidateResult(
            candidate=Candidate(
                iteration=0,
                candidate_idx=1,
                code=CandidateCode(obs_source="def enhance_observation(s): pass\n"),
            ),
            metrics=CandidateMetrics(M1_success_rate=0.4, M2_avg_return=1.0),
            verdict="progress",
        ),
    ]
    # Need evolution_history.json on disk so write_summary reads it.
    writer.write_evolution_history(run_dir=tmp_path, results=history)
    summary = LeroRunSummary(
        exp_id="lero_smoke",
        seed=0,
        n_iterations_completed=1,
        n_candidates_total=2,
        total_cost_usd=0.0123,
        best_candidate_metrics=CandidateMetrics(M1_success_rate=0.4, M2_avg_return=1.0),
        best_candidate_verdict="progress",
        best_candidate_full_metrics=CandidateMetrics(
            M1_success_rate=0.85, M2_avg_return=15.0
        ),
        fallback_chain=[],
        full_training_succeeded=True,
    )
    writer.write_summary(run_dir=tmp_path, summary=summary)

    doc_md = tmp_path / "output" / "lero" / "evolution_doc.md"
    assert doc_md.is_file()
    text = doc_md.read_text()
    # Headline carries both inner + full M1 (Phase 1 metric capture).
    assert "0.400" in text and "0.850" in text
    # Per-iter table mentions both candidates.
    assert "## Iterations" in text
    assert "| 0 |" in text and "| 1 |" in text
    # Relative links point at the prompts/ folder (no absolute paths).
    assert "prompts/iter_0/system.md" in text
    assert "prompts/iter_0/cand_0/obs_source.py" in text
    # No CRLF / no path leak — files live under output/lero/.
    assert "/output/lero/" not in text


def test_writer_satisfies_protocol():
    """Structural check: FilesystemTraceWriter has every TraceWriter method."""
    import inspect

    from multi_scenario.domain.ports import TraceWriter  # noqa: F401

    expected = {
        "write_prompt",
        "write_response",
        "write_reasoning",
        "write_candidate_result",
        "write_evolution_history",
        "write_summary",
    }
    actual = {
        name
        for name, _ in inspect.getmembers(FilesystemTraceWriter, predicate=callable)
    }
    missing = expected - actual
    assert not missing, f"FilesystemTraceWriter missing methods: {missing}"
