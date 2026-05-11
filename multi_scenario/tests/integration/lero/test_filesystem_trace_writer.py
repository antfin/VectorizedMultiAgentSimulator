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
