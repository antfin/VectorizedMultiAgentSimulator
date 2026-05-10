"""F9.0 — domain/lero/ pure-data type contracts."""

# pylint: disable=missing-function-docstring

import json

import pytest

from multi_scenario.domain.lero import (
    Candidate,
    CandidateCode,
    CandidateGenerationFailed,
    CandidateMetrics,
    CandidateResult,
    FairnessViolation,
    FallbackEntry,
    LeroError,
    LeroRunSummary,
    LlmCompletion,
    LlmCostCapExceeded,
    PromptTrace,
    ReasoningTrace,
    ResponseTrace,
)
from pydantic import ValidationError


# ── CandidateCode ──────────────────────────────────────────────────────


def test_candidate_code_allows_one_or_both_sources():
    """Either reward_source or obs_source may be None — F9.4 enforces "at
    least one" at extraction time, the type itself is permissive."""
    CandidateCode(reward_source="def compute_reward(s): return s['agent_pos']")
    CandidateCode(obs_source="def enhance_observation(s): return s['agent_pos']")
    CandidateCode(reward_source="x", obs_source="y")
    # Both None is also schematically valid — only F9.4's contract layer
    # rejects this; the domain type stays permissive so it round-trips
    # cleanly from a partially-populated trace file.
    CandidateCode()


# ── CandidateMetrics ───────────────────────────────────────────────────


def test_candidate_metrics_all_fields_optional():
    """Newly-built metrics start blank; orchestrator fills them in."""
    m = CandidateMetrics()
    assert m.M1_success_rate is None
    assert m.M5_tokens is None  # comm-only, expected None for non-comm runs


def test_candidate_metrics_strict_rejects_extra_keys():
    with pytest.raises(ValidationError):
        CandidateMetrics.model_validate({"M1_success_rate": 0.5, "M99_alien": 0})


# ── Candidate / CandidateResult ───────────────────────────────────────


def test_candidate_round_trips_through_json():
    cand = Candidate(
        iteration=2,
        candidate_idx=1,
        code=CandidateCode(reward_source="x"),
    )
    blob = cand.model_dump_json()
    parsed = Candidate.model_validate_json(blob)
    assert parsed == cand


@pytest.mark.parametrize(
    "verdict,bad",
    [
        ("invalid", False),
        ("regression", False),
        ("flat", False),
        ("progress", False),
        ("nope", True),  # not a Verdict literal
    ],
)
def test_candidate_result_verdict_literal_enforced(verdict: str, bad: bool):
    payload = {
        "candidate": {
            "iteration": 0,
            "candidate_idx": 0,
            "code": {"reward_source": "x"},
        },
        "metrics": {},
        "verdict": verdict,
    }
    if bad:
        with pytest.raises(ValidationError):
            CandidateResult.model_validate(payload)
    else:
        CandidateResult.model_validate(payload)


# ── LlmCompletion ──────────────────────────────────────────────────────


def test_llm_completion_minimal_construction():
    c = LlmCompletion(text="hello")
    assert c.text == "hello"
    assert c.usage.prompt_tokens == 0
    assert c.usage.estimated_cost_usd == 0.0


def test_llm_completion_round_trips():
    c = LlmCompletion(
        text="```python\ndef f(s): return s\n```",
        reasoning="thinking text",
        finish_reason="stop",
        system_fingerprint="fp_abc123",
    )
    blob = c.model_dump_json()
    parsed = LlmCompletion.model_validate_json(blob)
    assert parsed == c


# ── Trace types ────────────────────────────────────────────────────────


def test_prompt_trace_default_role_is_inner_codegen():
    t = PromptTrace(
        prompt_version="v2_fewshot_k2_local",
        messages=[{"role": "user", "content": "x"}],
    )
    assert t.role == "inner_codegen"


def test_prompt_trace_accepts_meta_roles_for_F9_7B_forward_compat():
    """F9.7.B will write meta_strategist / meta_editor / meta_critic traces.
    The Literal already includes them so trace files written today don't
    need a schema migration when meta-prompting lands."""
    t = PromptTrace(
        role="meta_strategist",
        prompt_version="v2_fewshot_k2_local",
        messages=[],
    )
    assert t.role == "meta_strategist"


def test_response_trace_round_trips():
    r = ResponseTrace(text="hello", finish_reason="stop")
    parsed = ResponseTrace.model_validate_json(r.model_dump_json())
    assert parsed == r


def test_reasoning_trace_round_trips():
    r = ReasoningTrace(text="step 1: …\nstep 2: …")
    parsed = ReasoningTrace.model_validate_json(r.model_dump_json())
    assert parsed == r


# ── LeroRunSummary / FallbackEntry ─────────────────────────────────────


def test_lero_run_summary_minimal():
    summary = LeroRunSummary(
        exp_id="er1_lero_s3b_local",
        seed=0,
        n_iterations_completed=4,
        n_candidates_total=12,
        total_cost_usd=0.27,
        best_candidate_metrics=CandidateMetrics(M1_success_rate=0.88),
        best_candidate_verdict="progress",
        full_training_succeeded=True,
    )
    blob = summary.model_dump_json()
    assert "0.88" in blob


def test_fallback_entry_outcome_literal_enforced():
    base = {
        "rank": 0,
        "iteration": 1,
        "candidate_idx": 0,
        "eval_metrics": {},
    }
    FallbackEntry.model_validate(base | {"outcome": "success"})
    FallbackEntry.model_validate(base | {"outcome": "crashed", "error": "NaN"})
    FallbackEntry.model_validate(base | {"outcome": "skipped"})
    with pytest.raises(ValidationError):
        FallbackEntry.model_validate(base | {"outcome": "kapow"})


# ── Exceptions ─────────────────────────────────────────────────────────


def test_lero_error_is_runtime_error_subclass():
    """``except LeroError`` is the orchestrator's wide catch — must be a
    runtime error so it doesn't shadow base ``Exception`` semantics in
    the fallback path."""
    assert issubclass(LeroError, RuntimeError)


def test_llm_cost_cap_exceeded_carries_amounts():
    err = LlmCostCapExceeded("over budget", spent_usd=5.5, cap_usd=5.0)
    assert isinstance(err, LeroError)
    assert err.spent_usd == 5.5
    assert err.cap_usd == 5.0


def test_lero_specific_exceptions_descend_from_base():
    """``except LeroError`` catches every LERO-specific failure."""
    for exc_cls in (
        LlmCostCapExceeded,
        CandidateGenerationFailed,
        FairnessViolation,
    ):
        # LlmCostCapExceeded has a custom __init__; the others use the
        # plain RuntimeError default which is fine.
        if exc_cls is LlmCostCapExceeded:
            instance = exc_cls("x", spent_usd=1.0, cap_usd=1.0)
        else:
            instance = exc_cls("x")
        assert isinstance(instance, LeroError)


# ── Hex-arch invariant ─────────────────────────────────────────────────


def test_domain_lero_imports_no_torch_or_litellm():
    """domain/lero/ MUST stay pure: no torch, no litellm. Catches a
    regression where someone adds a "convenience" tensor accessor and
    bloats the layer."""
    # pylint: disable=import-outside-toplevel
    import multi_scenario.domain.lero as lero_pkg

    src_files = [f for f in dir(lero_pkg) if not f.startswith("_")]
    # Walk the modules' source for forbidden imports.
    import importlib
    import inspect

    for module_name in [
        "candidate",
        "exceptions",
        "llm_completion",
        "summary",
        "traces",
    ]:
        mod = importlib.import_module(f"multi_scenario.domain.lero.{module_name}")
        src = inspect.getsource(mod)
        for forbidden in (
            "import torch",
            "import litellm",
            "from torch",
            "from litellm",
        ):
            assert forbidden not in src, (
                f"domain/lero/{module_name}.py imports forbidden "
                f"{forbidden!r} — keep this layer pure"
            )


def test_traces_round_trip_through_json_for_filesystem_writer():
    """The trace writer at F9.3 atomic-writes these as JSON. Ensure the
    Pydantic surface doesn't introduce a non-JSON-serialisable field
    (e.g., a tensor or a path) that would break atomic writes silently.
    """
    p = PromptTrace(
        prompt_version="v2_fewshot_k2_local",
        messages=[{"role": "user", "content": "hi"}],
        render_context={"x": 1, "y": [1, 2.5, "z"], "nested": {"k": None}},
    )
    blob = p.model_dump_json()
    # JSON-loadable — no NaN / inf / bytes leaking through.
    json.loads(blob)
