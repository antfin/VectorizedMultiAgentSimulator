"""Tests for the inner-LLM retry loop (LERO-MP v3 §3.1)."""

from __future__ import annotations

from typing import Dict, List

import pytest

from src.lero.inner_llm import (
    CandidateGenerationFailed,
    InnerLLM,
)


class _StubLLMClient:
    """Minimal LLMClient replacement that returns canned responses."""

    def __init__(self, responses: List[str]):
        self._responses = list(responses)
        self.calls = 0

    def generate(self, messages, n=1, seed_base=None):
        out = []
        for _ in range(n):
            self.calls += 1
            if not self._responses:
                raise RuntimeError("stub exhausted")
            out.append(self._responses.pop(0))
        return out

    def generate_structured(self, *a, **kw):  # unused in retry tests
        raise NotImplementedError


_VALID_OBS = """```python
def enhance_observation(scenario_state: dict):
    import torch
    return torch.zeros(1)
```"""

_BAD_SYNTAX = """```python
def enhance_observation(scenario_state: dict)
    return torch.zeros(1
```"""

_WRONG_FUNC = """```python
def wrong_name(scenario_state: dict):
    return 0
```"""


def _initial_messages() -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "generate"},
    ]


def test_retry_passes_on_first_attempt():
    stub = _StubLLMClient([_VALID_OBS])
    inner = InnerLLM(
        stub, evolve_reward=False, evolve_observation=True,
        max_attempts=3,
    )
    cand = inner.generate(_initial_messages())
    assert cand.obs_source is not None
    assert "enhance_observation" in cand.obs_source
    assert cand.attempts == 1
    assert stub.calls == 1


def test_retry_recovers_after_bad_syntax():
    stub = _StubLLMClient([_BAD_SYNTAX, _VALID_OBS])
    inner = InnerLLM(
        stub, evolve_reward=False, evolve_observation=True,
        max_attempts=3,
    )
    cand = inner.generate(_initial_messages())
    assert cand.obs_source is not None
    assert cand.attempts == 2
    assert stub.calls == 2
    # Attempt records should document the failure
    records = cand.attempt_records  # type: ignore[attr-defined]
    assert records[0].error is not None
    assert records[1].error is None


def test_retry_exhausts_after_max_attempts():
    stub = _StubLLMClient([_WRONG_FUNC, _BAD_SYNTAX, _WRONG_FUNC])
    inner = InnerLLM(
        stub, evolve_reward=False, evolve_observation=True,
        max_attempts=3,
    )
    with pytest.raises(CandidateGenerationFailed):
        inner.generate(_initial_messages())
    assert stub.calls == 3


def test_retry_conversation_includes_error_feedback():
    """After a failure, the retry message should include the error."""
    stub = _StubLLMClient([_WRONG_FUNC, _VALID_OBS])
    inner = InnerLLM(
        stub, evolve_reward=False, evolve_observation=True,
        max_attempts=3,
    )

    # Wrap stub.generate to capture the messages passed on retry
    captured = []
    orig = stub.generate

    def _capture(messages, n=1, seed_base=None):
        captured.append(list(messages))
        return orig(messages, n, seed_base)

    stub.generate = _capture  # type: ignore[method-assign]
    cand = inner.generate(_initial_messages())
    assert cand.attempts == 2
    # Second call should have at least one extra user message describing
    # the failure
    assert len(captured[1]) > len(captured[0])
    assert "error" in captured[1][-1]["content"].lower()


def test_per_candidate_seed_derivation():
    """Sibling candidates get distinct seeds via seed_base + i."""
    seeds_seen: List[int] = []

    class _SeedRecorderStub(_StubLLMClient):
        def generate(self, messages, n=1, seed_base=None):
            seeds_seen.append(seed_base)
            return super().generate(messages, n, seed_base)

    stub = _SeedRecorderStub([_VALID_OBS, _VALID_OBS])
    inner = InnerLLM(
        stub, evolve_reward=False, evolve_observation=True,
        max_attempts=3,
    )
    inner.generate(_initial_messages(), seed_base=42)
    inner.generate(_initial_messages(), seed_base=43)
    # Each generate call must see the correct seed
    assert seeds_seen == [42, 43]
