"""Standalone InnerLLM callable with retry loop (LERO-MP v3 §3.1+§3.3).

The inner LLM is the one that writes the actual PyTorch reward /
observation code that the RL policy trains on. In v2.x this was a
plain ``LLMClient.generate(...)`` + regex parsing in
``codegen.extract_candidates``. v3 wraps that into a standalone
callable with:

  - Retry loop (max 3 attempts) that feeds compile/validation errors
    back to the LLM so it can fix itself.
  - Pydantic ``InnerLLMOutput`` as the typed return shape.
  - Per-candidate seed derivation for reproducibility (§5.2).
  - A free-text fallback parser that reuses the existing regex
    extractor so v3 runs on providers with partial structured-output
    support without flipping a feature flag.

v4 migration: subclass ``dspy.Module``, rename ``generate`` →
``forward``, keep the Pydantic schema as the declared signature.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .codegen import (
    _CODE_BLOCK_RE,
    ALLOWED_IMPORTS,
    CandidateCode,
    validate_function,
)
from .llm_client import LLMClient
from .schemas import InnerLLMOutput

_log = logging.getLogger("rendezvous.lero.inner")


MAX_GEN_ATTEMPTS = 3


class CandidateGenerationFailed(RuntimeError):
    """Raised when the inner LLM exhausts ``MAX_GEN_ATTEMPTS`` without
    producing code that passes AST + signature validation.
    """


@dataclass
class CandidateAttempt:
    """Record of a single retry attempt (for provenance)."""

    attempt_index: int  # 1-based
    error: Optional[str]  # None on success
    response: str


def _parse_free_text_to_inner(
    text: str,
    evolve_reward: bool,
    evolve_observation: bool,
) -> InnerLLMOutput:
    """Regex-based parser used as the structured-outputs fallback.

    Mirrors the behavior of ``codegen.extract_candidates`` for a single
    response but returns the typed schema instead of a CandidateCode.
    """
    if not text:
        raise ValueError("empty LLM response")
    blocks = _CODE_BLOCK_RE.findall(text)
    if not blocks:
        raise ValueError("no ```python block in LLM response")
    reward_src: Optional[str] = None
    obs_src: Optional[str] = None
    for block in blocks:
        block = block.strip()
        if ("def compute_reward" in block) and evolve_reward:
            func_name = (
                "compute_reward_bonus"
                if "def compute_reward_bonus" in block
                else "compute_reward"
            )
            if validate_function(block, func_name, ["scenario_state"]):
                reward_src = block
        elif "def enhance_observation" in block and evolve_observation:
            if validate_function(
                block, "enhance_observation", ["scenario_state"],
            ):
                obs_src = block
    # Enforce the requested signatures: if the template asked for a
    # function and we couldn't extract it, surface a parse error so
    # the retry loop can ask the LLM to fix it.
    if evolve_observation and obs_src is None:
        raise ValueError(
            "enhance_observation(scenario_state) function missing or "
            "failed AST/signature validation."
        )
    if evolve_reward and reward_src is None:
        raise ValueError(
            "compute_reward(scenario_state) function missing or "
            "failed AST/signature validation."
        )
    # ``rationale`` is free-text only in the structured-output path;
    # pull best-effort from a leading paragraph if present.
    rationale = None
    first_para = text.split("```")[0].strip()
    if first_para and len(first_para) < 600:
        rationale = first_para
    return InnerLLMOutput(
        obs_code=obs_src,
        reward_code=reward_src,
        rationale=rationale,
    )


class InnerLLM:
    """Standalone inner-LLM callable. Wraps an ``LLMClient`` with retry."""

    def __init__(
        self,
        llm_client: LLMClient,
        evolve_reward: bool = True,
        evolve_observation: bool = True,
        max_attempts: int = MAX_GEN_ATTEMPTS,
        use_structured: bool = False,
    ):
        self.llm = llm_client
        self.evolve_reward = evolve_reward
        self.evolve_observation = evolve_observation
        self.max_attempts = max_attempts
        self.use_structured = use_structured

    def generate(
        self,
        messages: List[Dict[str, str]],
        seed_base: Optional[int] = None,
    ) -> "CandidateCode":
        """Generate one validated candidate with retries.

        Returns a ``CandidateCode`` with an ``attempts`` attribute
        recording how many retries were required.
        """
        convo = [dict(m) for m in messages]  # copy so we can mutate
        attempts: List[CandidateAttempt] = []
        last_err: Optional[Exception] = None

        for i in range(self.max_attempts):
            call_seed = (
                (seed_base + i) % (2 ** 31) if seed_base is not None else None
            )
            try:
                if self.use_structured:
                    out = self.llm.generate_structured(
                        convo,
                        schema=InnerLLMOutput,
                        seed=call_seed,
                        fallback_parser=lambda t: _parse_free_text_to_inner(
                            t, self.evolve_reward, self.evolve_observation,
                        ),
                    )
                    raw = ""  # structured path doesn't expose raw
                else:
                    responses = self.llm.generate(
                        convo, n=1,
                        seed_base=call_seed,
                    )
                    raw = responses[0]
                    out = _parse_free_text_to_inner(
                        raw,
                        self.evolve_reward,
                        self.evolve_observation,
                    )
                attempts.append(
                    CandidateAttempt(
                        attempt_index=i + 1, error=None, response=raw,
                    )
                )
                cand = CandidateCode(
                    reward_source=out.reward_code,
                    obs_source=out.obs_code,
                    raw_response=raw or _serialize_inner(out),
                )
                # Tag the candidate with retry metadata (monkey-patched;
                # dataclass doesn't have slots).
                cand.attempts = i + 1  # type: ignore[attr-defined]
                cand.attempt_records = attempts  # type: ignore[attr-defined]
                return cand
            except Exception as e:
                last_err = e
                err_msg = f"{type(e).__name__}: {e}"
                attempts.append(
                    CandidateAttempt(
                        attempt_index=i + 1,
                        error=err_msg,
                        response="",
                    )
                )
                _log.warning(
                    "InnerLLM attempt %d/%d failed: %s",
                    i + 1, self.max_attempts, err_msg[:200],
                )
                if i < self.max_attempts - 1:
                    # Append a retry message so the LLM can see its
                    # own mistake. We keep the system + initial-user
                    # intact so the task context doesn't drift.
                    convo.append({
                        "role": "user",
                        "content": (
                            f"Your previous response could not be used.\n"
                            f"Error: {err_msg}\n"
                            f"Please return a corrected version that "
                            f"addresses this error. Return ONLY the "
                            f"Python function(s) in a ```python code "
                            f"block; no prose around the code."
                        ),
                    })
        raise CandidateGenerationFailed(
            f"Max retries ({self.max_attempts}) exceeded. "
            f"Last error: {last_err}"
        )


def _serialize_inner(out: InnerLLMOutput) -> str:
    """Reassemble a human-readable response for provenance storage.

    Used when the structured-output path succeeds and we no longer
    have a raw string. Format mirrors what the regex parser expects
    so downstream code that re-parses the raw file still works.
    """
    parts: List[str] = []
    if out.rationale:
        parts.append(out.rationale)
        parts.append("")
    if out.reward_code:
        parts.append("```python")
        parts.append(out.reward_code)
        parts.append("```")
        parts.append("")
    if out.obs_code:
        parts.append("```python")
        parts.append(out.obs_code)
        parts.append("```")
    return "\n".join(parts)
