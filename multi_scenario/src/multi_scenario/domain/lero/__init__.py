"""F9.0 — domain types for the LERO evolutionary loop.

Hex-clean: this package imports from stdlib + pydantic only. No torch,
no LiteLLM, no BenchMARL, no VMAS. Adapters wrap these types when they
cross the I/O boundary.

Surface (curated re-exports):

- :class:`Candidate` — one (reward_source, obs_source) pair from the LLM.
- :class:`CandidateMetrics` — M1–M9 + LERO-specific eval scalars after
  short training.
- :class:`CandidateResult` — Candidate + Metrics + verdict, the unit the
  composer's feedback iterates over.
- :class:`PromptTrace` / :class:`ResponseTrace` / :class:`ReasoningTrace` —
  per-call provenance written under ``output/lero/iter_<n>/cand_<m>/``.
- :class:`LlmCompletion` — pure model output (text + token counts +
  reasoning); separate from our trace metadata so the LlmClient port is
  agnostic of where the result is recorded.
- :class:`LeroRunSummary` — final aggregate (best candidate + fallback
  chain + cost totals). Written to ``output/lero/final_summary.json``.

Exception hierarchy (all subclass :class:`LeroError`):

- :class:`LlmCostCapExceeded` — cost cap hit; orchestrator catches and
  falls back to "use whatever candidates we already have".
- :class:`CandidateGenerationFailed` — LLM returned no parseable code
  blocks (or all candidates failed AST validation).
- :class:`FairnessViolation` — patched scenario tried to read a key
  outside the per-mode whitelist.
"""

from multi_scenario.domain.lero.candidate import (
    Candidate,
    CandidateCode,
    CandidateMetrics,
    CandidateResult,
    Verdict,
)
from multi_scenario.domain.lero.codegen import (
    ALLOWED_IMPORTS,
    extract_candidates,
    validate_function,
)
from multi_scenario.domain.lero.exceptions import (
    CandidateGenerationFailed,
    FairnessViolation,
    LeroError,
    LlmCostCapExceeded,
)
from multi_scenario.domain.lero.llm_completion import LlmCompletion, LlmUsage
from multi_scenario.domain.lero.strategy import SignalTier, StrategyCard
from multi_scenario.domain.lero.summary import FallbackEntry, LeroRunSummary
from multi_scenario.domain.lero.traces import PromptTrace, ReasoningTrace, ResponseTrace
from multi_scenario.domain.lero.whitelist import (
    AllowedKeysDict,
    LOCAL_ALLOWED_KEYS,
    LOCAL_FORBIDDEN_KEYS,
)


__all__ = [
    "ALLOWED_IMPORTS",
    "AllowedKeysDict",
    "Candidate",
    "CandidateCode",
    "CandidateGenerationFailed",
    "CandidateMetrics",
    "CandidateResult",
    "extract_candidates",
    "LOCAL_ALLOWED_KEYS",
    "LOCAL_FORBIDDEN_KEYS",
    "FairnessViolation",
    "FallbackEntry",
    "LeroError",
    "LeroRunSummary",
    "LlmCompletion",
    "LlmCostCapExceeded",
    "LlmUsage",
    "PromptTrace",
    "ReasoningTrace",
    "ResponseTrace",
    "SignalTier",
    "StrategyCard",
    "validate_function",
    "Verdict",
]
