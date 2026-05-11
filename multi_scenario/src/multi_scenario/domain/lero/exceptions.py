"""F9.0 — LERO exception hierarchy.

A single base class (:class:`LeroError`) makes ``except LeroError`` a useful
defensive catch in the orchestrator's fallback paths without swallowing
unrelated errors.
"""


class LeroError(RuntimeError):
    """Base class for every LERO-specific error."""


class LlmCostCapExceeded(LeroError):
    """Raised by the cost-cap decorator when cumulative spend > cap.

    The orchestrator catches this and falls back to re-ranking the
    candidates accumulated so far — better to ship a degraded result
    than to silently overspend.
    """

    def __init__(self, message: str, *, spent_usd: float, cap_usd: float) -> None:
        super().__init__(message)
        #: Cumulative spend at the point the cap was tripped (USD).
        self.spent_usd = spent_usd
        #: The cap that was tripped (USD).
        self.cap_usd = cap_usd


class CandidateGenerationFailed(LeroError):
    """Raised when the LLM returned no parseable / valid candidate code.

    Distinct from a generic LLM failure: the call succeeded, but every
    response failed AST validation (or had no code blocks at all). The
    orchestrator records this on the iteration's trace and continues to
    the next iteration — sometimes the next prompt round produces valid
    code even when the current one didn't.
    """


class FairnessViolation(LeroError, KeyError):
    """Raised by ``AllowedKeysDict`` when LLM code reads a forbidden key.

    Used in CTDE-fair (``local`` observation mode) runs to enforce that
    the LLM only sees keys explicitly whitelisted for the per-agent
    sensor view — preventing it from accidentally fishing for global
    state via dictionary access. The orchestrator treats this as an
    invalid candidate (same as a NaN-action crash) and the fallback
    chain skips to the next-ranked candidate.

    Multiple inheritance from :class:`KeyError` is intentional: LLM
    code commonly uses ``state[key]`` patterns and catches ``KeyError``
    defensively. Subclassing both means ``except FairnessViolation``,
    ``except LeroError``, and ``except KeyError`` all catch the same
    exception, matching rendezvous_comm's semantics so the byte-parity
    test for downstream LLM code paths holds.
    """
