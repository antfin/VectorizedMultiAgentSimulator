"""F9.1 — :class:`LlmClient` Protocol.

The orchestrator only ever talks to this Protocol. Real LiteLLM, the
fake test client, the cost-cap decorator, and the disk cache are all
:class:`LlmClient` implementations — composition, not inheritance.

Surface kept narrow on purpose:

- ``generate(messages, n, seed, response_format=None) -> list[LlmCompletion]``

Anything LiteLLM-specific (streaming, retry policy, structured outputs
JSON schema) lives behind this — the orchestrator never imports
``litellm``.

Decorator composition order (outermost first):
    CostCapDecorator → DiskCacheDecorator → LiteLlmClient

The cap sees every uncached call's cost (cache hits are free); the
cache sees every request the cap allowed through.
"""

from typing import Any, Protocol

from multi_scenario.domain.lero import LlmCompletion


class LlmClient(Protocol):
    """Narrow port the orchestrator uses to ask the LLM for completions."""

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        n: int = 1,
        seed: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> list[LlmCompletion]:
        """Return ``n`` completions for ``messages``.

        Args:
            messages: OpenAI-shape chat messages —
                ``[{"role": "system" | "user" | "assistant", "content": "..."}]``.
            n: How many sibling completions to request. Decorators may
                fan out a single ``n=k`` call into ``k`` ``n=1`` calls
                (e.g., to derive distinct seeds per sibling) — the
                contract is "callers get a list of ``n`` completions",
                not "exactly one underlying API call".
            seed: Optional reproducibility seed (the integer OpenAI's
                ``seed=`` field accepts). When ``None``, the underlying
                provider chooses.
            response_format: Optional structured-output spec —
                LiteLLM-shape ``{"type": "json_schema", ...}``. ``None``
                = free-form text completion. Used today only by the
                future F9.7.B Strategist/Editor/Critic round-table; the
                inner code generator works with free-form text.

        Returns:
            Exactly ``n`` :class:`LlmCompletion` objects.

        Raises:
            LlmCostCapExceeded: when the cost-cap decorator detects the
                rolling-window budget would be exceeded.
        """
        ...
