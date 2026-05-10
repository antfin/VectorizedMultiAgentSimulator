"""F9.0 — LlmCompletion: the model-only output from a single LLM call.

Deliberately separate from :class:`PromptTrace` / :class:`ResponseTrace`
(those carry our orchestration metadata — when, why, with what context).
Splitting them keeps the :class:`LlmClient` Protocol agnostic of where
the result is recorded: a fake client for tests can construct
``LlmCompletion`` directly without touching the trace writer.

Field choices ported from rendezvous_comm's ``LLMClient.generate`` return
type (``llm_client.py``) — usage / fingerprint / reasoning all matter
for reproducibility and post-hoc analysis.
"""

from pydantic import BaseModel, Field

from multi_scenario.domain.models._common import STRICT


class LlmUsage(BaseModel):
    """Token counts + cost estimate for one LLM call."""

    model_config = STRICT

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    #: Estimated USD cost for this call. LiteLLM exposes per-model price
    #: tables; the adapter computes this so callers don't have to know
    #: pricing. ``0.0`` for FakeLlmClient and offline cache hits.
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)


class LlmCompletion(BaseModel):
    """One completed LLM call's output.

    Multi-completion calls (e.g., ``generate(messages, n=3)``) return a
    list of these — one per sibling — so a downstream codegen step can
    iterate without caring about how the API delivered them.
    """

    model_config = STRICT

    #: The actual generated text. For LERO this contains
    #: ``\`\`\`python ... \`\`\``` blocks the codegen extracts.
    text: str
    #: Optional separate reasoning trace (Anthropic / OpenAI reasoning
    #: models surface this as a distinct field). ``None`` when the
    #: provider doesn't separate reasoning from the response.
    reasoning: str | None = None
    #: ``finish_reason`` string from the provider — typically ``"stop"``,
    #: ``"length"``, or provider-specific values. Useful in trace
    #: analysis ("did the LLM hit max_tokens?") but never load-bearing.
    finish_reason: str | None = None
    #: Provider-reported fingerprint (OpenAI's ``system_fingerprint``).
    #: Pinned in the run's ``llm_provenance.json`` so we can reproduce
    #: results across model-version drift.
    system_fingerprint: str | None = None
    usage: LlmUsage = Field(default_factory=LlmUsage)
