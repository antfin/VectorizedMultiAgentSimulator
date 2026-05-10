"""F9.0 — Per-call trace records written by :class:`TraceWriter`.

Layout (per F9.3) under ``<run_dir>/output/lero/iter_<n>/cand_<m>/attempt_<a>/``:

- ``prompt.json``    → :class:`PromptTrace`
- ``response.json``  → :class:`ResponseTrace`
- ``reasoning.json`` → :class:`ReasoningTrace`

Trace records carry orchestration-level provenance (which composer fired
this call, what role the call played, what context the prompt was rendered
with) — distinct from :class:`LlmCompletion`, which is the bare model
output. The split lets the LlmClient port stay agnostic of where its
results land.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from multi_scenario.domain.lero.llm_completion import LlmUsage
from multi_scenario.domain.models._common import STRICT


#: Which LLM-role produced this prompt/response pair. The inner code
#: generator is the only role today; the F9.7.B Strategist / Editor /
#: Critic add three more values here when meta-prompting lands. Listed
#: now so existing trace files don't need a schema migration later.
TraceRole = Literal[
    "inner_codegen",
    "meta_strategist",
    "meta_editor",
    "meta_critic",
]


class PromptTrace(BaseModel):
    """Snapshot of one outbound LLM call's prompt + render context."""

    model_config = STRICT

    role: TraceRole = "inner_codegen"
    #: Prompt registry version (``v2_fewshot_k2_local`` etc.). Lets
    #: post-hoc analysis filter by which prompt produced which result.
    prompt_version: str
    #: The rendered messages list as it would be sent to the LLM —
    #: list of ``{"role": "system" | "user" | "assistant", "content": "..."}``
    #: dicts. We keep the OpenAI shape rather than a custom type so the
    #: trace round-trips cleanly back into a ``LlmClient.generate(...)``
    #: call (e.g., for a "replay this prompt" debug command).
    messages: list[dict[str, str]]
    #: The raw context dict the renderer used. Pickled-safe — only str /
    #: int / float / bool / None / list / dict values. Streamlit's Run
    #: Detail page reads this to reconstruct the prompt with substitutions
    #: highlighted.
    render_context: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ResponseTrace(BaseModel):
    """Snapshot of one inbound LLM response."""

    model_config = STRICT

    role: TraceRole = "inner_codegen"
    #: Verbatim model text (no truncation). Pair this with PromptTrace at
    #: the same ``attempt_<a>`` to see exactly what the LLM was asked and
    #: replied.
    text: str
    finish_reason: str | None = None
    system_fingerprint: str | None = None
    usage: LlmUsage = Field(default_factory=LlmUsage)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ReasoningTrace(BaseModel):
    """Provider-separated reasoning trace, when the model emits one.

    Stored separately from ResponseTrace so the visible / billed text
    stays clean. Anthropic's ``thinking`` blocks and OpenAI's
    ``reasoning`` field both land here.
    """

    model_config = STRICT

    role: TraceRole = "inner_codegen"
    text: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
