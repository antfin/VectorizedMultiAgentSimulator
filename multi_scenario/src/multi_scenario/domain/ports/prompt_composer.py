"""F9.6.a — :class:`PromptComposer` Protocol.

The composer is the seam between the orchestrator and the prompt
strategy. The default :class:`InitialAndFeedbackComposer` (F9.6.a) is
what F8.4's S3b-local replication uses; the F9.7.A
:class:`MetaPromptComposer` stub plugs in for the meta-prompt
experiment without orchestrator changes; the F9.7.B full Strategist /
Editor / Critic round-table replaces the stub when meta-prompting
lands post-extraction.

Returns a :class:`ComposedPrompt` carrying the message list plus
provenance (which prompt version, which signal tier, what context dict
was rendered) so the orchestrator can trace the call without coupling
to the composer's internals.
"""

from dataclasses import dataclass, field
from typing import Protocol

from multi_scenario.domain.lero import CandidateResult, SignalTier, StrategyCard


@dataclass(frozen=True)
class ComposedPrompt:
    """Output of :meth:`PromptComposer.compose`.

    The orchestrator forwards ``messages`` to :class:`LlmClient.generate`
    and writes the remaining fields into the per-call
    :class:`PromptTrace`. Frozen so accidentally mutating after compose
    is a hard error (would otherwise desync messages from the trace).
    """

    #: OpenAI-shape chat messages, ready to ship to
    #: :class:`LlmClient.generate`.
    messages: list[dict[str, str]]
    #: Prompt registry version (``"v2_fewshot_k2_local"`` etc.) so the
    #: trace records what template family produced these messages.
    prompt_version: str
    #: The substitution context passed to the renderer. Must be
    #: JSON-serialisable (str / int / float / bool / list / dict / None)
    #: so trace persistence (F9.3) round-trips cleanly.
    render_context: dict[str, object] = field(default_factory=dict)
    #: Behavioral feedback tier this composer wanted (F9.7.B noise control).
    #: Default ``"scalar"`` because the F9.6.a composer just sends M1/M2/M3
    #: in the feedback; F9.7.B's MetaPromptComposer escalates as needed.
    signal_tier: SignalTier = "scalar"


class PromptComposer(Protocol):
    """Compose a chat-message list for the current LERO iteration."""

    def compose(
        self,
        *,
        iteration: int,
        history: list[CandidateResult],
        task_params: dict[str, object],
        strategy_card: StrategyCard | None = None,
    ) -> ComposedPrompt:
        """Build the messages for this iteration's LLM call.

        Args:
            iteration: 0-indexed iteration number. The default composer
                uses ``iteration == 0`` to mean "send the initial prompt"
                and ``iteration > 0`` to mean "send the feedback prompt
                with the accumulated history".
            history: every :class:`CandidateResult` seen so far across
                all iterations (in registration order). The composer
                ranks / formats it into the feedback prompt.
            task_params: scenario-level parameters that show up as
                substitution vars in the prompt template (``n_agents``,
                ``n_targets``, ``covering_range``, …). The orchestrator
                derives this from the YAML's ``scenario.params``.
            strategy_card: optional Strategist output (F9.7.B). The
                default composer ignores it; ``MetaPromptComposer``
                threads its slot edits into the message list.

        Returns:
            A :class:`ComposedPrompt` carrying both the OpenAI-shape
            messages and the provenance needed for the trace record.
        """
        ...
