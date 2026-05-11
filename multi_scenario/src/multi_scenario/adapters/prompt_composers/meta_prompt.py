"""F9.7.A — :class:`MetaPromptComposer` stub for the meta-prompt seam.

This is **the seam**, not the full Strategist/Editor/Critic round-table
(which is deferred to F9.7.B per the locked decision in the plan: full
meta-prompting lands post-extraction + post-experiments). The stub
exists so that:

1. The orchestrator's contract (``PromptComposer`` Protocol) is
   exercised end-to-end with a non-default composer — proving
   ``cfg.lero.meta_prompting=true`` works mechanically before the real
   meta logic arrives.
2. When F9.7.B replaces this stub, the orchestrator + factory wiring
   doesn't change — only the body of :meth:`compose`.

What the stub does: delegate to :class:`InitialAndFeedbackComposer` and
append a placeholder ``"\\n\\n[meta-prompt placeholder]"`` to the last
user message. That single character mutation is enough to prove the
plumbing — the F9.7.B real composer will replace this with Strategist /
Editor / Critic-driven slot edits.
"""

from typing import Any

from multi_scenario.adapters.prompt_composers.initial_and_feedback import (
    InitialAndFeedbackComposer,
)
from multi_scenario.domain.lero import CandidateResult, StrategyCard
from multi_scenario.domain.ports import ComposedPrompt, PromptRenderer


_PLACEHOLDER_MARKER = "\n\n[meta-prompt placeholder]"


class MetaPromptComposer:
    """Stub meta-composer — wraps the default + injects a placeholder.

    The :class:`PromptComposer` Protocol is satisfied structurally —
    the orchestrator doesn't know whether it's talking to the default
    or the meta version. F9.7.B will replace the wrapping logic with
    a Strategist/Editor/Critic round-table that consumes
    ``strategy_card`` to mutate slot text; the placeholder is the
    minimal proof that the seam holds.
    """

    def __init__(
        self,
        *,
        renderer: PromptRenderer,
        prompt_version: str,
        n_candidates: int,
        top_k_with_code: int = 3,
    ) -> None:
        self._inner = InitialAndFeedbackComposer(
            renderer=renderer,
            prompt_version=prompt_version,
            n_candidates=n_candidates,
            top_k_with_code=top_k_with_code,
        )

    def compose(
        self,
        *,
        iteration: int,
        history: list[CandidateResult],
        task_params: dict[str, Any],
        strategy_card: StrategyCard | None = None,
    ) -> ComposedPrompt:
        base = self._inner.compose(
            iteration=iteration,
            history=history,
            task_params=task_params,
            strategy_card=strategy_card,
        )
        # Mutate the last user message — that's where F9.7.B's real
        # composer will inject Editor slot edits. Today: trivial marker.
        last = base.messages[-1]
        mutated_messages = [
            *base.messages[:-1],
            {**last, "content": last["content"] + _PLACEHOLDER_MARKER},
        ]
        return ComposedPrompt(
            messages=mutated_messages,
            prompt_version=base.prompt_version,
            render_context={**base.render_context, "meta_placeholder": True},
            signal_tier=base.signal_tier,
        )
