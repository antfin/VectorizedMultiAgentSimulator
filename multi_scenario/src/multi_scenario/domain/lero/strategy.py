"""F9.6.a / F9.7 — :class:`StrategyCard` (stub for meta-prompt seam).

The Strategist (one of three roles in F9.7.B's full meta-prompt
round-table) emits a StrategyCard per meta-iteration describing what
sub-slot of the inner prompt to edit and with what focus. F9.6.a's
:class:`PromptComposer` Protocol accepts this card as an optional input;
the default :class:`InitialAndFeedbackComposer` ignores it; a future
:class:`MetaPromptComposer` consumes it.

The full schema (target_domain / target_slot / focus / avoid /
confidence / include_signals / rationale) is documented in the plan's
F9.7.B notes. We ship a **minimal** Pydantic shape now so the Protocol
signature has a real type to reference today; F9.7.B widens the model
when the round-table lands.
"""

from typing import Literal

from pydantic import BaseModel, Field

from multi_scenario.domain.models._common import STRICT


#: Behavioral feedback tier — controls how much information feeds back
#: from the inner loop to the Strategist. Default ``scalar`` keeps the
#: signal narrow until evidence says otherwise (F9.7.B's noise-control
#: knob). Listed here so the Literal exists today; the consumer
#: doesn't ship until F9.7.B.
SignalTier = Literal["scalar", "fingerprint", "curve_shape"]


class StrategyCard(BaseModel):
    """Stub for the F9.7.B meta-prompt Strategist's output.

    Today this carries only the rationale text — enough for the
    F9.7.A stub composer to thread something non-empty through the
    Protocol. F9.7.B will expand the model to match the
    rendezvous_comm ``schemas.StrategyCard`` (target_domain,
    target_slot, focus[], avoid[], confidence, include_signals[]).
    """

    model_config = STRICT

    #: 2-4 sentence rationale citing specific evidence (record name,
    #: verdict, feature delta). Empty string when the composer doesn't
    #: care (the F9.7.A stub uses this code path).
    rationale: str = Field(default="")
