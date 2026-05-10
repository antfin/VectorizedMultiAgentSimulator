"""F9.1 — :class:`CostCapDecorator` over :class:`LlmClient`.

Enforces **rolling-window** cost caps (€10/day + €100/month by default)
against a persistent :class:`CostLedger`. The two windows are checked
*independently* — exceeding either trips the cap.

Composes cleanly with the cache decorator: outermost-first, the
intended composition is::

    CostCapDecorator(
        DiskCacheDecorator(
            LiteLlmClient(cfg),
        ),
        ledger=FilesystemCostLedger(),
        cfg_llm=cfg.llm,
    )

— so cache HITS are free (no cost ledger entry); cache MISSES bubble
through to LiteLLM, which writes the cost back into the LlmCompletion
``usage.estimated_cost_usd``, and the decorator records that against
the ledger before returning. A miss + write-back also populates the
cache so a re-run is free.

Why pre-flight check the cap (instead of charging-then-raising):
because LLM calls cost real money. If we let the call go through, then
discover the budget was already exhausted, we've spent money the user
asked us not to spend. Pre-flight rejects without contacting the API.
"""

import logging
from datetime import timedelta
from typing import Any

from multi_scenario.domain.lero import LlmCompletion, LlmCostCapExceeded
from multi_scenario.domain.models import LlmSection
from multi_scenario.domain.ports import CostLedger, LlmClient


_log = logging.getLogger(__name__)

#: Window definitions for the two caps. Stored as constants so unit
#: tests can reference the same values the production decorator uses.
_DAY_WINDOW = timedelta(days=1)
_MONTH_WINDOW = timedelta(days=30)


class CostCapDecorator:
    """Wraps any :class:`LlmClient` with rolling-window EUR cost caps."""

    def __init__(
        self,
        inner: LlmClient,
        *,
        ledger: CostLedger,
        cfg_llm: LlmSection,
    ) -> None:
        self._inner = inner
        self._ledger = ledger
        self._cfg = cfg_llm

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        n: int = 1,
        seed: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> list[LlmCompletion]:
        """Pre-flight cap check, then delegate to the inner client.

        We can't predict the call's exact cost in advance, but we *can*
        refuse to start a call when the existing rolling-window total
        already exceeds the cap. This catches the common case (a sweep
        that has spent €11 today on €10/day cap) without burning more
        budget. The post-call ledger record lets the next call see the
        new total.
        """
        self._raise_if_window_exceeded(
            _DAY_WINDOW, self._cfg.cost_cap_per_day_eur, "day"
        )
        self._raise_if_window_exceeded(
            _MONTH_WINDOW, self._cfg.cost_cap_per_month_eur, "month"
        )
        completions = self._inner.generate(
            messages=messages, n=n, seed=seed, response_format=response_format
        )
        for completion in completions:
            cost_eur = completion.usage.estimated_cost_usd * self._cfg.usd_to_eur_rate
            self._ledger.record(
                cost_eur=cost_eur,
                model=self._cfg.model,
                prompt_tokens=completion.usage.prompt_tokens,
                completion_tokens=completion.usage.completion_tokens,
            )
        return completions

    def _raise_if_window_exceeded(
        self, window: timedelta, cap_eur: float, window_label: str
    ) -> None:
        spent = self._ledger.sum_window(window)
        if spent >= cap_eur:
            msg = (
                f"cost cap reached: €{spent:.2f} ≥ €{cap_eur:.2f} "
                f"(window={window_label})"
            )
            _log.warning(
                msg,
                extra={
                    "spent_eur": spent,
                    "cap_eur": cap_eur,
                    "window": window_label,
                },
            )
            raise LlmCostCapExceeded(msg, spent_usd=spent, cap_usd=cap_eur)
