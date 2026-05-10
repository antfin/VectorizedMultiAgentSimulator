"""F9.1 — :class:`CostLedger` Protocol — persistent rolling-window spend tracker.

Designed for **host-wide rolling windows** (€10/day, €100/month). The
ledger persists across processes / sweeps / days so a crashed run that
spent €4 doesn't reset the budget on retry — accidental overspend is
structurally impossible without flipping the cap fields explicitly.

Two implementations:

- ``FilesystemCostLedger`` (production): JSONL append-on-record at
  ``~/.multi_scenario/cost_ledger.jsonl``, prune-on-read for entries
  older than 31 days.
- ``InMemoryCostLedger`` (tests): pure in-memory list, deterministic.
"""

from datetime import timedelta
from typing import Protocol


class CostLedger(Protocol):
    """Append-only spend record with rolling-window queries."""

    def record(
        self,
        *,
        cost_eur: float,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Append one entry to the ledger. Timestamp = now (UTC)."""
        ...

    def sum_window(self, window: timedelta) -> float:
        """Total EUR spend within the last ``window`` (rolling, ending now)."""
        ...
