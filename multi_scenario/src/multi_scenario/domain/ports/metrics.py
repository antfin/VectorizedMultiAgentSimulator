"""MetricsBundle port — single always-on bundle that produces M1-M9.

The Protocol receives the rollout from ``Algorithm.evaluate`` and the
``Scenario`` adapter (used for the four DI primitives that feed scenario-
specific metric calculations). Returns the M1–M9 dict directly; values are
``float`` or ``None`` (= not applicable for this run, see §3.5.3).
"""

from typing import Any, Protocol, runtime_checkable

from .scenario import Scenario


@runtime_checkable
class MetricsBundle(Protocol):
    """Domain port for the always-on M1-M9 metric bundle."""

    # Single-method Protocols are intentional in this architecture; pylint
    # flags them as candidates-for-functions, which doesn't apply to
    # structurally-typed ports.
    # pylint: disable=too-few-public-methods

    def compute(self, rollout: Any, scenario: Scenario) -> dict[str, float | None]:
        """Compute the M1-M9 dict; values are float or None (=N/A)."""
