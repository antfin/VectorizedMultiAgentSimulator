"""F9.1 — LLM adapter package.

The LERO orchestrator constructs LLM clients via decorator composition
(see ``application.factories`` for the build path), so most modules
here are wrappers/decorators over the :class:`LlmClient` Protocol
rather than free-standing implementations.

Surface:

- :class:`InMemoryCostLedger` / :class:`FilesystemCostLedger`
- :class:`FakeLlmClient` (canned responses for tests)
- :class:`LiteLlmClient` (the real LiteLLM-backed adapter; lazy-imports
  ``litellm`` so domain tests don't pay the heavy import cost)
- :class:`CostCapDecorator` (rolling-window €/day + €/month enforcer)
- :class:`DiskCacheDecorator` (on-disk response cache, default off)
"""

from multi_scenario.adapters.llm.cost_cap import CostCapDecorator
from multi_scenario.adapters.llm.disk_cache import DiskCacheDecorator
from multi_scenario.adapters.llm.fake_adapter import FakeLlmClient
from multi_scenario.adapters.llm.filesystem_cost_ledger import (
    FilesystemCostLedger,
    InMemoryCostLedger,
)
from multi_scenario.adapters.llm.litellm_adapter import LiteLlmClient


__all__ = [
    "CostCapDecorator",
    "DiskCacheDecorator",
    "FakeLlmClient",
    "FilesystemCostLedger",
    "InMemoryCostLedger",
    "LiteLlmClient",
]
