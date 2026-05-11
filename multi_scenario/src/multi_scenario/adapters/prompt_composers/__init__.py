"""F9.6.a — :class:`PromptComposer` adapters.

Two implementations of the :class:`PromptComposer` Protocol:

- :class:`InitialAndFeedbackComposer` (F9.6.a, default) — sends
  ``system.j2`` + ``initial_user.j2`` on iteration 0; sends
  ``system.j2`` + ``initial_user.j2`` + ``feedback.j2`` (with history)
  thereafter.
- :class:`MetaPromptComposer` (F9.7.A stub) — wraps the default and
  injects a placeholder mutation; F9.7.B replaces the wrapper with the
  full Strategist/Editor/Critic round-table.
"""

from multi_scenario.adapters.prompt_composers.initial_and_feedback import (
    InitialAndFeedbackComposer,
)
from multi_scenario.adapters.prompt_composers.meta_prompt import MetaPromptComposer


__all__ = ["InitialAndFeedbackComposer", "MetaPromptComposer"]
