"""F9.1 — :class:`FakeLlmClient` for tests.

In-memory canned-response client. Tests register ``(messages_matcher,
LlmCompletion)`` pairs; ``generate(...)`` returns the first matching
canned response (or raises ``LookupError`` so a misconfigured test
fails loudly rather than hiding behind a default).

Three matcher modes:
- ``exact``: compare full ``messages`` list (most strict).
- ``contains_user``: the matcher's substring appears in the last
  ``user`` message.
- ``always``: matches any call (use for orchestrator end-to-end tests
  where prompt content isn't being asserted).
"""

from dataclasses import dataclass, field
from typing import Any, Callable

from multi_scenario.domain.lero import LlmCompletion


@dataclass
class _Rule:
    matcher: Callable[[list[dict[str, str]]], bool]
    completion: LlmCompletion
    n_sibling: int = 1


@dataclass
class FakeLlmClient:
    """Returns canned :class:`LlmCompletion` objects for matching prompts.

    Use the ``register_*`` helpers to set up rules in tests; ``generate``
    walks rules in registration order and returns the first match's
    completion (replicated to ``n``).

    For test ergonomics every method returns ``self`` so registration
    can chain: ``client.register_always(comp).register_exact(...)``.
    """

    _rules: list[_Rule] = field(default_factory=list)
    #: Records every call so tests can assert prompt shape after the fact.
    calls: list[dict[str, Any]] = field(default_factory=list)

    def register_always(self, completion: LlmCompletion) -> "FakeLlmClient":
        """Match any call — useful when prompt content isn't under test."""
        self._rules.append(_Rule(matcher=lambda _msgs: True, completion=completion))
        return self

    def register_exact(
        self, messages: list[dict[str, str]], completion: LlmCompletion
    ) -> "FakeLlmClient":
        """Match calls whose ``messages`` list equals ``messages``."""
        self._rules.append(
            _Rule(
                matcher=lambda msgs, expected=messages: msgs == expected,
                completion=completion,
            )
        )
        return self

    def register_contains_user(
        self, substring: str, completion: LlmCompletion
    ) -> "FakeLlmClient":
        """Match when the last user message contains ``substring``."""

        def _match(msgs: list[dict[str, str]], _sub: str = substring) -> bool:
            user_msgs = [m for m in msgs if m.get("role") == "user"]
            return bool(user_msgs) and _sub in user_msgs[-1].get("content", "")

        self._rules.append(_Rule(matcher=_match, completion=completion))
        return self

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        n: int = 1,
        seed: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> list[LlmCompletion]:
        self.calls.append(
            {
                "messages": messages,
                "n": n,
                "seed": seed,
                "response_format": response_format,
            }
        )
        for rule in self._rules:
            if rule.matcher(messages):
                return [rule.completion for _ in range(n)]
        raise LookupError(
            "FakeLlmClient: no rule matched. Register one with "
            "register_always / register_exact / register_contains_user "
            "before calling generate(). messages="
            f"{messages!r}"
        )
