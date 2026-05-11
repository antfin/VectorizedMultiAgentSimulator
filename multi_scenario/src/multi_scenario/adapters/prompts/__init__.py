"""F9.2 — Jinja-based prompt registry (ported from rendezvous_comm).

The :class:`JinjaPromptRenderer` adapter resolves prompts from
``<adapter_root>/<version>/{system,initial_user,feedback}.j2`` and
substitutes context vars via Jinja2.

Prompt-text content is ported from ``rendezvous_comm/src/lero/prompts/<v>/*.txt``
with the ``$variable`` → ``{{ variable }}`` syntax migration applied;
the byte-parity test
(``tests/integration/prompts/test_prompt_byte_parity.py``) renders the
Jinja templates and the rendezvous_comm ``string.Template`` source
against the same context and asserts the two byte sequences are equal.
"""

from multi_scenario.adapters.prompts.jinja_renderer import (
    DEFAULT_PROMPTS_ROOT,
    JinjaPromptRenderer,
)


__all__ = ["DEFAULT_PROMPTS_ROOT", "JinjaPromptRenderer"]
