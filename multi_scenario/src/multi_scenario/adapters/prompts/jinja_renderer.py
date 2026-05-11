"""F9.2 — :class:`JinjaPromptRenderer` — Jinja2 prompt-template renderer.

Why Jinja (not :mod:`string.Template` like rendezvous_comm did): future
meta-prompt composition (F9.7.B) needs loops / conditionals / filters.
The migration is mechanical — every ``$variable`` in the rendezvous_comm
templates becomes ``{{ variable }}`` in our ``.j2`` files.

Byte-parity contract: for the prompts F8.4 needs to reproduce
(``v1``, ``v1_global``, ``v2``, ``v2_min``, ``v2_fewshot``, ``v2_twofn``,
``v2_fewshot_k2_local``), our Jinja-rendered output is byte-identical to
the rendezvous_comm ``string.Template.safe_substitute`` output for the
same context. Pinned by
:mod:`tests.integration.prompts.test_prompt_byte_parity`. If that test
ever drifts, F8.4's S3b-local replication will silently change too —
that's the failure mode it catches early.

Configuration choices that preserve byte-parity:

- ``StrictUndefined``: fail loudly on missing variables instead of
  rendering empty (matches rendezvous_comm's "every var supplied" call
  contract; safe_substitute would leave ``$missing`` literal in the
  output, which we reject — F9.2 callers always provide the full context).
- ``keep_trailing_newline=True``: Jinja's default strips trailing newlines.
  Off matches the verbatim file contents.
- No ``trim_blocks`` / ``lstrip_blocks``: the ported prompts have no
  Jinja control blocks (only ``{{ var }}``), so block-trimming would
  be a no-op anyway — but turning them off explicitly documents the
  intent and avoids surprise if a future prompt adds ``{% if %}``.
- ``autoescape=False``: prompts are plain text, not HTML.
"""

from pathlib import Path

import jinja2

from multi_scenario.domain.ports import (  # noqa: F401  (Protocol satisfied)
    PromptRenderer,
)


#: Module-level default. The renderer uses this when the caller doesn't
#: pass an explicit ``root``. Can be overridden by setting
#: ``MULTI_SCENARIO_PROMPTS_ROOT`` for tests / experiments.
DEFAULT_PROMPTS_ROOT = Path(__file__).parent


class JinjaPromptRenderer:
    """Implements :class:`PromptRenderer` using Jinja2.

    The Jinja environment is created once per renderer instance with a
    :class:`FileSystemLoader` rooted at ``root`` so each prompt version
    sits in its own subdirectory: ``<root>/<version>/<template>.j2``.
    """

    def __init__(self, root: Path | None = None) -> None:
        self._root = (root or DEFAULT_PROMPTS_ROOT).resolve()
        self._env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self._root)),
            undefined=jinja2.StrictUndefined,
            autoescape=False,
            keep_trailing_newline=True,
            trim_blocks=False,
            lstrip_blocks=False,
        )

    def render(
        self,
        *,
        version: str,
        template: str,
        context: dict[str, object],
    ) -> str:
        """Render ``<version>/<template>.j2`` with ``context``."""
        rel = f"{version}/{template}.j2"
        try:
            tmpl = self._env.get_template(rel)
        except jinja2.TemplateNotFound as exc:
            raise FileNotFoundError(
                f"prompt template not found: {self._root / rel}"
            ) from exc
        return tmpl.render(**context)
