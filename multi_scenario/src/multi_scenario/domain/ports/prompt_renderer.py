"""F9.2 — :class:`PromptRenderer` Protocol.

Three template kinds per prompt version: ``system``, ``initial_user``,
``feedback``. The composer (F9.6.a) decides which to render at each
iteration; the renderer just substitutes context vars into the chosen
template and returns a string.

Versioning lives in the registry path (``adapters/prompts/<version>/``).
The renderer is version-agnostic — it takes a version string and the
template name and looks them up.
"""

from typing import Protocol


class PromptRenderer(Protocol):
    """Substitute context vars into a versioned prompt template."""

    def render(
        self,
        *,
        version: str,
        template: str,
        context: dict[str, object],
    ) -> str:
        """Render ``<version>/<template>.j2`` with ``context``.

        Args:
            version: prompt registry directory name
                (``"v2_fewshot_k2_local"``, ``"v1_global"``, …).
            template: file stem — ``"system"``, ``"initial_user"``, or
                ``"feedback"``.
            context: substitution dict; values must be JSON-serialisable
                (str / int / float / bool / list / dict / None) so the
                rendered prompt and the trace's ``render_context`` round-trip.

        Returns:
            The rendered prompt as a UTF-8 string.

        Raises:
            FileNotFoundError: when ``<version>/<template>.j2`` doesn't exist.
            jinja2.UndefinedError: when ``context`` is missing a variable
                the template references (StrictUndefined; loud-fail by design).
        """
        ...
