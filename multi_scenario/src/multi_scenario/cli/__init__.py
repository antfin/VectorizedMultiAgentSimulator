"""multi_scenario CLI — Typer multi-command app.

The package is split per-command for readability; each module attaches its own
``@app.command()`` to the shared :data:`app` defined in :mod:`._app`. Importing
the modules below at the bottom of this file is what triggers the registration
— deletion of one of those imports is what removes a subcommand.

Public surface:

- ``app`` — the Typer app. Tests import this for ``CliRunner().invoke(app, ...)``.
- ``main()`` — entry point used by the ``multi-scenario`` console script
  (declared in ``pyproject.toml``) and by ``cli/__main__.py`` for
  ``python -m multi_scenario.cli`` (used by the OVH container's job command,
  see :class:`OvhJobConfig.command_template`).
"""

from ._app import app

# Side-effect imports register each command on ``app``. Order doesn't matter;
# Typer derives the subcommand name from the function name (or ``name=`` arg).
# Keep this list alphabetic so subcommands appear in a predictable order in
# ``--help``.
from . import (  # noqa: F401
    consolidate,
    eval_run,
    regenerate_videos,
    resume,
    run,
    sweep,
    upload_code,
    validate,
    version,
)


def main() -> None:
    """Entry point used by the ``multi-scenario`` console script."""
    app()
