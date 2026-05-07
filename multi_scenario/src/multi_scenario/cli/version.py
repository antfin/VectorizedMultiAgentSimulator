"""``multi-scenario version`` — prints the package version."""

import typer

from multi_scenario import __version__

from ._app import app


@app.command()
def version() -> None:
    """Print the multi_scenario package version.

    Also forces typer's multi-command mode so ``run`` stays a real subcommand.
    """
    typer.echo(__version__)
