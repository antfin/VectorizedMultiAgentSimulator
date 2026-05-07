"""Leaf module that owns the shared Typer ``app`` instance.

Lives separately from ``cli/__init__.py`` so command modules can import
``app`` *without* importing the package init (which itself imports those
command modules to register them). This breaks what would otherwise be a
``cli/__init__.py`` ↔ ``cli.<command>`` cyclic-import warning under pylint.
"""

import typer

app = typer.Typer(help="multi_scenario CLI")
