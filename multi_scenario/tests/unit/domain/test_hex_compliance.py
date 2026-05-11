"""F9.9 — Hex-architecture invariant: domain layer stays pure.

Scans every Python file under ``src/multi_scenario/domain/`` for
forbidden imports. A regression here means someone added a torch /
LiteLLM / VMAS / BenchMARL / Streamlit dependency at the domain layer,
which would (a) bloat the domain test surface and (b) break the
"domain is the contract" architectural promise.

Lives in ``tests/unit/`` because it's pure stdlib (just walks the
filesystem + AST-parses each file).
"""

# pylint: disable=missing-function-docstring

import ast
from pathlib import Path

import pytest


_DOMAIN_ROOT = Path(__file__).resolve().parents[3] / "src" / "multi_scenario" / "domain"

#: Top-level module names the domain layer is FORBIDDEN to import.
#: Anything I/O-heavy, ML-framework-specific, or UI goes here.
_FORBIDDEN = frozenset(
    {
        "torch",
        "litellm",
        "vmas",
        "benchmarl",
        "torchrl",
        "streamlit",
        "boto3",
        "imageio",
        "pyglet",
        "jinja2",
    }
)


def _python_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]


def _imported_top_levels(source: str) -> set[str]:
    tree = ast.parse(source)
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.add(node.module.split(".")[0])
    return out


@pytest.mark.parametrize(
    "py_file",
    _python_files(_DOMAIN_ROOT),
    ids=lambda p: str(p.relative_to(_DOMAIN_ROOT)),
)
def test_domain_file_imports_no_forbidden_top_levels(py_file: Path) -> None:
    """Each file under ``domain/`` imports nothing forbidden."""
    src = py_file.read_text(encoding="utf-8")
    imported = _imported_top_levels(src)
    leaks = imported & _FORBIDDEN
    assert not leaks, (
        f"{py_file.relative_to(_DOMAIN_ROOT)} imports forbidden top-level "
        f"module(s) at the domain layer: {sorted(leaks)}. The domain "
        f"layer is pure Python + pydantic + dataclasses; move framework "
        f"types behind a Protocol in ``domain/ports/`` and put the "
        f"concrete implementation under ``adapters/``."
    )


def test_at_least_one_domain_file_exists():
    """Smoke: the scan covers a non-trivial number of files (so the
    parametrize isn't silently empty)."""
    assert len(_python_files(_DOMAIN_ROOT)) >= 10
