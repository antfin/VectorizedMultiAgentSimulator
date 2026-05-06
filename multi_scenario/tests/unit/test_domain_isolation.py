"""F1.12 architecture lint: the domain layer is free of adapter-only deps.

Walks every ``.py`` under ``src/multi_scenario/domain/``, parses with ``ast``,
and asserts none of them import torch / vmas / benchmarl / streamlit / boto3.
This test is the executable form of the §1.2 / §2 hexagonal invariant.
"""

import ast
from pathlib import Path

FORBIDDEN_MODULES = {"vmas", "benchmarl", "streamlit", "boto3", "torch"}


def _imported_top_levels(file_path: Path) -> set[str]:
    """Top-level package names imported anywhere in `file_path` (incl. TYPE_CHECKING blocks)."""
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_no_forbidden_imports_in_domain(repo_root: Path):
    """Every .py under src/multi_scenario/domain/ avoids the forbidden modules."""
    domain_dir = repo_root / "src" / "multi_scenario" / "domain"
    files = sorted(domain_dir.rglob("*.py"))
    assert files, f"no domain files discovered under {domain_dir}"

    violations: dict[str, set[str]] = {}
    for file_path in files:
        forbidden = _imported_top_levels(file_path) & FORBIDDEN_MODULES
        if forbidden:
            violations[str(file_path.relative_to(repo_root))] = forbidden

    assert not violations, f"forbidden imports in domain layer: {violations}"
