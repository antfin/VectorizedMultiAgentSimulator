"""Prompt template loading and rendering.

Design note: This is intentionally simple (string.Template + file I/O).
When migrating to DSPy, replace this module with DSPy Signatures.
The file-based structure maps 1:1 to DSPy modules:
  - system.txt     -> Signature docstring
  - initial_user.txt -> InputField descriptions
  - feedback.txt   -> ChainOfThought module prompt
"""

import string
from pathlib import Path
from typing import Dict


_PROMPTS_DIR = Path(__file__).parent


class PromptLoader:
    """Load and render versioned prompt templates."""

    def __init__(self, version: str = "v1"):
        self.template_dir = _PROMPTS_DIR / version
        if not self.template_dir.exists():
            raise FileNotFoundError(
                f"Prompt version directory not found: {self.template_dir}"
            )
        self._cache: Dict[str, string.Template] = {}

    def render(self, template_name: str, **kwargs) -> str:
        """Render a prompt template with variable substitution.

        Uses Python's string.Template ($variable syntax) for simplicity.
        Missing variables are left as-is (safe_substitute).
        """
        if template_name not in self._cache:
            path = self.template_dir / template_name
            if not path.exists():
                raise FileNotFoundError(f"Template not found: {path}")
            self._cache[template_name] = string.Template(path.read_text())
        return self._cache[template_name].safe_substitute(**kwargs)

    def load_raw(self, template_name: str) -> str:
        """Load a template without substitution."""
        path = self.template_dir / template_name
        return path.read_text()
