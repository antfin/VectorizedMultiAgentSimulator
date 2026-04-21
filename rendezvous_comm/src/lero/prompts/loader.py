"""Prompt template loading and rendering.

Design note: This is intentionally simple (string.Template + file I/O).
When migrating to DSPy, replace this module with DSPy Signatures.
The file-based structure maps 1:1 to DSPy modules:
  - system.txt     -> Signature docstring
  - initial_user.txt -> InputField descriptions
  - feedback.txt   -> ChainOfThought module prompt

LERO-MP extension (2026-04-21): a template directory may ship with a
``meta.yaml`` that declares ``initial_user_slots`` — an ordered list of
slot files that are concatenated to form ``initial_user.txt`` at render
time. When no slots are declared, the loader falls back to the
monolithic ``initial_user.txt`` (backward-compatible with v1/v2/…).
"""

import hashlib
import string
from pathlib import Path
from typing import Dict, List, Optional

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover — yaml is ubiquitous in this repo
    yaml = None


_PROMPTS_DIR = Path(__file__).parent


class FrozenSlotMismatch(RuntimeError):
    """Raised when a slot marked ``frozen: true`` in meta.yaml has a
    content hash different from the ``frozen_hashes`` entry.

    Used as the meta-prompt fairness guard: the outer loop cannot
    rewrite a frozen slot (e.g. fairness.txt) without explicitly
    re-pinning the hash, which is a loud, reviewable action.
    """


class PromptLoader:
    """Load and render versioned prompt templates."""

    def __init__(self, version: str = "v1"):
        self.template_dir = _PROMPTS_DIR / version
        if not self.template_dir.exists():
            raise FileNotFoundError(
                f"Prompt version directory not found: {self.template_dir}"
            )
        self._cache: Dict[str, string.Template] = {}
        self._meta: Optional[Dict] = None

    # ── public API ────────────────────────────────────────────────

    def render(self, template_name: str, **kwargs) -> str:
        """Render a prompt template with variable substitution.

        Uses Python's string.Template ($variable syntax) for simplicity.
        Missing variables are left as-is (safe_substitute).

        When ``template_name == "initial_user.txt"`` and the template
        version's ``meta.yaml`` declares ``initial_user_slots``, the
        initial_user body is assembled by concatenating the slot files
        in declared order (with frozen-slot hash verification). This is
        the hook that LERO-MP's meta-prompt optimizer uses to edit
        individual slots without touching the rest of the template.
        """
        if template_name == "initial_user.txt":
            assembled = self._assemble_initial_user()
            if assembled is not None:
                return string.Template(assembled).safe_substitute(**kwargs)
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

    def slot_names(self) -> List[str]:
        """List the slot names declared in meta.yaml, in declared order.

        Returns [] if this template does not use slot decomposition.
        """
        meta = self._load_meta()
        if not meta:
            return []
        return [s["name"] for s in meta.get("initial_user_slots", [])]

    def slot_text(self, name: str) -> str:
        """Read the current text of a named slot."""
        meta = self._load_meta()
        for s in (meta or {}).get("initial_user_slots", []):
            if s["name"] == name:
                return (self.template_dir / s["file"]).read_text()
        raise KeyError(f"Slot not declared in meta.yaml: {name}")

    def frozen_slot_names(self) -> List[str]:
        """List slots flagged ``frozen: true`` — never edited by the
        meta-prompt optimizer.
        """
        meta = self._load_meta()
        return [
            s["name"]
            for s in (meta or {}).get("initial_user_slots", [])
            if s.get("frozen", False)
        ]

    # ── internals ─────────────────────────────────────────────────

    def _load_meta(self) -> Optional[Dict]:
        if self._meta is not None:
            return self._meta
        path = self.template_dir / "meta.yaml"
        if not path.exists() or yaml is None:
            self._meta = {}
            return self._meta
        self._meta = yaml.safe_load(path.read_text()) or {}
        return self._meta

    def _assemble_initial_user(self) -> Optional[str]:
        """Concatenate slot files into a single initial_user body.

        Returns None if the template does not declare ``initial_user_slots``
        (falls back to monolithic initial_user.txt).
        """
        meta = self._load_meta()
        slots = (meta or {}).get("initial_user_slots")
        if not slots:
            return None

        frozen_hashes = (meta or {}).get("frozen_hashes", {}) or {}
        parts: List[str] = []
        for slot in slots:
            fpath = self.template_dir / slot["file"]
            if not fpath.exists():
                raise FileNotFoundError(
                    f"Slot file not found: {fpath} "
                    f"(declared in {self.template_dir}/meta.yaml)"
                )
            text = fpath.read_text()
            if slot.get("frozen", False):
                pinned = frozen_hashes.get(slot["name"], "")
                if pinned:
                    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    if h != pinned:
                        raise FrozenSlotMismatch(
                            f"Frozen slot '{slot['name']}' content hash "
                            f"{h[:12]}… does not match pinned "
                            f"{pinned[:12]}… in meta.yaml. Either revert "
                            f"the slot or re-pin the hash explicitly."
                        )
            parts.append(text)
        return "".join(parts)
