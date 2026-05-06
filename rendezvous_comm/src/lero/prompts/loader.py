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

    def render(
        self,
        template_name: str,
        output_spec_variant: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Render a prompt template with variable substitution.

        Uses Python's string.Template ($variable syntax) for simplicity.
        Missing variables are left as-is (safe_substitute).

        When ``template_name == "initial_user.txt"`` and the template
        version's ``meta.yaml`` declares ``initial_user_slots``, the
        initial_user body is assembled by concatenating the slot files
        in declared order (with frozen-slot hash verification). This is
        the hook that LERO-MP's meta-prompt optimizer uses to edit
        individual slots without touching the rest of the template.

        ``output_spec_variant`` (LERO-MP v3 §3.2): when set to
        ``"both" | "reward_only" | "obs_only"``, the ``output_spec``
        slot is read from ``output_spec_<variant>.txt`` instead of the
        default ``output_spec.txt``. Lets the outer loop drop the
        unused function's signature from the inner-LLM prompt when
        evolve_reward or evolve_observation is False (~40% token
        savings on obs-only runs).

        v9 (2026-05-02): if meta.yaml declares ``task_domain: <name>``,
        the loader reads ``task_domains/<name>.yaml`` and merges its
        fields (``task_framing``, ``coordination_challenges_bullets``)
        into the substitution context. Caller-supplied ``**kwargs``
        win over task_domain values (so per-run ``$n_agents`` etc.
        substitute into ``task_framing`` correctly).
        """
        td_kwargs = self._task_domain_kwargs(kwargs)
        full_kwargs = {**td_kwargs, **kwargs}
        if template_name == "initial_user.txt":
            assembled = self._assemble_initial_user(
                output_spec_variant=output_spec_variant,
            )
            if assembled is not None:
                # Two-pass substitute: the task_framing pulled from the
                # task_domain YAML may itself contain $n_agents etc.,
                # which only resolve once we know the per-run kwargs.
                tpl = string.Template(assembled).safe_substitute(**full_kwargs)
                return string.Template(tpl).safe_substitute(**full_kwargs)
        if template_name not in self._cache:
            path = self.template_dir / template_name
            if not path.exists():
                raise FileNotFoundError(f"Template not found: {path}")
            self._cache[template_name] = string.Template(path.read_text())
        out = self._cache[template_name].safe_substitute(**full_kwargs)
        return string.Template(out).safe_substitute(**full_kwargs)

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

    def task_domain(self) -> Optional[Dict]:
        """Return the parsed task_domain YAML (or None if not declared).

        v9: meta.yaml may declare ``task_domain: <name>`` pointing at
        ``task_domains/<name>.yaml``. The dict is returned verbatim;
        callers (loader / meta-strategist) decide which fields to use.

        Resolves the YAML relative to ``self.template_dir.parent`` so the
        lookup works whether or not the module-level ``_PROMPTS_DIR`` has
        been redirected (the v9 outer loop redirects it to the run's
        per-outer prompt copies).
        """
        meta = self._load_meta()
        td_name = (meta or {}).get("task_domain")
        if not td_name:
            return None
        td_path = self.template_dir.parent / "task_domains" / f"{td_name}.yaml"
        if not td_path.exists() and yaml is not None:
            # Fallback: try the original module-level prompts dir
            # (in case template_dir was redirected to a copy that
            # doesn't include task_domains/).
            module_root = Path(__file__).parent
            td_path = module_root / "task_domains" / f"{td_name}.yaml"
        if not td_path.exists() or yaml is None:
            return None
        return yaml.safe_load(td_path.read_text()) or {}

    def _task_domain_kwargs(self, caller_kwargs: Dict) -> Dict[str, str]:
        """Derive substitution kwargs from the task_domain YAML.

        Currently produces:
          - task_framing: verbatim from YAML
          - coordination_challenges_bullets: bulleted markdown list

        Returns {} when no task_domain is declared.
        """
        td = self.task_domain()
        if not td:
            return {}
        out: Dict[str, str] = {}
        framing = td.get("task_framing", "")
        if framing:
            out["task_framing"] = framing.rstrip()
        ch_list = td.get("coordination_challenges") or []
        if ch_list:
            out["coordination_challenges_bullets"] = "\n".join(
                f"- {c}" for c in ch_list
            )
        return out

    def _load_meta(self) -> Optional[Dict]:
        if self._meta is not None:
            return self._meta
        path = self.template_dir / "meta.yaml"
        if not path.exists() or yaml is None:
            self._meta = {}
            return self._meta
        self._meta = yaml.safe_load(path.read_text()) or {}
        return self._meta

    def _assemble_initial_user(
        self,
        output_spec_variant: Optional[str] = None,
    ) -> Optional[str]:
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
            # v3 §3.2: swap output_spec file for the conditional variant
            file_name = slot["file"]
            if slot["name"] == "output_spec" and output_spec_variant in (
                "both",
                "reward_only",
                "obs_only",
            ):
                variant_name = f"output_spec_{output_spec_variant}.txt"
                variant_path = self.template_dir / variant_name
                if variant_path.exists():
                    file_name = variant_name
            fpath = self.template_dir / file_name
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
