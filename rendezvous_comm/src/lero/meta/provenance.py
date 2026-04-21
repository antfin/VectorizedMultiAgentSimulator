"""Prompt-template provenance tracking for LERO-MP.

Every mutation the outer loop emits lands as a fresh directory under
``src/lero/prompts/<new_version>/`` with a populated ``meta.yaml``
recording parent, target slot, rationale, and frozen-slot hashes.
The outer loop never edits an existing version — templates are
immutable once written, so lineage is unambiguous.

This module is pure file I/O + hashing. No LLM calls live here — the
meta-LLM lives in ``mutation.py`` and calls into here to materialize
its proposed edit.
"""

import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _resolve_prompts_dir(prompts_dir: Optional[Path]) -> Path:
    """Return the caller's override or the CURRENT module-level default.

    Lazy lookup (not a mutable default argument) is what lets tests
    ``monkeypatch.setattr(provenance, "_PROMPTS_DIR", tmp_path)`` and
    have every helper below pick up the new root. Default-argument
    binding happens at def time and would ignore the patch.
    """
    return prompts_dir if prompts_dir is not None else _PROMPTS_DIR


def sha256_text(text: str) -> str:
    """Canonical hex digest for slot content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_meta(
    version: str, prompts_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Parse meta.yaml for a given template version. Empty dict if none."""
    if yaml is None:
        raise RuntimeError("PyYAML is required for provenance I/O")
    root = _resolve_prompts_dir(prompts_dir)
    path = root / version / "meta.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def recompute_frozen_hashes(
    version: str, prompts_dir: Optional[Path] = None,
) -> Dict[str, str]:
    """Return the sha256 of every slot flagged ``frozen: true``.

    Useful for re-pinning after an intentional edit: write the new
    hashes back into meta.yaml with ``write_frozen_hashes``.
    """
    root = _resolve_prompts_dir(prompts_dir)
    meta = load_meta(version, root)
    slots = meta.get("initial_user_slots", [])
    out: Dict[str, str] = {}
    for slot in slots:
        if slot.get("frozen", False):
            path = root / version / slot["file"]
            out[slot["name"]] = sha256_text(path.read_text())
    return out


def write_frozen_hashes(
    version: str,
    hashes: Dict[str, str],
    prompts_dir: Optional[Path] = None,
) -> None:
    """Persist frozen-slot hashes into ``meta.yaml``."""
    if yaml is None:
        raise RuntimeError("PyYAML is required for provenance I/O")
    root = _resolve_prompts_dir(prompts_dir)
    meta_path = root / version / "meta.yaml"
    meta = load_meta(version, root)
    meta.setdefault("frozen_hashes", {})
    meta["frozen_hashes"].update(hashes)
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False))


def propose_version_name(parent: str, outer_iter: int) -> str:
    """Generate a deterministic name for the next template version.

    Format: ``<parent>_mp_<NNN>`` where NNN is zero-padded outer-iter.
    E.g. ``v2_fewshot_modular_mp_001``. Never collides — if the
    directory already exists, callers should bump outer_iter.
    """
    return f"{parent}_mp_{outer_iter:03d}"


def materialize_mutation(
    parent_version: str,
    new_version: str,
    slot_edits: Dict[str, str],
    rationale: str,
    mutation_operator: str = "rewrite_slot",
    generated_by: str = "",
    prompts_dir: Optional[Path] = None,
) -> Path:
    """Write a new prompt-template directory derived from ``parent``.

    Args:
        parent_version: Directory name of the template being mutated.
        new_version: Directory name to create. Must not already exist.
        slot_edits: Mapping of slot name → new content. Slots not in
            this dict are copied verbatim from the parent.
        rationale: Why this mutation was proposed (meta-LLM's output).
        mutation_operator: One of ``rewrite_slot``, ``add_example``,
            ``crossover`` etc. — bookkeeping only.
        generated_by: Meta-LLM model identifier (for audit trail).
        prompts_dir: Root prompts directory (override for tests).

    Returns:
        Absolute Path of the newly-written directory.

    Raises:
        FileExistsError: if ``new_version`` already exists.
        ValueError: if a slot_edit name is not declared in parent meta.yaml
                    OR targets a frozen slot.
    """
    root = _resolve_prompts_dir(prompts_dir)
    parent_dir = root / parent_version
    new_dir = root / new_version
    if not parent_dir.exists():
        raise FileNotFoundError(f"Parent template not found: {parent_dir}")
    if new_dir.exists():
        raise FileExistsError(f"Version already exists: {new_dir}")

    parent_meta = load_meta(parent_version, root)
    declared_slots = parent_meta.get("initial_user_slots", [])
    declared_names = {s["name"] for s in declared_slots}
    frozen_names = {s["name"] for s in declared_slots if s.get("frozen", False)}

    # Validate edits before any filesystem change.
    for name in slot_edits:
        if name not in declared_names:
            raise ValueError(
                f"Cannot edit slot {name!r}: not declared in "
                f"{parent_version}/meta.yaml (known: {sorted(declared_names)})."
            )
        if name in frozen_names:
            raise ValueError(
                f"Cannot edit slot {name!r}: it is marked frozen in "
                f"{parent_version}/meta.yaml. The fairness contract is "
                f"not a mutation target; see plan §4.3 / §7.3."
            )

    # Copy parent → new directory, then overlay the edits.
    shutil.copytree(parent_dir, new_dir)

    for slot in declared_slots:
        if slot["name"] in slot_edits:
            (new_dir / slot["file"]).write_text(slot_edits[slot["name"]])

    # Update meta.yaml with mutation provenance + refreshed frozen hashes.
    new_meta = dict(parent_meta)
    new_meta["version"] = new_version
    new_meta["parent"] = parent_version
    new_meta["date"] = datetime.now(tz=timezone.utc).date().isoformat()
    new_meta["mutation"] = {
        "operator": mutation_operator,
        "target_slots": sorted(slot_edits.keys()),
        "rationale": rationale,
    }
    new_meta["provenance"] = {
        "generated_by": generated_by,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(
            timespec="seconds",
        ),
    }
    # Frozen hashes: parent's values still pinned (we never touched the
    # slots). Recompute to be safe — if the hash changed, materialize
    # will raise at render time via the loader's FrozenSlotMismatch.
    new_meta["frozen_hashes"] = recompute_frozen_hashes(new_version, root)

    (new_dir / "meta.yaml").write_text(
        yaml.safe_dump(new_meta, sort_keys=False)
    )
    return new_dir


def lineage(
    version: str, prompts_dir: Optional[Path] = None,
) -> List[str]:
    """Return the chain of ancestors back to the root, most-recent first.

    ``lineage("v2_fewshot_modular_mp_002")`` → e.g.
    ``["v2_fewshot_modular_mp_002", "v2_fewshot_modular_mp_001",
    "v2_fewshot_modular", "v2_fewshot"]``.
    """
    root = _resolve_prompts_dir(prompts_dir)
    chain: List[str] = []
    cur: Optional[str] = version
    seen: set = set()
    while cur and cur not in seen:
        chain.append(cur)
        seen.add(cur)
        meta = load_meta(cur, root)
        cur = meta.get("parent")
    return chain
