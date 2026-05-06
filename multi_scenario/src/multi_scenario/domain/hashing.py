"""Pure hashing utilities used to compute Provenance fields.

Stdlib only — no torch, no vmas, no benchmarl. The Provenance model that
holds the resulting hashes lives in ``domain/models/provenance.py``.
"""

import hashlib
import json
from pathlib import Path


def compute_config_hash(config: dict) -> str:
    """sha256 of a canonical-JSON encoding of ``config`` (sort_keys=True).

    Key-order invariant: programmatically-reordered configs produce the same
    hash, so reorderings don't falsely flag results as stale.
    """
    canonical = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def compute_code_hash(file_paths: list[Path] | list[str]) -> str:
    """sha256 over the concatenated bytes of ``file_paths`` in the given order.

    Order matters: reordering the input list changes the hash. Callers should
    pass a curated, deterministically-sorted list (typically the
    ``hashed_source_files`` array on the resulting Provenance).
    """
    digest = hashlib.sha256()
    for path in file_paths:
        with open(path, "rb") as handle:
            digest.update(handle.read())
    return f"sha256:{digest.hexdigest()}"
