"""On-disk LLM-call cache for reproducibility replay (LERO-MP v3 §5.1).

Four modes:
  - "off"         : no reads, no writes
  - "read_write"  : reads + writes (default when cache enabled)
  - "read_only"   : reads only (for bit-exact replay of a past run)
  - "write_only"  : writes only (to seed a cache without replaying)

Key = sha256(model, messages, temperature, seed, response_format).
Value = the raw assistant text.

Usage:
    from lero.llm_cache import LLMCache
    cache = LLMCache(mode="read_write", root="/path/to/cache")
    llm = LLMClient(llm_config, cache=cache)

Environment override: ``LERO_LLM_CACHE_MODE`` takes precedence over
the constructor mode, so you can flip replay semantics for a single
run without editing the config.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal, Optional

_log = logging.getLogger("rendezvous.lero.cache")

CacheMode = Literal["off", "read_write", "read_only", "write_only"]
_VALID_MODES = ("off", "read_write", "read_only", "write_only")


class LLMCache:
    """Disk-backed key-value cache for LLM completions."""

    def __init__(
        self,
        mode: CacheMode = "off",
        root: Optional[Path] = None,
    ):
        env_mode = os.environ.get("LERO_LLM_CACHE_MODE")
        if env_mode:
            if env_mode not in _VALID_MODES:
                raise ValueError(
                    f"LERO_LLM_CACHE_MODE={env_mode!r} must be one of "
                    f"{_VALID_MODES}"
                )
            mode = env_mode  # type: ignore[assignment]
        if mode not in _VALID_MODES:
            raise ValueError(f"mode={mode!r} must be one of {_VALID_MODES}")
        self.mode: CacheMode = mode
        self.root = Path(
            root
            or os.environ.get(
                "LERO_LLM_CACHE_DIR",
                str(Path.home() / ".cache" / "lero_llm"),
            )
        )
        if self.mode != "off":
            self.root.mkdir(parents=True, exist_ok=True)
        _log.info("LLMCache: mode=%s root=%s", self.mode, self.root)

    def read(self, key: str) -> Optional[str]:
        if self.mode not in ("read_write", "read_only"):
            return None
        path = self.root / f"{key}.txt"
        if not path.exists():
            if self.mode == "read_only":
                _log.warning(
                    "LLMCache MISS in read_only mode: key=%s… "
                    "(run with read_write first to populate)",
                    key[:10],
                )
            return None
        return path.read_text()

    def write(self, key: str, value: str) -> None:
        if self.mode not in ("read_write", "write_only"):
            return
        path = self.root / f"{key}.txt"
        path.write_text(value)

    def clear(self) -> int:
        """Delete all cached entries. Returns number of files deleted."""
        n = 0
        if not self.root.exists():
            return 0
        for p in self.root.glob("*.txt"):
            p.unlink()
            n += 1
        return n
