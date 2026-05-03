"""v9 memory — append-only JSONL of meta-LLM CoT + outcomes.

Per §5 of docs/v9_plan.md. After every outer iter the v9 outer loop
appends one row capturing:

  - the strategy CoT that was active
  - the predicted M1/M6 from the success_signature
  - the actual analyzer facts + diagnosis label
  - the meta-LLM's post-hoc reflection_chain_of_thought

Before each subsequent reflect_decide call, the outer loop reads the
last N=3 rows and embeds them in the user prompt. The full file
remains on disk for post-hoc analysis.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_log = logging.getLogger("rendezvous.lero.v9.memory")


@dataclass
class MemoryRow:
    outer_idx: int
    ts: str
    strategy_name: str
    predicted: Dict[str, Any] = field(default_factory=dict)
    actual: Dict[str, Any] = field(default_factory=dict)
    delta: Dict[str, Any] = field(default_factory=dict)
    chain_of_thought: Dict[str, Any] = field(default_factory=dict)
    post_hoc_reflection: Dict[str, Any] = field(default_factory=dict)


class MemoryStore:
    """Append-only JSONL store with last-N reader.

    Caller workflow per outer iter:
      mem.append(MemoryRow(...))
      rows = mem.read_recent(n=3)   # before next reflect call
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Optional[List[Dict]] = None

    def append(self, row: MemoryRow) -> None:
        line = json.dumps(asdict(row), default=str)
        with self.path.open("a") as f:
            f.write(line + "\n")
        self._cache = None  # invalidate
        _log.info(
            "memory.append outer=%d strategy=%s actual_M1=%s",
            row.outer_idx,
            row.strategy_name,
            row.actual.get("M1"),
        )

    def read_all(self) -> List[Dict]:
        if self._cache is not None:
            return list(self._cache)
        if not self.path.exists():
            self._cache = []
            return []
        rows: List[Dict] = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                _log.warning("memory: skipped malformed line")
        self._cache = rows
        return list(rows)

    def read_recent(self, n: int = 3) -> List[Dict]:
        rows = self.read_all()
        if n <= 0:
            return []
        return rows[-n:]

    def __len__(self) -> int:
        return len(self.read_all())
