"""F9.1 — :class:`InMemoryCostLedger` + :class:`FilesystemCostLedger`.

Both implement the :class:`multi_scenario.domain.ports.CostLedger` Protocol.
Tests use the in-memory variant; production uses the filesystem-backed one.

The filesystem ledger is **host-wide** (default at ``~/.multi_scenario/
cost_ledger.jsonl``) so the rolling-window cap holds across processes,
sweeps, restarts. Every record is one JSONL line; reads filter the
in-memory list by timestamp window. Old entries (> 31 days) are pruned
on read so the file doesn't grow unboundedly across years of runs.

Atomic writes: use ``open(..., "a")`` line-append on POSIX (atomic for
< PIPE_BUF). The ledger is append-only — no compaction race possible.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── shared dataclass --------------------------------------------------


@dataclass
class _LedgerEntry:
    """One spend record. Internal — Pydantic shape isn't needed (no I/O
    boundary; JSON serialisation is hand-written for atomic appends)."""

    timestamp: datetime
    cost_eur: float
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def to_jsonl(self) -> str:
        return (
            json.dumps(
                {
                    "ts": self.timestamp.isoformat(),
                    "cost_eur": self.cost_eur,
                    "model": self.model,
                    "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": self.completion_tokens,
                },
                separators=(",", ":"),
            )
            + "\n"
        )

    @classmethod
    def from_jsonl(cls, line: str) -> "_LedgerEntry | None":
        try:
            d = json.loads(line)
            return cls(
                timestamp=datetime.fromisoformat(d["ts"]),
                cost_eur=float(d["cost_eur"]),
                model=str(d["model"]),
                prompt_tokens=int(d.get("prompt_tokens", 0)),
                completion_tokens=int(d.get("completion_tokens", 0)),
            )
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            # Corrupt line — skip. Defensive: a half-written line from a
            # crashed process shouldn't break future ledger reads.
            return None


_PRUNE_WINDOW = timedelta(days=31)
_DEFAULT_LEDGER_PATH = Path.home() / ".multi_scenario" / "cost_ledger.jsonl"


# ── In-memory (tests) ------------------------------------------------


@dataclass
class InMemoryCostLedger:
    """Pure-Python ledger — used by tests + as a fallback when the FS
    ledger isn't writable (e.g., read-only /home)."""

    _entries: list[_LedgerEntry] = field(default_factory=list)
    #: Injectable clock so tests can simulate the rolling-window cutoff
    #: without ``time.sleep``. ``None`` means use real wall clock.
    _now: "callable | None" = None

    def _utcnow(self) -> datetime:
        return self._now() if self._now is not None else datetime.now(timezone.utc)

    def record(
        self,
        *,
        cost_eur: float,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        self._entries.append(
            _LedgerEntry(
                timestamp=self._utcnow(),
                cost_eur=cost_eur,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )

    def sum_window(self, window: timedelta) -> float:
        cutoff = self._utcnow() - window
        return sum(e.cost_eur for e in self._entries if e.timestamp >= cutoff)


# ── Filesystem (production) ------------------------------------------


class FilesystemCostLedger:
    """JSONL-backed ledger persisted at :attr:`path`.

    The default location (``~/.multi_scenario/cost_ledger.jsonl``) is
    **host-wide** and shared across all multi_scenario runs on this
    machine — this is the whole point of switching from per-run to
    rolling-window caps. Override via the ``MULTI_SCENARIO_COST_LEDGER``
    env var (used by tests + sandboxed CI).
    """

    def __init__(self, path: Path | None = None, *, now: "callable | None" = None):
        env_override = os.environ.get("MULTI_SCENARIO_COST_LEDGER")
        if path is not None:
            self._path = Path(path)
        elif env_override:
            self._path = Path(env_override)
        else:
            self._path = _DEFAULT_LEDGER_PATH
        self._now = now or (lambda: datetime.now(timezone.utc))
        # Lazily ensure parent dir exists on first write — cheap and
        # avoids creating ``~/.multi_scenario/`` on import.
        self._dir_ensured = False

    def _ensure_dir(self) -> None:
        if self._dir_ensured:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._dir_ensured = True

    def record(
        self,
        *,
        cost_eur: float,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        self._ensure_dir()
        entry = _LedgerEntry(
            timestamp=self._now(),
            cost_eur=cost_eur,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        # POSIX append is atomic for writes < PIPE_BUF (4 KiB on Linux,
        # ≥ 512 B on macOS). One JSONL line is well within that — no
        # cross-process locking needed.
        with self._path.open("a", encoding="utf-8") as f:
            f.write(entry.to_jsonl())

    def sum_window(self, window: timedelta) -> float:
        if not self._path.is_file():
            return 0.0
        cutoff = self._now() - window
        prune_cutoff = self._now() - _PRUNE_WINDOW
        total = 0.0
        keep: list[str] = []
        try:
            with self._path.open("r", encoding="utf-8") as f:
                for line in f:
                    entry = _LedgerEntry.from_jsonl(line)
                    if entry is None:
                        continue
                    if entry.timestamp >= prune_cutoff:
                        keep.append(line)
                    if entry.timestamp >= cutoff:
                        total += entry.cost_eur
        except OSError:
            # File gone between the is_file() check and open() — race
            # with another process; treat as empty ledger.
            return 0.0
        # Prune-on-read: rewrite the file with only entries in the
        # 31-day window. Cheap because LERO produces O(100) entries/day.
        # Skip the rewrite if nothing was pruned (no-op on a fresh file).
        try:
            current_size = self._path.stat().st_size
        except OSError:
            current_size = -1
        new_blob = "".join(keep)
        if len(new_blob.encode("utf-8")) != current_size:
            tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
            with tmp_path.open("w", encoding="utf-8") as f:
                f.write(new_blob)
            os.replace(tmp_path, self._path)
        return total
