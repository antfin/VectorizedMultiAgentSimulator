"""F9.1 — :class:`DiskCacheDecorator` over :class:`LlmClient`.

Optional response cache, ``cache_enabled=false`` by default per the
reproducibility-locked decision. Useful for development iteration (a
re-run of the orchestrator on the same prompts is free) but disabled
on reproducibility runs so we don't accidentally pin yesterday's
LLM behaviour into today's experiment.

Cache key = SHA-256 over (model, messages, seed, response_format,
``n`` index for sibling completions). On a hit the cached
:class:`LlmCompletion` is returned with ``usage.estimated_cost_usd=0``
because the API wasn't actually contacted — the cost-cap decorator
that wraps this then records €0 for the call, so cache hits don't
debit the rolling-window budget.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from multi_scenario.domain.lero import LlmCompletion, LlmUsage
from multi_scenario.domain.ports import LlmClient


def _cache_key(
    *,
    model: str,
    messages: list[dict[str, str]],
    seed: int | None,
    response_format: dict[str, Any] | None,
    sibling_idx: int,
) -> str:
    """Stable SHA-256 over the call signature."""
    payload = {
        "model": model,
        "messages": messages,
        "seed": seed,
        "response_format": response_format,
        "sibling": sibling_idx,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class DiskCacheDecorator:
    """JSON-on-disk cache wrapping any :class:`LlmClient`.

    Layout: one ``<cache_dir>/<sha256>.json`` file per cached completion.
    Atomic write-rename so a crashed write doesn't poison the cache.
    """

    def __init__(self, inner: LlmClient, *, model: str, cache_dir: Path):
        self._inner = inner
        self._model = model
        self._cache_dir = Path(cache_dir)

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        n: int = 1,
        seed: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> list[LlmCompletion]:
        out: list[LlmCompletion | None] = [None] * n
        misses: list[int] = []
        for i in range(n):
            key = _cache_key(
                model=self._model,
                messages=messages,
                seed=seed,
                response_format=response_format,
                sibling_idx=i,
            )
            hit = self._read(key)
            if hit is not None:
                # Zero cost on cache hit — the cap decorator wrapping
                # this should record €0 (no API call happened).
                out[i] = hit.model_copy(update={"usage": LlmUsage()})
            else:
                misses.append(i)

        if misses:
            fresh = self._inner.generate(
                messages=messages,
                n=len(misses),
                seed=seed,
                response_format=response_format,
            )
            for sibling_idx, completion in zip(misses, fresh):
                key = _cache_key(
                    model=self._model,
                    messages=messages,
                    seed=seed,
                    response_format=response_format,
                    sibling_idx=sibling_idx,
                )
                self._write(key, completion)
                out[sibling_idx] = completion

        return [c for c in out if c is not None]

    def _read(self, key: str) -> LlmCompletion | None:
        path = self._cache_dir / f"{key}.json"
        if not path.is_file():
            return None
        try:
            return LlmCompletion.model_validate_json(path.read_text("utf-8"))
        except (OSError, ValueError):
            # Corrupt cache file — fall through to miss; the writer
            # will overwrite it. Defensive: a partial write from a
            # crashed prior run shouldn't poison subsequent runs.
            return None

    def _write(self, key: str, completion: LlmCompletion) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_dir / f"{key}.json"
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(completion.model_dump_json(), encoding="utf-8")
        tmp.replace(path)  # atomic on POSIX
