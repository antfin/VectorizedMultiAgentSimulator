"""LLM interface via LiteLLM.

LiteLLM provides a unified OpenAI-compatible interface to 100+ providers:
  - Anthropic: model="claude-sonnet-4-6"
  - OpenAI:    model="gpt-4o"
  - OVH/custom: model="openai/my-model", api_base="https://..."

All providers use the same chat completions API.
API keys are read from env vars (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)
or passed explicitly via LLMConfig.api_key.

LERO-MP v3 additions:
  - ``generate_structured(messages, schema)`` validates output against
    a Pydantic model, falling back to regex + model.model_validate_json
    if the provider doesn't support JSON-schema.
  - ``openai_seed`` + ``system_fingerprint`` logging — reproducibility
    (§5.2/§5.3 of lero_metaprompt_v3_plan.md).
  - Per-candidate seed: the ``seed_suffix`` arg lets the caller derive a
    unique OpenAI ``seed`` per candidate in an n>1 generate batch, so
    sibling candidates are not identical at temperature=1.0.
  - LLM-call cache (optional): when a ``LLMCache`` is passed in, every
    completion's (model, messages, temperature, seed) → response is
    cached on disk. Four modes: off / read_write / read_only / write_only.

Future DSPy migration: replace litellm.completion() with dspy.LM().
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from dotenv import load_dotenv

from .config import LLMConfig

# Auto-load .env from rendezvous_comm/ (gitignored, never committed)
_ENV_PATH = Path(__file__).parent.parent.parent / ".env"
load_dotenv(_ENV_PATH, override=False)

_log = logging.getLogger("rendezvous.lero")

# Default context window for custom endpoints where auto-detect fails
_DEFAULT_CONTEXT_WINDOW = 16_000


def _pydantic_to_json_schema(schema_cls) -> Dict[str, Any]:
    """Convert a Pydantic BaseModel to an OpenAI json_schema payload.

    OpenAI's structured-outputs API requires:
      {"type": "json_schema",
       "json_schema": {"name": "...", "schema": {...}, "strict": true}}
    """
    name = schema_cls.__name__
    raw_schema = schema_cls.model_json_schema()
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": raw_schema,
            "strict": False,  # strict=True rejects Optional unless required
        },
    }


def _cache_key_for(call_kwargs: Dict[str, Any]) -> str:
    """sha256 hash over (model, messages, temperature, seed, response_format).

    Keyed via canonical JSON so equivalent payloads hash identically.
    """
    payload = {
        "model": call_kwargs.get("model"),
        "temperature": call_kwargs.get("temperature"),
        "seed": call_kwargs.get("seed"),
        "messages": call_kwargs.get("messages"),
        "response_format": call_kwargs.get("response_format"),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class LLMClient:
    """Unified LLM client using LiteLLM."""

    def __init__(self, config: LLMConfig, cache=None):
        self.config = config
        self.cache = cache  # Optional LLMCache (see llm_cache.py)
        try:
            import litellm

            self._litellm = litellm
        except ImportError:
            raise ImportError(
                "LiteLLM is required for LERO. Install it with:\n"
                "  pip install litellm"
            )
        litellm.suppress_debug_info = True

        # Track last-seen system_fingerprint for drift detection.
        self.last_system_fingerprint: Optional[str] = None

        # Resolve context window once at init
        self._context_window = self._resolve_context_window()
        _log.info(
            "LLM: %s (context=%d tokens, max_output=%s tokens)",
            config.model,
            self._context_window,
            config.max_tokens if config.max_tokens is not None else "provider-default",
        )

    @property
    def context_window(self) -> int:
        return self._context_window

    # ── public API ───────────────────────────────────────────────

    def generate(
        self,
        messages: List[Dict[str, str]],
        n: int = 1,
        seed_base: Optional[int] = None,
    ) -> List[str]:
        """Generate *n* independent completions.

        When ``seed_base`` is provided, each of the *n* calls gets a
        distinct derived seed (``seed_base + i``) so sibling candidates
        do not collide at temperature=1.0 with structured outputs.
        This is required for §5.2 reproducibility — without per-call
        seed derivation, ``llm.generate(n=3)`` returns the same text
        three times when an OpenAI seed is set.
        """
        responses: List[str] = []
        for i in range(n):
            call_seed = None
            if seed_base is not None:
                call_seed = (seed_base + i) % (2**31)
            text = self._call_with_retry(messages, seed=call_seed)
            responses.append(text)
            if n > 1:
                _log.info("  candidate %d/%d generated", i + 1, n)
        return responses

    def generate_structured(
        self,
        messages: List[Dict[str, str]],
        schema: "Type[Any]",
        seed: Optional[int] = None,
        fallback_parser=None,
    ):
        """Generate a single response validated against a Pydantic schema.

        Args:
            messages: chat completion messages.
            schema: Pydantic BaseModel subclass.
            seed: OpenAI seed for reproducibility.
            fallback_parser: optional callable ``(text) -> schema`` used
                when JSON-schema endpoint support is unavailable.
                Typically this is a regex-based parser from the v2 era.

        Returns:
            An instance of ``schema`` (Pydantic-validated).

        Raises:
            ValueError if the response cannot be parsed or validated.

        Implementation: tries LiteLLM's ``response_format={"type":
        "json_schema", ...}`` first. If the provider rejects the call
        (ProviderNotSupportedError, BadRequestError, etc.) AND a
        ``fallback_parser`` is provided, falls back to raw generate +
        parser. This lets v3 ship on providers with partial structured-
        output support (OVH-hosted custom models) without blocking on
        the happy path.
        """
        # Try strict JSON-schema mode first
        try:
            text = self._call_with_retry(
                messages,
                seed=seed,
                response_format=_pydantic_to_json_schema(schema),
            )
            return schema.model_validate_json(text)
        except Exception as primary_err:
            # If the provider rejected structured-output OR the JSON
            # was valid-but-Pydantic-invalid: fall back to raw + parser.
            if fallback_parser is None:
                raise
            _log.warning(
                "Structured-output path failed (%s: %s). "
                "Falling back to regex parser.",
                type(primary_err).__name__,
                str(primary_err)[:200],
            )
            raw = self._call_with_retry(messages, seed=seed)
            parsed = fallback_parser(raw)
            # Accept either a dict (to be validated) or an already-built
            # schema instance from legacy parsers.
            if isinstance(parsed, schema):
                return parsed
            if isinstance(parsed, dict):
                return schema.model_validate(parsed)
            # Legacy parsers return dataclass-like objects with a
            # ``.to_dict()`` method — use that if available.
            if hasattr(parsed, "to_dict"):
                return schema.model_validate(parsed.to_dict())
            raise TypeError(
                f"fallback_parser returned {type(parsed).__name__}; "
                f"expected dict or {schema.__name__}."
            )

    # ── internal ─────────────────────────────────────────────────

    def _resolve_context_window(self) -> int:
        """Determine context window for the configured model.

        Priority:
        1. Explicit context_window in LLMConfig (for custom endpoints)
        2. LiteLLM's model registry (knows Claude, GPT, etc.)
        3. Conservative default (16K) for unknown models
        """
        # 1. Explicit config
        if self.config.context_window is not None:
            return self.config.context_window

        # 2. LiteLLM model registry
        try:
            info = self._litellm.get_model_info(self.config.model)
            if info and info.get("max_input_tokens"):
                # max_input_tokens + max_output_tokens = context window
                max_in = info["max_input_tokens"]
                max_out = info.get("max_output_tokens", self.config.max_tokens)
                return max_in + max_out
        except Exception:
            pass

        # 3. Conservative default for custom/unknown endpoints
        _log.warning(
            "Could not determine context window for '%s'. "
            "Using conservative default %d tokens. "
            "Set 'context_window' in llm config for accuracy.",
            self.config.model,
            _DEFAULT_CONTEXT_WINDOW,
        )
        return _DEFAULT_CONTEXT_WINDOW

    def _call_with_retry(
        self,
        messages: List[Dict[str, str]],
        seed: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        last_err = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                return self._call(
                    messages,
                    seed=seed,
                    response_format=response_format,
                )
            except Exception as e:
                last_err = e
                # Don't retry schema-rejection errors — caller needs to
                # see them to trigger its fallback path.
                msg = str(e).lower()
                if "response_format" in msg or "json_schema" in msg:
                    raise
                wait = self.config.retry_delay * attempt
                _log.warning(
                    "LLM call failed (attempt %d/%d): %s. " "Retrying in %.1fs ...",
                    attempt,
                    self.config.max_retries,
                    e,
                    wait,
                )
                time.sleep(wait)
        raise RuntimeError(
            f"LLM call failed after {self.config.max_retries} attempts: " f"{last_err}"
        )

    def _call(
        self,
        messages: List[Dict[str, str]],
        seed: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        # Check prompt fits in context window
        total_chars = sum(len(m.get("content", "")) for m in messages)
        approx_prompt_tokens = total_chars // 4
        max_out = self.config.max_tokens or 4096  # estimate for warning
        total_needed = approx_prompt_tokens + max_out
        headroom = self._context_window - total_needed

        if headroom < 0:
            _log.warning(
                "Prompt (~%d tokens) + output (~%d) = ~%d total, "
                "EXCEEDS context window (%d) by ~%d tokens.",
                approx_prompt_tokens,
                max_out,
                total_needed,
                self._context_window,
                -headroom,
            )
        elif headroom < 2000:
            _log.warning(
                "Prompt (~%d tokens) + output (~%d) = ~%d total, "
                "only ~%d tokens headroom in context window (%d).",
                approx_prompt_tokens,
                max_out,
                total_needed,
                headroom,
                self._context_window,
            )

        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        if self.config.max_tokens is not None:
            kwargs["max_tokens"] = self.config.max_tokens

        if self.config.api_base:
            kwargs["api_base"] = self.config.api_base

        if seed is not None:
            kwargs["seed"] = int(seed) % (2**31)
        if response_format is not None:
            kwargs["response_format"] = response_format

        api_key = self.config.api_key
        if not api_key and self.config.api_base:
            api_key = os.environ.get("OVH_AI_ENDPOINTS_ACCESS_TOKEN")
        if api_key:
            kwargs["api_key"] = api_key

        # LLM-cache read path (Step 7). Keyed on the full call payload so
        # sibling candidates (distinct seed) don't collide.
        cache_key = None
        if self.cache is not None:
            cache_key = _cache_key_for(kwargs)
            cached = self.cache.read(cache_key)
            if cached is not None:
                _log.info(
                    "LLM cache HIT (key=%s…, len=%d)",
                    cache_key[:10],
                    len(cached),
                )
                return cached

        response = self._litellm.completion(**kwargs)
        # Log system_fingerprint for drift detection (§5.3)
        fp = getattr(response, "system_fingerprint", None)
        if fp:
            if (
                self.last_system_fingerprint is not None
                and fp != self.last_system_fingerprint
            ):
                _log.warning(
                    "system_fingerprint drift: %s → %s",
                    self.last_system_fingerprint,
                    fp,
                )
            self.last_system_fingerprint = fp
        content = response.choices[0].message.content
        if content is None:
            _log.warning("LLM returned None content, treating as empty")
            content = ""

        if self.cache is not None and cache_key is not None:
            self.cache.write(cache_key, content)
        return content
