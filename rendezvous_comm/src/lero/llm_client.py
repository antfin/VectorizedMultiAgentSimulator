"""LLM interface via LiteLLM.

LiteLLM provides a unified OpenAI-compatible interface to 100+ providers:
  - Anthropic: model="claude-sonnet-4-6"
  - OpenAI:    model="gpt-4o"
  - OVH/custom: model="openai/my-model", api_base="https://..."

All providers use the same chat completions API.
API keys are read from env vars (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)
or passed explicitly via LLMConfig.api_key.

Future DSPy migration: replace litellm.completion() with dspy.LM().
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

from .config import LLMConfig

# Auto-load .env from rendezvous_comm/ (gitignored, never committed)
_ENV_PATH = Path(__file__).parent.parent.parent / ".env"
load_dotenv(_ENV_PATH, override=False)

_log = logging.getLogger("rendezvous.lero")

# Default context window for custom endpoints where auto-detect fails
_DEFAULT_CONTEXT_WINDOW = 16_000


class LLMClient:
    """Unified LLM client using LiteLLM."""

    def __init__(self, config: LLMConfig):
        self.config = config
        try:
            import litellm
            self._litellm = litellm
        except ImportError:
            raise ImportError(
                "LiteLLM is required for LERO. Install it with:\n"
                "  pip install litellm"
            )
        litellm.suppress_debug_info = True

        # Resolve context window once at init
        self._context_window = self._resolve_context_window()
        _log.info(
            "LLM: %s (context=%d tokens, max_output=%d tokens)",
            config.model, self._context_window, config.max_tokens,
        )

    @property
    def context_window(self) -> int:
        return self._context_window

    # ── public API ───────────────────────────────────────────────

    def generate(
        self,
        messages: List[Dict[str, str]],
        n: int = 1,
    ) -> List[str]:
        """Generate *n* independent completions."""
        responses: List[str] = []
        for i in range(n):
            text = self._call_with_retry(messages)
            responses.append(text)
            if n > 1:
                _log.info("  candidate %d/%d generated", i + 1, n)
        return responses

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
            self.config.model, _DEFAULT_CONTEXT_WINDOW,
        )
        return _DEFAULT_CONTEXT_WINDOW

    def _call_with_retry(self, messages: List[Dict[str, str]]) -> str:
        last_err = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                return self._call(messages)
            except Exception as e:
                last_err = e
                wait = self.config.retry_delay * attempt
                _log.warning(
                    "LLM call failed (attempt %d/%d): %s. "
                    "Retrying in %.1fs ...",
                    attempt, self.config.max_retries, e, wait,
                )
                time.sleep(wait)
        raise RuntimeError(
            f"LLM call failed after {self.config.max_retries} attempts: "
            f"{last_err}"
        )

    def _call(self, messages: List[Dict[str, str]]) -> str:
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
                approx_prompt_tokens, max_out,
                total_needed, self._context_window, -headroom,
            )
        elif headroom < 2000:
            _log.warning(
                "Prompt (~%d tokens) + output (~%d) = ~%d total, "
                "only ~%d tokens headroom in context window (%d).",
                approx_prompt_tokens, max_out,
                total_needed, headroom, self._context_window,
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

        api_key = self.config.api_key
        if not api_key and self.config.api_base:
            api_key = os.environ.get("OVH_AI_ENDPOINTS_ACCESS_TOKEN")
        if api_key:
            kwargs["api_key"] = api_key

        response = self._litellm.completion(**kwargs)
        content = response.choices[0].message.content
        if content is None:
            _log.warning("LLM returned None content, treating as empty")
            return ""
        return content
