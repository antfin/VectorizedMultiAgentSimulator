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


class LLMClient:
    """Unified LLM client using LiteLLM."""

    def __init__(self, config: LLMConfig):
        self.config = config
        # Import check at init time — fail fast if not installed
        try:
            import litellm
            self._litellm = litellm
        except ImportError:
            raise ImportError(
                "LiteLLM is required for LERO. Install it with:\n"
                "  pip install litellm"
            )
        # Suppress litellm's own verbose logging
        litellm.suppress_debug_info = True

    # ── public API ───────────────────────────────────────────────

    def generate(
        self,
        messages: List[Dict[str, str]],
        n: int = 1,
    ) -> List[str]:
        """Generate *n* independent completions.

        Args:
            messages: conversation history [{role, content}, ...]
            n: number of independent completions (candidates)

        Returns:
            List of response text strings.
        """
        responses: List[str] = []
        for i in range(n):
            text = self._call_with_retry(messages)
            responses.append(text)
            if n > 1:
                _log.info("  candidate %d/%d generated", i + 1, n)
        return responses

    # ── internal ─────────────────────────────────────────────────

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
        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        # Custom endpoint (OVH, local, etc.)
        if self.config.api_base:
            kwargs["api_base"] = self.config.api_base

        # API key: explicit config > OVH env var > provider default env var
        api_key = self.config.api_key
        if not api_key and self.config.api_base:
            # Custom endpoint — try OVH token as fallback
            api_key = os.environ.get("OVH_AI_ENDPOINTS_ACCESS_TOKEN")
        if api_key:
            kwargs["api_key"] = api_key

        response = self._litellm.completion(**kwargs)
        return response.choices[0].message.content
